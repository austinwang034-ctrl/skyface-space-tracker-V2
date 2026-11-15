"""FastAPI backend for Skyface Space Tracker.   Run:  uvicorn server:app --port 8000   then open http://localhost:8000

Heavy work (archive search, downloads, plate solving) runs as a background job; the browser polls for progress and
then pulls the movie frames, tracks, plate-solving results and neighbourhood data as plain JSON/PNG.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
import uuid
import warnings
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.time import Time
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from spherex_viewer import (
    crosscheck,
    identify,
    logbook,
    measure,
    neighborhood,
    neossat,
    objects,
    overlays,
    pipeline,
    protect,
    render,
    signatures,
    smallbodies,
)
from spherex_viewer.align import offset_to_pix, pix_to_offset
from spherex_viewer.config import (
    ALLOWED_ORIGINS,
    APP_NAME,
    APP_OWNER,
    APP_VERSION,
    BAD_FLAG_BITS,
    BANDS,
    FRAME_ANCESTORS,
    JOBS_PER_MINUTE,
    MAX_FRAMES,
    OWN_HOSTS,
    RELEASES,
    RESOLVING_POWER,
    SPHEREX_ACKNOWLEDGEMENT,
)
from spherex_viewer.frames import FLAG_NAMES, decode_flags
from spherex_viewer.targets import (
    TARGETS,
    TargetUnavailable,
    builtin_near,
    custom_target,
    parse_coords,
    resolve_target,
)

WEB = Path(__file__).parent / "web"
MAX_JOBS, JOB_TTL_S = 24, 3600
ROOT = Path(__file__).parent
app = FastAPI(title=f"{APP_NAME} API", version=APP_VERSION)
JOB_LIMITER = protect.RateLimiter(JOBS_PER_MINUTE)


@app.middleware("http")
async def guard(request: Request, call_next):
    """Refuse to serve as another website's back end, and forbid framing of the full page."""
    path = request.url.path
    ours = protect.own_hosts(request.headers, OWN_HOSTS)
    if path.startswith("/api/") and protect.is_cross_site(request.headers, ours, ALLOWED_ORIGINS):
        return JSONResponse(
            {"detail": f"This API serves {APP_NAME} only. Other sites may embed it with ?embed=1."},
            status_code=403,
        )
    response = await call_next(request)
    response.headers.update(protect.security_headers(path, request.url.query, FRAME_ANCESTORS))
    return response


class JobRequest(BaseModel):
    source: Literal["irsa", "demo"] = "irsa"
    target: str | None = None  # preset name; omit for custom RA/Dec
    ra: str | None = None  # degrees or sexagesimal
    dec: str | None = None
    fov_arcmin: float = Field(5.0, ge=1.0, le=15.0)
    bands: list[str] = []
    max_frames: int = Field(100, ge=10, le=300)
    oversample: int = Field(2, ge=1, le=3)
    refine: bool = True
    epoch: float = Field(
        2000.0, ge=1900, le=2100
    )  # epoch of the entered RA/Dec (SIMBAD: 2000, Gaia: 2016)
    identify: bool = True  # look the object up in SIMBAD and track it by its proper motion
    refresh: bool = False  # ignore any cached result and re-query the archive for the newest data


class IdentifyRequest(BaseModel):
    ra: str
    dec: str
    epoch: float = Field(2000.0, ge=1900, le=2100)
    fov_arcmin: float = Field(5.0, ge=1.0, le=15.0)
    demo: bool = False


@dataclass
class Job:
    id: str
    key: str
    req: JobRequest
    state: str = "queued"  # queued | running | done | error
    progress: float = 0.0
    message: str = "Queued"
    error: str | None = None
    bundle: pipeline.Bundle | None = None
    created: float = field(default_factory=time.time)
    cubes: dict = field(default_factory=dict)  # stretch mode -> (N, ny, nx) in [0, 1]
    views: dict = field(default_factory=dict)  # cached display ranges and static-sky images
    overlays: dict = field(default_factory=dict)  # catalogue name -> overlay sources
    track: tuple | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    def public(self) -> dict:
        return {
            "id": self.id,
            "state": self.state,
            "progress": round(self.progress, 3),
            "message": self.message,
            "error": self.error,
            "n_frames": len(self.bundle.frames) if self.bundle else None,
        }


JOBS: "OrderedDict[str, Job]" = OrderedDict()
KEYS: dict[str, str] = {}
GLOCK = threading.Lock()
POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="job")


def run_job(job: Job) -> None:
    job.state = "running"

    def say(msg: str, frac: float) -> None:
        job.message, job.progress = msg, float(frac)

    try:
        r = job.req
        if r.source == "demo":
            target = TARGETS["Barnard's Star"]
            center = pipeline.patch_center(target)
        elif r.target:
            say("Resolving target…", 0.02)
            target = resolve_target(TARGETS[r.target])
            center = pipeline.patch_center(target)
        else:
            entered = parse_coords(r.ra or "", r.dec or "")
            target = None
            if r.identify:
                try:
                    say("Identifying the object…", 0.02)
                    m = identify.identify(entered.ra.deg, entered.dec.deg, r.epoch).match
                    target = identify.target_from_object(m) if m else None
                except Exception:  # noqa: BLE001  (SIMBAD unreachable: carry on with the raw coordinates)
                    target = None
            # SIMBAD down or silent: the built-in test objects are still recognised by position.
            target = target or builtin_near(entered, r.epoch) or custom_target(entered)
            center = pipeline.patch_center(target)
        job.bundle = pipeline.build_bundle(
            r.source,
            target,
            center,
            r.fov_arcmin,
            r.bands or [target.band],
            min(r.max_frames, MAX_FRAMES),
            r.oversample,
            r.refine,
            say,
        )
        job.state, job.progress, job.message = "done", 1.0, "Done"
    except (TargetUnavailable, ValueError, RuntimeError) as exc:
        job.state, job.error = "error", str(exc)
    except Exception as exc:  # noqa: BLE001  (network errors etc.: report, never crash the worker)
        job.state, job.error = "error", f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------------------------- helpers
def done_job(job_id: str) -> Job:
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown or expired job.")
    if job.state == "error":
        raise HTTPException(409, job.error or "Job failed.")
    if job.state != "done":
        raise HTTPException(409, "Job is still running.")
    return job


def offsets_for(b: pipeline.Bundle, coords: SkyCoord) -> np.ndarray:
    x, y = b.wcs_out.world_to_pixel(coords)
    e, n = pix_to_offset(x, y, b.shape, b.scale_arcsec)
    return np.column_stack([e, n])


def cube01(job: Job, mode: str) -> np.ndarray:
    with job.lock:
        if mode not in job.cubes:
            job.cubes[mode] = render.stretch(job.bundle.cube, per_frame=(mode == "frame"))
        return job.cubes[mode]


def band_group(job: Job, k: int) -> list[int]:
    """Frames from the same detector as frame k, in time order (only these may be combined with it)."""
    frames = job.bundle.frames
    return [i for i, f in enumerate(frames) if f.band == frames[k].band]


def view_cache(job: Job, key, compute):
    """Compute-once storage for display ranges and static-sky images."""
    with job.lock:
        if key not in job.views:
            job.views[key] = compute()
        return job.views[key]


def frame_rgba(job: Job, k: int, stretch: str, view: str, window: int) -> np.ndarray:
    """One tracker image as RGBA.

    view "image": the frame itself (grey). "minus_static": the frame minus the median of every
    frame from its detector, so only what changed is left. "minus_previous": the frame minus the
    one before it. Differences are drawn red (brighter) and blue (fainter).
    """
    cube = job.bundle.cube
    group = band_group(job, k)
    image = render.windowed(cube, group, k, window)
    if view == "image":
        if stretch == "frame":
            lo, hi = render.limits(image)
        else:
            lo, hi = view_cache(job, "limits", lambda: render.limits(cube))
        return render.gray_rgba(image, lo, hi)
    band = job.bundle.frames[k].band
    static = view_cache(job, ("static", band), lambda: render.static_sky(cube, group))
    if view == "minus_static":
        diff = image - static
    else:
        at = group.index(k)
        previous = render.windowed(cube, group, group[at - 1], window) if at else image
        diff = image - previous
    limit = view_cache(
        job,
        ("diff_limit", band),
        lambda: render.limits(np.abs(cube[group] - static), 0.0, 99.8)[1],
    )
    return render.diverging_rgba(diff, limit)


def records(df) -> list[dict]:
    return json.loads(df.to_json(orient="records"))  # NaN -> null, numpy types -> plain JSON


def get_track(job: Job):
    with job.lock:
        if job.track is None:
            b = job.bundle
            tr = measure.measure_track(b.frames, b.target)
            tr["meas_e"] = np.nan
            tr["meas_n"] = np.nan
            if "ra" in tr:
                m = tr.ok.astype(bool) & tr.ra.notna()
                if m.any():
                    o = offsets_for(
                        b, SkyCoord(tr.ra[m].to_numpy(), tr.dec[m].to_numpy(), unit="deg")
                    )
                    tr.loc[m, "meas_e"], tr.loc[m, "meas_n"] = o[:, 0], o[:, 1]
            # Fit a straight-line motion even when no catalogue motion is known: that is how an
            # uncatalogued mover would be noticed.
            fit = measure.fit_linear_motion(tr)
            job.track = (tr, {k: float(v) for k, v in fit.items()} if fit else None)
        return job.track


# ---------------------------------------------------------------------------------------------- endpoints
@app.get("/api/config")
def config():
    return {
        "name": APP_NAME,
        "version": APP_VERSION,
        "owner": APP_OWNER,
        "max_frames": MAX_FRAMES,
        "bands": {k: list(v) for k, v in BANDS.items()},
        "resolving_power": RESOLVING_POWER,
        "releases": RELEASES,
        "acknowledgement": SPHEREX_ACKNOWLEDGEMENT,
        "targets": [
            {
                "name": t.name,
                "kind": t.kind,
                "band": t.band,
                "notes": t.notes,
                "source": t.source,
                "needs_lookup": not t.has_position,
            }
            for t in TARGETS.values()
        ],
    }


@app.post("/api/jobs", status_code=202)
def create_job(req: JobRequest, request: Request):
    visitor = request.client.host if request.client else "unknown"
    if not JOB_LIMITER.allow(visitor):
        raise HTTPException(429, "Too many trackers started at once. Wait a minute and try again.")
    if req.source == "irsa":
        bad = [b for b in req.bands if b not in BANDS]
        if bad:
            raise HTTPException(422, f"Unknown detector bands: {', '.join(bad)}")
        if req.target and req.target not in TARGETS:
            raise HTTPException(422, "Unknown target preset.")
        if not req.target:
            if not (req.ra and req.dec):
                raise HTTPException(422, "Choose a target preset or enter both RA and Dec.")
            try:
                parse_coords(req.ra, req.dec)
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from exc
    key = hashlib.sha1(req.model_dump_json().encode()).hexdigest()
    with GLOCK:
        old = JOBS.get(KEYS.get(key, ""))
        if (
            old
            and not req.refresh
            and old.state != "error"
            and time.time() - old.created < JOB_TTL_S
        ):
            return old.public()  # identical request: reuse the result
        job = Job(id=uuid.uuid4().hex[:12], key=key, req=req)
        JOBS[job.id], KEYS[key] = job, job.id
        now = time.time()
        for jid in list(JOBS):  # bound memory: drop expired jobs, then the oldest finished ones
            old_job = JOBS[jid]
            finished = old_job.state in ("done", "error") and jid != job.id
            if finished and (now - old_job.created > JOB_TTL_S or len(JOBS) > MAX_JOBS):
                if KEYS.get(old_job.key) == jid:
                    del KEYS[old_job.key]
                del JOBS[jid]
    POOL.submit(run_job, job)
    return job.public()


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown or expired job.")
    return job.public()


@app.get("/api/jobs/{job_id}/tracker")
@app.get("/api/jobs/{job_id}/movie", include_in_schema=False)  # old name kept as an alias
def tracker_info(job_id: str):
    job = done_job(job_id)
    b = job.bundle
    tg = b.target
    cs = [tg.coord_at(f.time) if tg.has_motion else tg.coord_ref() for f in b.frames]
    pred = offsets_for(b, SkyCoord([c.ra for c in cs], [c.dec for c in cs]))

    def wl(v):
        return float(v) if np.isfinite(v) else None

    return {
        "synthetic": b.synthetic,
        "refine_note": b.refine_note,
        "n_found": b.n_found,
        "center": {"ra": float(b.center.ra.deg), "dec": float(b.center.dec.deg)},
        "grid": {"nx": b.shape[1], "ny": b.shape[0], "scale_arcsec": b.scale_arcsec},
        "native_pixel_arcsec": float(b.frames[0].pixscale_arcsec),
        "bands": sorted({f.band for f in b.frames}),
        "first_obs_utc": b.frames[0].time.iso,
        "latest_obs_utc": b.frames[-1].time.iso,
        "target": {
            "name": tg.name,
            "kind": tg.kind,
            "source": tg.source,
            "notes": tg.notes,
            "has_motion": tg.has_motion,
            "pmra": tg.pmra,
            "pmdec": tg.pmdec,
        },
        "skipped": [m for _, m in b.failed][:20],
        "n_skipped": len(b.failed),
        "frames": [
            {
                "k": i,
                "label": f.label(),
                "time_utc": f.time.iso,
                "mjd": f.mjd,
                "band": f.band,
                "wavelength_um": wl(f.wavelength_um),
                "pred": [float(pred[i, 0]), float(pred[i, 1])],
            }
            for i, f in enumerate(b.frames)
        ],
    }


@app.get("/api/jobs/{job_id}/frames/{k}.png")
def frame_png(
    job_id: str,
    k: int,
    stretch: Literal["global", "frame"] = "global",
    view: Literal["image", "minus_static", "minus_previous"] = "image",
    window: int = Query(1, ge=1, le=15),
):
    """One tracker frame, North up. `window` averages it with its neighbours in time; `view` can
    subtract the unchanging sky (or the previous frame) so that only what moved or varied is left."""
    job = done_job(job_id)
    if not 0 <= k < len(job.bundle.frames):
        raise HTTPException(404, "No such frame.")
    return Response(
        render.png_bytes(frame_rgba(job, k, stretch, view, window)),
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=3600"},
    )


@app.get("/api/jobs/{job_id}/overlay/{catalogue}")
def catalogue_overlay(job_id: str, catalogue: Literal["gaia", "wise"]):
    """Catalogued sources to draw over the tracker: offsets from the patch centre (arcsec) at a
    reference year, plus yearly motion, so each ring follows its star from frame to frame."""
    job = done_job(job_id)
    b = job.bundle
    with job.lock:
        cached = job.overlays.get(catalogue)
    if cached:
        return cached
    radius = 0.75 * max(b.shape) * b.scale_arcsec  # reaches the corners of the square patch
    mid_year = float(b.frames[len(b.frames) // 2].time.jyear)
    try:
        if catalogue == "gaia":
            stars = b.gaia
            if stars is None:
                stars = neighborhood.gaia_cone(b.center, radius / 60.0 + 1.0, 18.0)
            out = overlays.gaia_overlay(stars, b.center, mid_year, radius)
        elif b.synthetic:
            out = {"catalogue": "WISE", "sources": [], "note": "The demo sky has no WISE sources."}
        else:
            out = overlays.wise_overlay(b.center, radius)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            502, f"The {catalogue.upper()} catalogue could not be reached: {exc}"
        ) from exc
    with job.lock:
        job.overlays[catalogue] = out
    return out


@app.get("/api/jobs/{job_id}/tracker.gif")
@app.get("/api/jobs/{job_id}/movie.gif", include_in_schema=False)
def tracker_gif(job_id: str):
    job = done_job(job_id)
    b = job.bundle
    data = render.gif_bytes(
        cube01(job, "global"),
        [f.label() for f in b.frames],
        watermark=f"{APP_NAME} · "
        + ("SYNTHETIC DEMO" if b.synthetic else "SPHEREx / NASA-IPAC IRSA"),
    )
    return Response(
        data,
        media_type="image/gif",
        headers={"Content-Disposition": 'attachment; filename="skyface_tracker.gif"'},
    )


@app.get("/api/jobs/{job_id}/metadata")
def frame_metadata(job_id: str):
    """Per-frame quality metadata, including the pixel flags at the target position (names for documented bits)."""
    b = done_job(job_id).bundle
    tg = b.target
    rows = []
    for i, f in enumerate(b.frames):
        pred = tg.coord_at(f.time) if tg.has_motion else tg.coord_ref()
        x, y = (float(v) for v in f.wcs.world_to_pixel(pred))
        flag = f.flag_at(x, y)
        rows.append(
            {
                "k": i,
                "time_utc": f.time.iso,
                "band": f.band,
                "wavelength_um": float(f.wavelength_um) if np.isfinite(f.wavelength_um) else None,
                "obs_key": f.obs.obs_key,
                "version": f.obs.version,
                "release": f.obs.release,
                "time_source": f.time_source,
                "finite_pct": float(100 * np.isfinite(f.data).mean()),
                "flagged_pct": None if f.flagged_fraction() is None else 100 * f.flagged_fraction(),
                "flag_at_target": flag,
                "flag_names": decode_flags(flag) if flag else [],
            }
        )
    return {
        "rows": rows,
        "flag_legend": {str(k): v for k, v in FLAG_NAMES.items()},
        "masked_bits": [k for k in range(32) if BAD_FLAG_BITS & (1 << k)],
    }


@app.get("/api/jobs/{job_id}/spectrum")
def spectrum_here(
    job_id: str,
    east: float = Query(..., ge=-3000, le=3000),
    north: float = Query(..., ge=-3000, le=3000),
):
    """Spectrum extracted on the fly at a position picked in the tracker (arcsec offsets, +East / +North from the patch centre)."""
    b = done_job(job_id).bundle
    px, py = offset_to_pix(east, north, b.shape, b.scale_arcsec)
    coord = b.wcs_out.pixel_to_world(float(px), float(py))
    df = measure.spectrum_at(b.frames, coord)
    return {
        "east": east,
        "north": north,
        "ra": float(coord.ra.deg),
        "dec": float(coord.dec.deg),
        "rows": records(df),
    }


def compare_arrays(job: Job, split: float, band: str | None):
    """Early stack, late stack and late-minus-early difference (frames are chronological; optional single detector band)."""
    b = job.bundle
    idx = [i for i, f in enumerate(b.frames) if band in (None, "", f.band)]
    if len(idx) < 2:
        raise HTTPException(422, "Need at least two frames (try another band).")
    s = int(min(max(round(len(idx) * split), 1), len(idx) - 1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN pixels at the edges are expected
        early, late = np.nanmean(b.cube[idx[:s]], axis=0), np.nanmean(b.cube[idx[s:]], axis=0)
    return early, late, late - early, idx, s


@app.get("/api/jobs/{job_id}/compare/meta")
def compare_meta(job_id: str, split: float = Query(0.5, ge=0.1, le=0.9), band: str | None = None):
    job = done_job(job_id)
    _, _, _, idx, s = compare_arrays(job, split, band)
    f = job.bundle.frames
    return {
        "n_early": s,
        "n_late": len(idx) - s,
        "early_from": f[idx[0]].time.iso,
        "early_to": f[idx[s - 1]].time.iso,
        "late_from": f[idx[s]].time.iso,
        "late_to": f[idx[-1]].time.iso,
        "bands": sorted({x.band for x in f}),
    }


@app.get("/api/jobs/{job_id}/compare/{part}.png")
def compare_png(
    job_id: str,
    part: Literal["early", "late", "diff"],
    split: float = Query(0.5, ge=0.1, le=0.9),
    band: str | None = None,
):
    """Stacking improves signal-to-noise; subtracting the early stack from the late one makes movers appear as a blue/red pair."""
    job = done_job(job_id)
    early, late, diff, _, _ = compare_arrays(job, split, band)
    both = np.concatenate([early[np.isfinite(early)], late[np.isfinite(late)]])
    if part == "diff":
        d = diff[np.isfinite(diff)]
        rgba = render.diverging_rgba(diff, float(np.percentile(np.abs(d), 99.5)) if d.size else 1.0)
    else:
        lo, hi = np.percentile(both, [1, 99.7]) if both.size else (0, 1)
        rgba = render.gray_rgba(early if part == "early" else late, float(lo), float(hi))
    return Response(
        render.png_bytes(rgba), media_type="image/png", headers={"Cache-Control": "no-store"}
    )


def detector_stack(job: Job, band: str) -> np.ndarray:
    """Every frame from one detector averaged together. More frames, less noise."""
    b = job.bundle
    idx = [i for i, f in enumerate(b.frames) if f.band == band]
    if not idx:
        raise HTTPException(404, f"This tracker has no frames from {band}.")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN pixels at the edges are expected
        return np.nanmean(b.cube[idx], axis=0)


def _stretched(stack: np.ndarray) -> np.ndarray:
    """One stack as 0-255 brightness, using the same stretch as the grey images."""
    good = stack[np.isfinite(stack)]
    lo, hi = np.percentile(good, [1, 99.7]) if good.size else (0.0, 1.0)
    return render.gray_rgba(stack, float(lo), float(hi))


@app.get("/api/jobs/{job_id}/detectors")
def detectors(job_id: str):
    """Which of the six SPHEREx detectors this tracker holds, for the per-detector stacks."""
    frames = done_job(job_id).bundle.frames
    out = []
    for band in sorted({f.band for f in frames}):
        waves = [f.wavelength_um for f in frames if f.band == band and np.isfinite(f.wavelength_um)]
        out.append(
            {
                "band": band,
                "short": band.replace("SPHEREx-", ""),
                "n_frames": sum(f.band == band for f in frames),
                "range_um": list(BANDS.get(band, (None, None))),
                "observed_um": [float(min(waves)), float(max(waves))] if waves else None,
            }
        )
    return {"detectors": out, "colour": len(out) >= 3}


@app.get("/api/jobs/{job_id}/detectors/colour.png")
def detector_colour(job_id: str):
    """An infrared colour picture: the longest-wavelength detector in red, the shortest in blue,
    one in between in green. Needs frames from at least three detectors."""
    job = done_job(job_id)
    bands = sorted({f.band for f in job.bundle.frames})
    if len(bands) < 3:
        raise HTTPException(422, "A colour picture needs frames from at least three detectors.")
    blue, green, red = bands[0], bands[len(bands) // 2], bands[-1]
    rgba = _stretched(detector_stack(job, red)).copy()
    rgba[..., 1] = _stretched(detector_stack(job, green))[..., 0]
    rgba[..., 2] = _stretched(detector_stack(job, blue))[..., 0]
    return Response(render.png_bytes(rgba), media_type="image/png")


@app.get("/api/jobs/{job_id}/detectors/{band}.png")
def detector_png(job_id: str, band: str):
    job = done_job(job_id)
    full = band if band.startswith("SPHEREx-") else f"SPHEREx-{band}"
    return Response(render.png_bytes(_stretched(detector_stack(job, full))), media_type="image/png")


@app.get("/api/signatures")
def signature_list():
    """The chemical and temperature bands a visitor can pick, with what each one reveals."""
    return {"signatures": [sig.as_dict() for sig in signatures.SIGNATURES]}


def signature_frames(job: Job, lo: float, hi: float) -> tuple[list[int], list[int]]:
    if hi <= lo:
        raise HTTPException(422, "The longer wavelength must be larger than the shorter one.")
    return signatures.split_frames([f.wavelength_um for f in job.bundle.frames], lo, hi)


@app.get("/api/jobs/{job_id}/signature/meta")
def signature_meta(
    job_id: str,
    lo: float = Query(..., ge=0.5, le=5.5),
    hi: float = Query(..., ge=0.5, le=5.5),
):
    """Which frames fall in the chosen wavelength range, and how the target behaves there."""
    job = done_job(job_id)
    inside, beside = signature_frames(job, lo, hi)
    tr, _ = get_track(job)
    flux = tr["flux_mjy"].to_numpy(dtype=float)
    err = tr["flux_err_mjy"].to_numpy(dtype=float)
    usable = np.isfinite(flux) & np.isfinite(err) & (err > 0)
    strength = signatures.band_strength(
        flux[[i for i in inside if usable[i]]], flux[[i for i in beside if usable[i]]]
    )
    loaded = sorted({f.band.replace("SPHEREx-", "") for f in job.bundle.frames})
    needed = signatures.detectors_covering(lo, hi)
    return {
        "lo_um": lo,
        "hi_um": hi,
        "frames_in_band": [int(i) for i in inside],
        "frames_beside": [int(i) for i in beside],
        "detectors_needed": needed,
        "detectors_loaded": loaded,
        "missing_detectors": [d for d in needed if d not in loaded],
        "target": strength,
    }


@app.get("/api/jobs/{job_id}/signature/{part}.png")
def signature_png(
    job_id: str,
    part: Literal["band", "beside", "diff"],
    lo: float = Query(..., ge=0.5, le=5.5),
    hi: float = Query(..., ge=0.5, le=5.5),
):
    """Stack of the frames inside the band, of the frames just outside it, or their difference
    (red: brighter inside the band, so emitting; blue: fainter inside, so absorbing)."""
    job = done_job(job_id)
    inside, beside = signature_frames(job, lo, hi)
    wanted = inside if part == "band" else beside
    if not inside or (part != "band" and not beside):
        raise HTTPException(404, "No frames at these wavelengths in this tracker.")
    cube = job.bundle.cube
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN pixels at the edges are expected
        if part == "diff":
            diff = np.nanmean(cube[inside], axis=0) - np.nanmean(cube[beside], axis=0)
            d = diff[np.isfinite(diff)]
            limit = float(np.percentile(np.abs(d), 99.5)) if d.size else 1.0
            rgba = render.diverging_rgba(diff, limit)
        else:
            rgba = _stretched(np.nanmean(cube[wanted], axis=0))
    return Response(render.png_bytes(rgba), media_type="image/png")


@app.get("/api/jobs/{job_id}/temperature")
def colour_temperature(job_id: str):
    """Rough temperature of the target from how its brightness changes with wavelength."""
    job = done_job(job_id)
    tr, _ = get_track(job)
    estimate = signatures.colour_temperature(
        tr["wavelength_um"].to_numpy(dtype=float),
        tr["flux_mjy"].to_numpy(dtype=float),
        tr["flux_err_mjy"].to_numpy(dtype=float),
    )
    return {"estimate": estimate, "synthetic": job.bundle.synthetic}


@app.get("/api/health")
def health():
    return {"ok": True, "jobs": len(JOBS), "version": app.version}


@app.get("/api/jobs/{job_id}/track")
def track_data(job_id: str):
    job = done_job(job_id)
    tr, fit = get_track(job)
    tg = job.bundle.target
    return {
        "rows": records(tr),
        "fit": fit,
        "catalogue_pm": {"pmra": tg.pmra, "pmdec": tg.pmdec},
        "native_pixel_arcsec": float(job.bundle.frames[0].pixscale_arcsec),
    }


@app.get("/api/jobs/{job_id}/logbook")
def job_logbook(job_id: str):
    """The observer's log: what was observed, how well it lined up, and whether anything moved."""
    job = done_job(job_id)
    b = job.bundle
    tr, fit = get_track(job)
    tg = b.target
    frames = b.frames
    solved = [r for r in (b.solves or []) if r.ok]
    residuals = sorted(r.rms_after_arcsec for r in solved)
    return logbook.write_log(
        name=tg.name,
        n_frames=len(frames),
        first_date=frames[0].time.iso[:10],
        last_date=frames[-1].time.iso[:10],
        span_days=float(frames[-1].mjd - frames[0].mjd),
        bands=sorted({f.band for f in frames}),
        n_solved=len(solved) if b.solves else None,
        median_residual_arcsec=float(residuals[len(residuals) // 2]) if residuals else None,
        n_detected=int(tr.ok.astype(bool).sum()),
        fit=fit,
        catalogue=logbook.Motion(tg.pmra, tg.pmdec) if tg.has_motion else None,
        parallax_mas=tg.parallax,
        pixel_arcsec=float(frames[0].pixscale_arcsec),
        synthetic=b.synthetic,
    )


@app.get("/api/smallbodies")
def known_small_bodies(
    ra: float = Query(..., ge=0, lt=360),
    dec: float = Query(..., ge=-90, le=90),
    mjd: float = Query(..., ge=40000, le=80000),
    radius_arcmin: float = Query(15.0, ge=1, le=60),
):
    """Catalogued asteroids and comets near the patch at that time (NASA JPL)."""
    try:
        return smallbodies.search(ra, dec, mjd, radius_arcmin / 60.0)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            502, f"NASA JPL's small-body service could not be reached: {exc}"
        ) from exc


@app.get("/api/jobs/{job_id}/track.csv")
def track_csv(job_id: str):
    tr, _ = get_track(done_job(job_id))
    return Response(
        tr.to_csv(index=False),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="spherex_track.csv"'},
    )


@app.get("/api/jobs/{job_id}/solves")
def solves(job_id: str):
    b = done_job(job_id).bundle
    rs = b.solves or []
    return {
        "ran": bool(rs),
        "note": b.refine_note,
        "frames": [
            {
                "k": i,
                "time_utc": f.time.iso,
                "ok": r.ok,
                "message": r.message,
                "n_detected": r.n_detected,
                "n_matched": r.n_matched,
                "shift_arcsec": r.shift_arcsec if r.ok else None,
                "rms_after_arcsec": r.rms_after_arcsec if r.ok else None,
            }
            for i, (f, r) in enumerate(zip(b.frames, rs))
        ],
    }


@app.get("/api/jobs/{job_id}/candidates/{k}")
def candidates(job_id: str, k: int):
    b = done_job(job_id).bundle
    if not b.solves or not 0 <= k < len(b.solves):
        raise HTTPException(404, "No plate-solving result for this frame.")
    u = b.solves[k].unmatched
    return {"rows": records(u[["ra", "dec", "x", "y", "flux"]].head(25)) if len(u) else []}


@app.get("/api/jobs/{job_id}/neighborhood")
def neighbourhood(
    job_id: str,
    radius_arcmin: float = Query(8.0, ge=2, le=20),
    mag_limit: float = Query(17.0, ge=12, le=19),
    year_start: float = Query(2010, ge=1990, le=2060),
    year_end: float = Query(2040, ge=1990, le=2060),
    n_epochs: int = Query(7, ge=2, le=12),
):
    b = done_job(job_id).bundle
    if year_end <= year_start:
        raise HTTPException(422, "year_end must be after year_start.")
    try:
        tbl = (
            b.gaia_demo
            if b.synthetic
            else neighborhood.gaia_cone(b.center, radius_arcmin, mag_limit)
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(502, f"Gaia query failed: {exc}") from exc
    years = [float(y) for y in np.linspace(year_start, year_end, n_epochs)]
    E, N = [], []
    for y in years:
        dx, dy = b.center.spherical_offsets_to(neighborhood.propagate(tbl, Time(y, format="jyear")))
        E.append([float(v) for v in dx.arcsec])
        N.append([float(v) for v in dy.arcsec])
    mt = neighborhood.motion_table(tbl)
    mt["source_id"] = mt["source_id"].astype(str)  # Gaia ids exceed 2**53: keep them as text for JS
    path = None
    if b.target.has_motion:
        c = [b.target.coord_at(Time(y, format="jyear")) for y in years]
        dx, dy = b.center.spherical_offsets_to(SkyCoord([x.ra for x in c], [x.dec for x in c]))
        path = {"e": [float(v) for v in dx.arcsec], "n": [float(v) for v in dy.arcsec]}
    gmag = [
        None if not np.isfinite(g) else float(g) for g in np.asarray(tbl["phot_g_mean_mag"], float)
    ]
    pm = np.hypot(
        np.nan_to_num(np.asarray(tbl["pmra"], float)),
        np.nan_to_num(np.asarray(tbl["pmdec"], float)),
    )
    return {
        "years": years,
        "e": E,
        "n": N,
        "g": gmag,
        "pm": [float(v) for v in pm],
        "ids": [str(s) for s in tbl["source_id"]],
        "target_path": path,
        "movers": records(
            mt[["source_id", "phot_g_mean_mag", "pm_total_masyr", "parallax", "vtan_kms"]].head(15)
        ),
    }


@app.post("/api/identify")
def api_identify(req: IdentifyRequest):
    """Which object is at these coordinates, plus sky-survey images of the spot. Never fails just because SIMBAD is down."""
    try:
        entered = parse_coords(req.ra, req.dec)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    lookup_failed = False
    try:
        res = (
            identify.canned_demo()
            if req.demo
            else identify.identify(entered.ra.deg, entered.dec.deg, req.epoch)
        )
    except Exception as exc:  # noqa: BLE001
        res = identify.IdResult(
            None,
            [],
            f"SIMBAD lookup failed ({type(exc).__name__}); showing images of your coordinates only.",
        )
        lookup_failed = True
    if res.match is None and not req.demo:
        known = builtin_near(entered, req.epoch)
        if known is not None:
            res = identify.IdResult(
                identify.object_from_target(known, 0.0),
                res.nearby,
                "Recognised from this site's built-in catalogue"
                + (" (SIMBAD could not be reached)." if lookup_failed else "."),
            )
    m = res.match
    fast = bool(
        m
        and (m.pmra is not None or m.pmdec is not None)
        and np.hypot(m.pmra or 0, m.pmdec or 0) > 200
    )
    ra, dec = (m.ra, m.dec) if fast else (float(entered.ra.deg), float(entered.dec.deg))
    note = (
        "Survey images are decades old, so they are centred on the catalogue (J2000) position of this fast-moving star; it has moved since."
        if fast
        else "Images are centred on your coordinates."
    )
    return {
        "match": m.to_dict() if m else None,
        "nearby": [o.to_dict() for o in res.nearby],
        "note": res.note,
        "canned": res.canned,
        "lookup_failed": lookup_failed,
        "images": identify.reference_images(ra, dec, req.fov_arcmin),
        "image_note": note,
    }


@app.get("/api/resolve")
def resolve(name: str = Query(..., min_length=2, max_length=80)):
    """Name -> coordinates (SIMBAD), so example buttons and name lookups can fill the RA/Dec boxes."""
    try:
        return identify.resolve_name(name)
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(502, f"SIMBAD lookup failed: {exc}") from exc


# ---------------------------------------------------------------------------------------------- sky map + featured objects
@app.get("/api/sky")
def sky():
    """Everything the all-sky map labels: featured objects (positions today), bright reference stars, planets today."""
    now = Time.now()
    feats = []
    for o in objects.FEATURED:
        ra, dec = objects.position_now(
            objects.profile_target(o, online=False), now
        )  # cached/fallback only: never blocks on the network
        feats.append(
            {
                "id": o["id"],
                "name": o["name"],
                "category": o["category"],
                "headline": o["headline"],
                "ra": ra,
                "dec": dec,
                "band": o["band"],
            }
        )
    return {
        "epoch": float(now.jyear),
        "objects": feats,
        "stars": objects.star_labels(),
        "planets": objects.planet_positions(now),
    }


def _obj_or_404(obj_id: str) -> dict:
    if obj_id not in objects.BY_ID:
        raise HTTPException(404, "Unknown object.")
    return objects.BY_ID[obj_id]


@app.get("/api/objects/{obj_id}")
def object_profile(
    obj_id: str,
    y0: float = Query(2000, ge=1900, le=2100),
    y1: float = Query(2040, ge=1900, le=2100),
):
    """Researched profile, illustrative spectrum, and the object's motion path for the zoomed-in view."""
    o = _obj_or_404(obj_id)
    if y1 <= y0:
        raise HTTPException(422, "y1 must be after y0.")
    t = objects.profile_target(
        o, online=True
    )  # user-initiated: allowed to ask SIMBAD for exact coordinates
    exact = t.source.startswith("SIMBAD") or o["fallback"] is None
    return {
        "id": o["id"],
        "name": o["name"],
        "category": o["category"],
        "headline": o["headline"],
        "facts": [{"label": a, "value": b} for a, b in o["facts"]],
        "light": o["light"],
        "spherex": o["spherex"],
        "features": [{"um": a, "label": b} for a, b in o["features"]],
        "sources": [{"title": a, "url": b} for a, b in o["sources"]],
        "model": objects.model_curves(o["components"]),
        "bands": {k: list(v) for k, v in BANDS.items()},
        "path": objects.motion_path(t, y0, y1, 41, o["min_fov_deg"]),
        "position_is_exact": exact,
        "position_source": t.source,
        "tracker": {
            "ra": t.ra_deg,
            "dec": t.dec_deg,
            "epoch": t.ref_epoch,
            "fov_arcmin": o["tracker_fov"],
            "band": o["band"],
        },
        "pm_masyr": float((t.pmra**2 + t.pmdec**2) ** 0.5),
    }


@app.get("/api/neossat")
def neossat_search(
    ra: float = Query(..., ge=0, lt=360),
    dec: float = Query(..., ge=-90, le=90),
    radius_arcmin: float = Query(3.0, ge=0.5, le=30),
    max: int = Query(24, ge=1, le=60),
):
    """NEOSSat (Canadian Space Agency) images covering this position, oldest first, from CADC."""
    try:
        visits = neossat.search(ra, dec, radius_arcmin / 60.0)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            502, f"The Canadian Astronomy Data Centre could not be searched: {exc}"
        ) from exc
    shown = neossat.thin(visits, max)
    rows = []
    for v in shown:
        d = v.as_dict()
        d["has_image"] = neossat.valid_uri(v.uri)
        rows.append(d)
    return {
        "summary": neossat.summarise(visits),
        "visits": rows,
        "search_page": neossat.SEARCH_PAGE,
        "telescope": {
            "aperture_cm": neossat.APERTURE_CM,
            "fov_deg": neossat.FOV_DEG,
            "pixscale_arcsec": neossat.PIXSCALE_ARCSEC,
        },
    }


@app.get("/api/neossat/cutout.png")
def neossat_cutout(
    uri: str = Query(..., max_length=200),
    ra: float = Query(..., ge=0, lt=360),
    dec: float = Query(..., ge=-90, le=90),
    size_arcmin: float = Query(6.0, ge=1, le=30),
):
    if not neossat.valid_uri(uri):
        raise HTTPException(422, "Not a NEOSSat file identifier.")
    try:
        png = neossat.cutout_png(uri, ra, dec, size_arcmin / 120.0)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(502, f"NEOSSat cutout failed: {exc}") from exc
    return Response(png, media_type="image/png", headers={"Cache-Control": "public, max-age=86400"})


@app.get("/api/links")
def cross_links(
    ra: float = Query(..., ge=0, lt=360),
    dec: float = Query(..., ge=-90, le=90),
    fov_arcmin: float = Query(6.0, ge=1, le=60),
):
    """Links to other tools for the same patch (WiseView, IRSA, CADC, SIMBAD, MAST) and the data credits."""
    return {"links": crosscheck.links(ra, dec, fov_arcmin), "credits": crosscheck.CREDITS}


# The page's stylesheet and script (web/styles.css, web/app.js) are served from /assets.
app.mount("/assets", StaticFiles(directory=WEB), name="assets")


@app.get("/license", response_class=PlainTextResponse)
def licence():
    """The copyright notice and the full licence text."""
    return (ROOT / "NOTICE").read_text() + "\n" + "=" * 78 + "\n\n" + (ROOT / "LICENSE").read_text()


@app.get("/robots.txt", response_class=PlainTextResponse)
def robots():
    """Search engines may index the page; the API is for the page only."""
    return "User-agent: *\nAllow: /\nDisallow: /api/\n"


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="127.0.0.1", port=8000)
