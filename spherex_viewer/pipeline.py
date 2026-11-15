"""Glue: search -> download -> load -> sort -> align. The UI and tests both call this."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.time import Time
from astropy.wcs import WCS

from . import archive, astrometry, neighborhood
from .align import common_wcs, reproject_stack
from .frames import Frame, load_frame, sort_chronologically, usable
from .synthetic import make_synthetic
from .targets import Target

NOMINAL_EPOCH = 2026.2  # centre of the SPHEREx data currently public; used to place the patch for moving targets


@dataclass
class Bundle:
    target: Target
    center: SkyCoord
    frames: list[Frame]  # chronological
    wcs_out: WCS
    shape: tuple[int, int]
    scale_arcsec: float
    cube: np.ndarray  # (N, ny, nx), NaN outside footprint
    failed: list
    synthetic: bool = False
    truth: dict | None = None
    gaia_demo: Table | None = None
    n_found: int = 0
    raw_frames: list | None = None  # frames exactly as downloaded (before plate solving)
    solves: list | None = None  # one SolveResult per frame, or None if not run
    refine_note: str = ""  # human-readable status of the plate-solving step
    gaia: Table | None = None  # Gaia stars used for plate solving (reused for the tracker overlay)


def patch_center(target: Target) -> SkyCoord:
    """Where to centre the patch: for movers, where the target is at the nominal mid-epoch of the data."""
    if target.has_motion:
        c = target.coord_at(Time(NOMINAL_EPOCH, format="jyear"))
        return SkyCoord(c.ra, c.dec, frame="icrs")
    return target.coord_ref()


def make_cube(frames: list[Frame], center: SkyCoord, fov_arcsec: float, oversample: float = 1.0):
    scale = np.mean([f.pixscale_arcsec for f in frames]) / oversample
    wcs_out, shape = common_wcs(center, fov_arcsec, scale)
    return wcs_out, shape, float(scale), reproject_stack(frames, wcs_out, shape)


def solve_frames(frames: list[Frame], gaia: Table, **kw):
    """Plate-solve; returns (frames_with_refined_wcs, results, note)."""
    solved, res = astrometry.solve_all(frames, gaia, **kw)
    n_ok = sum(r.ok for r in res)
    if n_ok == 0:
        return (
            frames,
            res,
            "Plate solving found too few Gaia matches; using the original coordinates.",
        )
    med = float(np.median([r.shift_arcsec for r in res if r.ok]))
    return (
        solved,
        res,
        f"Plate-solved {n_ok}/{len(res)} frames against Gaia (median pointing correction {med:.2f}″).",
    )


def build_bundle(
    source: str,
    target: Target,
    center: SkyCoord,
    fov_arcmin: float,
    bands: list[str],
    max_frames: int = 120,
    oversample: float = 1.0,
    refine: bool = True,
    progress: Callable[[str, float], None] | None = None,
) -> Bundle:
    say = progress or (lambda msg, frac: None)
    fov_arcsec = fov_arcmin * 60.0
    truth = gaia = None
    failed: list = []
    n_found = 0
    if source == "demo":
        say("Generating synthetic frames…", 0.2)
        frames, truth, gaia = make_synthetic(center, n_frames=min(max_frames, 40))
        n_found = len(frames)
        target = Target(
            "Synthetic mover (Barnard-like)",
            truth["mover_ra"][0],
            truth["mover_dec"][0],
            pmdec=10300.0,
            ref_epoch=frames[0].time.jyear,
            kind="mover",
            source="SYNTHETIC. Not SPHEREx data.",
        )
    else:
        say("Searching the IRSA SPHEREx archive…", 0.05)
        # request 1.5x the displayed field so rotated cutouts still fill the whole square grid
        obs = archive.query_observations(
            center.ra.deg, center.dec.deg, 1.5 * fov_arcmin / 60.0, bands
        )
        n_found = len(obs)
        if not obs:
            raise RuntimeError("No SPHEREx images cover this position in the selected bands (yet).")
        obs = archive.thin_evenly(obs, max_frames)
        say(f"Downloading {len(obs)} of {n_found} cutouts…", 0.15)
        ok, failed = archive.fetch_all(
            obs, progress=lambda i, n: say(f"Downloading cutouts… {i}/{n}", 0.15 + 0.6 * i / n)
        )
        frames = []
        for o, path in ok:
            try:
                f = load_frame(o, path, center)
                if usable(f):
                    frames.append(f)
                else:
                    failed.append((o, "mostly empty cutout (target near detector edge)"))
            except Exception as exc:  # noqa: BLE001
                path.unlink(
                    missing_ok=True
                )  # corrupt download: drop from cache so a retry refetches
                failed.append((o, f"unreadable FITS: {exc}"))
    if not frames:
        raise RuntimeError("No usable frames were retrieved.")
    frames = sort_chronologically(frames)
    raw, solves, note = frames, None, "Plate solving was switched off."
    if refine:
        say("Plate-solving frames against Gaia…", 0.8)
        try:
            if gaia is None:
                gaia = neighborhood.gaia_cone(center, 0.75 * 1.5 * fov_arcmin + 1.0, 18.0)
            frames, solves, note = solve_frames(frames, gaia)
        except Exception as exc:  # noqa: BLE001
            note = f"Gaia step skipped ({exc}); using the archive's own coordinates."
    say("Aligning frames onto a common sky grid…", 0.92)
    wcs_out, shape, scale, cube = make_cube(frames, center, fov_arcsec, oversample)
    return Bundle(
        target,
        center,
        frames,
        wcs_out,
        shape,
        scale,
        cube,
        failed,
        source == "demo",
        truth,
        gaia if source == "demo" else None,
        n_found,
        raw,
        solves,
        note,
        gaia,
    )


def with_frames(
    b: Bundle, frames: list[Frame], solves: list | None = None, note: str = ""
) -> Bundle:
    """New bundle with different frames (e.g. re-solved with other settings); grid unchanged, cube re-projected."""
    frames = sort_chronologically(frames)
    return replace(
        b,
        frames=frames,
        cube=reproject_stack(frames, b.wcs_out, b.shape),
        solves=solves if solves is not None else b.solves,
        refine_note=note or b.refine_note,
    )
