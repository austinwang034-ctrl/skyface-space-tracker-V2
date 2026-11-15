"""Guided plate solving: SPHEREx images already carry a WCS, so we verify it against Gaia (propagated to the
exposure time), measure and remove any pointing offset, report residuals, and list unmatched sources.

Unmatched sources are the interesting ones: objects too faint/red for Gaia (e.g. Y dwarfs) or things that
moved. A fully blind solve (no starting WCS) is intentionally not included: it is not needed for SPHEREx.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from astropy.table import Table
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from scipy.spatial import cKDTree

from .align import clean
from .frames import Frame
from .measure import fit_gaussian
from .neighborhood import propagate


@dataclass
class SolveResult:
    ok: bool
    message: str
    wcs: WCS
    dx_pix: float = 0.0
    dy_pix: float = 0.0
    shift_arcsec: float = 0.0
    n_detected: int = 0
    n_matched: int = 0
    rms_before_arcsec: float = float("nan")
    rms_after_arcsec: float = float("nan")
    matches: pd.DataFrame = field(default_factory=pd.DataFrame)
    unmatched: pd.DataFrame = field(default_factory=pd.DataFrame)


def detect_sources(img: np.ndarray, fwhm_pix: float = 2.5, nsigma: float = 5.0) -> pd.DataFrame:
    """DAOStarFinder on a background-subtracted image (SPHEREx PSFs are close to undersampled)."""
    from astropy.stats import sigma_clipped_stats
    from photutils.detection import DAOStarFinder

    mask = ~np.isfinite(img)
    data = np.where(mask, 0.0, img)
    _, med, std = sigma_clipped_stats(data, mask=mask, sigma=3.0)
    if not np.isfinite(std) or std <= 0:
        return pd.DataFrame(columns=["x", "y", "flux"])
    tbl = DAOStarFinder(threshold=nsigma * std, fwhm=fwhm_pix)(data - med, mask=mask)
    if tbl is None or len(tbl) == 0:
        return pd.DataFrame(columns=["x", "y", "flux"])
    xc, yc = (
        ("x_centroid", "y_centroid") if "x_centroid" in tbl.colnames else ("xcentroid", "ycentroid")
    )
    return pd.DataFrame(
        {
            "x": np.asarray(tbl[xc], float),
            "y": np.asarray(tbl[yc], float),
            "flux": np.asarray(tbl["flux"], float),
        }
    )


def refine_centroids(img: np.ndarray, det: pd.DataFrame, max_shift: float = 1.5) -> pd.DataFrame:
    """Replace coarse DAOStarFinder centroids by Gaussian-fit centroids (much better for undersampled PSFs).
    Detections whose fit fails keep their original position and are flagged refined=False."""
    xs, ys, ok = [], [], []
    for x, y in zip(det.x, det.y):
        fit = fit_gaussian(img, float(x), float(y), half=3, max_shift=max_shift)
        good = bool(fit and fit["ok"])
        xs.append(fit["x"] if good else x)
        ys.append(fit["y"] if good else y)
        ok.append(good)
    return det.assign(x=xs, y=ys, refined=ok)


def _vote_shift(
    det: np.ndarray, ref: np.ndarray, max_shift: float, bin_pix: float = 0.5
) -> tuple[float, float, int]:
    """Most common (dx, dy) between detections and reference positions: robust to false matches."""
    d = (det[:, None, :] - ref[None, :, :]).reshape(-1, 2)
    d = d[(np.abs(d[:, 0]) <= max_shift) & (np.abs(d[:, 1]) <= max_shift)]
    if len(d) == 0:
        return 0.0, 0.0, 0
    nb = int(2 * max_shift / bin_pix) + 1
    H, xe, ye = np.histogram2d(d[:, 0], d[:, 1], bins=nb, range=[[-max_shift, max_shift]] * 2)
    i, j = np.unravel_index(np.argmax(H), H.shape)
    cx, cy = (xe[i] + xe[i + 1]) / 2, (ye[j] + ye[j + 1]) / 2
    sel = d[np.hypot(d[:, 0] - cx, d[:, 1] - cy) <= 1.0]
    return float(np.median(sel[:, 0])), float(np.median(sel[:, 1])), int(H[i, j])


def shifted_wcs(wcs: WCS, dx: float, dy: float) -> WCS:
    """WCS' with new(p) = old(p - d): a pure pointing correction that keeps any SIP distortion intact."""
    from astropy.wcs import Sip

    w = wcs.deepcopy()
    w.wcs.crpix = w.wcs.crpix + np.array([dx, dy])
    if w.sip is not None:
        s = w.sip
        w.sip = Sip(s.a, s.b, s.ap, s.bp, np.asarray(s.crpix) + np.array([dx, dy]))
    w.wcs.set()
    return w


def solve_frame(
    frame: Frame,
    gaia: Table,
    fwhm_pix: float = 2.5,
    nsigma: float = 5.0,
    max_shift_pix: float = 4.0,
    match_tol_pix: float = 1.2,
    min_matches: int = 3,
) -> SolveResult:
    img, _ = clean(frame)
    det = detect_sources(img, fwhm_pix, nsigma)
    if len(det):
        det = refine_centroids(img, det)
    scale = float(np.mean(proj_plane_pixel_scales(frame.wcs)) * 3600.0)
    fail = lambda msg: SolveResult(False, msg, frame.wcs, n_detected=len(det))  # noqa: E731
    if len(det) < min_matches:
        return fail(f"Only {len(det)} sources detected (need {min_matches}).")

    gc = propagate(gaia, frame.time)
    gx, gy = frame.wcs.world_to_pixel(gc)
    ny, nx = img.shape
    inside = (
        (gx > -max_shift_pix)
        & (gx < nx + max_shift_pix)
        & (gy > -max_shift_pix)
        & (gy < ny + max_shift_pix)
    )
    if inside.sum() < min_matches:
        return fail(f"Only {int(inside.sum())} Gaia stars fall in this cutout.")
    ref = np.column_stack([np.asarray(gx)[inside], np.asarray(gy)[inside]])
    dpos = det[["x", "y"]].to_numpy()

    dx, dy, votes = _vote_shift(dpos, ref, max_shift_pix)
    for _ in range(3):  # match with the shift, then re-estimate it
        tree = cKDTree(ref + [dx, dy])
        dist, idx = tree.query(dpos, distance_upper_bound=match_tol_pix)
        hit = np.isfinite(dist)
        if hit.sum() < 1:
            break
        # keep the closest detection per reference star
        order = np.argsort(dist)
        used, keep = set(), np.zeros(len(dpos), bool)
        for k in order:
            if hit[k] and idx[k] not in used:
                used.add(idx[k])
                keep[k] = True
        resid = dpos[keep] - ref[idx[keep]]
        if len(resid) == 0:
            break
        dx, dy = float(np.median(resid[:, 0])), float(np.median(resid[:, 1]))
    n = int(keep.sum()) if hit.sum() else 0
    if n < min_matches:
        return fail(f"Only {n} Gaia stars matched (need {min_matches}); WCS left unchanged.")

    r_before = np.hypot(*(dpos[keep] - ref[idx[keep]]).T) * scale
    r_after = np.hypot(*(dpos[keep] - ref[idx[keep]] - [dx, dy]).T) * scale
    new = shifted_wcs(frame.wcs, dx, dy)
    sky = new.pixel_to_world(dpos[:, 0], dpos[:, 1])
    det = det.assign(ra=sky.ra.deg, dec=sky.dec.deg)
    matches = det[keep].assign(
        gaia_source_id=np.asarray(gaia["source_id"])[inside][idx[keep]], resid_arcsec=r_after
    )
    unmatched = det[~keep].sort_values("flux", ascending=False).reset_index(drop=True)
    return SolveResult(
        True,
        "Pointing offset measured and applied.",
        new,
        dx,
        dy,
        float(np.hypot(dx, dy) * scale),
        len(det),
        n,
        float(np.sqrt(np.mean(r_before**2))),
        float(np.sqrt(np.mean(r_after**2))),
        matches.reset_index(drop=True),
        unmatched,
    )


def solve_all(
    frames: list[Frame], gaia: Table, progress=None, **kw
) -> tuple[list[Frame], list[SolveResult]]:
    """Plate-solve every frame; frames with a good solution get the refined WCS (others are untouched)."""
    from dataclasses import replace

    out, results = [], []
    for i, f in enumerate(frames):
        r = solve_frame(f, gaia, **kw)
        results.append(r)
        out.append(replace(f, wcs=r.wcs) if r.ok else f)
        if progress:
            progress(i + 1, len(frames))
    return out, results
