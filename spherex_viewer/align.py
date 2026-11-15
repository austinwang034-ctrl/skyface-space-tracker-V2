"""Common sky grid, background removal and reprojection of all frames onto that grid."""

from __future__ import annotations

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.stats import sigma_clipped_stats
from astropy.wcs import WCS
from reproject import reproject_interp

from .frames import Frame


def common_wcs(
    center: SkyCoord, fov_arcsec: float, scale_arcsec: float
) -> tuple[WCS, tuple[int, int]]:
    """North-up, East-left TAN grid centred exactly on `center` (odd size so the target sits on a pixel centre)."""
    n = int(np.ceil(fov_arcsec / scale_arcsec))
    n += 1 - n % 2
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [center.ra.deg, center.dec.deg]
    w.wcs.crpix = [(n + 1) / 2.0, (n + 1) / 2.0]  # FITS 1-based
    w.wcs.cdelt = [-scale_arcsec / 3600.0, scale_arcsec / 3600.0]
    w.wcs.cunit = ["deg", "deg"]
    w.array_shape = (n, n)
    return w, (n, n)


def clean(frame: Frame) -> tuple[np.ndarray, float]:
    """Background-subtracted copy of the native image and the background RMS (sigma-clipped)."""
    d = frame.data.astype(np.float64)
    good = np.isfinite(d)
    if good.sum() < 10:
        return d, float("nan")
    _, med, std = sigma_clipped_stats(d, mask=~good, sigma=3.0, maxiters=5)
    return d - med, float(std)


def reproject_stack(
    frames: list[Frame], wcs_out: WCS, shape: tuple[int, int], order: str = "bilinear"
) -> np.ndarray:
    """(N, ny, nx) float32 cube; NaN outside each frame's footprint. Frames are used in the order given."""
    cube = np.full((len(frames),) + shape, np.nan, dtype=np.float32)
    for i, f in enumerate(frames):
        img, _ = clean(f)
        out = reproject_interp(
            (img, f.wcs), wcs_out, shape_out=shape, order=order, return_footprint=False
        )
        cube[i] = out
    return cube


# Offsets in the tangent plane (arcsec): +east, +north. Pixel coords are 0-based array indices.
def pix_to_offset(x, y, shape: tuple[int, int], scale_arcsec: float):
    ny, nx = shape
    return (
        ((nx - 1) / 2.0 - np.asarray(x)) * scale_arcsec,
        (np.asarray(y) - (ny - 1) / 2.0) * scale_arcsec,
    )


def offset_to_pix(east, north, shape: tuple[int, int], scale_arcsec: float):
    ny, nx = shape
    return (
        (nx - 1) / 2.0 - np.asarray(east) / scale_arcsec,
        (ny - 1) / 2.0 + np.asarray(north) / scale_arcsec,
    )
