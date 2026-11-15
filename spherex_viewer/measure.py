"""Sub-pixel position fitting, aperture photometry, per-frame tracks and linear proper-motion fits."""

from __future__ import annotations

import numpy as np
import pandas as pd
from astropy.time import Time
from scipy.optimize import least_squares

from .align import clean
from .config import ARCSEC_PER_RAD
from .frames import Frame
from .targets import Target


def fit_gaussian(
    img: np.ndarray, x0: float, y0: float, half: int = 4, max_shift: float = 3.0
) -> dict | None:
    """Least-squares circular Gaussian + constant on a stamp around (x0, y0). Returns position, amplitude,
    width, S/N and 1-sigma position errors (pixels), or None if the stamp is unusable."""
    ny, nx = img.shape
    xi, yi = int(round(x0)), int(round(y0))
    xs = slice(max(xi - half, 0), min(xi + half + 1, nx))
    ys = slice(max(yi - half, 0), min(yi + half + 1, ny))
    sub = img[ys, xs]
    yy, xx = np.mgrid[ys, xs]
    m = np.isfinite(sub)
    if m.sum() < 12:
        return None
    base = float(np.nanmedian(sub))

    def model(p):
        a, x, y, s, b = p
        return a * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * s * s)) + b

    def resid(p):
        return (model(p) - sub)[m]

    p0 = [max(float(np.nanmax(sub)) - base, 1e-6), x0, y0, 1.1, base]
    lo = [0.0, x0 - max_shift, y0 - max_shift, 0.5, -np.inf]
    hi = [np.inf, x0 + max_shift, y0 + max_shift, 3.0, np.inf]
    try:
        sol = least_squares(
            resid, p0, bounds=(lo, hi), x_scale=[max(p0[0], 1e-3), 1, 1, 1, max(abs(base), 1e-3)]
        )
    except Exception:  # noqa: BLE001
        return None
    a, x, y, s, b = sol.x
    r = sol.fun
    dof = max(int(m.sum()) - 5, 1)
    rms = float(np.sqrt(np.sum(r**2) / dof))
    try:
        cov = np.linalg.inv(sol.jac.T @ sol.jac) * rms**2
        sx, sy = float(np.sqrt(abs(cov[1, 1]))), float(np.sqrt(abs(cov[2, 2])))
    except np.linalg.LinAlgError:
        sx = sy = float("nan")
    at_bound = (
        abs(x - x0) > max_shift - 0.05 or abs(y - y0) > max_shift - 0.05 or s <= 0.51 or s >= 2.99
    )
    snr = float(a / rms) if rms > 0 else float("inf")
    return {
        "x": float(x),
        "y": float(y),
        "amp": float(a),
        "sigma_pix": float(s),
        "snr": snr,
        "sx_pix": sx,
        "sy_pix": sy,
        "ok": bool(a > 0 and snr >= 4.0 and not at_bound),
    }


def aperture_flux(
    img: np.ndarray,
    x: float,
    y: float,
    pixscale_arcsec: float,
    r_ap: float = 2.0,
    r_in: float = 4.0,
    r_out: float = 7.0,
) -> tuple[float, float]:
    """Background-subtracted aperture flux in mJy and its 1-sigma error (annulus scatter).
    Pixel solid angle is approximated as scale^2 (the SPHEREx solid-angle pixel map would refine this).
    """
    from photutils.aperture import CircularAperture, aperture_photometry

    mask = ~np.isfinite(img)
    data = np.where(mask, 0.0, img)
    ap = CircularAperture((x, y), r_ap)
    tot = float(aperture_photometry(data, ap, mask=mask)["aperture_sum"][0])
    yy, xx = np.indices(img.shape)
    rr = np.hypot(xx - x, yy - y)
    ring = img[(rr >= r_in) & (rr <= r_out) & ~mask]
    if ring.size < 8:
        return float("nan"), float("nan")
    bkg, scat = float(np.median(ring)), float(np.std(ring))
    area = float(ap.area)
    omega = (pixscale_arcsec / ARCSEC_PER_RAD) ** 2
    to_mjy = omega * 1e9  # (MJy/sr * sr) -> mJy
    return (tot - bkg * area) * to_mjy, scat * np.sqrt(area) * to_mjy


def measure_track(
    frames: list[Frame], target: Target, half: int = 4, max_shift: float = 3.0
) -> pd.DataFrame:
    """Per frame: predicted position, fitted position, offset and aperture flux."""
    rows = []
    for f in frames:
        img, _ = clean(f)
        pred = target.coord_at(f.time) if target.has_motion else target.coord_ref()
        xp, yp = (float(v) for v in f.wcs.world_to_pixel(pred))
        row = {
            "mjd": f.mjd,
            "time_utc": f.time.iso,
            "band": f.band,
            "wavelength_um": f.wavelength_um,
            "pred_ra": pred.ra.deg,
            "pred_dec": pred.dec.deg,
            "x_pred": xp,
            "y_pred": yp,
            "ok": False,
        }
        ny, nx = img.shape
        if -0.5 <= xp < nx - 0.5 and -0.5 <= yp < ny - 0.5:
            fit = fit_gaussian(img, xp, yp, half=half, max_shift=max_shift)
            if fit is not None:
                sky = f.wcs.pixel_to_world(fit["x"], fit["y"])
                scale = f.pixscale_arcsec
                row.update(
                    ok=fit["ok"],
                    x=fit["x"],
                    y=fit["y"],
                    ra=float(sky.ra.deg),
                    dec=float(sky.dec.deg),
                    sx_arcsec=fit["sx_pix"] * scale,
                    sy_arcsec=fit["sy_pix"] * scale,
                    amp=fit["amp"],
                    snr=fit["snr"],
                    offset_arcsec=float(sky.separation(pred).arcsec),
                )
            fx, fy = (row["x"], row["y"]) if row["ok"] else (xp, yp)
            row["flux_mjy"], row["flux_err_mjy"] = aperture_flux(img, fx, fy, f.pixscale_arcsec)
        rows.append(row)
    return pd.DataFrame(rows)


def spectrum_at(frames: list[Frame], coord, r_ap: float = 2.0) -> pd.DataFrame:
    """Spectrum at a fixed sky position: aperture flux in every frame, with the wavelength valid at THAT pixel."""
    rows = []
    for f in frames:
        img, _ = clean(f)
        x, y = (float(v) for v in f.wcs.world_to_pixel(coord))
        ny, nx = img.shape
        row = {
            "mjd": f.mjd,
            "time_utc": f.time.iso,
            "band": f.band,
            "wavelength_um": f.wavelength_at(x, y),
            "x": x,
            "y": y,
            "ok": False,
            "flux_mjy": float("nan"),
            "flux_err_mjy": float("nan"),
            "flag": f.flag_at(x, y),
        }
        if r_ap <= x < nx - r_ap and r_ap <= y < ny - r_ap:
            flux, err = aperture_flux(img, x, y, f.pixscale_arcsec, r_ap=r_ap)
            row.update(
                flux_mjy=flux,
                flux_err_mjy=err,
                ok=bool(np.isfinite(flux) and np.isfinite(err) and err > 0 and flux / err >= 3.0),
            )
        rows.append(row)
    return pd.DataFrame(rows)


def fit_linear_motion(
    track: pd.DataFrame, min_points: int = 5, min_span_days: float = 60.0
) -> dict | None:
    """Straight-line fit of measured positions -> proper motion (mas/yr) with 3-sigma clipping.
    Needs a long enough baseline; returns None otherwise."""
    d = track[track.get("ok", False) == True].dropna(subset=["ra", "dec"])  # noqa: E712
    if len(d) < min_points or (d.mjd.max() - d.mjd.min()) < min_span_days:
        return None
    ra0, dec0, t0 = d.ra.mean(), d.dec.mean(), d.mjd.mean()
    t = ((d.mjd - t0) / 365.25).to_numpy()
    east = ((d.ra - ra0) * np.cos(np.radians(dec0)) * 3600.0).to_numpy()
    north = ((d.dec - dec0) * 3600.0).to_numpy()
    sig = np.maximum(np.hypot(d.sx_arcsec, d.sy_arcsec).to_numpy() / np.sqrt(2), 0.3)
    keep = np.ones(len(d), bool)
    for _ in range(3):
        if keep.sum() < min_points:
            return None
        pe, ce = np.polyfit(t[keep], east[keep], 1, w=1 / sig[keep], cov=True)
        pn, cn = np.polyfit(t[keep], north[keep], 1, w=1 / sig[keep], cov=True)
        res = np.hypot(east - np.polyval(pe, t), north - np.polyval(pn, t))
        s = np.std(res[keep]) or 1e-9
        new = res < 3 * max(s, 0.3)
        if new.sum() == keep.sum():
            break
        keep = new
    return {
        "pmra_masyr": pe[0] * 1000,
        "pmra_err": np.sqrt(ce[0, 0]) * 1000,
        "pmdec_masyr": pn[0] * 1000,
        "pmdec_err": np.sqrt(cn[0, 0]) * 1000,
        "ra_at_epoch": ra0 + pe[1] / 3600.0 / np.cos(np.radians(dec0)),
        "dec_at_epoch": dec0 + pn[1] / 3600.0,
        "epoch_jyear": float(Time(t0, format="mjd").jyear),
        "n_used": int(keep.sum()),
        "rms_arcsec": float(np.sqrt(np.mean(res[keep] ** 2))),
    }
