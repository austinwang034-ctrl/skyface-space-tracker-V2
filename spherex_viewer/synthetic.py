"""Synthetic SPHEREx-like frames for offline demos and tests. NOT real data: every UI path that uses this
must label it as synthetic. Frames are rendered with the TRUE WCS but carry a deliberately wrong pointing,
so the plate-solving step has something real to recover."""

from __future__ import annotations

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.time import Time
from astropy.wcs import WCS

from .archive import Observation
from .frames import Frame


def _wcs(center: SkyCoord, scale: float, n: int, roll_deg: float) -> WCS:
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [center.ra.deg, center.dec.deg]
    w.wcs.crpix = [(n + 1) / 2, (n + 1) / 2]
    c, s = np.cos(np.radians(roll_deg)), np.sin(np.radians(roll_deg))
    sc = scale / 3600.0
    w.wcs.cd = [[-sc * c, sc * s], [sc * s, sc * c]]
    w.wcs.cunit = ["deg", "deg"]
    return w


def make_synthetic(
    center: SkyCoord,
    n_frames: int = 40,
    seed: int = 7,
    npix: int = 41,
    scale: float = 6.15,
    mover_pm=(0.0, 10300.0),
    variable: bool = True,
    pointing_error_pix: float = 0.8,
    start_mjd: float = 60900.0,
    span_days: float = 400.0,
    band: str = "SPHEREx-D2",
    mover_in_gaia: bool = True,
):
    """Returns (frames, truth, gaia_like_table). truth holds per-frame true mover/variable positions."""
    rng = np.random.default_rng(seed)
    mjds = np.sort(start_mjd + rng.uniform(0, span_days, n_frames))
    years = Time(mjds, format="mjd").jyear
    t_mid = years.mean()
    half = npix * scale / 2.0 * 1.6

    # background stars (Gaia-like epoch 2016), uniform in a disc around the centre
    ns = 28
    r = half * np.sqrt(rng.uniform(0, 1, ns))
    th = rng.uniform(0, 2 * np.pi, ns)
    st_ra = center.ra.deg + (r * np.cos(th)) / 3600 / np.cos(center.dec.rad)
    st_dec = center.dec.deg + (r * np.sin(th)) / 3600
    st_amp = 10 ** rng.uniform(1.6, 3.0, ns)
    st_pmra, st_pmdec = rng.normal(0, 15, ns), rng.normal(0, 15, ns)

    # mover: positioned so that it crosses the field centre at the mid-epoch
    dt = t_mid - 2016.0
    m_ra0 = center.ra.deg - mover_pm[0] * dt / 3.6e6 / np.cos(center.dec.rad)
    m_dec0 = center.dec.deg - mover_pm[1] * dt / 3.6e6
    m_amp = 600.0
    # variable star offset from centre by ~40 arcsec
    v_ra, v_dec = center.ra.deg + 40 / 3600 / np.cos(center.dec.rad), center.dec.deg - 25 / 3600

    frames, truth = (
        [],
        {
            "mover_ra": [],
            "mover_dec": [],
            "var_flux": [],
            "true_wcs": [],
            "pointing_err_pix": [],
        },
    )
    sigma = 1.1
    for k in range(n_frames):
        roll = rng.uniform(-180, 180)
        wtrue = _wcs(center, scale, npix, roll)
        yy, xx = np.mgrid[0:npix, 0:npix]
        img = rng.normal(0, 2.0, (npix, npix)) + 40.0 + 0.002 * (xx + yy)

        def stamp(ra, dec, amp):
            x, y = wtrue.world_to_pixel(SkyCoord(ra * u.deg, dec * u.deg))
            return amp * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma**2))

        dty = years[k] - 2016.0
        for i in range(ns):
            img += stamp(
                st_ra[i] + st_pmra[i] * dty / 3.6e6 / np.cos(center.dec.rad),
                st_dec[i] + st_pmdec[i] * dty / 3.6e6,
                st_amp[i],
            )
        mra = m_ra0 + mover_pm[0] * dty / 3.6e6 / np.cos(center.dec.rad)
        mdec = m_dec0 + mover_pm[1] * dty / 3.6e6
        img += stamp(mra, mdec, m_amp)
        vf = (
            300.0 * (1 + 0.6 * np.sin(2 * np.pi * (mjds[k] - start_mjd) / 245.0))
            if variable
            else 0.0
        )
        img += stamp(v_ra, v_dec, vf)
        truth["mover_ra"].append(mra)
        truth["mover_dec"].append(mdec)
        truth["var_flux"].append(vf)

        # the WCS we hand to the pipeline is wrong by a random pointing error
        err = rng.uniform(-pointing_error_pix, pointing_error_pix, 2)
        wbad = wtrue.deepcopy()
        wbad.wcs.crpix = wbad.wcs.crpix + err
        truth["true_wcs"].append(wtrue)
        truth["pointing_err_pix"].append(err)
        obs = Observation(
            url=f"synthetic://frame/{k}",
            mjd=float(mjds[k]),
            band=band,
            obs_key=f"SYN_{k:04d}",
            version=1,
        )
        frames.append(
            Frame(
                obs=obs,
                data=img.astype(np.float32),
                wcs=wbad,
                mjd=float(mjds[k]),
                time_source="synthetic",
                wavelength_um=float(rng.uniform(1.1, 1.62)),
            )
        )

    ids = list(range(1, ns + 1))
    ra = list(st_ra)
    dec = list(st_dec)
    pmra = list(st_pmra)
    pmdec = list(st_pmdec)
    gmag = list(17.5 - 2.5 * np.log10(st_amp / 40))
    if mover_in_gaia:
        ids.append(999)
        ra.append(m_ra0)
        dec.append(m_dec0)
        pmra.append(mover_pm[0])
        pmdec.append(mover_pm[1])
        gmag.append(12.0)
    gaia = Table(
        {
            "source_id": ids,
            "ra": ra,
            "dec": dec,
            "parallax": np.full(len(ids), 5.0),
            "pmra": pmra,
            "pmdec": pmdec,
            "phot_g_mean_mag": gmag,
            "ref_epoch": np.full(len(ids), 2016.0),
        }
    )
    truth["variable_coord"] = SkyCoord(v_ra * u.deg, v_dec * u.deg)
    return frames, truth, gaia
