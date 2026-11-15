"""Catalogue overlays for the tracker: Gaia stars and WISE sources drawn on top of the SPHEREx images.

An overlay answers the first question a mover-hunter asks: "is that dot already known?" A source in
the image with no ring around it is in neither catalogue. Positions are sent as offsets from the patch
centre at a reference year plus a yearly motion, so the browser can place each ring correctly in every
frame, including for fast-moving stars.
"""

from __future__ import annotations

import csv
import io

import numpy as np
import requests
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.time import Time

from .config import IRSA_TAP_SYNC
from .neighborhood import GAIA_REF_EPOCH, propagate

# WISE catalogues at IRSA, best first. CatWISE2020 measures motions (in arcsec/yr); AllWISE is the fallback.
WISE_TABLES = (
    {
        "name": "CatWISE2020",
        "table": "catwise_2020",
        "columns": "ra, dec, w1mpro, w2mpro, pmra, pmdec, meanobsmjd",
        "pm_to_arcsec": 1.0,
    },
    {
        "name": "AllWISE",
        "table": "allwise_p3as_psd",
        "columns": "ra, dec, w1mpro, w2mpro, pmra, pmdec",
        "pm_to_arcsec": 0.001,  # AllWISE lists apparent motion in mas/yr
        "epoch": 2010.5,
    },
)
MAX_SOURCES = 600


def _number(value) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def _offsets(center: SkyCoord, coords: SkyCoord) -> tuple[np.ndarray, np.ndarray]:
    east, north = center.spherical_offsets_to(coords)
    return east.arcsec, north.arcsec


def gaia_overlay(tbl: Table, center: SkyCoord, year: float, radius_arcsec: float) -> dict:
    """Gaia stars inside the patch: offsets at `year` and yearly motion, both in arcsec."""
    east, north = _offsets(center, propagate(tbl, Time(year, format="jyear")))
    pm_e = np.nan_to_num(np.asarray(tbl["pmra"], dtype=float)) / 1000.0
    pm_n = np.nan_to_num(np.asarray(tbl["pmdec"], dtype=float)) / 1000.0
    mag = np.asarray(tbl["phot_g_mean_mag"], dtype=float)
    keep = np.flatnonzero(np.hypot(east, north) <= radius_arcsec)
    keep = keep[np.argsort(np.nan_to_num(mag[keep], nan=99.0))][:MAX_SOURCES]
    return {
        "catalogue": "Gaia DR3",
        "epoch": year,
        "sources": [
            {
                "id": str(tbl["source_id"][i]),
                "e": float(east[i]),
                "n": float(north[i]),
                "pm_e": float(pm_e[i]),
                "pm_n": float(pm_n[i]),
                "mag": _number(mag[i]),
            }
            for i in keep
        ],
    }


def build_wise_query(spec: dict, ra: float, dec: float, radius_deg: float) -> str:
    return (
        f"SELECT TOP {MAX_SOURCES} {spec['columns']} FROM {spec['table']} "
        f"WHERE CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {ra:.6f}, {dec:.6f}, {radius_deg:.5f})) = 1"
    )


def _tap_csv(query: str, timeout: float = 60.0) -> list[dict]:
    reply = requests.get(IRSA_TAP_SYNC, params={"QUERY": query, "FORMAT": "CSV"}, timeout=timeout)
    if reply.status_code >= 400:
        raise RuntimeError(f"IRSA answered {reply.status_code}")
    return list(csv.DictReader(io.StringIO(reply.text)))


def parse_wise_rows(rows: list[dict], spec: dict, center: SkyCoord) -> list[dict]:
    """Catalogue rows -> overlay sources. Rows without a position are dropped; missing motion counts as zero."""
    good = [
        r for r in rows if _number(r.get("ra")) is not None and _number(r.get("dec")) is not None
    ]
    if not good:
        return []
    coords = SkyCoord([float(r["ra"]) for r in good], [float(r["dec"]) for r in good], unit="deg")
    east, north = _offsets(center, coords)
    scale = spec["pm_to_arcsec"]
    out = []
    for r, e, n in zip(good, east, north):
        mjd = _number(r.get("meanobsmjd"))
        epoch = float(Time(mjd, format="mjd").jyear) if mjd else spec.get("epoch", 2014.0)
        pm_e, pm_n = (
            (_number(r.get("pmra")) or 0.0) * scale,
            (_number(r.get("pmdec")) or 0.0) * scale,
        )
        if np.hypot(pm_e, pm_n) > 15.0:  # faster than any known star: a bad fit, not a motion
            pm_e = pm_n = 0.0
        out.append(
            {
                "e": float(e),
                "n": float(n),
                "pm_e": pm_e,
                "pm_n": pm_n,
                "epoch": epoch,
                "w1": _number(r.get("w1mpro")),
                "w2": _number(r.get("w2mpro")),
            }
        )
    return out


def wise_overlay(center: SkyCoord, radius_arcsec: float, fetch=None) -> dict:
    """WISE sources inside the patch. Tries CatWISE2020 first, then AllWISE. `fetch(query) -> rows` is injectable."""
    fetch = fetch or _tap_csv
    errors = []
    for spec in WISE_TABLES:
        try:
            rows = fetch(
                build_wise_query(spec, center.ra.deg, center.dec.deg, radius_arcsec / 3600.0)
            )
            return {"catalogue": spec["name"], "sources": parse_wise_rows(rows, spec, center)}
        except Exception as exc:  # noqa: BLE001  (try the next catalogue)
            errors.append(f"{spec['name']}: {exc}")
    raise RuntimeError(" | ".join(errors))


__all__ = ["GAIA_REF_EPOCH", "gaia_overlay", "wise_overlay", "build_wise_query", "parse_wise_rows"]
