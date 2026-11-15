"""IRSA access: TAP search for overlapping SPHEREx spectral images, cutout download with retries + disk cache."""

from __future__ import annotations

import hashlib
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import requests

from .config import CUTOUT_DIR, RELEASES, SIA_COLLECTIONS


@dataclass(frozen=True)
class Observation:
    url: str  # full cutout URL (query string included)
    mjd: float  # observation time from the archive (MJD, TAP time_bounds_lower)
    band: str  # e.g. 'SPHEREx-D2'
    obs_key: str  # exposure identifier without processing version
    version: int  # processing version number (higher = newer)

    @property
    def release(self) -> str:
        """Data release this file belongs to ('dr1', 'qr3', ...), read from its archive path."""
        m = _REL_RE.search(self.url)
        return m.group(1).lower() if m else "unknown"

    @property
    def release_rank(self) -> int:
        """Smaller is better: position in config.RELEASES (unknown releases rank last)."""
        r = self.release
        return RELEASES.index(r) if r in RELEASES else len(RELEASES)


_REL_RE = re.compile(r"/spherex/([a-z]+\d+)(?:_deep)?/", re.I)
_DET_RE = re.compile(r"_\dD([1-6])_spx_")
_NAME_RE = re.compile(r"level2_(?P<key>.+?)_spx_l2b-v(?P<ver>\d+)-[^/?]*?\.fits")


def parse_observation(url: str, mjd: float, band: str) -> Observation:
    m = _NAME_RE.search(url)
    key, ver = (m.group("key"), int(m.group("ver"))) if m else (url.split("?")[0], 0)
    return Observation(url=url, mjd=float(mjd), band=band, obs_key=key, version=ver)


def build_query(ra: float, dec: float, size_deg: float, bands: list[str]) -> str:
    """ADQL pattern from the IRSA SPHEREx cutout tutorial (spherex.artifact JOIN spherex.plane)."""
    band_list = ", ".join(f"'{b}'" for b in bands)
    return f"""
SELECT
    'https://irsa.ipac.caltech.edu/' || a.uri || '?center={ra:.8f},{dec:.8f}d&size={size_deg:.6f}' AS uri,
    p.time_bounds_lower,
    p.energy_bandpassname
FROM spherex.artifact a
JOIN spherex.plane p ON a.planeid = p.planeid
WHERE 1 = CONTAINS(POINT('ICRS', {ra:.8f}, {dec:.8f}), p.poly)
    AND p.energy_bandpassname IN ({band_list})
ORDER BY p.time_bounds_lower
"""


def dedupe_versions(obs: list[Observation]) -> list[Observation]:
    """If the same exposure appears more than once, keep the copy from the best data release
    (DR1, then QR3, ... see config.RELEASES) and, within a release, the newest processing version.
    """
    best: dict[str, Observation] = {}
    for o in obs:
        cur = best.get(o.obs_key)
        if cur is None or (o.release_rank, -o.version) < (cur.release_rank, -cur.version):
            best[o.obs_key] = o
    return sorted(best.values(), key=lambda o: (o.mjd, o.obs_key))


def _query_tap(ra: float, dec: float, size_deg: float, bands: list[str]) -> list[Observation]:
    from astroquery.ipac.irsa import Irsa

    res = Irsa.query_tap(build_query(ra, dec, size_deg, bands))
    res = res.to_table() if hasattr(res, "to_table") else res
    cols = {c.lower(): c for c in res.colnames}
    return [
        parse_observation(
            str(r[cols["uri"]]),
            float(r[cols["time_bounds_lower"]]),
            str(r[cols["energy_bandpassname"]]),
        )
        for r in res
    ]


def sia_rows_to_observations(
    rows, ra: float, dec: float, size_deg: float, bands: list[str]
) -> list[Observation]:
    """Turn SIA (ObsCore) rows into cutout Observations. Rows need 'access_url' and 't_min' (MJD)."""
    out = []
    for r in rows:
        url = str(r["access_url"])
        m = _DET_RE.search(url)
        if not m or not _NAME_RE.search(url):
            continue  # not a level-2 spectral image
        band = f"SPHEREx-D{m.group(1)}"
        if band in bands:
            out.append(
                parse_observation(
                    f"{url}?center={ra:.8f},{dec:.8f}d&size={size_deg:.6f}", float(r["t_min"]), band
                )
            )
    return out


def _query_sia(ra: float, dec: float, size_deg: float, bands: list[str]) -> list[Observation]:
    """Same search through IRSA's Simple Image Access service (the route used in the IRSA data-overview tutorial)."""
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from astroquery.ipac.irsa import Irsa

    res = Irsa.query_sia(
        pos=(SkyCoord(ra, dec, unit="deg"), 1 * u.arcsec), collection=SIA_COLLECTIONS
    )
    return sia_rows_to_observations(res, ra, dec, size_deg, bands)


def query_observations(
    ra: float, dec: float, size_deg: float, bands: list[str]
) -> list[Observation]:
    """Search IRSA for every SPHEREx spectral image covering the position.
    TAP first (it includes newly ingested images immediately); if TAP fails or finds nothing, fall back to SIA.
    """
    errors = []
    for name, fn in (("TAP", _query_tap), ("SIA", _query_sia)):
        try:
            out = fn(ra, dec, size_deg, bands)
            if out:
                return dedupe_versions(out)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{name}: {exc}")
    if errors:
        raise RuntimeError("IRSA search failed (" + " | ".join(errors) + ")")
    return []


def thin_evenly(obs: list[Observation], max_n: int) -> list[Observation]:
    """Keep at most max_n observations, spread evenly through time (input must be time-sorted)."""
    if max_n <= 0 or len(obs) <= max_n:
        return obs
    step = (len(obs) - 1) / (max_n - 1) if max_n > 1 else 0
    idx = sorted({round(i * step) for i in range(max_n)})
    return [obs[i] for i in idx]


# ------------------------------------------------------------------------------------ downloading
class DownloadError(RuntimeError):
    pass


def cutout_path(url: str) -> Path:
    return CUTOUT_DIR / f"{hashlib.sha1(url.encode()).hexdigest()[:24]}.fits"


def download_cutout(url: str, retries: int = 4, timeout: float = 120.0) -> Path:
    path = cutout_path(url)
    if path.exists() and path.stat().st_size > 0:
        return path
    CUTOUT_DIR.mkdir(parents=True, exist_ok=True)
    last: Exception | None = None
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            tmp = path.with_suffix(".part")
            tmp.write_bytes(r.content)
            tmp.replace(path)
            return path
        except requests.RequestException as exc:
            last = exc
            time.sleep(1.5 * 2**attempt)
    raise DownloadError(f"{url}: {last}")


def fetch_all(
    obs: list[Observation], max_workers: int = 8, progress: Callable[[int, int], None] | None = None
):
    """Download in parallel. Returns (ok=[(obs, path)], failed=[(obs, message)]); input order is not preserved."""
    ok, failed = [], []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {pool.submit(download_cutout, o.url): o for o in obs}
        for i, fut in enumerate(as_completed(futs), 1):
            o = futs[fut]
            try:
                ok.append((o, fut.result()))
            except Exception as exc:  # noqa: BLE001
                failed.append((o, str(exc)))
            if progress:
                progress(i, len(obs))
    return ok, failed
