"""NEOSSat (Canadian Space Agency) images of the same patch of sky, from the Canadian Astronomy Data Centre.

NEOSSat is a 15 cm visible-light space telescope (0.85 deg field, 3 arcsec pixels) that has been observing since
2013, so where it happens to have looked it gives an independent, much longer time baseline than SPHEREx.
The search uses CADC's public TAP service (CAOM-2 tables); cutouts use CADC's position-cutout service.
"""

from __future__ import annotations

import csv
import hashlib
import io
import re
from dataclasses import asdict, dataclass
from urllib.parse import quote

import numpy as np
import requests
from astropy.time import Time

from .config import CACHE_DIR, CADC_CUTOUT, CADC_DATA, CADC_TAP_SYNC

COLLECTION = "NEOSSAT"
FOV_DEG, PIXSCALE_ARCSEC, APERTURE_CM = 0.85, 3.0, 15.0
SEARCH_PAGE = "https://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/en/search/?collection=NEOSSAT"
# Processing levels, best first (CADC NEOSSat page): CORD = clipped + overscan + dark-subtracted, COR = clipped +
# overscan, CLEAN = overscan only, RAW = as downlinked.
PRODUCT_RANK = {"cord": 0, "cor": 1, "clean": 2, "raw": 3}
_URI_RE = re.compile(r"^(?:cadc|ad):NEOSSAT/[A-Za-z0-9_.\-]+$")
NEOSSAT_DIR = CACHE_DIR / "neossat"


@dataclass(frozen=True)
class Visit:
    observation_id: str
    target: str
    mjd: float
    exposure_s: float | None
    product: str  # cord / cor / clean / raw
    uri: str  # CADC file identifier, e.g. cadc:NEOSSAT/NEOS_SCI_2021..._cord.fits

    @property
    def iso(self) -> str:
        return Time(self.mjd, format="mjd", scale="utc").iso[:19]

    @property
    def year(self) -> float:
        return float(Time(self.mjd, format="mjd").jyear)

    @property
    def file_url(self) -> str:
        return file_url(self.uri)

    def as_dict(self) -> dict:
        return {
            **asdict(self),
            "iso": self.iso,
            "year": round(self.year, 3),
            "file_url": self.file_url,
        }


def valid_uri(uri: str) -> bool:
    """Only NEOSSat file identifiers are accepted (the server fetches these, so nothing else may get through)."""
    return bool(_URI_RE.match(uri or ""))


def file_url(uri: str) -> str:
    """Public download address for a CADC file identifier ('cadc:NEOSSAT/x.fits' -> .../data/pub/NEOSSAT/x.fits)."""
    return f"{CADC_DATA}/{uri.split(':', 1)[-1]}"


def cutout_url(uri: str, ra: float, dec: float, radius_deg: float) -> str:
    return f"{CADC_CUTOUT}?ID={quote(uri, safe='')}&CIRCLE={ra:.6f}+{dec:.6f}+{radius_deg:.5f}"


def build_query(
    ra: float, dec: float, radius_deg: float, with_artifact: bool = True, limit: int = 3000
) -> str:
    """ADQL against CADC's CAOM-2 tables: every NEOSSat image whose footprint touches the search circle."""
    art_cols = ", a.uri, a.productType" if with_artifact else ""
    art_join = " JOIN caom2.Artifact AS a ON a.planeID = p.planeID" if with_artifact else ""
    return (
        f"SELECT TOP {int(limit)} o.observationID, o.target_name, p.productID, p.time_bounds_lower, p.time_exposure{art_cols} "
        f"FROM caom2.Observation AS o JOIN caom2.Plane AS p ON o.obsID = p.obsID{art_join} "
        f"WHERE o.collection = '{COLLECTION}' "
        f"AND INTERSECTS(p.position_bounds, CIRCLE('ICRS', {ra:.6f}, {dec:.6f}, {radius_deg:.5f})) = 1 "
        f"ORDER BY p.time_bounds_lower"
    )


def _product_of(product_id: str, uri: str) -> str:
    text = f"{product_id} {uri}".lower()
    for name in ("cord", "cor", "clean"):
        if re.search(rf"(^|[_\s/]){name}($|[.\s_])", text):
            return name
    return "raw"


def _f(v):
    try:
        x = float(v)
        return x if np.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def parse_rows(rows: list[dict]) -> list[Visit]:
    """CSV rows from the TAP service -> one Visit per observation (best processing level), oldest first."""
    best: dict[str, Visit] = {}
    for r in rows:
        r = {k.lower().split(".")[-1]: v for k, v in r.items()}
        mjd = _f(r.get("time_bounds_lower"))
        obs_id = str(r.get("observationid") or "").strip()
        if mjd is None or not obs_id:
            continue
        if str(r.get("producttype") or "science").lower() not in ("science", ""):
            continue  # skip previews / thumbnails / auxiliary files
        uri = str(r.get("uri") or "").strip()
        if uri and not uri.lower().split("?")[0].endswith((".fits", ".fits.gz", ".fits.fz")):
            continue
        v = Visit(
            obs_id,
            str(r.get("target_name") or "").strip(),
            mjd,
            _f(r.get("time_exposure")),
            _product_of(str(r.get("productid") or ""), uri),
            uri,
        )
        cur = best.get(obs_id)
        if cur is None or PRODUCT_RANK[v.product] < PRODUCT_RANK[cur.product]:
            best[obs_id] = v
    return sorted(best.values(), key=lambda v: (v.mjd, v.observation_id))


def _tap(query: str, timeout: float) -> list[dict]:
    r = requests.post(
        CADC_TAP_SYNC,
        data={"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "CSV", "QUERY": query},
        timeout=timeout,
    )
    if r.status_code >= 400:
        raise RuntimeError(f"CADC answered {r.status_code}: {r.text[:300].strip()}")
    return list(csv.DictReader(io.StringIO(r.text)))


def search(
    ra: float, dec: float, radius_deg: float = 0.05, timeout: float = 60.0, fetch=None
) -> list[Visit]:
    """All NEOSSat observations covering the position, chronological. `fetch(query)->rows` can be injected for tests.
    If the richer query (with file names) is rejected, a simpler one is tried so the visit list still works.
    """
    fetch = fetch or (lambda q: _tap(q, timeout))
    try:
        return parse_rows(fetch(build_query(ra, dec, radius_deg, with_artifact=True)))
    except RuntimeError:
        return parse_rows(fetch(build_query(ra, dec, radius_deg, with_artifact=False)))


def thin(visits: list[Visit], max_n: int) -> list[Visit]:
    """At most max_n visits spread evenly in time (NEOSSat often takes hundreds of frames in one staring run)."""
    if max_n <= 0 or len(visits) <= max_n:
        return visits
    idx = (
        sorted({round(i * (len(visits) - 1) / (max_n - 1)) for i in range(max_n)})
        if max_n > 1
        else [0]
    )
    return [visits[i] for i in idx]


def summarise(visits: list[Visit]) -> dict:
    if not visits:
        return {"count": 0}
    years = [v.year for v in visits]
    return {
        "count": len(visits),
        "first": visits[0].iso,
        "last": visits[-1].iso,
        "span_years": round(max(years) - min(years), 2),
        "targets": sorted({v.target for v in visits if v.target})[:12],
    }


def image_from_fits(blob: bytes) -> np.ndarray:
    """First 2-D image in a FITS file, as float32 (NaN where non-finite)."""
    from astropy.io import fits

    with fits.open(io.BytesIO(blob), memmap=False) as hdul:
        for h in hdul:
            if getattr(h, "data", None) is not None and np.ndim(h.data) >= 2:
                a = np.array(h.data, dtype=np.float32)
                while a.ndim > 2:
                    a = a[0]
                return a
    raise ValueError("no image found in the FITS file")


def to_png(img: np.ndarray, size: int = 320) -> bytes:
    """Asinh-stretched greyscale PNG (background-subtracted with the median, robust percentile limits)."""
    from PIL import Image

    a = img - np.nanmedian(img)
    good = a[np.isfinite(a)]
    if good.size == 0:
        raise ValueError("cutout is empty")
    lo, hi = np.percentile(good, [5.0, 99.7])
    hi = max(hi, lo + 1e-6)
    z = np.clip((np.nan_to_num(a, nan=lo) - lo) / (hi - lo), 0, 1)
    z = np.arcsinh(z / 0.08) / np.arcsinh(1 / 0.08)
    im = Image.fromarray((255 * z[::-1]).astype(np.uint8), "L")  # FITS rows run bottom-up
    im = im.resize(
        (size, max(1, round(size * im.height / im.width))),
        Image.NEAREST if im.width < size / 2 else Image.BILINEAR,
    )
    buf = io.BytesIO()
    im.save(buf, "PNG", optimize=True)
    return buf.getvalue()


def cutout_png(
    uri: str, ra: float, dec: float, radius_deg: float, size: int = 320, timeout: float = 90.0
) -> bytes:
    """PNG of the patch from one NEOSSat image (cached on disk). Orientation follows the detector, not North-up."""
    if not valid_uri(uri):
        raise ValueError("not a NEOSSat file identifier")
    key = hashlib.sha1(f"{uri}|{ra:.5f}|{dec:.5f}|{radius_deg:.5f}|{size}".encode()).hexdigest()[
        :24
    ]
    path = NEOSSAT_DIR / f"{key}.png"
    if path.exists() and path.stat().st_size:
        return path.read_bytes()
    r = requests.get(cutout_url(uri, ra, dec, radius_deg), timeout=timeout)
    if r.status_code >= 400:
        raise RuntimeError(f"CADC cutout service answered {r.status_code}")
    png = to_png(image_from_fits(r.content), size)
    NEOSSAT_DIR.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png)
    return png
