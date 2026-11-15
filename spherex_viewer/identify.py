"""What is at this RA/Dec? SIMBAD cone search, proper-motion-aware ranking, and reference-image URLs."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from urllib.parse import quote

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.time import Time

from . import targets
from .targets import Target, simbad_try

TYPE_LABELS = {
    "*": "Star",
    "PM*": "High-proper-motion star",
    "BD*": "Brown dwarf",
    "BD?": "Brown dwarf candidate",
    "Y*O": "Young stellar object",
    "TT*": "T Tauri star",
    "Or*": "Orion-type variable star",
    "V*": "Variable star",
    "Mi*": "Mira variable",
    "LP*": "Long-period variable",
    "Ir*": "Irregular variable",
    "Pu*": "Pulsating variable",
    "EB*": "Eclipsing binary",
    "SB*": "Spectroscopic binary",
    "**": "Double or multiple star",
    "RG*": "Red giant",
    "AB*": "Asymptotic giant branch star",
    "WD*": "White dwarf",
    "Em*": "Emission-line star",
    "Fl*": "Flare star",
    "BY*": "BY Draconis variable",
    "RR*": "RR Lyrae variable",
    "Ce*": "Cepheid variable",
    "G": "Galaxy",
    "IR": "Infrared source",
    "X": "X-ray source",
    "Pl": "Exoplanet",
    "Pl?": "Exoplanet candidate",
}
MAX_PM_MASYR_PER_YEAR = (
    11.0  # Barnard's Star is ~10.4 arcsec/yr: search radius grows by this per year of epoch gap
)


@dataclass
class ObjectInfo:
    name: str
    ra: float  # J2000 (SIMBAD epoch), degrees
    dec: float
    otype: str | None = None
    sp_type: str | None = None
    pmra: float | None = None  # mas/yr
    pmdec: float | None = None
    parallax: float | None = None  # mas
    mags: dict = field(default_factory=dict)
    sep_arcsec: float | None = None  # separation from the query position at the query epoch

    def to_dict(self) -> dict:
        has_pm = self.pmra is not None or self.pmdec is not None
        pm = float(np.hypot(self.pmra or 0.0, self.pmdec or 0.0)) if has_pm else None
        d = 1000.0 / self.parallax if self.parallax and self.parallax > 0.5 else None
        return {
            "name": self.name,
            "otype": self.otype,
            "type_label": TYPE_LABELS.get(self.otype or "", self.otype or "Unknown type"),
            "sp_type": self.sp_type,
            "pmra": self.pmra,
            "pmdec": self.pmdec,
            "pm_total": pm,
            "parallax": self.parallax,
            "distance_pc": d,
            "distance_ly": d * 3.26156 if d else None,
            "vtan_kms": 4.74047 * pm / 1000.0 * d if pm and d else None,
            "mags": self.mags,
            "ra": self.ra,
            "dec": self.dec,
            "sep_arcsec": self.sep_arcsec,
            "coords": SkyCoord(self.ra * u.deg, self.dec * u.deg).to_string("hmsdms", precision=2),
            "simbad_url": "https://simbad.cds.unistra.fr/simbad/sim-id?Ident=" + quote(self.name),
        }


@dataclass
class IdResult:
    match: ObjectInfo | None
    nearby: list[ObjectInfo]
    note: str
    canned: bool = False


# ------------------------------------------------------------------------------------------- SIMBAD query
def _num(v):
    try:
        f = float(np.ma.filled(v, np.nan)) if np.ma.is_masked(v) else float(v)
        return f if np.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _query_simbad(center: SkyCoord, radius_arcsec: float) -> list[dict]:
    """Cone search on whichever SIMBAD server answers. Returns plain dict rows."""
    return simbad_try(lambda server: _query_simbad_on(server, center, radius_arcsec))


def _query_simbad_on(server: str, center: SkyCoord, radius_arcsec: float) -> list[dict]:
    """Cone search; tolerant of astroquery's old and new SIMBAD column names."""
    from astroquery.simbad import Simbad

    s = Simbad()
    s.server = server
    s.TIMEOUT = 10
    s.ROW_LIMIT = 500
    for group in (
        ("otype",),
        ("sp_type", "sptype"),
        ("pmra",),
        ("pmdec",),
        ("plx_value", "plx"),
        ("V", "flux(V)"),
        ("J", "flux(J)"),
        ("K", "flux(K)"),
    ):
        for nm in group:
            try:
                s.add_votable_fields(nm)
                break
            except Exception:  # noqa: BLE001  (field unknown in this astroquery version: try the alias)
                continue
    t = s.query_region(center, radius=radius_arcsec * u.arcsec)
    if t is None:
        return []
    cols = {c.lower(): c for c in t.colnames}
    pick = lambda r, *names: next((r[cols[n]] for n in names if n in cols), None)  # noqa: E731
    rows = []
    for r in t:
        ra, dec = pick(r, "ra"), pick(r, "dec")
        if isinstance(ra, (str, bytes)):
            c = SkyCoord(str(ra), str(dec), unit=(u.hourangle, u.deg))
            ra, dec = c.ra.deg, c.dec.deg
        if _num(ra) is None:
            continue
        sp, ot = pick(r, "sp_type", "sptype"), pick(r, "otype")
        rows.append(
            {
                "name": str(pick(r, "main_id")).strip(),
                "ra": float(ra),
                "dec": float(dec),
                "otype": None if ot is None or np.ma.is_masked(ot) else str(ot).strip(),
                "sp_type": (
                    None
                    if sp is None or np.ma.is_masked(sp) or not str(sp).strip()
                    else str(sp).strip()
                ),
                "pmra": _num(pick(r, "pmra")),
                "pmdec": _num(pick(r, "pmdec")),
                "parallax": _num(pick(r, "plx_value", "plx")),
                "mags": {
                    b: v
                    for b, v in (
                        (b, _num(pick(r, b.lower(), f"flux_{b.lower()}", f"flux({b.lower()})")))
                        for b in "VJK"
                    )
                    if v is not None
                },
            }
        )
    return rows


def rank_candidates(
    rows: list[dict], center: SkyCoord, epoch: float, tol_arcsec: float = 15.0
) -> IdResult:
    """Move every catalogue object to `epoch` with its proper motion and rank by distance to `center`."""
    if not rows:
        return IdResult(None, [], "SIMBAD has no object near these coordinates.")
    f = lambda k, r: r[k] if r.get(k) is not None else 0.0  # noqa: E731
    c = SkyCoord(
        ra=[r["ra"] for r in rows] * u.deg,
        dec=[r["dec"] for r in rows] * u.deg,
        pm_ra_cosdec=[f("pmra", r) for r in rows] * u.mas / u.yr,
        pm_dec=[f("pmdec", r) for r in rows] * u.mas / u.yr,
        obstime=Time(2000.0, format="jyear"),
        frame="icrs",
    )
    moved = c.apply_space_motion(new_obstime=Time(epoch, format="jyear"))
    sep = SkyCoord(moved.ra, moved.dec, frame="icrs").separation(center).arcsec
    objs = []
    for i in np.argsort(sep):
        r = rows[i]
        objs.append(
            ObjectInfo(
                r["name"],
                r["ra"],
                r["dec"],
                r["otype"],
                r["sp_type"],
                r["pmra"],
                r["pmdec"],
                r["parallax"],
                r["mags"],
                float(sep[i]),
            )
        )
    best = objs[0]
    if best.sep_arcsec <= tol_arcsec:
        q = "Strong match" if best.sep_arcsec <= 5 else "Likely match"
        return IdResult(
            best,
            objs[1:8],
            f"{q}: {best.sep_arcsec:.1f}″ from your coordinates at epoch {epoch:g}.",
        )
    return IdResult(
        None,
        objs[:8],
        f"No catalogued object within {tol_arcsec:.0f}″ at epoch {epoch:g}. The closest is {best.name} "
        f"({best.sep_arcsec:.0f}″ away). If your star moves fast, check the epoch of your coordinates.",
    )


_CACHE: dict = {}


def identify(
    ra: float, dec: float, epoch: float = 2000.0, tol_arcsec: float = 15.0, fetch=None
) -> IdResult:
    """Identify the object at (ra, dec) [degrees] valid at `epoch`. Results are cached for an hour."""
    key = (round(ra, 5), round(dec, 5), round(epoch, 2))
    hit = _CACHE.get(key)
    if hit and time.time() - hit[0] < 3600:
        return hit[1]
    center = SkyCoord(ra * u.deg, dec * u.deg, frame="icrs")
    radius = max(
        30.0, tol_arcsec + MAX_PM_MASYR_PER_YEAR * abs(epoch - 2000.0)
    )  # PM-aware: J2000 entry may be far away
    res = rank_candidates(
        (fetch or _query_simbad)(center, min(radius, 600.0)), center, epoch, tol_arcsec
    )
    _CACHE[key] = (time.time(), res)
    return res


def canned_demo() -> IdResult:
    """Offline example (Barnard's Star, values from memory): clearly flagged as canned in the UI."""
    o = ObjectInfo(
        "Barnard's Star (demo)",
        269.452075,
        4.693361,
        "PM*",
        "M4.0V",
        -802.803,
        10362.394,
        546.976,
        {"V": 9.5},
        0.0,
    )
    return IdResult(o, [], "Demo mode: example record, no live lookup.", canned=True)


def target_from_object(o: ObjectInfo) -> Target:
    pm = float(np.hypot(o.pmra or 0.0, o.pmdec or 0.0))
    return Target(
        o.name,
        o.ra,
        o.dec,
        pmra=o.pmra or 0.0,
        pmdec=o.pmdec or 0.0,
        parallax=o.parallax or 0.0,
        ref_epoch=2000.0,
        kind="mover" if pm > 100 else "star",
        notes=f"{TYPE_LABELS.get(o.otype or '', o.otype or '')} {o.sp_type or ''}".strip(),
        source="SIMBAD (identified from your coordinates)",
    )


def resolve_name(name: str) -> dict:
    """Name -> coordinates. SIMBAD first (either server); if it cannot be reached, or does not
    know the name, the built-in test objects are tried before giving up."""
    try:
        r = targets._simbad_lookup(name)
        source = "SIMBAD"
    except Exception:  # noqa: BLE001
        known = targets.builtin_by_name(name)
        if known is None:
            raise
        r = {
            "ra_deg": known.ra_deg,
            "dec_deg": known.dec_deg,
            "ref_epoch": known.ref_epoch,
            "pmra": known.pmra,
            "pmdec": known.pmdec,
        }
        source = "built-in catalogue (SIMBAD could not be reached)"
    return {
        "ra": r["ra_deg"],
        "dec": r["dec_deg"],
        "epoch": r.get("ref_epoch", 2000.0),
        "pmra": _num(r["pmra"]),
        "pmdec": _num(r["pmdec"]),
        "source": source,
    }


def object_from_target(t: Target, sep_arcsec: float | None = None) -> ObjectInfo:
    """A built-in target shown the same way as a SIMBAD record."""
    return ObjectInfo(
        t.name,
        t.ra_deg,
        t.dec_deg,
        pmra=t.pmra or None,
        pmdec=t.pmdec or None,
        parallax=t.parallax or None,
        sep_arcsec=sep_arcsec,
    )


SURVEYS = [
    ("CDS/P/SPHEREx/QR2/color", "SPHEREx (all six detectors, 102 infrared colours, 2025–26)"),
    ("CDS/P/DSS2/color", "DSS2 (visible light, 1950s–90s plates)"),
    ("CDS/P/2MASS/color", "2MASS (near-infrared, ~1999)"),
    ("CDS/P/allWISE/color", "AllWISE (mid-infrared, 2010)"),
]


def reference_images(ra: float, dec: float, fov_arcmin: float, size: int = 480) -> list[dict]:
    """CDS hips2fits cutouts the browser loads directly (North up, East left)."""
    return [
        {
            "id": h,
            "label": label,
            "url": f"https://alasky.cds.unistra.fr/hips-image-services/hips2fits?hips={quote(h, safe='')}"
            f"&width={size}&height={size}&fov={fov_arcmin / 60:.5f}&projection=TAN&coordsys=icrs&ra={ra:.6f}&dec={dec:.6f}&format=jpg",
        }
        for h, label in SURVEYS
    ]
