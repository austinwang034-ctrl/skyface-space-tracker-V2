"""Check that every real data source works from this computer.

Run it from the project folder, with the virtual environment active:

    python check_live.py

It tries each archive the site depends on, one at a time, using Barnard's Star and
RW Aurigae as test positions, and prints OK or FAILED with the reason. Nothing is
changed on disk except the normal download cache. If a line says FAILED, copy the
whole output and send it to whoever maintains the code.
"""

from __future__ import annotations

import sys
import time
import traceback

from astropy.coordinates import SkyCoord
from astropy.time import Time

BARNARD = (269.452, 4.6934)  # degrees, J2000
RW_AUR = (76.9566, 30.4011)

results: list[tuple[str, bool, str]] = []


def check(name: str):
    """Decorator: run one check, time it, and record OK / FAILED instead of crashing."""

    def wrap(fn):
        start = time.time()
        try:
            detail = fn()
            ok = True
        except Exception as exc:  # noqa: BLE001
            detail = f"{type(exc).__name__}: {exc}"
            ok = False
            traceback.print_exc(limit=2)
        results.append((name, ok, detail))
        print(
            f"[{'OK' if ok else 'FAILED':6}] {name} ({time.time() - start:.0f} s)\n         {detail}\n"
        )
        return fn

    return wrap


@check("SPHEREx image search (NASA/IPAC IRSA)")
def _search():
    from spherex_viewer import archive

    found = archive.query_observations(*BARNARD, 0.05, ["SPHEREx-D2"])
    if not found:
        raise RuntimeError("the search worked but returned no images")
    releases = sorted({o.release for o in found})
    _search.sample = found[len(found) // 2]
    return f"{len(found)} images of Barnard's Star in band D2, releases: {', '.join(releases)}"


@check("SPHEREx image download and reading (IRSA cutout)")
def _download():
    from spherex_viewer import archive, frames

    sample = getattr(_search, "sample", None)
    if sample is None:
        raise RuntimeError("skipped because the search failed")
    path = archive.download_cutout(sample.url)
    frame = frames.load_frame(sample, path)
    date = Time(frame.mjd, format="mjd").iso[:10]
    return f"{frame.data.shape[1]}x{frame.data.shape[0]} pixels, taken {date}, wavelength {frame.wavelength_um:.2f} µm"


@check("Object identification (SIMBAD)")
def _simbad():
    from spherex_viewer import identify

    result = identify.identify(*BARNARD, epoch=2000.0)
    if not result.match:
        raise RuntimeError(f"no object returned for Barnard's Star's position ({result.note})")
    return f"identified as: {result.match.name}"


@check("Gaia stars for plate solving (ESA, then MAST as backup)")
def _gaia():
    from spherex_viewer import neighborhood

    stars = neighborhood.gaia_cone(SkyCoord(*BARNARD, unit="deg"), 4.0, mag_limit=17.0)
    return f"{len(stars)} stars, served by: {stars.meta.get('source', 'cache')}"


@check("Gaia backup copy on its own (MAST)")
def _mast():
    from spherex_viewer import neighborhood

    stars = neighborhood._gaia_mast(SkyCoord(*BARNARD, unit="deg"), 4.0, 17.0)
    return f"{len(stars)} stars"


@check("NEOSSat image search (CADC)")
def _neossat():
    from spherex_viewer import neossat

    visits = neossat.search(*RW_AUR, radius_deg=0.2)
    _neossat.visits = visits
    if not visits:
        return "search worked; NEOSSat has no images of RW Aurigae within 0.2 degrees"
    with_files = sum(neossat.valid_uri(v.uri) for v in visits)
    return f"{len(visits)} images, {visits[0].iso[:10]} to {visits[-1].iso[:10]}, {with_files} with file names"


@check("NEOSSat image cutout (CADC)")
def _neossat_cutout():
    from spherex_viewer import neossat

    usable = [v for v in getattr(_neossat, "visits", []) if neossat.valid_uri(v.uri)]
    if not usable:
        return "skipped: no NEOSSat image to cut out"
    png = neossat.cutout_png(usable[0].uri, *RW_AUR, radius_deg=0.05)
    return f"cutout of {usable[0].uri} rendered ({len(png)} bytes)"


@check("Known asteroids and comets (NASA JPL)")
def _jpl():
    from spherex_viewer import smallbodies

    mjd = Time("2026-01-15T00:00:00").mjd
    # A field on the ecliptic, where asteroids are common.
    found = smallbodies.search(100.0, 23.0, mjd, half_width_deg=0.5)
    nearest = found["bodies"][0]["name"] if found["bodies"] else "none"
    return f"{len(found['bodies'])} known small bodies in the test field; nearest: {nearest}"


failed = [name for name, ok, _ in results if not ok]
print("=" * 60)
print(f"{len(results) - len(failed)} of {len(results)} checks passed.")
if failed:
    print("Failed: " + "; ".join(failed))
sys.exit(1 if failed else 0)
