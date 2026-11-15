"""Known asteroids and comets near a patch of sky at a given time, from NASA JPL.

When something moves in the tracker, the first question is whether it is already known.
JPL's free Small-Body Identification service answers that: give it a place and a time and
it lists every catalogued asteroid and comet in the field.

Service documentation: https://ssd-api.jpl.nasa.gov/doc/sb_ident.html

One limitation to know about: the service needs an observatory on the ground, and SPHEREx is
in low Earth orbit. We ask from Mauna Kea (code 568) instead. For main-belt asteroids the
difference is a few arcseconds, less than one SPHEREx pixel; for objects passing very close
to Earth it can be much larger, so the search field is made generously wide.
"""

from __future__ import annotations

import math

import requests
from astropy.time import Time

API_URL = "https://ssd-api.jpl.nasa.gov/sb_ident.api"
OBSERVATORY = "568"  # Mauna Kea; the service does not accept a geocentric observer
MIN_HALF_WIDTH_DEG = 0.25


def _sexagesimal(value: float, places: int = 2) -> str:
    """Degrees or hours -> 'dd-mm-ss.ss', with 'M' in place of a minus sign (the API's format)."""
    sign = "M" if value < 0 else ""
    value = abs(value)
    whole = int(value)
    minutes = int((value - whole) * 60)
    seconds = (value - whole - minutes / 60) * 3600
    if round(seconds, places) >= 60:  # avoid '59-60.00' after rounding
        seconds = 0.0
        minutes += 1
    if minutes >= 60:
        minutes = 0
        whole += 1
    return f"{sign}{whole:02d}-{minutes:02d}-{seconds:0{3 + places}.{places}f}"


def build_params(ra_deg: float, dec_deg: float, mjd: float, half_width_deg: float) -> dict:
    """Query parameters for one field at one time."""
    when = Time(mjd, format="mjd", scale="utc").strftime("%Y-%m-%d_%H:%M:%S")
    half_width = max(half_width_deg, MIN_HALF_WIDTH_DEG)
    return {
        "mpc-code": OBSERVATORY,
        "obs-time": when,
        "fov-ra-center": _sexagesimal(ra_deg / 15.0),
        "fov-dec-center": _sexagesimal(dec_deg),
        # RA half-width is in degrees of RA, so widen it away from the equator.
        "fov-ra-hwidth": f"{min(half_width / max(math.cos(math.radians(dec_deg)), 0.05), 5.0):.4f}",
        "fov-dec-hwidth": f"{half_width:.4f}",
        "two-pass": "true",
        "suppress-first-pass": "true",
        "req-elem": "false",
        "mag-required": "false",
    }


def _number(text) -> float | None:
    try:
        return float(str(text).strip())
    except (TypeError, ValueError):
        return None


def parse_response(payload: dict) -> list[dict]:
    """Turn the service's table (a list of column names plus rows) into plain dictionaries,
    nearest to the field centre first."""
    fields = payload.get("fields_second") or payload.get("fields_first") or []
    rows = payload.get("data_second_pass") or payload.get("data_first_pass") or []

    def column(*words: str) -> int | None:
        for i, heading in enumerate(fields):
            if all(word.lower() in heading.lower() for word in words):
                return i
        return None

    i_name, i_ra, i_dec = (
        column("object"),
        column("astrometric", "ra"),
        column("astrometric", "dec"),
    )
    i_dist, i_mag = column("norm"), column("magnitude")

    def cell(row, index):
        return row[index] if index is not None and index < len(row) else None

    found = []
    for row in rows:
        found.append(
            {
                "name": str(cell(row, i_name) or "").strip(),
                "ra": str(cell(row, i_ra) or "").strip(),
                "dec": str(cell(row, i_dec) or "").strip(),
                "offset_arcsec": _number(cell(row, i_dist)),
                "vmag": _number(cell(row, i_mag)),
            }
        )
    found.sort(
        key=lambda body: body["offset_arcsec"] if body["offset_arcsec"] is not None else 1e12
    )
    return found


def search(
    ra_deg: float,
    dec_deg: float,
    mjd: float,
    half_width_deg: float = 0.25,
    timeout: float = 60.0,
    fetch=None,
) -> dict:
    """Known small bodies in the field at that time. `fetch(params) -> dict` can be injected for tests."""
    params = build_params(ra_deg, dec_deg, mjd, half_width_deg)
    if fetch is None:

        def fetch(p):
            reply = requests.get(API_URL, params=p, timeout=timeout)
            if reply.status_code >= 400:
                raise RuntimeError(f"JPL answered {reply.status_code}: {reply.text[:200].strip()}")
            return reply.json()

    payload = fetch(params)
    return {
        "time_utc": Time(mjd, format="mjd", scale="utc").iso[:19],
        "half_width_arcmin": float(params["fov-dec-hwidth"]) * 60,
        "observer": "Mauna Kea (MPC 568), standing in for SPHEREx",
        "bodies": parse_response(payload),
        "source": "NASA/JPL Small-Body Identification API",
    }
