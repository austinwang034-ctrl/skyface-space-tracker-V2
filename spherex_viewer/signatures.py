"""Picking out wavelengths: chemical signatures and temperature.

Each SPHEREx exposure of a patch is taken at one near-infrared wavelength. Molecules absorb or
emit at particular wavelengths, so stacking only the exposures inside a molecule's band, and
comparing them with exposures just outside it, shows where that molecule is at work. An object
that is fainter inside the band than beside it is absorbing there; one that is brighter is
emitting.

Temperature shows up as colour: cool objects are brighter at long wavelengths, hot ones at
short wavelengths. Fitting a blackbody curve to brightness against wavelength gives a rough
"colour temperature".

This module holds the list of signatures and the arithmetic. It does no downloading.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .config import BANDS

# Physical constants for the blackbody curve (SI units).
PLANCK_H = 6.62607015e-34
LIGHT_C = 2.99792458e8
BOLTZMANN_K = 1.380649e-23

# A temperature estimate needs brightness measured across at least this much wavelength.
MIN_COLOUR_SPAN_UM = 1.0


@dataclass(frozen=True)
class Signature:
    """A wavelength range worth looking at, and what it tells you."""

    key: str
    name: str
    lo_um: float
    hi_um: float
    kind: str  # "molecule", "hydrogen" or "temperature"
    tells_you: str

    def as_dict(self) -> dict:
        return {**asdict(self), "detectors": detectors_covering(self.lo_um, self.hi_um)}


# Wavelengths are the usual band centres from infrared spectroscopy; ranges are widened to
# roughly SPHEREx's spectral resolution so each holds a few exposures.
SIGNATURES = [
    Signature(
        "water_ice",
        "Water ice",
        2.90,
        3.20,
        "molecule",
        "Frozen water on dust grains absorbs here. Deep in cold clouds where stars and planets "
        "form, and on comets.",
    ),
    Signature(
        "co2_ice",
        "Carbon dioxide ice",
        4.20,
        4.35,
        "molecule",
        "Frozen carbon dioxide absorbs here. Earth's air blocks this band, so only space "
        "telescopes see it.",
    ),
    Signature(
        "co",
        "Carbon monoxide",
        4.55,
        4.75,
        "molecule",
        "Carbon monoxide gas and ice absorb here: cool stars, brown dwarfs and icy clouds.",
    ),
    Signature(
        "methane",
        "Methane",
        3.20,
        3.45,
        "molecule",
        "Methane absorbs here in the coldest brown dwarfs and giant planets. In dusty regions "
        "the same band glows from carbon-rich molecules (PAHs).",
    ),
    Signature(
        "water_vapour",
        "Water vapour",
        1.33,
        1.48,
        "molecule",
        "Steam in the atmospheres of cool stars and brown dwarfs absorbs here.",
    ),
    Signature(
        "co_bandhead",
        "Carbon monoxide in cool stars",
        2.29,
        2.40,
        "molecule",
        "Cool giant stars such as R Draconis show a sharp drop here from carbon monoxide.",
    ),
    Signature(
        "paschen_alpha",
        "Hydrogen (Paschen alpha)",
        1.85,
        1.90,
        "hydrogen",
        "Glowing hydrogen gas: young stars pulling in material, such as RW Aurigae, and "
        "star-forming clouds.",
    ),
    Signature(
        "brackett_alpha",
        "Hydrogen (Brackett alpha)",
        4.02,
        4.08,
        "hydrogen",
        "Glowing hydrogen seen through dust that hides it at shorter wavelengths.",
    ),
    Signature(
        "warm",
        "Hot objects (shortest wavelengths)",
        0.75,
        1.30,
        "temperature",
        "Stars like the Sun and hotter are brightest at this end of SPHEREx's range.",
    ),
    Signature(
        "cool",
        "Cold objects (longest wavelengths)",
        3.80,
        5.00,
        "temperature",
        "The coldest brown dwarfs, below about 500 K, show up almost only here.",
    ),
]


def detectors_covering(lo_um: float, hi_um: float) -> list[str]:
    """Short names (D1..D6) of the SPHEREx detectors whose range overlaps [lo, hi]."""
    return [
        band.replace("SPHEREx-", "")
        for band, (band_lo, band_hi) in BANDS.items()
        if band_hi >= lo_um and band_lo <= hi_um
    ]


def split_frames(wavelengths_um, lo_um: float, hi_um: float) -> tuple[list[int], list[int]]:
    """Indices of frames inside the band, and of comparison frames just outside it.

    The comparison ("continuum") frames come from a strip on each side of the band as wide as
    the band itself, and at least 0.1 µm, so both sides are sampled when the data allow.
    """
    waves = np.asarray(wavelengths_um, dtype=float)
    margin = max(hi_um - lo_um, 0.1)
    inside = np.isfinite(waves) & (waves >= lo_um) & (waves <= hi_um)
    beside = np.isfinite(waves) & ~inside & (waves >= lo_um - margin) & (waves <= hi_um + margin)
    return list(np.flatnonzero(inside)), list(np.flatnonzero(beside))


def band_strength(flux_in, flux_beside) -> dict | None:
    """How much fainter or brighter the object is inside the band than beside it.

    Returns {"ratio", "percent", "verdict"} or None when either side has no usable brightness.
    percent is negative for absorption (fainter inside the band).
    """
    inside = np.asarray(flux_in, dtype=float)
    beside = np.asarray(flux_beside, dtype=float)
    inside, beside = inside[np.isfinite(inside)], beside[np.isfinite(beside)]
    if inside.size == 0 or beside.size == 0:
        return None
    reference = float(np.median(beside))
    if reference <= 0:
        return None
    ratio = float(np.median(inside)) / reference
    percent = (ratio - 1.0) * 100.0
    # Scatter between exposures tells us how big a difference is believable.
    spread = float(np.std(beside) / reference * 100.0) if beside.size > 1 else 25.0
    threshold = max(10.0, 2.0 * spread)
    if percent <= -threshold:
        verdict = "absorbs"
    elif percent >= threshold:
        verdict = "emits"
    else:
        verdict = "no clear feature"
    return {"ratio": ratio, "percent": percent, "threshold_percent": threshold, "verdict": verdict}


def blackbody_fnu(wavelength_um, temperature_k: float) -> np.ndarray:
    """Shape of a blackbody's brightness per unit frequency (the unit SPHEREx fluxes use)."""
    wavelength_m = np.asarray(wavelength_um, dtype=float) * 1e-6
    frequency = LIGHT_C / wavelength_m
    exponent = np.clip(PLANCK_H * frequency / (BOLTZMANN_K * temperature_k), None, 700.0)
    return frequency**3 / np.expm1(exponent)


def colour_temperature(wavelengths_um, fluxes, errors=None) -> dict | None:
    """Rough temperature from the shape of brightness against wavelength.

    Tries blackbody curves from 200 K to 12,000 K, scales each to the data, and keeps the best.
    Needs at least 8 measurements spanning 1 µm, which usually means two or more detectors. Returns None when the data cannot say.

    This is a colour temperature, not a physical one: real stars and brown dwarfs have
    absorption bands, and dust reddens light, so treat it as a guide.
    """
    waves = np.asarray(wavelengths_um, dtype=float)
    flux = np.asarray(fluxes, dtype=float)
    err = np.ones_like(flux) if errors is None else np.asarray(errors, dtype=float)
    good = np.isfinite(waves) & np.isfinite(flux) & np.isfinite(err) & (err > 0) & (flux > 0)
    waves, flux, err = waves[good], flux[good], err[good]
    if waves.size < 8 or np.ptp(waves) < MIN_COLOUR_SPAN_UM:
        return None

    temperatures = np.geomspace(200.0, 12000.0, 240)
    misfit = np.empty_like(temperatures)
    for i, temperature in enumerate(temperatures):
        model = blackbody_fnu(waves, temperature)
        scale = np.sum(flux * model / err**2) / np.sum(model**2 / err**2)
        misfit[i] = np.sum(((flux - scale * model) / err) ** 2)
    best = int(np.argmin(misfit))

    # If the best curve sits at the edge of the range, the data only give a limit.
    limit = None
    if best == 0:
        limit = "cooler than"
    elif best == len(temperatures) - 1:
        limit = "hotter than"
    # SPHEREx stops at 5 µm. Above about 6,000 K every blackbody has the same slope there,
    # so the fit cannot tell hot from very hot.
    if limit is None and temperatures[best] > 6000:
        limit = "hotter than"
        best = int(np.searchsorted(temperatures, 6000.0))
    return {
        "temperature_k": float(temperatures[best]),
        "limit": limit,
        "n_points": int(waves.size),
        "range_um": [float(waves.min()), float(waves.max())],
        "peak_um": 2897.8 / float(temperatures[best]),
    }
