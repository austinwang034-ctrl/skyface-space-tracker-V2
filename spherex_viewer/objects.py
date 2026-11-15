"""Researched profiles of the featured objects, map labels (bright stars, planets today) and illustrative spectra.

Facts were compiled from the sources listed on each profile (papers, NASA Exoplanet Archive, Wikipedia summaries of
the literature). Model spectra are BLACKBODIES at each object's effective temperature: they show where the light peaks,
not the real molecular absorption bands, and are labelled as illustrative in the UI."""

from __future__ import annotations

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord, get_body
from astropy.time import Time

from .config import BANDS
from .targets import TARGETS, Target, TargetUnavailable, _sexa, resolve_target

H, C, K = 6.62607015e-34, 2.99792458e8, 1.380649e-23


def planck(wavelength_um, teff_k: float) -> np.ndarray:
    lam = np.asarray(wavelength_um, float) * 1e-6
    return 1.0 / (lam**5 * np.expm1(H * C / (lam * K * teff_k)))


def wien_peak_um(teff_k: float) -> float:
    return 2897.771955 / teff_k


def channel_centres() -> list[float]:
    """Nominal 102 SPHEREx channels: 17 per detector, evenly spaced in wavelength within each band (approximate)."""
    out = []
    for lo, hi in BANDS.values():
        w = (hi - lo) / 17.0
        out += [lo + (i + 0.5) * w for i in range(17)]
    return out


def model_curves(components: list[tuple[str, float]], n: int = 140) -> dict:
    lam = np.geomspace(0.7, 5.2, n)
    comps = []
    for label, teff in components:
        f = planck(lam, teff)
        comps.append(
            {
                "label": label,
                "teff_k": teff,
                "peak_um": wien_peak_um(teff),
                "flux": [float(max(v, 1e-12)) for v in f / f.max()],
            }
        )
    ch = np.array(channel_centres())
    s = planck(ch, components[0][1])
    strip = np.clip(np.log10(s / s.max()), -6, 0)
    return {
        "wavelength_um": [float(v) for v in lam],
        "components": comps,
        "channels_um": [float(v) for v in ch],
        "strip_log10": [float(v) for v in strip],
    }


FEATURED = [
    {
        "id": "barnard",
        "name": "Barnard’s Star",
        "target_key": "Barnard's Star",
        "category": "Red dwarf star (M4 V)",
        "headline": "The fastest-moving star in our sky: it drifts about 10 arcseconds every year.",
        "facts": [
            ("Spectral type", "M4 V red dwarf"),
            ("Temperature", "≈ 3,200 K"),
            ("Mass", "≈ 0.16 Suns (150–170 Jupiters)"),
            ("Radius", "≈ 0.19 Suns (about 2 Jupiters)"),
            ("Luminosity", "≈ 0.0034 Suns in total, but only 0.0004 Suns in visible light"),
            ("Distance", "1.83 parsecs (5.96 light-years)"),
            ("Brightness", "V ≈ 9.5: too faint for the naked eye"),
            ("Motion", "10.3″ per year, mostly northward"),
            ("Age", "≈ 10 billion years, among the oldest stars nearby"),
            (
                "Planets",
                "Four sub-Earth-mass planets (b, c, d, e) orbiting every 2.3 to 6.7 days, found in 2024–25",
            ),
        ],
        "light": "A cool red dwarf. Its light is mostly red and near-infrared, with a blackbody peak near 0.9 µm. Titanium oxide and "
        "water vapour carve dark bands into its spectrum. Because it is far brighter in the infrared than in visible light, "
        "it is an easy target for SPHEREx.",
        "spherex": "Brightest in detectors D1–D2 (0.75–1.6 µm) and fading toward 5 µm. At 10″ per year it shifts about 1.7 SPHEREx "
        "pixels each year, so the SPHEREx data (about one year) show a clear drift of roughly two pixels.",
        "features": [
            (0.84, "TiO"),
            (0.99, "FeH"),
            (1.4, "H₂O"),
            (1.9, "H₂O"),
            (2.3, "CO"),
            (4.65, "CO"),
        ],
        "components": [("Barnard’s Star (≈3,200 K)", 3195)],
        "min_fov_deg": 0.12,
        "tracker_fov": 5,
        "band": "SPHEREx-D2",
        "fallback": None,
        "sources": [
            ("Wikipedia: Barnard’s Star", "https://en.wikipedia.org/wiki/Barnard%27s_Star"),
            (
                "NASA Exoplanet Archive: GJ 699",
                "https://exoplanetarchive.ipac.caltech.edu/overview/GJ-699",
            ),
        ],
    },
    {
        "id": "epsindi",
        "name": "Epsilon Indi",
        "target_key": "Epsilon Indi B (Ba/Bb)",
        "category": "Two brown dwarfs + a giant planet around a nearby star",
        "headline": "A nearby system with a Sun-like host star, a directly imaged giant planet and a pair of brown dwarfs.",
        "facts": [
            ("Distance", "≈ 3.6 parsecs (11.9 light-years)"),
            ("Ba (brown dwarf)", "Type T1–T1.5, ≈ 1,350–1,385 K, ≈ 67 Jupiter masses"),
            ("Bb (brown dwarf)", "Type T6, ≈ 975–1,010 K, ≈ 53 Jupiter masses"),
            ("Ba–Bb orbit", "0.73″ apart; orbit of about 11 years"),
            ("Ba+Bb from the host star", "≈ 1,460 au, about 6.7 arcminutes away"),
            ("Host star (A)", "K dwarf, ≈ 4,630 K, ≈ 0.76 Suns in mass, ≈ 0.22 Suns in luminosity"),
            (
                "Planet (Ab)",
                "Gas giant, ≈ 6.5–7.6 Jupiter masses, ≈ 275 K; the nearest exoplanet imaged directly (JWST, 2024)",
            ),
            ("Age", "≈ 3.5 billion years"),
            ("Motion", "≈ 4.7″ per year"),
        ],
        "light": "Brown dwarfs are too light to fuse hydrogen, so they glow from leftover heat. T dwarfs show strong methane absorption near "
        "1.6–1.7, 2.2 and 3.3 µm and water bands, so their light escapes through narrow windows near 1.1, 1.3, 1.6 and 4–5 µm. "
        "Hotter Ba (T1) sits at the transition from cloudy L dwarfs to methane-rich T dwarfs; cooler Bb (T6) is strongly methane-dominated.",
        "spherex": "Ba and Bb are only 0.73″ apart, so SPHEREx (6.15″ pixels) sees them as one blended source. The planet Ab is far too close to its "
        "bright host and too faint to separate. The host star shares the same fast motion.",
        "features": [
            (1.4, "H₂O"),
            (1.67, "CH₄"),
            (1.9, "H₂O"),
            (2.2, "CH₄"),
            (3.3, "CH₄"),
            (4.65, "CO"),
        ],
        "components": [
            ("Ba (T1, ≈1,370 K)", 1370),
            ("Bb (T6, ≈990 K)", 990),
            ("Host star A (≈4,630 K)", 4630),
            ("Planet Ab (≈275 K)", 275),
        ],
        "min_fov_deg": 0.35,
        "tracker_fov": 15,
        "band": "SPHEREx-D2",
        "fallback": Target(
            "Epsilon Indi (host star A, offline fallback)",
            *_sexa("22h03m21.65s", "-56d47m09.5s"),
            pmra=3981.977,
            pmdec=-2466.832,
            parallax=270.658,
            ref_epoch=2000.0,
        ),
        "sources": [
            (
                "Dynamical masses of ε Indi Ba and Bb (Chen et al.)",
                "https://arxiv.org/abs/2205.08077",
            ),
            (
                "ε Indi Ab: precise mass and luminosity (Sanghi et al. 2026)",
                "https://arxiv.org/abs/2603.08787",
            ),
            ("Wikipedia: Epsilon Indi Ab", "https://en.wikipedia.org/wiki/Epsilon_Indi_Ab"),
        ],
    },
    {
        "id": "wise0855",
        "name": "WISE 0855−0714",
        "target_key": "WISE 0855-0714",
        "category": "Planetary-mass brown dwarf (Y4), the coldest known",
        "headline": "About as warm as a winter day, it is the coldest known brown dwarf and one of our closest neighbours.",
        "facts": [
            ("Spectral type", "Y4 (the coldest brown-dwarf class)"),
            ("Temperature", "≈ 265–285 K (roughly −8 °C to +12 °C)"),
            ("Mass", "≈ 3–10 Jupiter masses (some analyses lower, ≈ 2–4)"),
            ("Radius", "≈ 0.9 Jupiters (derived)"),
            (
                "Distance",
                "2.28 parsecs (7.43 light-years), the 4th-closest star or brown-dwarf system",
            ),
            ("Brightness", "≈ 13.9 mag at 4.6 µm; invisible in visible light"),
            ("Motion", "≈ 8.1″ per year (very high)"),
            (
                "Atmosphere",
                "Methane; water vapour, probably water clouds that change thickness as it rotates (JWST)",
            ),
            ("Luminosity", "≈ 5 × 10⁻⁸ Suns"),
        ],
        "light": "Almost all of its energy comes out as thermal infrared, with a blackbody peak near 10 µm, beyond SPHEREx. Inside SPHEREx’s range "
        "only a trickle escapes, strongest in the 4–5 µm window. Methane absorbs around 3.3 µm. JWST has detected it at wavelengths as "
        "short as 1.15 µm.",
        "spherex": "We expect it only in the longest-wavelength detectors (D5–D6, 3.8–5 µm), and even there it is faint, so a single-exposure "
        "detection is not guaranteed. The tracker will show whether SPHEREx catches it (unverified expectation).",
        "features": [(1.4, "H₂O"), (3.3, "CH₄"), (4.65, "CO")],
        "components": [("WISE 0855 (≈285 K)", 285), ("WISE 0855 (≈265 K estimate)", 265)],
        "min_fov_deg": 0.2,
        "tracker_fov": 5,
        "band": "SPHEREx-D6",
        "fallback": None,
        "sources": [
            ("Wright & Foley: Flux and colour of WISE 0855", "https://arxiv.org/abs/2505.12105"),
            (
                "Water depletion in WISE 0855 with JWST/MIRI (A&A 695, A224)",
                "https://doi.org/10.1051/0004-6361/202452547",
            ),
            ("Luhman & Esplin: spectral energy distribution", "https://arxiv.org/abs/1605.06655"),
        ],
    },
    {
        "id": "rwaur",
        "name": "RW Aurigae",
        "target_key": "RW Aurigae",
        "category": "Young binary of T Tauri stars with a dusty disk",
        "headline": "A newborn star system that keeps dimming as clouds of dust pass in front of it.",
        "facts": [
            ("System", "RW Aur A (K1–K3, ≈ 1.3 Suns) and B (K5–K6, ≈ 0.8–0.9 Suns)"),
            ("Separation", "≈ 1.45–1.49″ (≈ 270 au)"),
            ("Age", "≈ 3 million years"),
            ("Distance", "≈ 140–180 parsecs (≈ 460–600 light-years)"),
            ("Brightness", "V ≈ 9.6–10.5 normally"),
            (
                "Dimmings",
                "Deep fades of ≈ 2 mag in 2010–11, 2012–13, 2014–16, and a new one from Feb 2025",
            ),
            (
                "Cause",
                "Dusty material from a disk torn by a past stellar fly-by passing in front of star A",
            ),
            ("Motion", "Small: ≈ 25 mas per year"),
        ],
        "light": "Young stars shine from K-type photospheres (roughly 4,000–5,000 K) plus infrared excess from a warm dusty disk and gas falling onto the "
        "star. Dust blocks short wavelengths far more than long ones, so a dimming that is deep in visible light is shallower in the infrared: "
        "comparing colours tells you about the dust.",
        "spherex": "The two stars are blended in SPHEREx pixels. This star barely moves, so the interesting signal is brightness: compare dips "
        "across the detector bands in the spectrogram.",
        "features": [(1.28, "Paβ"), (2.17, "Brγ"), (2.3, "CO"), (4.05, "Brα")],
        "components": [("K-type photosphere (≈4,500 K, illustrative)", 4500)],
        "min_fov_deg": 0.05,
        "tracker_fov": 5,
        "band": "SPHEREx-D2",
        "fallback": Target(
            "RW Aurigae (offline fallback)",
            *_sexa("05h07m49.4562s", "+30d24m04.774s"),
            pmra=4.906,
            pmdec=-24.578,
            parallax=5.454,
            ref_epoch=2000.0,
            kind="variable",
        ),
        "sources": [
            ("Wikipedia: RW Aurigae", "https://en.wikipedia.org/wiki/RW_Aurigae"),
            ("Lux, Mugrauer & Bischoff: RW Aur deep eclipses", "https://arxiv.org/abs/2312.17598"),
            (
                "Rodriguez et al.: recurring occultations of RW Aur",
                "https://arxiv.org/abs/1512.03745",
            ),
        ],
    },
    {
        "id": "rdra",
        "name": "R Draconis",
        "target_key": "R Draconis",
        "category": "Mira variable (pulsating red giant)",
        "headline": "A swollen red giant that breathes in and out every 245 days, changing its brightness about 400-fold.",
        "facts": [
            ("Type", "Mira variable: a cool, pulsating red giant"),
            ("Period", "≈ 245.5 days"),
            ("Brightness range", "V ≈ 6.7 (maximum) to 13.2 (minimum), a change of about 400×"),
            ("Distance", "≈ 660 parsecs (≈ 2,150 light-years)"),
            ("Temperature", "Typical for Miras: roughly 2,500–3,500 K, varying through the cycle"),
            ("Motion", "Small"),
        ],
        "light": "Mira variables are very red and swing by 6+ magnitudes in visible light but only 1–2 magnitudes in the infrared, so SPHEREx sees "
        "much gentler changes than your eyes would. Water vapour and titanium oxide absorb strongly, and the bands deepen and weaken through the cycle.",
        "spherex": "Bright across D1–D3. SPHEREx covers the whole sky about every six months, so only a few points of the 245-day cycle are sampled per year.",
        "features": [
            (0.84, "TiO"),
            (1.4, "H₂O"),
            (1.9, "H₂O"),
            (2.7, "H₂O"),
            (2.3, "CO"),
            (4.65, "CO"),
        ],
        "components": [("Typical Mira (≈3,000 K, illustrative)", 3000)],
        "min_fov_deg": 0.05,
        "tracker_fov": 5,
        "band": "SPHEREx-D2",
        "fallback": Target(
            "R Draconis (offline fallback)",
            *_sexa("16h32m40.3s", "+66d45m18s"),
            ref_epoch=2000.0,
            kind="variable",
        ),
        "sources": [
            (
                "Wikipedia: Mira variable (list with R Dra)",
                "https://en.wikipedia.org/wiki/Mira_variable",
            ),
            (
                "Wikipedia: Draco (R Dra period and range)",
                "https://en.wikipedia.org/wiki/Draco_(constellation)",
            ),
        ],
    },
]
BY_ID = {o["id"]: o for o in FEATURED}

BRIGHT_STARS = [
    ("Sirius", "06h45m08.9s", "-16d42m58s", "Brightest star in the night sky"),
    ("Vega", "18h36m56.3s", "+38d47m01s", "Famous reference star for brightness"),
    ("Arcturus", "14h15m39.7s", "+19d10m57s", "Bright orange giant"),
    ("Betelgeuse", "05h55m10.3s", "+07d24m25s", "Red supergiant"),
    ("Rigel", "05h14m32.3s", "-08d12m06s", "Blue supergiant"),
    ("Polaris", "02h31m49.1s", "+89d15m51s", "The North Star"),
    ("Antares", "16h29m24.5s", "-26d25m55s", "Red supergiant"),
    ("Alpha Centauri", "14h39m36.5s", "-60d50m02s", "Nearest bright star system (4.4 light-years)"),
]
PLANET_NOTES = {
    "mercury": "Innermost planet",
    "venus": "Brightest planet, wrapped in cloud",
    "mars": "The red planet",
    "jupiter": "Largest planet, a gas giant",
    "saturn": "Gas giant with bright rings",
    "uranus": "Ice giant",
    "neptune": "Outermost planet, an ice giant",
}


def planet_positions(when: Time) -> list[dict]:
    out = []
    for name, note in PLANET_NOTES.items():
        c = get_body(name, when)
        out.append(
            {
                "name": name.capitalize(),
                "ra": float(c.ra.deg) % 360.0,
                "dec": float(c.dec.deg),
                "note": note,
            }
        )
    return out


def star_labels() -> list[dict]:
    return [
        {"name": n, "ra": _sexa(ra, dec)[0], "dec": _sexa(ra, dec)[1], "note": note}
        for n, ra, dec, note in BRIGHT_STARS
    ]


def profile_target(obj: dict, online: bool = False) -> Target:
    """Best available Target for a featured object: SIMBAD-resolved if possible, else a built-in fallback."""
    try:
        return resolve_target(TARGETS[obj["target_key"]], online=online)
    except TargetUnavailable:
        if obj["fallback"] is None:
            raise
        return obj["fallback"]


def position_now(t: Target, when: Time) -> tuple[float, float]:
    c = t.coord_at(when) if t.has_motion else t.coord_ref()
    return float(c.ra.deg) % 360.0, float(c.dec.deg)


def motion_path(
    t: Target, y0: float = 2000.0, y1: float = 2040.0, n: int = 41, min_fov_deg: float = 0.05
) -> dict:
    years = [float(y) for y in np.linspace(y0, y1, n)]
    cs = (
        [t.coord_at(Time(y, format="jyear")) for y in years]
        if t.has_motion
        else [t.coord_ref()] * n
    )
    ra, dec = np.array([c.ra.deg for c in cs]), np.array([c.dec.deg for c in cs])
    mid = SkyCoord(float(np.mean(ra)) * u.deg, float(np.mean(dec)) * u.deg)
    extent = float(max(mid.separation(SkyCoord(ra * u.deg, dec * u.deg)).deg))
    return {
        "years": years,
        "ra": [float(v) for v in ra],
        "dec": [float(v) for v in dec],
        "center": {"ra": float(mid.ra.deg), "dec": float(mid.dec.deg)},
        "extent_deg": extent,
        "fov_deg": float(max(min_fov_deg, 5.0 * extent)),
        "total_motion_arcsec": float(
            SkyCoord(ra[0] * u.deg, dec[0] * u.deg)
            .separation(SkyCoord(ra[-1] * u.deg, dec[-1] * u.deg))
            .arcsec
        ),
    }
