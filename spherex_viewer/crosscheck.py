"""Deep links to other tools and archives for the same patch of sky, plus the credits the data providers ask for."""

from __future__ import annotations

from urllib.parse import quote

from .config import SPHEREX_ACKNOWLEDGEMENT
from .neossat import SEARCH_PAGE as NEOSSAT_SEARCH

# WiseView (Backyard Worlds): blinks WISE/NEOWISE images 2010-2024. Parameters follow the tool's own URL format.
_WISEVIEW_DEFAULTS = [
    ("band", "2"),
    ("speed", "500"),
    ("minbright", "-50.0000"),
    ("maxbright", "500.0000"),
    ("window", "0.5"),
    ("diff_window", "1"),
    ("linear", "1"),
    ("color", ""),
    ("zoom", "9"),
    ("border", "0"),
    ("gaia", "1"),
    ("invert", "1"),
    ("maxdyr", "0"),
    ("scandir", "0"),
    ("neowise", "0"),
    ("diff", "0"),
    ("outer_epochs", "0"),
    ("unique_window", "1"),
    ("smooth_scan", "0"),
    ("shift", "0"),
    ("pmra", "0"),
    ("pmdec", "0"),
]


def wiseview_url(
    ra: float, dec: float, size_arcsec: float = 176, pmra: float = 0.0, pmdec: float = 0.0
) -> str:
    """WiseView link centred on the position. size is the field in arcsec (the tool's own examples use ~176)."""
    size = int(min(max(round(size_arcsec), 60), 2000))
    parts = [("ra", f"{ra:.6f}"), ("dec", f"{dec:.6f}"), ("size", str(size))] + _WISEVIEW_DEFAULTS
    return "http://byw.tools/wiseview#" + "&".join(f"{k}={quote(v, safe='-.')}" for k, v in parts)


def links(ra: float, dec: float, fov_arcmin: float = 6.0) -> list[dict]:
    coo = quote(f"{ra:.6f} {dec:+.6f}")
    return [
        {
            "group": "Watch it move in other surveys",
            "name": "WiseView (WISE / NEOWISE, 2010–2024)",
            "url": wiseview_url(ra, dec, fov_arcmin * 60 / 2),
            "why": "Blinks 14 years of WISE images at 3.4 and 4.6 µm. The tool Backyard Worlds volunteers use to find moving brown dwarfs; compare its motion with the SPHEREx tracker.",
        },
        {
            "group": "Watch it move in other surveys",
            "name": "NEOSSat archive at CADC",
            "url": NEOSSAT_SEARCH,
            "why": "Canada's 15 cm space telescope, visible light, observing since 2013. The NEOSSat panel on this page searches it for you.",
        },
        {
            "group": "SPHEREx at the source",
            "name": "SPHEREx Data Explorer (IRSA)",
            "url": "https://irsa.ipac.caltech.edu/applications/spherex",
            "why": "NASA/IPAC's own browser for the same spectral images this site downloads.",
        },
        {
            "group": "SPHEREx at the source",
            "name": "SPHEREx spectrophotometry tool",
            "url": "https://irsa.ipac.caltech.edu/applications/spherex/tool-spectrophotometry",
            "why": "Science-grade spectra. Use it to check the quick-look spectra shown here.",
        },
        {
            "group": "SPHEREx at the source",
            "name": "Moving Object Search Tool (MOST)",
            "url": "https://irsa.ipac.caltech.edu/applications/MOST",
            "why": "Finds SPHEREx images that caught a known asteroid or comet as it crossed the sky.",
        },
        {
            "group": "SPHEREx at the source",
            "name": "SPHEREx Mosaic Tool",
            "url": "https://irsa.ipac.caltech.edu/applications/spherex/tool-mosaic",
            "why": "Builds a mosaic of a region from many exposures at the wavelengths you choose.",
        },
        {
            "group": "SPHEREx at the source",
            "name": "SPHEREx mission page (IRSA)",
            "url": "https://irsa.ipac.caltech.edu/Missions/spherex.html",
            "why": "Data releases, documentation and every official tool in one place.",
        },
        {
            "group": "SPHEREx at the source",
            "name": "SPHEREx tutorials (IRSA)",
            "url": "https://caltech-ipac.github.io/irsa-tutorials/spherex/",
            "why": "Python notebooks this site's data access is modelled on.",
        },
        {
            "group": "Who is it?",
            "name": "SIMBAD at this position",
            "url": f"https://simbad.cds.unistra.fr/simbad/sim-coo?Coord={coo}&Radius=2&Radius.unit=arcmin",
            "why": "Every catalogued object within 2 arcminutes.",
        },
        {
            "group": "Who is it?",
            "name": "Gaia at MAST",
            "url": "https://archive.stsci.edu/missions-and-data/gaia",
            "why": "Positions, distances and motions for 1.8 billion stars; the reference catalogue for this site's plate solving.",
        },
        {
            "group": "Learn more",
            "name": "NEOSSat mission (Canadian Space Agency)",
            "url": "https://www.asc-csa.gc.ca/eng/satellites/neossat/",
            "why": "What the satellite is and what it looks for.",
        },
        {
            "group": "Learn more",
            "name": "NEOSSat open data and tutorial",
            "url": "https://github.com/asc-csa/NEOSSAT_Tutorial",
            "why": "CSA's notebooks for downloading and cleaning NEOSSat images; dataset page: https://data.asc-csa.gc.ca/dataset/neossat",
        },
    ]


CREDITS = [
    SPHEREX_ACKNOWLEDGEMENT,
    "SPHEREx data are served by the NASA/IPAC Infrared Science Archive (IRSA). Spectral images DOI 10.26131/IRSA678 (DR1). "
    "Mission: Bock et al. 2026 (ApJ 999, 139). Pipeline: Akeson et al. 2026 (ApJS 285, 67).",
    "NEOSSat: Canadian Space Agency and Defence Research and Development Canada; archived by the Canadian Astronomy Data Centre (NRC Canada).",
    "Gaia: European Space Agency (ESA), Gaia Data Processing and Analysis Consortium; mirror hosted by MAST at STScI.",
    "WISE / NEOWISE: NASA/JPL-Caltech; WiseView by the Backyard Worlds: Planet 9 team.",
    "Sky map and catalogue services: Aladin Lite, SIMBAD and VizieR, CDS, Strasbourg, France.",
]
