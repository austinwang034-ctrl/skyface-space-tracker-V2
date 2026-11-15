"""Shared constants. Everything archive-specific lives here so it is easy to update."""

import os
from pathlib import Path

APP_NAME = "Skyface Space Tracker"
APP_VERSION = "1.4.2"
APP_OWNER = "The Skyface Space Tracker team"  # shown in the copyright notice and on exported GIFs

# Other websites allowed to call this site's API from their own pages (comma-separated origins,
# e.g. "https://myteam.example"). Empty means only this site itself. Embedding with ?embed=1
# needs no entry here: the embedded page calls the API from this site's own address.
ALLOWED_ORIGINS = [
    o.strip().rstrip("/").lower()
    for o in os.environ.get("SKYFACE_ALLOWED_ORIGINS", "").split(",")
    if o.strip()
]
# Websites allowed to show the FULL site inside a frame. Hugging Face shows every Space inside a
# frame on huggingface.co, so that is allowed automatically when running there (it sets SPACE_ID).
FRAME_ANCESTORS = [
    o.strip()
    for o in os.environ.get(
        "SKYFACE_FRAME_ANCESTORS", "https://huggingface.co" if os.environ.get("SPACE_ID") else ""
    ).split(",")
    if o.strip()
]
# Addresses this site is known by besides the one in each request (Hugging Face sets SPACE_HOST).
OWN_HOSTS = [h.strip().lower() for h in os.environ.get("SPACE_HOST", "").split(",") if h.strip()]
# Most images one tracker may use. Small free hosts (0.1 CPU, 512 MB) should set this to about 60
# so a request finishes in reasonable time; render.yaml does.
MAX_FRAMES = max(10, int(os.environ.get("SKYFACE_MAX_FRAMES", "300")))
# New trackers one visitor (IP address) may start per minute; each one costs archive downloads.
JOBS_PER_MINUTE = int(os.environ.get("SKYFACE_JOBS_PER_MINUTE", "12"))

TAP_URL = "https://irsa.ipac.caltech.edu/TAP"
IRSA_BASE = "https://irsa.ipac.caltech.edu/"
IRSA_TAP_SYNC = "https://irsa.ipac.caltech.edu/TAP/sync"

CACHE_DIR = Path(os.environ.get("SPHEREX_VIEWER_CACHE", Path.home() / ".spherex_viewer"))
CUTOUT_DIR = CACHE_DIR / "cutouts"
CATALOG_DIR = CACHE_DIR / "catalogs"

# Nominal bandpass edges in micron (labels only; the true wavelength at a pixel comes from each
# image's WCS-WAVE extension because SPHEREx uses linear variable filters).
BANDS = {
    "SPHEREx-D1": (0.75, 1.09),
    "SPHEREx-D2": (1.10, 1.62),
    "SPHEREx-D3": (1.63, 2.41),
    "SPHEREx-D4": (2.42, 3.82),
    "SPHEREx-D5": (3.83, 4.41),
    "SPHEREx-D6": (4.42, 5.00),
}

# Spectral resolving power (lambda / delta-lambda) of each detector, from IRSA's SPHEREx mission page.
RESOLVING_POWER = {
    "SPHEREx-D1": 39,
    "SPHEREx-D2": 41,
    "SPHEREx-D3": 41,
    "SPHEREx-D4": 35,
    "SPHEREx-D5": 112,
    "SPHEREx-D6": 128,
}

# FLAGS-extension bits masked as bad (names from the IRSA SPHEREx data-overview tutorial):
# 0 TRANSIENT, 1 OVERFLOW, 2 SUR_ERROR, 6 NONFUNC, 9 MISSING_DATA, 10 HOT, 11 COLD, 19 OUTLIER.
# Bit 21 SOURCE is informational (it marks real sources) and is NOT masked. Override: SPHEREX_BAD_FLAG_BITS=<integer>.
BAD_FLAG_BITS = int(
    os.environ.get("SPHEREX_BAD_FLAG_BITS", sum(1 << b for b in (0, 1, 2, 6, 9, 10, 11, 19)))
)

# SPHEREx data releases, best first. IRSA's timeline: QR (Jul 2025), QR2 (Oct 2025), QR3 (Sep 2026),
# DR1 (Nov 2026). DR1 and QR3 supersede QR2. To adopt a future release,
# add its folder name at the front here (or set SPHEREX_RELEASES="dr2,dr1,qr3"); nothing else needs to change.
RELEASES = [
    r.strip().lower()
    for r in os.environ.get("SPHEREX_RELEASES", "dr1,qr3,qr2").split(",")
    if r.strip()
]
SIA_COLLECTIONS = [f"spherex_{r}" for r in RELEASES if r != "qr2"]

# SIMBAD servers, tried in order: CDS in Strasbourg, then its mirror at Harvard. Whichever answers
# is tried first from then on. Override with SKYFACE_SIMBAD_SERVERS="host1,host2".
SIMBAD_SERVERS = [
    s.strip()
    for s in os.environ.get(
        "SKYFACE_SIMBAD_SERVERS", "simbad.cds.unistra.fr,simbad.harvard.edu"
    ).split(",")
    if s.strip()
]

# Other archives used for cross-checks
CADC_TAP_SYNC = "https://ws.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/argus/sync"
CADC_CUTOUT = "https://ws.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/caom2ops/sync"
CADC_DATA = "https://ws.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/data/pub"
MAST_GAIA_CONE = "https://mast.stsci.edu/vo-conesearch/api/v0.1/gaiadr3"

# The acknowledgement IRSA asks every user of SPHEREx data to include.
SPHEREX_ACKNOWLEDGEMENT = (
    "This publication makes use of data products from the Spectro-Photometer for the History of the "
    "Universe, Epoch of Reionization and Ices Explorer (SPHEREx), which is a joint project of the Jet "
    "Propulsion Laboratory and the California Institute of Technology, and is funded by the National "
    "Aeronautics and Space Administration."
)

ARCSEC_PER_RAD = 206264.80624709636
