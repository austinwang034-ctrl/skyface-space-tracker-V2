"""Skyface Space Tracker: time-lapse viewer, astrometry and spectra for SPHEREx spectral images."""

import warnings

from .config import APP_VERSION as __version__  # noqa: F401

# ERFA warns when a catalogue entry has no distance; propagation is still correct.
warnings.filterwarnings("ignore", message=".*distance overridden.*")
warnings.filterwarnings("ignore", message=".*'datfix' made the change.*")
