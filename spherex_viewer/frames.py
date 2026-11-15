"""Reading cutouts into Frame objects and ordering them chronologically."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time
from astropy.utils.exceptions import AstropyWarning
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

from .archive import Observation
from .config import BAD_FLAG_BITS

# Bit names from the IRSA SPHEREx data-overview tutorial; refreshed from each file's FLAGS header (MSKNnnnn) on load.
FLAG_NAMES = {
    0: "TRANSIENT",
    1: "OVERFLOW",
    2: "SUR_ERROR",
    6: "NONFUNC",
    7: "DICHROIC",
    9: "MISSING_DATA",
    10: "HOT",
    11: "COLD",
    12: "FULLSAMPLE",
    14: "PHANMISS",
    15: "NONLINEAR",
    17: "PERSIST",
    19: "OUTLIER",
    20: "CROSSTALK",
    21: "SOURCE",
    22: "GHOST",
    23: "GHOST_FPA",
    24: "GHOST_EXT",
    25: "STREAK",
    26: "BLOOM",
    27: "SNOWBALL",
    28: "HALO",
}


def learn_flag_names(header) -> None:
    """Pick up bit names the archive defines in the FLAGS header, so new or renamed bits show correctly."""
    for key in header:
        if (
            key.startswith("MSKN")
            and key[4:].isdigit()
            and 0 <= int(key[4:]) < 32
            and str(header[key]).strip()
        ):
            FLAG_NAMES[int(key[4:])] = str(header[key]).strip()


def decode_flags(value: int) -> list[str]:
    """Names of the set bits (documented ones by name, others as 'bit N')."""
    return [FLAG_NAMES.get(b, f"bit {b}") for b in range(32) if int(value) & (1 << b)]


@dataclass
class Frame:
    obs: Observation
    data: np.ndarray  # native-pixel image (MJy/sr), NaN where bad/missing
    wcs: WCS  # native celestial WCS (incl. SIP if present)
    mjd: float
    time_source: str  # 'header' or 'archive'
    wavelength_um: float = float("nan")
    flags: np.ndarray | None = None  # raw FLAGS extension (bit mask per pixel)
    wave_wcs: WCS | None = None  # spectral WCS: wavelength at any pixel (linear variable filter)

    @property
    def band(self) -> str:
        return self.obs.band

    @property
    def time(self) -> Time:
        return Time(self.mjd, format="mjd", scale="utc")

    @property
    def pixscale_arcsec(self) -> float:
        return float(np.mean(proj_plane_pixel_scales(self.wcs)) * 3600.0)

    def wavelength_at(self, x: float, y: float) -> float:
        """Wavelength (µm) at a pixel; falls back to the patch-centre value when no spectral WCS is available."""
        if self.wave_wcs is not None:
            try:
                w, _ = self.wave_wcs.pixel_to_world(x, y)
                return float(w.to(u.um).value)
            except Exception:  # noqa: BLE001
                pass
        return self.wavelength_um

    def flag_at(self, x: float, y: float) -> int:
        if self.flags is None:
            return 0
        i, j = int(round(y)), int(round(x))
        return (
            int(self.flags[i, j])
            if 0 <= i < self.flags.shape[0] and 0 <= j < self.flags.shape[1]
            else 0
        )

    def flagged_fraction(self) -> float | None:
        return None if self.flags is None else float(((self.flags & BAD_FLAG_BITS) != 0).mean())

    def label(self) -> str:
        wl = f" · {self.wavelength_um:.2f} µm" if np.isfinite(self.wavelength_um) else ""
        return f"{self.time.strftime('%Y-%m-%d %H:%M')} UTC{wl}"


def _header_mjd(primary, image, archive_mjd: float):
    """Prefer an exposure-time keyword from the FITS header, but only if it agrees with the archive
    time to within a day (guards against unit/keyword surprises)."""
    for hdr in (image, primary):
        for key in ("MJD-AVG", "MJD-OBS"):
            v = hdr.get(key)
            if isinstance(v, (int, float)) and abs(float(v) - archive_mjd) < 1.0:
                return float(v), "header"
    return archive_mjd, "archive"


def load_frame(obs: Observation, path: Path, center: SkyCoord | None = None) -> Frame:
    with warnings.catch_warnings(), fits.open(path, memmap=False) as hdul:
        warnings.simplefilter("ignore", AstropyWarning)
        ext = hdul["IMAGE"] if "IMAGE" in hdul else hdul[1]
        data = np.array(ext.data, dtype=np.float32)
        hdr = ext.header
        wcs = WCS(hdr)
        flags = np.array(hdul["FLAGS"].data, dtype=np.int64) if "FLAGS" in hdul else None
        if flags is not None:
            learn_flag_names(hdul["FLAGS"].header)
        if BAD_FLAG_BITS and flags is not None:
            data[(flags & BAD_FLAG_BITS) != 0] = np.nan
        mjd, tsrc = _header_mjd(hdul[0].header, hdr, obs.mjd)
        wl, swcs = float("nan"), None
        if "WCS-WAVE" in hdul:
            try:
                swcs = WCS(hdr, fobj=hdul, key="W")
                swcs.sip = None
                if center is not None:
                    x, y = wcs.world_to_pixel(center)
                    wave, _ = swcs.pixel_to_world(x, y)
                    wl = float(wave.to(u.um).value)
            except Exception:  # noqa: BLE001  (wavelength is informational only)
                swcs = None
    return Frame(
        obs=obs,
        data=data,
        wcs=wcs,
        mjd=mjd,
        time_source=tsrc,
        wavelength_um=wl,
        flags=flags,
        wave_wcs=swcs,
    )


def sort_chronologically(frames: list[Frame]) -> list[Frame]:
    return sorted(frames, key=lambda f: (f.mjd, f.obs.obs_key))


def usable(frame: Frame, min_finite_fraction: float = 0.6) -> bool:
    """Reject cutouts that are mostly empty (target near a detector edge)."""
    return bool(np.isfinite(frame.data).mean() >= min_finite_fraction)
