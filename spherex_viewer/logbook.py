"""The observer's log: the tracker's measurements written out as a few plain sentences.

Astronomers who blinked photographic plates kept a logbook: what was observed, when, how well
the plates lined up, and what moved. This module writes the same kind of entry from numbers the
pipeline has already measured. It only does arithmetic and wording, so it is easy to test.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

KM_S_PER_AU_YR = 4.740470  # 1 astronomical unit per year, in km/s

COMPASS = ["north", "north-east", "east", "south-east", "south", "south-west", "west", "north-west"]


@dataclass(frozen=True)
class Motion:
    """A proper motion on the sky. East and north components in milliarcseconds per year."""

    east: float
    north: float
    east_err: float = 0.0
    north_err: float = 0.0

    @property
    def total(self) -> float:
        return math.hypot(self.east, self.north)

    @property
    def total_err(self) -> float:
        if self.total == 0:
            return math.hypot(self.east_err, self.north_err)
        return math.hypot(self.east * self.east_err, self.north * self.north_err) / self.total

    @property
    def significance(self) -> float:
        """How many times larger the motion is than its uncertainty."""
        return self.total / self.total_err if self.total_err > 0 else float("inf")

    @property
    def position_angle(self) -> float:
        """Direction of travel in degrees, measured from north through east (the usual convention)."""
        return math.degrees(math.atan2(self.east, self.north)) % 360.0

    @property
    def heading(self) -> str:
        return COMPASS[int((self.position_angle + 22.5) // 45) % 8]


def tangential_velocity(motion_masyr: float, parallax_mas: float) -> float | None:
    """Speed across the line of sight in km/s, from proper motion and parallax."""
    if parallax_mas <= 0:
        return None
    return KM_S_PER_AU_YR * motion_masyr / parallax_mas


def agreement_sigma(measured: Motion, catalogue: Motion) -> float:
    """How far the measured motion is from the catalogue's, in units of the measurement error."""
    d_east = (measured.east - catalogue.east) / max(measured.east_err, 1e-9)
    d_north = (measured.north - catalogue.north) / max(measured.north_err, 1e-9)
    return math.hypot(d_east, d_north) / math.sqrt(2)


def _arcsec(mas: float) -> str:
    return f"{mas / 1000:.2f}″" if abs(mas) >= 100 else f"{mas:.0f} mas"


def _with_error(value_mas: float, error_mas: float) -> str:
    """A value and its uncertainty in the same unit, e.g. '10.28 ± 0.01″'."""
    if abs(value_mas) >= 100:
        return f"{value_mas / 1000:.2f} ± {max(error_mas / 1000, 0.01):.2f}″"
    return f"{value_mas:.0f} ± {error_mas:.0f} mas"


def write_log(
    *,
    name: str,
    n_frames: int,
    first_date: str,
    last_date: str,
    span_days: float,
    bands: list[str],
    n_solved: int | None,
    median_residual_arcsec: float | None,
    n_detected: int,
    fit: dict | None,
    catalogue: Motion | None,
    parallax_mas: float = 0.0,
    pixel_arcsec: float = 6.15,
    synthetic: bool = False,
) -> dict:
    """Build the log entry. Returns {"verdict", "kind", "lines": [{"label", "text"}], "numbers"}.

    kind is one of: "moving", "still", "unseen", "short" (too little data to say).
    """
    lines: list[dict] = []
    numbers: dict = {}
    span_years = span_days / 365.25

    # 1. What was observed
    band_text = ", ".join(b.replace("SPHEREx-", "") for b in bands) or "unknown band"
    data_kind = "simulated demo images" if synthetic else "SPHEREx images"
    lines.append(
        {
            "label": "Observed",
            "text": f"{n_frames} {data_kind} in {band_text}, {first_date} to {last_date} "
            f"({span_days:.0f} days).",
        }
    )

    # 2. How well the images are lined up (plate solving against Gaia)
    if n_solved is None:
        lines.append(
            {"label": "Alignment", "text": "Not checked against Gaia; pointing is as delivered."}
        )
    elif median_residual_arcsec is not None:
        fraction = median_residual_arcsec / pixel_arcsec
        lines.append(
            {
                "label": "Alignment",
                "text": f"{n_solved} of {n_frames} images tied to Gaia stars. Typical star lands "
                f"within {median_residual_arcsec:.2f}″ of its catalogue place "
                f"({fraction:.2f} of a pixel).",
            }
        )
        numbers["median_residual_arcsec"] = median_residual_arcsec
    else:
        lines.append(
            {
                "label": "Alignment",
                "text": f"{n_solved} of {n_frames} images could be tied to Gaia stars.",
            }
        )

    # 3. Was the object seen at all?
    if n_detected == 0:
        lines.append(
            {
                "label": "Detection",
                "text": "Nothing significant at the expected place in any image. It may be too "
                "faint in this band, or the coordinates may be for a different year.",
            }
        )
        return {
            "verdict": f"{name}: not seen in these images.",
            "kind": "unseen",
            "lines": lines,
            "numbers": numbers,
        }
    lines.append({"label": "Detection", "text": f"Seen in {n_detected} of {n_frames} images."})

    # 4. Did it move?
    if fit is None:
        lines.append(
            {
                "label": "Motion",
                "text": "Too few detections or too short a time span to measure motion "
                "(needs 5 or more detections over 60 days).",
            }
        )
        return {
            "verdict": f"{name}: seen, but the record is too short to measure motion.",
            "kind": "short",
            "lines": lines,
            "numbers": numbers,
        }

    measured = Motion(fit["pmra_masyr"], fit["pmdec_masyr"], fit["pmra_err"], fit["pmdec_err"])
    travelled_arcsec = measured.total / 1000 * span_years
    numbers.update(
        total_masyr=measured.total,
        total_err_masyr=measured.total_err,
        position_angle_deg=measured.position_angle,
        significance=measured.significance,
        travelled_arcsec=travelled_arcsec,
    )
    moving = measured.significance >= 5 and travelled_arcsec >= 0.5 * pixel_arcsec

    if moving:
        lines.append(
            {
                "label": "Motion",
                "text": f"Moving {measured.heading} at {_with_error(measured.total, measured.total_err)} "
                f"per year (position angle "
                f"{measured.position_angle:.0f}°). It crossed {travelled_arcsec:.1f}″, "
                f"{travelled_arcsec / pixel_arcsec:.1f} pixels, during these observations.",
            }
        )
    else:
        limit = 3 * measured.total_err
        lines.append(
            {
                "label": "Motion",
                "text": f"No motion found. Anything faster than about {_arcsec(limit)} per year "
                "would have shown up.",
            }
        )
        numbers["upper_limit_masyr"] = limit

    # 5. Does that match what the catalogues say?
    if catalogue is not None and catalogue.total > 0:
        sigma = agreement_sigma(measured, catalogue)
        numbers["catalogue_sigma"] = sigma
        if sigma <= 3:
            verdict_text = "The two agree."
        else:
            verdict_text = (
                "They differ by more than the errors allow; over a short record the yearly "
                "parallax wobble can do this."
            )
        lines.append(
            {
                "label": "Catalogue",
                "text": f"Catalogue motion is {_arcsec(catalogue.total)} per year toward the "
                f"{catalogue.heading}. {verdict_text}",
            }
        )
    elif moving:
        lines.append(
            {
                "label": "Catalogue",
                "text": "No catalogued motion for these coordinates. A moving source with no "
                "catalogue entry is worth a second look: check the Astrometry tab for known "
                "asteroids before getting excited.",
            }
        )

    # 6. How fast is that in space?
    speed = tangential_velocity(measured.total, parallax_mas) if moving else None
    if speed is not None:
        distance_ly = 3261.56 / parallax_mas
        numbers["tangential_km_s"] = speed
        lines.append(
            {
                "label": "Speed",
                "text": f"At {distance_ly:.1f} light-years that is {speed:.0f} km/s across our "
                "line of sight.",
            }
        )

    if moving:
        verdict = (
            f"{name} moved {travelled_arcsec:.1f}″ toward the {measured.heading} "
            f"in {span_days:.0f} days."
        )
    else:
        verdict = f"{name} stayed put over {span_days:.0f} days."
    return {
        "verdict": verdict,
        "kind": "moving" if moving else "still",
        "lines": lines,
        "numbers": numbers,
    }
