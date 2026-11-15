"""Offline tests for wavelength selection: chemical signatures and colour temperature."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

import server
from spherex_viewer import signatures


def test_every_signature_sits_inside_spherex_and_names_its_detectors():
    keys = [s.key for s in signatures.SIGNATURES]
    assert len(keys) == len(set(keys)) and {"water_ice", "co2_ice", "methane"} <= set(keys)
    for sig in signatures.SIGNATURES:
        assert 0.75 <= sig.lo_um < sig.hi_um <= 5.0, sig.key
        assert sig.as_dict()["detectors"], sig.key
    assert signatures.detectors_covering(4.20, 4.35) == ["D5"]
    assert signatures.detectors_covering(2.90, 3.20) == ["D4"]
    assert signatures.detectors_covering(1.0, 1.2) == ["D1", "D2"]


def test_frames_are_split_into_band_and_neighbouring_wavelengths():
    waves = [2.5, 2.75, 2.95, 3.05, 3.15, 3.3, 3.45, 3.6, float("nan")]
    inside, beside = signatures.split_frames(waves, 2.9, 3.2)
    assert inside == [2, 3, 4] and beside == [1, 5, 6]  # a 0.3 µm strip on each side
    inside, beside = signatures.split_frames(waves, 4.2, 4.35)
    assert inside == [] and beside == []


def test_band_strength_recognises_absorption_emission_and_nothing():
    steady = [10.0, 10.2, 9.8, 10.1]
    assert signatures.band_strength([6.0, 6.2, 5.9], steady)["verdict"] == "absorbs"
    assert signatures.band_strength([14.0, 15.0], steady)["verdict"] == "emits"
    flat = signatures.band_strength([10.1, 9.9], steady)
    assert flat["verdict"] == "no clear feature" and abs(flat["percent"]) < 3
    assert signatures.band_strength([], steady) is None
    assert signatures.band_strength([1.0], [float("nan")]) is None


@pytest.mark.parametrize("true_temperature", [400.0, 1300.0, 3200.0])
def test_colour_temperature_recovers_a_blackbody(true_temperature):
    waves = np.linspace(0.8, 4.9, 60)
    flux = 5.0 * signatures.blackbody_fnu(waves, true_temperature)
    flux = flux / flux.max()
    out = signatures.colour_temperature(waves, flux, np.full_like(flux, 0.01))
    assert out["limit"] is None
    assert out["temperature_k"] == pytest.approx(true_temperature, rel=0.03)


def test_colour_temperature_knows_its_limits():
    waves = np.linspace(0.8, 4.9, 60)
    hot = signatures.blackbody_fnu(waves, 9000.0)
    out = signatures.colour_temperature(waves, hot / hot.max())
    assert out["limit"] == "hotter than" and out["temperature_k"] <= 6100
    narrow = np.linspace(1.1, 1.5, 30)  # one detector: too little colour to judge
    assert signatures.colour_temperature(narrow, np.ones(30)) is None
    assert signatures.colour_temperature(waves[:5], np.ones(5)) is None


def test_api_lists_signatures():
    c = TestClient(server.app)
    listed = c.get("/api/signatures").json()["signatures"]
    assert {"key", "name", "lo_um", "hi_um", "kind", "tells_you", "detectors"} <= set(listed[0])
