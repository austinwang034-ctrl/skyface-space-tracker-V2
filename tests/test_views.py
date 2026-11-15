"""Offline tests for the tracker's display views and the catalogue overlays."""

import time

import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from fastapi.testclient import TestClient

import server
from spherex_viewer import overlays, render, synthetic

c = TestClient(server.app)
CENTER = SkyCoord(133.7862, -7.2444, unit="deg")


def test_window_averages_neighbours_from_the_same_group_only():
    cube = np.arange(6, dtype=float)[:, None, None] * np.ones((6, 2, 2))
    group = [0, 2, 4]  # frames 1, 3, 5 belong to another detector
    assert render.windowed(cube, group, 2, 1)[0, 0] == 2
    assert render.windowed(cube, group, 2, 3)[0, 0] == pytest.approx(2.0)  # mean of 0, 2, 4
    assert render.windowed(cube, group, 0, 3)[0, 0] == pytest.approx(1.0)  # edge: mean of 0, 2
    assert render.windowed(cube, group, 1, 3)[0, 0] == 1  # not in the group: left alone


def test_static_sky_removes_what_does_not_change():
    rng = np.random.default_rng(3)
    sky = rng.normal(10, 1, (9, 9))
    cube = np.repeat(sky[None], 7, axis=0)
    cube[3, 4, 4] += 500  # a visitor in one frame
    static = render.static_sky(cube, list(range(7)))
    assert np.allclose(static, sky) and (cube[3] - static)[4, 4] == pytest.approx(500)
    assert render.limits(np.full((3, 3), np.nan)) == (0.0, 1.0)


def test_gaia_overlay_gives_offsets_and_yearly_motion():
    _, _, gaia = synthetic.make_synthetic(CENTER, n_frames=3)
    out = overlays.gaia_overlay(gaia, CENTER, 2026.0, radius_arcsec=200.0)
    assert out["sources"] and all(np.hypot(s["e"], s["n"]) <= 200 for s in out["sources"])
    mover = next(s for s in out["sources"] if s["id"] == "999")
    assert mover["pm_n"] == pytest.approx(10.3) and mover["pm_e"] == 0.0
    mags = [s["mag"] for s in out["sources"]]
    assert mags == sorted(mags)  # brightest first


def test_wise_overlay_parses_catwise_and_falls_back_to_allwise():
    catwise = [
        {"ra": "133.7862", "dec": "-7.2444", "w1mpro": "17.1", "w2mpro": "13.9",
         "pmra": "-8.1", "pmdec": "0.7", "meanobsmjd": "57000.0"},
        {"ra": "133.7900", "dec": "-7.2400", "w1mpro": "12.0", "w2mpro": "12.1",
         "pmra": "99", "pmdec": "0", "meanobsmjd": ""},
        {"ra": "", "dec": "", "w1mpro": "", "w2mpro": "", "pmra": "", "pmdec": "", "meanobsmjd": ""},
    ]  # fmt: skip
    out = overlays.wise_overlay(CENTER, 300.0, fetch=lambda q: catwise)
    assert out["catalogue"] == "CatWISE2020" and len(out["sources"]) == 2
    first, second = out["sources"]
    assert abs(first["e"]) < 0.01 and first["pm_e"] == -8.1 and 2014.8 < first["epoch"] < 2015.0
    assert second["pm_e"] == 0.0  # an impossible motion is treated as no motion

    seen = []

    def fetch(query):
        seen.append(query)
        if "catwise_2020" in query:
            raise RuntimeError("no such table")
        return [
            {
                "ra": "133.7862",
                "dec": "-7.2444",
                "w1mpro": "15",
                "w2mpro": "14",
                "pmra": "500",
                "pmdec": "0",
            }
        ]

    out = overlays.wise_overlay(CENTER, 300.0, fetch=fetch)
    assert out["catalogue"] == "AllWISE" and out["sources"][0]["pm_e"] == 0.5  # mas/yr -> arcsec/yr
    assert (
        "allwise_p3as_psd" in seen[1]
        and "CIRCLE('ICRS', 133.786200, -7.244400, 0.08333)" in seen[1]
    )
    with pytest.raises(RuntimeError, match="CatWISE2020.*AllWISE"):
        overlays.wise_overlay(
            CENTER, 300.0, fetch=lambda q: (_ for _ in ()).throw(RuntimeError("down"))
        )


def test_api_frame_views_and_overlays():
    jid = c.post("/api/jobs", json={"source": "demo", "max_frames": 20}).json()["id"]
    for _ in range(300):
        if c.get(f"/api/jobs/{jid}").json()["state"] in ("done", "error"):
            break
        time.sleep(0.2)
    pngs = {}
    for view in ("image", "minus_static", "minus_previous"):
        r = c.get(f"/api/jobs/{jid}/frames/5.png", params={"view": view, "window": 3})
        assert r.status_code == 200 and r.content[:4] == b"\x89PNG"
        pngs[view] = r.content
    assert len(set(pngs.values())) == 3
    assert c.get(f"/api/jobs/{jid}/frames/5.png", params={"view": "nope"}).status_code == 422
    assert c.get(f"/api/jobs/{jid}/frames/5.png", params={"window": 99}).status_code == 422
    assert c.get(f"/api/jobs/{jid}/frames/999.png").status_code == 404

    gaia = c.get(f"/api/jobs/{jid}/overlay/gaia").json()
    assert gaia["catalogue"] == "Gaia DR3" and len(gaia["sources"]) > 5
    assert c.get(f"/api/jobs/{jid}/overlay/wise").json()["sources"] == []  # demo sky
    assert c.get(f"/api/jobs/{jid}/overlay/sdss").status_code == 422

    cfg = c.get("/api/config").json()
    assert cfg["version"] == server.APP_VERSION and cfg["resolving_power"]["SPHEREx-D6"] == 128
    assert "SPHEREx" in cfg["acknowledgement"]
