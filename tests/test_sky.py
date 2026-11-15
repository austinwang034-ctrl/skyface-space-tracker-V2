import re

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.time import Time
from fastapi.testclient import TestClient

import server
from spherex_viewer import frames as fr
from spherex_viewer import objects, render, targets

c = TestClient(server.app)


def offline(monkeypatch):
    def boom(*a, **k):
        raise ConnectionError("no network")

    monkeypatch.setattr(targets, "_simbad_lookup", boom)


def test_sky_endpoint_labels(monkeypatch):
    offline(monkeypatch)
    j = c.get("/api/sky").json()
    assert [o["id"] for o in j["objects"]] == ["barnard", "epsindi", "wise0855", "rwaur", "rdra"]
    assert all(
        0 <= o["ra"] < 360 and -90 <= o["dec"] <= 90
        for o in j["objects"] + j["stars"] + j["planets"]
    )
    assert len(j["stars"]) == 8 and [p["name"] for p in j["planets"]][:2] == ["Mercury", "Venus"]
    b = next(o for o in j["objects"] if o["id"] == "barnard")  # label sits where the star is TODAY
    now = targets.TARGETS["Barnard's Star"].coord_at(Time(j["epoch"], format="jyear"))
    assert SkyCoord(b["ra"], b["dec"], unit="deg").separation(now).arcsec < 1.0


def test_planets_stay_near_the_ecliptic():
    for p in objects.planet_positions(Time("2026-11-09")):
        lat = SkyCoord(p["ra"], p["dec"], unit="deg").barycentrictrueecliptic.lat.deg
        assert abs(lat) < 8, (p["name"], lat)


def test_object_profiles(monkeypatch):
    offline(monkeypatch)
    for oid in ("barnard", "epsindi", "wise0855", "rwaur", "rdra"):
        o = c.get(f"/api/objects/{oid}").json()
        assert (
            o["facts"]
            and o["light"]
            and o["sources"]
            and len(o["model"]["channels_um"]) == 102
            and len(o["model"]["strip_log10"]) == 102
        )
        assert all(max(cmp["flux"]) == 1.0 for cmp in o["model"]["components"])
        assert (
            len(o["path"]["years"]) == 41
            and o["path"]["fov_deg"] >= 0.05
            and o["tracker"]["ra"] is not None
        )
    assert not c.get("/api/objects/epsindi").json()[
        "position_is_exact"
    ]  # offline fallback is flagged as approximate
    w = c.get("/api/objects/wise0855").json()
    assert abs(w["path"]["total_motion_arcsec"] - 40 * 8.147) < 6  # 8.147"/yr over 40 years
    assert (
        c.get("/api/objects/nope").status_code == 404
        and c.get("/api/objects/barnard?y0=2040&y1=2000").status_code == 422
    )


def test_blackbody_physics():
    lam = np.geomspace(0.3, 20, 4000)
    for t in (3000.0, 1000.0):
        assert (
            abs(lam[np.argmax(objects.planck(lam, t))] - objects.wien_peak_um(t))
            / objects.wien_peak_um(t)
            < 0.01
        )
    assert abs(objects.wien_peak_um(285) - 10.17) < 0.05  # WISE 0855 peaks beyond SPHEREx's 5 µm


def test_flag_decoding():
    assert (
        fr.decode_flags(0) == []
        and fr.decode_flags(1 | 64) == ["TRANSIENT", "NONFUNC"]
        and fr.decode_flags(1 << 21) == ["SOURCE"]
    )
    assert fr.decode_flags(1 << 3) == ["bit 3"]  # undocumented bits are never guessed


def test_difference_image_shows_a_mover_as_a_pair():
    from dataclasses import replace

    from spherex_viewer import align, synthetic

    centre = SkyCoord(269.4, 4.75, unit="deg")
    frames, truth, _ = synthetic.make_synthetic(centre, n_frames=30, pointing_error_pix=0.0)
    good = [replace(f, wcs=w) for f, w in zip(frames, truth["true_wcs"])]
    wcs_out, shape = align.common_wcs(centre, 250, 6.15)
    cube = align.reproject_stack(good, wcs_out, shape)
    s = 15
    diff = np.nanmean(cube[s:], axis=0) - np.nanmean(cube[:s], axis=0)
    cy, cx = (shape[0] - 1) // 2, (shape[1] - 1) // 2
    assert (
        diff[cy + 1, cx] > 50 and diff[cy - 1, cx] < -50
    )  # brighter north (later), fainter south (earlier)
    rgba = render.diverging_rgba(diff, float(np.nanpercentile(np.abs(diff), 99.5)))
    assert (
        rgba[cy + 1, cx, 0] > rgba[cy + 1, cx, 2] and rgba[cy - 1, cx, 2] > rgba[cy - 1, cx, 0]
    )  # red north, blue south


def _site():
    """The whole front end as served: the page, its stylesheet and its script."""
    page, css, js = c.get("/"), c.get("/assets/styles.css"), c.get("/assets/app.js")
    assert page.status_code == css.status_code == js.status_code == 200
    return page.text, css.text, js.text


def test_frontend_has_the_requested_pieces():
    html, _, js = _site()
    for needle in (
        "Welcome to Skyface Space Tracker",
        'id="start_name"',
        'id="start_ra"',
        'id="start_dec"',
        'id="start_map"',
        'id="home"',
        "Track this patch",
        "Compare epochs",
        "Data quality",
        "Spectrum",
        "Next object",
        "Previous object",
        "backyardworlds.org",
        "Track it in SPHEREx images",
        'id="ra"',
        'id="dec"',
    ):
        assert needle in html, needle
    for needle in ("SPHEREx/color", "embed", "lookupName", "startTracking"):
        assert needle in js, needle
    for needle in ("Observer’s log", 'id="blink_go"', 'id="c_neg"', 'id="sb_go"', "Astrometry"):
        assert needle in html, needle
    for needle in ("/logbook", "/api/smallbodies", "blinkToggle", "/assets/vendor/aladin.js"):
        assert needle in js, needle
    assert "fonts.googleapis.com" not in html  # typefaces are bundled, not fetched
    for asset in (
        "fonts/atkinson-hyperlegible-next-latin-wght-normal.woff2",
        "vendor/plotly.min.js",
        "vendor/aladin.js",
    ):
        assert c.get(f"/assets/{asset}").status_code == 200, asset
    for needle in ('id="tab-chem"', 'id="chem_molecules"', 'id="chem_play"', "How warm is it?"):
        assert needle in html, needle
    for needle in ("/api/signatures", "playWavelengths", "waveFilter", "/temperature"):
        assert needle in js, needle
    # Accessibility basics: landmarks, skip link, live regions, keyboard-reachable tracker.
    for needle in (
        '<main id="main"',
        'class="skip-link"',
        'role="alert"',
        'role="status"',
        'role="tabpanel"',
        'aria-describedby="cv_help"',
        'lang="en"',
        "Where the data comes from",
    ):
        assert needle in html, needle
    for needle in ("trackerKey", "tabKey", "labelMapControls", "describeFrame"):
        assert needle in js, needle
    # Tracker display controls: views, stacking, catalogue rings, stepping by hand.
    for needle in (
        'id="v_view"',
        'id="v_window"',
        'id="c_gaia"',
        'id="c_wise"',
        'id="prev"',
        'id="demo"',
    ):
        assert needle in html, needle
    for needle in ("/overlay/", "minus_static", "toggleOverlay", "DISPLAY_FLAGS"):
        assert needle in js or needle in html, needle
    for gone in ("Sky movie", "Make the movie", "No movie", "style="):
        assert gone not in html, gone  # no leftover wording, no inline styles in the page


def test_no_global_canvas_style_that_would_blacken_the_sky_map():
    _, css, _ = _site()
    assert not re.search(
        r"(^|[},\s])canvas\s*[{,]", css
    )  # a bare canvas rule once hid the whole map under black
    ov = css.split("#ov {")[1].split("}")[0]
    assert "background: transparent" in ov
