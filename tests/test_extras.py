"""Offline tests for the data-release handling, the NEOSSat/CADC module, the MAST Gaia fallback and cross-check links."""

import io

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table
from fastapi.testclient import TestClient

import server
from spherex_viewer import archive, crosscheck, frames, neighborhood, neossat

BASE = "https://irsa.ipac.caltech.edu/ibe/data/spherex"
F = "level2_2025W18_1B_0074_2D1_spx_l2b-v{v}-2026-196.fits"


def test_release_is_read_from_the_archive_path_and_dr1_beats_qr():
    dr1 = archive.parse_observation(
        f"{BASE}/dr1/level2/2025W18_1B/x/1/{F.format(v=27)}?center=1,2d&size=0.1",
        60800.0,
        "SPHEREx-D1",
    )
    qr2 = archive.parse_observation(
        f"{BASE}/qr2/level2/2025W18_1B/x/1/{F.format(v=31)}?center=1,2d&size=0.1",
        60800.0,
        "SPHEREx-D1",
    )
    assert (dr1.release, qr2.release) == ("dr1", "qr2") and dr1.obs_key == qr2.obs_key
    assert archive.dedupe_versions([qr2, dr1]) == [
        dr1
    ]  # better release wins even with a lower version number
    newer = archive.parse_observation(dr1.url.replace("v27", "v28"), 60800.0, "SPHEREx-D1")
    assert archive.dedupe_versions([dr1, newer]) == [
        newer
    ]  # within a release the newest processing wins


def test_sia_rows_become_cutout_urls_and_respect_the_band_filter():
    rows = [
        {"access_url": f"{BASE}/dr1/level2/2025W18_1B/x/1/{F.format(v=27)}", "t_min": 60800.5},
        {
            "access_url": f"{BASE}/dr1/level2/2025W18_1B/x/1/{F.format(v=27).replace('2D1', '2D4')}",
            "t_min": 60801.5,
        },
        {"access_url": f"{BASE}/dr1/hips5_cube/something.fits", "t_min": 60802.5},
    ]
    out = archive.sia_rows_to_observations(rows, 10.0, -5.0, 0.1, ["SPHEREx-D1"])
    assert (
        len(out) == 1
        and out[0].band == "SPHEREx-D1"
        and out[0].url.endswith("?center=10.00000000,-5.00000000d&size=0.100000")
    )


def test_search_falls_back_to_sia_when_tap_fails(monkeypatch):
    monkeypatch.setattr(
        archive, "_query_tap", lambda *a: (_ for _ in ()).throw(RuntimeError("TAP down"))
    )
    o = archive.parse_observation(
        f"{BASE}/dr1/level2/a/b/1/{F.format(v=27)}", 60800.0, "SPHEREx-D1"
    )
    monkeypatch.setattr(archive, "_query_sia", lambda *a: [o])
    assert archive.query_observations(1, 2, 0.1, ["SPHEREx-D1"]) == [o]
    monkeypatch.setattr(
        archive, "_query_sia", lambda *a: (_ for _ in ()).throw(RuntimeError("SIA down"))
    )
    with pytest.raises(RuntimeError, match="TAP down.*SIA down"):
        archive.query_observations(1, 2, 0.1, ["SPHEREx-D1"])


def test_flag_names_follow_the_irsa_tutorial_and_learn_from_headers():
    assert frames.decode_flags((1 << 21) | (1 << 27)) == ["SOURCE", "SNOWBALL"]
    h = fits.Header()
    h["MSKN0030"] = "NEWBIT"
    frames.learn_flag_names(h)
    assert frames.decode_flags(1 << 30) == ["NEWBIT"]
    frames.FLAG_NAMES.pop(30)


# ---------------------------------------------------------------- NEOSSat
ROWS = [
    {
        "observationID": "2019001",
        "target_name": "RW Aur",
        "productID": "raw",
        "time_bounds_lower": "58485.2",
        "time_exposure": "30.0",
        "uri": "cadc:NEOSSAT/NEOS_SCI_2019001.fits",
        "productType": "science",
    },
    {
        "observationID": "2019001",
        "target_name": "RW Aur",
        "productID": "cord",
        "time_bounds_lower": "58485.2",
        "time_exposure": "30.0",
        "uri": "cadc:NEOSSAT/NEOS_SCI_2019001_cord.fits.gz",
        "productType": "science",
    },
    {
        "observationID": "2019001",
        "target_name": "RW Aur",
        "productID": "cord",
        "time_bounds_lower": "58485.2",
        "time_exposure": "30.0",
        "uri": "cadc:NEOSSAT/NEOS_SCI_2019001_cord_prev.png",
        "productType": "preview",
    },
    {
        "observationID": "2015007",
        "target_name": "",
        "productID": "cor",
        "time_bounds_lower": "57030.9",
        "time_exposure": "",
        "uri": "cadc:NEOSSAT/NEOS_SCI_2015007_cor.fits.gz",
        "productType": "science",
    },
    {
        "observationID": "broken",
        "target_name": "",
        "productID": "raw",
        "time_bounds_lower": "",
        "time_exposure": "",
        "uri": "",
        "productType": "science",
    },
]


def test_neossat_rows_are_deduplicated_to_best_product_and_sorted_by_time():
    v = neossat.parse_rows(ROWS)
    assert [x.observation_id for x in v] == ["2015007", "2019001"]
    assert v[1].product == "cord" and v[1].uri.endswith("_cord.fits.gz") and v[0].exposure_s is None
    assert (
        v[0].iso.startswith("2015-01-08")
        and v[1].file_url
        == "https://ws.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/data/pub/NEOSSAT/NEOS_SCI_2019001_cord.fits.gz"
    )
    s = neossat.summarise(v)
    assert s["count"] == 2 and 3.9 < s["span_years"] < 4.1 and s["targets"] == ["RW Aur"]


def test_neossat_query_and_fallback_to_the_simpler_query():
    q = neossat.build_query(76.9566, 30.4011, 0.05)
    assert (
        "o.collection = 'NEOSSAT'" in q
        and "INTERSECTS(p.position_bounds, CIRCLE('ICRS', 76.956600, 30.401100, 0.05000)) = 1" in q
        and "caom2.Artifact" in q
    )
    calls = []

    def fetch(query):
        calls.append(query)
        if "caom2.Artifact" in query:
            raise RuntimeError("CADC answered 400")
        return [{k: v for k, v in r.items() if k not in ("uri", "productType")} for r in ROWS]

    v = neossat.search(1, 2, fetch=fetch)
    assert len(calls) == 2 and len(v) == 2 and not neossat.valid_uri(v[0].uri)


@pytest.mark.parametrize(
    "uri,ok",
    [
        ("cadc:NEOSSAT/NEOS_SCI_2019001_cord.fits.gz", True),
        ("cadc:CFHT/x.fits", False),
        ("http://evil.example/x.fits", False),
        ("cadc:NEOSSAT/../../etc/passwd", False),
        ("", False),
    ],
)
def test_only_neossat_identifiers_are_accepted(uri, ok):
    assert neossat.valid_uri(uri) is ok


def test_neossat_thin_keeps_first_and_last():
    v = [neossat.Visit(str(i), "", 57000.0 + i, 1.0, "cord", "") for i in range(100)]
    t = neossat.thin(v, 10)
    assert len(t) == 10 and t[0] is v[0] and t[-1] is v[-1]


def test_fits_cutout_renders_to_png():
    rng = np.random.default_rng(1)
    img = rng.normal(100, 3, (40, 40)).astype(np.float32)
    img[20, 20] = 5000
    buf = io.BytesIO()
    fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(img)]).writeto(buf)
    png = neossat.to_png(neossat.image_from_fits(buf.getvalue()), 160)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    with pytest.raises(ValueError):
        neossat.to_png(np.full((5, 5), np.nan, dtype=np.float32))


# ---------------------------------------------------------------- Gaia via MAST, links, API
def test_mast_gaia_table_is_normalised_whatever_the_capitalisation():
    raw = Table(
        {
            "SOURCE_ID": [1234567890123456789, 2],
            "RA": [10.0, 10.1],
            "Dec": [1.0, 1.1],
            "Parallax": [5.0, np.nan],
            "PMRA": [100.0, np.nan],
            "pmdec": [-50.0, 1.0],
            "phot_g_mean_mag": [12.0, 19.5],
            "extra": [0, 0],
        }
    )
    out = neighborhood.normalise_gaia(raw, mag_limit=18)
    assert out.colnames == list(neighborhood.GAIA_COLS) and len(out) == 1
    assert out["source_id"][0] == "1234567890123456789" and out["ref_epoch"][0] == 2016.0
    with pytest.raises(ValueError):
        neighborhood.normalise_gaia(Table({"ra": [1.0]}), 18)


def test_gaia_cone_falls_back_to_mast(monkeypatch, tmp_path):
    from astropy.coordinates import SkyCoord

    monkeypatch.setattr(neighborhood, "CATALOG_DIR", tmp_path)
    monkeypatch.setattr(
        neighborhood, "_gaia_esa", lambda *a: (_ for _ in ()).throw(RuntimeError("ESA down"))
    )
    good = neighborhood.normalise_gaia(
        Table(
            {
                "source_id": [1],
                "ra": [10.0],
                "dec": [1.0],
                "parallax": [5.0],
                "pmra": [1.0],
                "pmdec": [1.0],
                "phot_g_mean_mag": [12.0],
            }
        ),
        18,
    )
    monkeypatch.setattr(neighborhood, "_gaia_mast", lambda *a: good)
    t = neighborhood.gaia_cone(SkyCoord(10, 1, unit="deg"), 5.0)
    assert len(t) == 1 and len(list(tmp_path.iterdir())) == 1
    assert (
        len(
            neighborhood.propagate(
                t, __import__("astropy.time", fromlist=["Time"]).Time(2026, format="jyear")
            )
        )
        == 1
    )


def test_wiseview_link_matches_the_tools_format():
    u = crosscheck.wiseview_url(133.786245, -7.244372)
    assert u.startswith(
        "http://byw.tools/wiseview#ra=133.786245&dec=-7.244372&size=176&band=2&speed=500&minbright=-50.0000&maxbright=500.0000"
    )
    assert "size=60&" in crosscheck.wiseview_url(1, 1, 5) and all(
        l["url"].startswith("http") for l in crosscheck.links(1, 2)
    )


def test_api_neossat_links_and_cutout_guard(monkeypatch):
    c = TestClient(server.app)
    monkeypatch.setattr(neossat, "search", lambda ra, dec, radius_deg: neossat.parse_rows(ROWS))
    r = c.get("/api/neossat", params={"ra": 76.95, "dec": 30.4}).json()
    assert (
        r["summary"]["count"] == 2
        and [v["has_image"] for v in r["visits"]] == [True, True]
        and r["telescope"]["fov_deg"] == 0.85
    )
    monkeypatch.setattr(
        neossat, "search", lambda *a: (_ for _ in ()).throw(RuntimeError("offline"))
    )
    assert c.get("/api/neossat", params={"ra": 1, "dec": 2}).status_code == 502
    assert (
        c.get(
            "/api/neossat/cutout.png",
            params={"uri": "http://evil.example/a.fits", "ra": 1, "dec": 2},
        ).status_code
        == 422
    )
    L = c.get("/api/links", params={"ra": 133.786245, "dec": -7.244372}).json()
    assert any("wiseview" in l["url"] for l in L["links"]) and len(L["credits"]) >= 4
    assert c.get("/api/config").json()["releases"][0] == "dr1"


def test_page_has_the_other_telescopes_tab():
    html = (server.WEB / "index.html").read_text(encoding="utf-8")
    js = (server.WEB / "app.js").read_text(encoding="utf-8")
    for needle in ('data-tab="more"', 'id="tab-more"'):
        assert needle in html
    for needle in ("/api/neossat", "/api/links", '"more"'):
        assert needle in js
