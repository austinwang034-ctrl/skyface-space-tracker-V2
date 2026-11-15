import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from fastapi.testclient import TestClient

import server
from spherex_viewer import identify, targets

c = TestClient(server.app)
BARNARD = dict(
    name="V* V2500 Oph",
    ra=269.452075,
    dec=4.693361,
    otype="PM*",
    sp_type="M4.0V",
    pmra=-802.803,
    pmdec=10362.394,
    parallax=546.976,
    mags={"V": 9.5},
)
DECOY = dict(
    name="Decoy Star",
    ra=269.4521,
    dec=4.6936,
    otype="*",
    sp_type=None,
    pmra=None,
    pmdec=None,
    parallax=None,
    mags={},
)


def barnard_at(year):
    return targets.TARGETS["Barnard's Star"].coord_at(Time(year, format="jyear"))


def test_fast_mover_matches_despite_270_arcsec_of_motion():
    q = barnard_at(2026.85)
    assert (
        q.separation(SkyCoord(BARNARD["ra"] * u.deg, BARNARD["dec"] * u.deg)).arcsec > 250
    )  # J2000 entry is far away
    res = identify.rank_candidates([DECOY, BARNARD], q, 2026.85)
    assert (
        res.match.name == BARNARD["name"]
        and res.match.sep_arcsec < 1.0
        and "Strong match" in res.note
    )


def test_j2000_query_and_far_query():
    res = identify.rank_candidates(
        [DECOY, BARNARD], SkyCoord(269.452075 * u.deg, 4.693361 * u.deg), 2000.0
    )
    assert (
        res.match.name == BARNARD["name"] or res.match.name == "Decoy Star"
    )  # both are within a few arcsec
    far = identify.rank_candidates([BARNARD], SkyCoord(10 * u.deg, 10 * u.deg), 2000.0)
    assert far.match is None and "closest" in far.note.lower()
    assert identify.rank_candidates([], SkyCoord(1 * u.deg, 1 * u.deg), 2000.0).match is None


def test_identify_uses_pm_aware_radius_and_caches():
    calls = []

    def fake(center, radius):
        calls.append(radius)
        return [BARNARD]

    q = barnard_at(2026.85)
    r1 = identify.identify(q.ra.deg, q.dec.deg, 2026.85, fetch=fake)
    r2 = identify.identify(q.ra.deg, q.dec.deg, 2026.85, fetch=fake)
    assert (
        r1.match and r2 is r1 and len(calls) == 1 and calls[0] >= 290
    )  # ~5 arcmin cone for a 27-year gap


def test_object_details_and_reference_images():
    d = identify.ObjectInfo(**{**BARNARD, "sep_arcsec": 0.5}).to_dict()
    assert (
        abs(d["distance_pc"] - 1.828) < 0.01
        and abs(d["vtan_kms"] - 90.1) < 0.5
        and d["type_label"] == "High-proper-motion star"
    )
    imgs = identify.reference_images(269.45, 4.69, 6.0)
    assert (
        len(imgs) == 4
        and imgs[0]["id"].endswith("SPHEREx/QR2/color")
        and all("hips2fits" in i["url"] and "fov=0.10000" in i["url"] for i in imgs)
    )
    t = identify.target_from_object(identify.ObjectInfo(**BARNARD))
    assert t.kind == "mover" and t.has_motion and t.name == BARNARD["name"]


def test_identify_endpoint(monkeypatch):
    q = barnard_at(2026.85)
    monkeypatch.setattr(
        identify,
        "identify",
        lambda ra, dec, epoch, **k: identify.rank_candidates([BARNARD], q, epoch),
    )
    j = c.post(
        "/api/identify", json={"ra": str(q.ra.deg), "dec": str(q.dec.deg), "epoch": 2026.85}
    ).json()
    assert (
        j["match"]["name"] == BARNARD["name"] and len(j["images"]) == 4 and not j["lookup_failed"]
    )
    assert "ra=269.452075" in j["images"][0]["url"]  # fast mover: centred on the J2000 entry
    assert "Images are centred" not in j["image_note"]

    def boom(*a, **k):
        raise ConnectionError("no network")

    monkeypatch.setattr(identify, "identify", boom)
    j = c.post("/api/identify", json={"ra": "10", "dec": "20"}).json()
    assert (
        j["lookup_failed"] and j["match"] is None and len(j["images"]) == 4
    )  # images still work without SIMBAD
    assert "ra=10.000000" in j["images"][0]["url"]
    assert c.post("/api/identify", json={"ra": "abc", "dec": "x"}).status_code == 422
    demo = c.post("/api/identify", json={"ra": "0", "dec": "0", "demo": True}).json()
    assert demo["canned"] and "demo" in demo["match"]["name"].lower()


def test_job_tracks_the_identified_star(monkeypatch):
    q = barnard_at(2026.85)
    seen = {}
    monkeypatch.setattr(
        identify,
        "identify",
        lambda ra, dec, epoch, **k: identify.rank_candidates([BARNARD], q, epoch),
    )
    monkeypatch.setattr(
        server.pipeline,
        "build_bundle",
        lambda src, target, center, *a, **k: seen.update(target=target, center=center) or object(),
    )
    req = server.JobRequest(source="irsa", ra=str(q.ra.deg), dec=str(q.dec.deg), epoch=2026.85)
    job = server.Job(id="t", key="k", req=req)
    server.run_job(job)
    assert (
        job.state == "done" and seen["target"].name == BARNARD["name"] and seen["target"].has_motion
    )
    # patch is centred where the star is during the SPHEREx data (~2026.2), not at its old J2000 spot
    assert seen["center"].separation(barnard_at(2026.2)).arcsec < 1.0
