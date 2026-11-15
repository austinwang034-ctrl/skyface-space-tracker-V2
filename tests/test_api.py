import time

from fastapi.testclient import TestClient

import server

c = TestClient(server.app)


def _wait(jid, tries=300):
    for _ in range(tries):
        s = c.get(f"/api/jobs/{jid}").json()
        if s["state"] in ("done", "error"):
            return s
        time.sleep(0.2)
    raise AssertionError("job timed out")


def test_config_lists_all_presets():
    cfg = c.get("/api/config").json()
    names = {t["name"] for t in cfg["targets"]}
    assert {"Barnard's Star", "WISE 0855-0714", "RW Aurigae", "R Draconis"} <= names
    assert "SPHEREx-D6" in cfg["bands"]


def test_validation_and_unknown_job():
    assert (
        c.post(
            "/api/jobs", json={"source": "irsa", "bands": ["NOPE"], "ra": "1", "dec": "2"}
        ).status_code
        == 422
    )
    assert (
        c.post("/api/jobs", json={"source": "irsa", "ra": "abc", "dec": "def"}).status_code == 422
    )
    assert c.post("/api/jobs", json={"source": "irsa"}).status_code == 422
    assert c.post("/api/jobs", json={"source": "demo", "fov_arcmin": 99}).status_code == 422
    assert c.get("/api/jobs/doesnotexist").status_code == 404


def test_full_demo_flow():
    r = c.post("/api/jobs", json={"source": "demo", "max_frames": 40})
    assert r.status_code == 202
    jid = r.json()["id"]
    assert (
        c.post("/api/jobs", json={"source": "demo", "max_frames": 40}).json()["id"] == jid
    )  # cached
    s = _wait(jid)
    assert s["state"] == "done" and s["n_frames"] == 40

    mv = c.get(f"/api/jobs/{jid}/tracker").json()
    assert c.get(f"/api/jobs/{jid}/movie").json()["grid"] == mv["grid"]  # old name still works
    assert mv["first_obs_utc"] < mv["latest_obs_utc"]
    mjd = [f["mjd"] for f in mv["frames"]]
    assert mv["synthetic"] and mjd == sorted(mjd)  # chronological order
    assert mv["grid"]["nx"] == mv["grid"]["ny"]

    png = c.get(f"/api/jobs/{jid}/frames/0.png")
    assert png.content[:8] == b"\x89PNG\r\n\x1a\n"
    assert c.get(f"/api/jobs/{jid}/frames/999.png").status_code == 404
    assert c.get(f"/api/jobs/{jid}/frames/0.png?stretch=frame").status_code == 200

    det = c.get(f"/api/jobs/{jid}/detectors").json()
    assert det["detectors"] and det["detectors"][0]["n_frames"] > 0
    short = det["detectors"][0]["short"]
    assert c.get(f"/api/jobs/{jid}/detectors/{short}.png").content[:4] == b"\x89PNG"
    assert c.get(f"/api/jobs/{jid}/detectors/D9.png").status_code == 404
    if not det["colour"]:
        assert c.get(f"/api/jobs/{jid}/detectors/colour.png").status_code == 422
    # Wavelength selection: the demo tracker is one detector (D2, 1.10-1.62 µm).
    meta = c.get(f"/api/jobs/{jid}/signature/meta", params={"lo": 1.33, "hi": 1.48}).json()
    assert meta["frames_in_band"] and meta["missing_detectors"] == []
    assert (
        c.get(f"/api/jobs/{jid}/signature/band.png", params={"lo": 1.33, "hi": 1.48}).content[:4]
        == b"\x89PNG"
    )
    far = c.get(f"/api/jobs/{jid}/signature/meta", params={"lo": 4.2, "hi": 4.35}).json()
    assert far["frames_in_band"] == [] and far["missing_detectors"] == ["D5"]
    assert (
        c.get(f"/api/jobs/{jid}/signature/band.png", params={"lo": 4.2, "hi": 4.35}).status_code
        == 404
    )
    assert (
        c.get(f"/api/jobs/{jid}/signature/meta", params={"lo": 3.0, "hi": 2.0}).status_code == 422
    )
    assert "estimate" in c.get(f"/api/jobs/{jid}/temperature").json()
    log = c.get(f"/api/jobs/{jid}/logbook").json()
    assert (
        log["kind"] == "moving" and "north" in log["verdict"]
    )  # demo star copies Barnard's motion
    assert {"Observed", "Alignment", "Detection", "Motion", "Catalogue"} <= {
        x["label"] for x in log["lines"]
    }
    tr = c.get(f"/api/jobs/{jid}/track").json()
    assert (
        tr["fit"] and abs(tr["fit"]["pmdec_masyr"] - 10300) < 300
    )  # plate-solved -> accurate proper motion
    assert sum(1 for row in tr["rows"] if row["ok"]) > 35
    assert "time_utc" in c.get(f"/api/jobs/{jid}/track.csv").text

    sv = c.get(f"/api/jobs/{jid}/solves").json()
    assert sv["ran"] and sum(f["ok"] for f in sv["frames"]) == 40
    assert "rows" in c.get(f"/api/jobs/{jid}/candidates/0").json()

    nb = c.get(f"/api/jobs/{jid}/neighborhood?n_epochs=5").json()
    assert len(nb["years"]) == 5 and len(nb["e"]) == 5 and nb["movers"][0]["pm_total_masyr"] > 10000
    assert c.get(f"/api/jobs/{jid}/neighborhood?year_start=2040&year_end=2030").status_code == 422

    assert c.get(f"/api/jobs/{jid}/tracker.gif").content[:3] == b"GIF"
    assert c.get("/").status_code == 200

    md = c.get(f"/api/jobs/{jid}/metadata").json()
    assert (
        len(md["rows"]) == 40
        and md["flag_legend"]["21"] == "SOURCE"
        and 19 in md["masked_bits"]
        and 21 not in md["masked_bits"]
    )

    sp = c.get(
        f"/api/jobs/{jid}/spectrum?east=0&north=0"
    ).json()  # spectrum on the fly at the patch centre
    assert len(sp["rows"]) == 40 and all(abs(sp[k]) < 1e9 for k in ("ra", "dec"))
    assert c.get(f"/api/jobs/{jid}/spectrum?east=99999&north=0").status_code == 422

    cm = c.get(f"/api/jobs/{jid}/compare/meta?split=0.5").json()
    assert cm["n_early"] + cm["n_late"] == 40 and cm["early_to"] <= cm["late_from"]
    for part in ("early", "late", "diff"):
        assert (
            c.get(f"/api/jobs/{jid}/compare/{part}.png?split=0.3").content[:8]
            == b"\x89PNG\r\n\x1a\n"
        )
    assert c.get(f"/api/jobs/{jid}/compare/diff.png?band=SPHEREx-D9").status_code == 422
    assert c.get("/api/health").json()["ok"]
    assert (
        c.post("/api/jobs", json={"source": "demo", "max_frames": 40, "refresh": True}).json()["id"]
        != jid
    )  # refresh bypasses the cache
