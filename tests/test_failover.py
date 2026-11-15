"""Offline tests for SIMBAD failover: the Harvard mirror, then the built-in test objects."""

import pytest
from astropy.coordinates import SkyCoord
from astropy.time import Time
from fastapi.testclient import TestClient

import server
from spherex_viewer import config, identify, targets

c = TestClient(server.app)


@pytest.fixture
def servers(monkeypatch):
    monkeypatch.setattr(config, "SIMBAD_SERVERS", ["main.example", "mirror.example"])
    monkeypatch.setattr(targets, "SIMBAD_SERVERS", config.SIMBAD_SERVERS)
    return config.SIMBAD_SERVERS


def test_the_mirror_is_used_when_the_main_server_is_down_and_then_preferred(servers):
    asked = []

    def ask(server_name):
        asked.append(server_name)
        if server_name == "main.example":
            raise RuntimeError("404 Not Found\nlong explanation")
        return "answer"

    assert targets.simbad_try(ask) == "answer" and asked == ["main.example", "mirror.example"]
    assert servers[0] == "mirror.example"  # next time the working server is asked first
    asked.clear()
    assert targets.simbad_try(ask) == "answer" and asked == ["mirror.example"]


def test_unknown_names_are_an_answer_and_total_failure_names_every_server(servers):
    def unknown(server_name):
        raise LookupError("no such object")

    with pytest.raises(LookupError):
        targets.simbad_try(unknown)

    def down(server_name):
        raise RuntimeError("404 Not Found\nmore text")

    with pytest.raises(ConnectionError, match="main.example: 404 Not Found.*mirror.example"):
        targets.simbad_try(down)


@pytest.mark.parametrize(
    "typed,key",
    [
        ("Barnard's Star", "Barnard's Star"),
        ("barnard’s star", "Barnard's Star"),
        ("GJ 699", "Barnard's Star"),
        ("WISE 0855-0714", "WISE 0855-0714"),
        ("eps Ind Ba", "Epsilon Indi B (Ba/Bb)"),
        ("RW Aur", "RW Aurigae"),
        ("r draconis", "R Draconis"),
    ],
)
def test_test_objects_are_known_without_a_network(typed, key):
    assert targets.builtin_by_name(typed) is targets.TARGETS[key]
    assert targets.builtin_by_name("Vega") is None


def test_builtin_objects_are_recognised_by_position_allowing_for_motion():
    barnard = targets.TARGETS["Barnard's Star"]
    now = barnard.coord_at(Time(2026.5, format="jyear"))
    here = SkyCoord(now.ra, now.dec, frame="icrs")
    assert targets.builtin_near(here, 2026.5) is barnard
    assert targets.builtin_near(here, 2000.0) is None  # 270 arcsec from where it was in 2000
    assert targets.builtin_near(barnard.coord_ref(), 2000.0) is barnard
    assert targets.builtin_near(SkyCoord(10, 10, unit="deg"), 2000.0) is None


def test_name_lookup_survives_a_simbad_outage(monkeypatch):
    def down(name):
        raise ConnectionError("no SIMBAD server answered")

    monkeypatch.setattr(targets, "_simbad_lookup", down)
    r = c.get("/api/resolve", params={"name": "Barnard's Star"}).json()
    assert abs(r["ra"] - 269.452075) < 1e-6 and r["pmdec"] > 10000 and "built-in" in r["source"]
    w = c.get("/api/resolve", params={"name": "WISE 0855-0714"}).json()
    assert w["epoch"] == 2010.34  # the year its built-in position is valid
    assert c.get("/api/resolve", params={"name": "Vega"}).status_code == 502


def test_tracker_and_identity_card_use_the_builtin_object_when_simbad_is_down(monkeypatch):
    def down(*a, **k):
        raise ConnectionError("no SIMBAD server answered")

    monkeypatch.setattr(identify, "identify", down)
    card = c.post("/api/identify", json={"ra": "269.452075", "dec": "4.693361"}).json()
    assert card["lookup_failed"] and card["match"]["name"] == "Barnard's Star"
    assert "built-in" in card["note"]

    seen = {}
    monkeypatch.setattr(
        server.pipeline,
        "build_bundle",
        lambda src, target, center, *a, **k: seen.update(target=target) or object(),
    )
    job = server.Job(id="t", key="k", req=server.JobRequest(ra="269.452075", dec="4.693361"))
    server.run_job(job)
    assert job.state == "done" and seen["target"].has_motion  # tracked with its real motion
