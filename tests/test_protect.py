"""Offline tests for the protections: cross-site API use, framing, rate limits, licence."""

from fastapi.testclient import TestClient

import server
from spherex_viewer import protect

c = TestClient(server.app)


def test_cross_site_detection():
    own = "skyface.example"
    assert not protect.is_cross_site({}, own, [])  # scripts and tests send no Origin
    assert not protect.is_cross_site({"origin": "https://skyface.example"}, own, [])
    assert not protect.is_cross_site({"referer": "https://skyface.example/?ra=1&dec=2"}, own, [])
    assert protect.is_cross_site({"origin": "https://copycat.example"}, own, [])
    assert protect.is_cross_site({"referer": "https://copycat.example/page"}, own, [])
    assert not protect.is_cross_site(
        {"origin": "https://Friend.example"}, own, ["https://friend.example"]
    )


def test_api_refuses_other_websites_but_serves_itself():
    assert c.get("/api/health", headers={"origin": "https://copycat.example"}).status_code == 403
    assert c.get("/api/health", headers={"referer": "https://copycat.example/x"}).status_code == 403
    assert c.get("/api/health", headers={"origin": "http://testserver"}).status_code == 200
    assert c.get("/api/health").status_code == 200
    # The page and its assets are not part of the API guard: browsers must be able to load them.
    assert c.get("/", headers={"referer": "https://search.example/"}).status_code == 200


def test_only_the_embed_view_may_be_framed():
    full = c.get("/").headers
    assert (
        full["x-frame-options"] == "SAMEORIGIN"
        and "frame-ancestors" in full["content-security-policy"]
    )
    embed = c.get("/?ra=1&dec=2&embed=1").headers
    assert "x-frame-options" not in embed and "content-security-policy" not in embed
    assert embed["x-content-type-options"] == "nosniff"
    assert "frame-ancestors" in c.get("/api/health").headers["content-security-policy"]


def test_rate_limiter_allows_a_burst_then_waits():
    limiter = protect.RateLimiter(limit=3, window_s=60)
    assert [limiter.allow("a", now=t) for t in (0, 1, 2, 3)] == [True, True, True, False]
    assert limiter.allow("b", now=3)  # other visitors are unaffected
    assert limiter.allow("a", now=61.5)  # the oldest event has aged out


def test_job_creation_is_rate_limited(monkeypatch):
    monkeypatch.setattr(server, "JOB_LIMITER", protect.RateLimiter(limit=1))
    assert c.post("/api/jobs", json={"source": "irsa"}).status_code == 422  # counted, then rejected
    assert c.post("/api/jobs", json={"source": "irsa"}).status_code == 429


def test_licence_notice_and_logo_are_served():
    text = c.get("/license").text
    assert "Skyface Space Tracker team" in text and "GNU AFFERO GENERAL PUBLIC LICENSE" in text
    assert "Disallow: /api/" in c.get("/robots.txt").text
    for asset in ("logo.svg", "favicon.svg"):
        r = c.get(f"/assets/{asset}")
        assert r.status_code == 200 and "<svg" in r.text
    html = c.get("/").text
    assert "/assets/logo.svg" in html and 'href="/license"' in html and "©" in html


def test_a_proxy_or_space_address_counts_as_this_site():
    headers = {"host": "10.0.0.5:7860", "x-forwarded-host": "team-skyface.hf.space"}
    ours = protect.own_hosts(headers, ["other-name.hf.space"])
    assert ours == {"10.0.0.5:7860", "team-skyface.hf.space", "other-name.hf.space"}
    assert not protect.is_cross_site({"origin": "https://team-skyface.hf.space"}, ours, [])
    assert protect.is_cross_site({"origin": "https://copycat.example"}, ours, [])


def test_the_hosting_site_may_frame_the_full_page_when_listed():
    plain = protect.security_headers("/", "")
    assert (
        plain["Content-Security-Policy"] == "frame-ancestors 'self'" and "X-Frame-Options" in plain
    )
    hosted = protect.security_headers("/", "", ["https://huggingface.co"])
    assert hosted["Content-Security-Policy"] == "frame-ancestors 'self' https://huggingface.co"
    assert "X-Frame-Options" not in hosted


def test_deployment_files_target_the_free_render_plan():
    blueprint = (server.ROOT / "render.yaml").read_text(encoding="utf-8")
    assert (
        "plan: free" in blueprint and "disk:" not in blueprint and "SKYFACE_MAX_FRAMES" in blueprint
    )
    docker = (server.ROOT / "Dockerfile").read_text(encoding="utf-8")
    for needle in ("--uid 1000", "USER user", "${PORT:-7860}", "COPY --chown=user spherex_viewer"):
        assert needle in docker, needle
    assert c.get("/api/config").json()["max_frames"] == server.MAX_FRAMES
