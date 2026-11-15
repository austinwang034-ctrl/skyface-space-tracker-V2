"""Keeping the site ours: who may call the API, who may frame the page, and how fast.

Nothing can stop a visitor from reading a web page's HTML, CSS and JavaScript: the browser needs
them to show the page. What a server *can* do is refuse to work for other people's websites, and
that is what this module decides:

  - another site may not use our API as its own back end (cross-site requests are refused);
  - another site may not show our full page inside a frame and pass it off as its own
    (only the deliberate ?embed=1 view may be framed);
  - nobody may start trackers faster than a person plausibly would.

Ownership itself is a matter of the licence (see LICENSE and NOTICE), not of code.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from urllib.parse import urlsplit


def _origin_of(url: str) -> str:
    """'https://Example.org/page?x=1' -> 'https://example.org'. Empty for anything unparseable."""
    parts = urlsplit(url.strip())
    return f"{parts.scheme}://{parts.netloc}".lower() if parts.scheme and parts.netloc else ""


def own_hosts(headers, extra: list[str] = ()) -> set[str]:
    """Every address this site answers to for this request: the Host header, the host a proxy
    says the visitor asked for, and any configured ones."""
    hosts = {h.strip().lower() for h in extra}
    for name in ("host", "x-forwarded-host"):
        hosts.update(h.strip().lower() for h in (headers.get(name) or "").split(",") if h.strip())
    return hosts


def is_cross_site(headers, own_host, allowed_origins: list[str]) -> bool:
    """True when a request was made by a page on someone else's website.

    Browsers say where a request comes from in the Origin header (or Referer, for images).
    Requests with neither, such as scripts, curl and tests, are not cross-site: this guards
    against other websites, not against people using the public API directly.
    """
    source = _origin_of(headers.get("origin") or headers.get("referer") or "")
    if not source:
        return False
    ours = {own_host.lower()} if isinstance(own_host, str) else set(own_host)
    if urlsplit(source).netloc in ours:
        return False
    return source not in allowed_origins


def security_headers(path: str, query: str, frame_ancestors: list[str] = ()) -> dict[str, str]:
    """Headers added to every reply. Only the embed view (/?embed=1) may be shown in a frame by
    anyone; the full site only by itself and the sites listed in `frame_ancestors`."""
    headers = {
        "X-Content-Type-Options": "nosniff",
        "Referrer-Policy": "strict-origin-when-cross-origin",
        "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
    }
    embed_view = path == "/" and "embed=1" in query.split("&")
    if not embed_view:
        headers["Content-Security-Policy"] = " ".join(["frame-ancestors 'self'", *frame_ancestors])
        if (
            not frame_ancestors
        ):  # the older header cannot list other sites, so it is only sent alone
            headers["X-Frame-Options"] = "SAMEORIGIN"
    return headers


class RateLimiter:
    """At most `limit` events per `window_s` seconds for each key (an IP address)."""

    def __init__(self, limit: int, window_s: float = 60.0):
        self.limit, self.window_s = limit, window_s
        self._events: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        with self._lock:
            events = self._events[key]
            while events and now - events[0] > self.window_s:
                events.popleft()
            if len(events) >= self.limit:
                return False
            events.append(now)
            if len(self._events) > 10_000:  # forget idle visitors so memory stays bounded
                for other in [
                    k for k, v in self._events.items() if not v or now - v[-1] > self.window_s
                ]:
                    del self._events[other]
            return True
