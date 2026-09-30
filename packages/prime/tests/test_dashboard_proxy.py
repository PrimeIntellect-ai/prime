"""Loopback dashboard proxy: request mapping, SSE passthrough, error hygiene."""

import http.client
import json
import os
import signal
import socket
import string
import subprocess
import sys
import threading
import time
import urllib.parse
from pathlib import Path
from typing import Any, Optional

import httpx
import pytest
from prime_cli.dashboard_proxy import (
    _proxy_fingerprint,
    make_dashboard_proxy_server,
    proxy_state_path,
    start_detached_dashboard_proxy,
)

BASE_URL = "https://api.example.com"


@pytest.fixture
def proxy_factory():
    servers: list[Any] = []

    def start(upstream: httpx.Client, run_id: str = "run-1") -> tuple[str, int, str]:
        server, url = make_dashboard_proxy_server(
            run_id,
            base_url=BASE_URL,
            api_key="test-key",
            upstream=upstream,
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        servers.append(server)
        return url, server.server_address[1], server.capability_token

    yield start

    for server in servers:
        server.shutdown()
        server.server_close()


def _get(
    port: int,
    token: str,
    path: str,
    headers: Optional[dict[str, str]] = None,
) -> http.client.HTTPResponse:
    """GET ``path`` with the capability cookie (a warmed-up browser)."""
    merged = {"Cookie": f"t={token}"}
    merged.update(headers or {})
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    conn.request("GET", path, headers=merged)
    return conn.getresponse()


def _entry(port: int, token: str, path: str = "") -> http.client.HTTPResponse:
    """GET the printed URL: the one-time token path-segment handoff."""
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    conn.request("GET", f"/{token}{path}")
    return conn.getresponse()


def test_loopback_url_embeds_unguessable_capability_token(proxy_factory) -> None:
    """Every spawn mints a fresh token and embeds it in the URL path."""
    url, port, token = proxy_factory(
        httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200)))
    )

    assert url == f"http://127.0.0.1:{port}/{token}"
    assert len(token) >= 32  # unguessable: ~256 bits of entropy
    _, _, token_2 = proxy_factory(
        httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200)))
    )
    assert token_2 != token  # per-spawn randomness, never derived from the run


def test_printed_url_is_glob_safe_for_command_substitution(proxy_factory) -> None:
    """The stdout URL must survive UNQUOTED command substitution on zsh.

    `open $(prime train dashboard <id> --no-browser)` runs the printed
    URL through zsh's glob expansion: a ``?`` (the earlier ``?t=``
    handoff) makes zsh's default NOMATCH abort with "no matches found".
    The token is token_urlsafe output used as the sole path segment, so
    the URL must contain only glob-safe characters — letters, digits,
    ``-``, ``_``, ``:`` and ``/`` — and no ``?``, ``*``, ``[``, ``]`` or
    ``=``. (Quoting the substitution is still good practice, but the
    CLI's documented command-substitution contract must not require it.)
    """
    url, _, token = proxy_factory(
        httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200)))
    )
    # token_urlsafe alphabet plus the authority separators only.
    assert set(token) <= set(string.ascii_letters + string.digits + "_-")
    assert set(url) <= set(string.ascii_letters + string.digits + ".:/_-")
    assert not set(url) & set("?*[]=~^")


def test_proxy_rejects_requests_without_capability_cookie(proxy_factory) -> None:
    """Another local user who port-scans the loopback port gets nothing.

    Without the capability cookie a request is rejected with 403 BEFORE
    the Host/Origin checks and BEFORE any upstream work — the victim's
    bearer token is never used.
    """
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, content=b"secret dashboard")

    _, port, token = proxy_factory(httpx.Client(transport=httpx.MockTransport(handler)))

    # No capability at all: no cookie, no token path segment.
    status, body = _raw_request(port, [f"Host: 127.0.0.1:{port}"], path="/")
    assert status == 403
    assert seen == []
    assert b"secret" not in body

    # A wrong cookie and a wrong token path segment (including
    # near-misses on the real token), plus a cookie jar without a
    # "t"-keyed value.
    for wrong in ("", "wrong-token", token[:-1], token + "x", token.upper()):
        status, body = _raw_request(
            port, [f"Host: 127.0.0.1:{port}", f"Cookie: t={wrong}"], path="/"
        )
        assert status == 403, wrong
        status, body = _raw_request(port, [f"Host: 127.0.0.1:{port}"], path=f"/{wrong}")
        assert status == 403, wrong
        assert seen == []
        assert b"secret" not in body
    status, body = _raw_request(
        port, [f"Host: 127.0.0.1:{port}", f"Cookie: other={token}"], path="/"
    )
    assert status == 403

    # Malformed and non-ASCII targets without a capability: clean 403,
    # never a crash (and never a 500).
    for path in ("//[::1", "/[::1", "/%C3%A9/x", "/\u00e9/x"):
        status, body = _raw_request(port, [f"Host: 127.0.0.1:{port}"], path=path)
        assert status == 403, path
        assert seen == []
        assert b"secret" not in body

    # The right cookie goes through.
    response = _get(port, token, "/")
    assert response.status == 200
    assert response.read() == b"secret dashboard"
    assert len(seen) == 1

    # An authorized request with a weird-but-in-scope target never crashes
    # either: it stays mapped inside the run-scoped dashboard route.
    status, body = _raw_request(
        port, [f"Host: 127.0.0.1:{port}", f"Cookie: t={token}"], path="/[::1"
    )
    assert status == 200
    assert len(seen) == 2
    assert seen[1].endswith("/dashboard/[::1")


def test_entry_request_with_path_token_sets_capability_cookie(proxy_factory) -> None:
    """The printed URL's token segment becomes the loopback cookie.

    ``GET /<token>`` serves the dashboard root directly (no redirect)
    and sets the cookie that authorizes every subsequent request.
    """
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        return httpx.Response(200, headers={"Content-Type": "text/html"}, content=b"<html>x</html>")

    _, port, token = proxy_factory(httpx.Client(transport=httpx.MockTransport(handler)))

    response = _entry(port, token)

    assert response.status == 200
    assert response.read() == b"<html>x</html>"
    # HttpOnly keeps the token out of page JavaScript; SameSite=Strict
    # blocks cross-site sends; Path=/ covers every root-relative request.
    assert response.getheader("Set-Cookie") == f"t={token}; Path=/; HttpOnly; SameSite=Strict"
    # The handoff token never reaches the upstream path.
    assert seen["url"] == f"{BASE_URL}/api/v1/rft/runs/run-1/dashboard/"

    # The token segment also authorizes subpaths directly (no cookie yet)
    # and strips itself: /<token>/static/x maps to the dashboard subpath.
    seen.clear()
    response = _entry(port, token, path="/static/x")
    assert response.status == 200
    assert response.getheader("Set-Cookie") == f"t={token}; Path=/; HttpOnly; SameSite=Strict"
    assert seen["url"] == f"{BASE_URL}/api/v1/rft/runs/run-1/dashboard/static/x"


def test_root_relative_dashboard_requests_work_with_capability_cookie(proxy_factory) -> None:
    """THE regression the path-prefix design broke (Bugbot t028).

    The dashboard is a root-relative app: its assets, API calls and SSE
    streams all request loopback-ROOT paths (/static/..., /api/...).
    Those must pass with the capability cookie alone — no token in any
    path.
    """
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        if str(request.url).endswith("events"):
            return httpx.Response(
                200,
                headers={"Content-Type": "text/event-stream"},
                content=b"data: 1\n\n",
            )
        return httpx.Response(200, content=b"asset-or-api")

    _, port, token = proxy_factory(httpx.Client(transport=httpx.MockTransport(handler)))

    # Browser opens the printed URL (handoff), then loads root-relative assets.
    assert _entry(port, token).status == 200
    for path in ("/static/app.js?v=2", "/api/view/events", "/api/v1/rft/runs/run-1/x"):
        response = _get(port, token, path)
        assert response.status == 200, path
        assert response.read() in (b"asset-or-api", b"data: 1\n\n")

    assert seen == [
        f"{BASE_URL}/api/v1/rft/runs/run-1/dashboard/",
        f"{BASE_URL}/api/v1/rft/runs/run-1/dashboard/static/app.js?v=2",
        f"{BASE_URL}/api/v1/rft/runs/run-1/dashboard/api/view/events",
        f"{BASE_URL}/api/v1/rft/runs/run-1/dashboard/api/v1/rft/runs/run-1/x",
    ]


def test_proxy_maps_root_to_run_scoped_route(proxy_factory) -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["url"] = str(request.url)
        seen["authorization"] = request.headers.get("authorization")
        seen["accept_encoding"] = request.headers.get("accept-encoding")
        return httpx.Response(
            200,
            headers={"Content-Type": "text/html; charset=utf-8"},
            content=b"<html>dashboard</html>",
        )

    _, port, token = proxy_factory(httpx.Client(transport=httpx.MockTransport(handler)))

    response = _get(port, token, "/")

    assert response.status == 200
    assert response.getheader("Content-Type") == "text/html; charset=utf-8"
    assert response.read() == b"<html>dashboard</html>"
    assert seen["method"] == "GET"
    assert seen["url"] == f"{BASE_URL}/api/v1/rft/runs/run-1/dashboard/"
    assert seen["authorization"] == "Bearer test-key"
    assert seen["accept_encoding"] == "identity"


def test_proxy_maps_root_relative_paths_and_query_strings(proxy_factory) -> None:
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        return httpx.Response(
            200,
            headers={"Content-Type": "text/javascript"},
            content=b"console.log(1)",
        )

    _, port, token = proxy_factory(httpx.Client(transport=httpx.MockTransport(handler)))

    response = _get(port, token, "/static/app.js?v=2")

    assert response.status == 200
    assert response.getheader("Content-Type") == "text/javascript"
    assert response.read() == b"console.log(1)"
    assert seen["url"] == f"{BASE_URL}/api/v1/rft/runs/run-1/dashboard/static/app.js?v=2"


def test_proxy_forwards_the_raw_query_verbatim(proxy_factory) -> None:
    """The browser's query bytes reach upstream UNCHANGED.

    Rebuilding the query (parse_qsl + urlencode) would rewrite commas,
    colons, slashes and spaces — dashboard API calls can change meaning
    even when no handoff parameter is present. The proxy must not
    re-encode anything.
    """
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["query"] = urllib.parse.urlsplit(str(request.url)).query
        return httpx.Response(200, content=b"ok")

    _, port, token = proxy_factory(httpx.Client(transport=httpx.MockTransport(handler)))

    raw_query = "q=a,b:c/d%20e&filter=x%2Fy&empty=&flag"
    response = _get(port, token, f"/search?{raw_query}")

    assert response.status == 200
    assert seen["query"] == raw_query  # byte-identical, no re-encoding


def test_proxy_streams_sse_events_through_incrementally(proxy_factory) -> None:
    upstream_done = threading.Event()

    def sse_body():
        yield b"data: 1\n\n"
        time.sleep(0.25)
        yield b"data: 2\n\n"
        upstream_done.set()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"Content-Type": "text/event-stream", "Cache-Control": "no-store"},
            content=sse_body(),
        )

    _, port, token = proxy_factory(httpx.Client(transport=httpx.MockTransport(handler)))

    response = _get(port, token, "/api/view/events")

    assert response.status == 200
    assert response.getheader("Content-Type") == "text/event-stream"
    assert response.getheader("Cache-Control") == "no-store"
    # The first event must be delivered before the upstream stream is done:
    # buffering the whole body would break live dashboards.
    assert response.readline() == b"data: 1\n"
    assert response.readline() == b"\n"
    assert not upstream_done.is_set()
    assert response.read() == b"data: 2\n\n"
    assert upstream_done.is_set()


def test_proxy_detects_client_disconnect_on_quiet_sse_streams(monkeypatch) -> None:
    """A browser tab closed mid-SSE must release the idle watchdog.

    The handler may be blocked in a quiet upstream stream indefinitely
    (upstream reads have no timeout); the proxy must notice the
    downstream FIN and finish the request so the idle watchdog can shut
    a detached proxy down instead of pinning it (and its API
    credential) forever.
    """
    monkeypatch.setattr("prime_cli.dashboard_proxy._STREAM_LIVENESS_POLL_SECONDS", 0.2)
    release = threading.Event()

    def sse_body():
        yield b"data: 1\n\n"
        release.wait(timeout=30)  # quiet: no further upstream events

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            content=sse_body(),
        )

    server, _url = make_dashboard_proxy_server(
        "run-1",
        base_url=BASE_URL,
        api_key="test-key",
        upstream=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    serve_thread = threading.Thread(target=server.serve_forever, daemon=True)
    serve_thread.start()
    try:
        port = server.server_address[1]
        # Raw socket client: http.client's close() keeps the fd open behind
        # the response's file object, so it never delivers the FIN the way
        # a real browser tab close does. shutdown() does.
        sock = socket.create_connection(("127.0.0.1", port), timeout=5)
        sock.sendall(f"GET /{server.capability_token} HTTP/1.1\r\n".encode())
        sock.sendall(f"Host: 127.0.0.1:{port}\r\nConnection: close\r\n\r\n".encode())
        data = b""
        while b"data: 1\n" not in data:
            chunk = sock.recv(65536)
            assert chunk, "stream closed before the first event"
            data += chunk
        assert b"HTTP/1.1 200" in data.split(b"\r\n")[0]
        assert not server.is_idle_for(0.05)  # the request is in flight

        sock.shutdown(socket.SHUT_RDWR)  # the browser tab closes
        sock.close()

        deadline = time.monotonic() + 10
        while not server.is_idle_for(0.05) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert server.is_idle_for(0.05), "handler must finish after the browser disconnects"
    finally:
        release.set()
        server.shutdown()
        server.server_close()


def test_proxy_forwards_accept_and_last_event_id_headers(proxy_factory) -> None:
    """SSE negotiation: the browser's Accept and Last-Event-ID reach upstream.

    Accept lets the EventSource negotiate text/event-stream; Last-Event-ID
    lets a reconnecting stream RESUME instead of restarting (which would
    duplicate events).
    """
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["accept"] = request.headers.get("accept", "")
        seen["last_event_id"] = request.headers.get("last-event-id", "")
        return httpx.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            content=b"data: 1\n\n",
        )

    _, port, token = proxy_factory(httpx.Client(transport=httpx.MockTransport(handler)))

    response = _get(
        port,
        token,
        "/api/view/events",
        headers={"Accept": "text/event-stream", "Last-Event-ID": "42"},
    )

    assert response.status == 200
    assert seen["accept"] == "text/event-stream"
    assert seen["last_event_id"] == "42"


def test_proxy_forwards_no_other_request_headers(proxy_factory) -> None:
    """Only Accept and Last-Event-ID cross to the platform, nothing else."""
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["cookie"] = request.headers.get("cookie", "")
        seen["x_custom"] = request.headers.get("x-custom", "")
        seen["forwarded"] = request.headers.get("forwarded", "")
        seen["user_agent"] = request.headers.get("user-agent", "")
        return httpx.Response(200, content=b"ok")

    _, port, token = proxy_factory(httpx.Client(transport=httpx.MockTransport(handler)))

    response = _get(
        port,
        token,
        "/",
        headers={
            # A real browser would send its own jar alongside the
            # capability cookie; the merge below must keep both.
            "Cookie": f"t={token}; session=attacker",
            "X-Custom": "leak-me",
            "Forwarded": "for=1.2.3.4",
            "User-Agent": "evil-browser",
        },
    )

    assert response.status == 200
    assert seen["cookie"] == ""
    assert seen["x_custom"] == ""
    assert seen["forwarded"] == ""
    assert seen["user_agent"] not in ("", "evil-browser")  # CLI default only


def test_proxy_surfaces_upstream_http_errors_as_clean_local_errors(proxy_factory) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        # Raw upstream failures must not leak internal hosts to the browser.
        return httpx.Response(
            503,
            headers={"Content-Type": "application/json"},
            content=b'{"detail": "upstream http://rl-dashboard.tail-9.ts.net:7788 down"}',
        )

    _, port, token = proxy_factory(httpx.Client(transport=httpx.MockTransport(handler)))

    response = _get(port, token, "/")

    body = response.read()
    assert response.status == 502
    assert b"Dashboard backend returned HTTP 503." in body
    assert b"tail-" not in body  # no internal tailnet URL leaks


def test_proxy_surfaces_transport_failures_as_clean_local_errors(proxy_factory) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused to 10.0.0.1:7788")

    _, port, token = proxy_factory(httpx.Client(transport=httpx.MockTransport(handler)))

    response = _get(port, token, "/")

    body = response.read()
    assert response.status == 502
    assert b"Dashboard backend is unreachable." in body
    assert b"10.0.0.1" not in body  # no internal address leaks


# --- Redirect handling ------------------------------------------------------


def test_proxy_rewrites_same_origin_redirect_to_loopback_root(proxy_factory) -> None:
    redirect_target = f"{BASE_URL}/api/v1/rft/runs/run-1/dashboard/static/index.html?v=2"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"Location": redirect_target}, content=b"")

    _, port, token = proxy_factory(httpx.Client(transport=httpx.MockTransport(handler)))

    response = _get(port, token, "/")

    # Absolute in-prefix Location must become a plain root-relative loopback
    # path (scheme, host, run prefix and query preserved); the browser's
    # capability cookie covers the follow-up, so no token is embedded.
    assert response.status == 302
    location = response.getheader("Location")
    assert location == "/static/index.html?v=2"
    assert token not in location


def test_proxy_forwards_relative_redirect_unchanged(proxy_factory) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(308, headers={"Location": "/static/app.js"}, content=b"")

    _, port, token = proxy_factory(httpx.Client(transport=httpx.MockTransport(handler)))

    response = _get(port, token, "/")

    assert response.status == 308
    # Ordinary root-relative locations pass through semantically unchanged.
    assert response.getheader("Location") == "/static/app.js"


def test_proxy_passes_truly_relative_redirect_through_verbatim(proxy_factory) -> None:
    """A location WITHOUT a leading slash resolves below the token segment.

    It must be relayed verbatim: prefixing it would corrupt the path the
    browser resolves against the token-prefixed current URL.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(308, headers={"Location": "static/app.js"}, content=b"")

    _, port, token = proxy_factory(httpx.Client(transport=httpx.MockTransport(handler)))

    response = _get(port, token, "/")
    assert response.status == 308
    assert response.getheader("Location") == "static/app.js"


def test_proxy_preserves_fragments_in_rewritten_redirects(proxy_factory) -> None:
    """Absolute same-origin redirect with a fragment keeps it on the rewrite."""
    redirect_target = f"{BASE_URL}/api/v1/rft/runs/run-1/dashboard/#/metrics"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"Location": redirect_target}, content=b"")

    _, port, token = proxy_factory(httpx.Client(transport=httpx.MockTransport(handler)))

    response = _get(port, token, "/")
    assert response.status == 302
    assert response.getheader("Location") == "/#/metrics"


def test_proxy_preserves_fragments_in_platform_shaped_redirects(proxy_factory) -> None:
    """Platform-shaped relative redirect with query AND fragment keeps both."""
    redirect_target = "/api/v1/rft/runs/run-1/dashboard/static/index?next=1#/metrics"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"Location": redirect_target}, content=b"")

    _, port, token = proxy_factory(httpx.Client(transport=httpx.MockTransport(handler)))

    response = _get(port, token, "/")
    assert response.status == 302
    assert response.getheader("Location") == "/static/index?next=1#/metrics"


def test_proxy_rejects_cross_origin_redirects(proxy_factory) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            302,
            headers={"Location": "https://evil.example.com/dashboard/index.html"},
            content=b"",
        )

    _, port, token = proxy_factory(httpx.Client(transport=httpx.MockTransport(handler)))

    response = _get(port, token, "/")

    body = response.read()
    assert response.status == 502
    assert "redirect rejected" in body.decode("utf-8")
    assert response.getheader("Location") is None
    assert b"evil.example.com" not in body


def test_proxy_rejects_same_origin_redirects_outside_the_run_scope(proxy_factory) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            302,
            headers={"Location": f"{BASE_URL}/api/v1/rft/runs/run-1/logs"},
            content=b"",
        )

    _, port, token = proxy_factory(httpx.Client(transport=httpx.MockTransport(handler)))

    response = _get(port, token, "/")

    assert response.status == 502
    assert response.read().decode("utf-8").startswith("Dashboard backend redirect rejected")


@pytest.mark.parametrize(
    ("location", "expected"),
    [
        ("/static/x", "/static/x"),
        (f"{BASE_URL}/api/v1/rft/runs/run-1/dashboard/", "/"),
        (f"{BASE_URL}/api/v1/rft/runs/run-1/dashboard", "/"),
        (f"{BASE_URL}/api/v1/rft/runs/run-1/dashboard/a/b?c=1", "/a/b?c=1"),
        ("https://evil.example.com/dashboard/", None),
        (f"{BASE_URL}/api/v1/rft/runs/run-1/logs", None),
        (f"{BASE_URL}/api/v1/rft/runs/run-2/dashboard/", None),
        # Platform-rewritten relative locations: strip the run prefix.
        ("/api/v1/rft/runs/run-1/dashboard/static/x", "/static/x"),
        ("/api/v1/rft/runs/run-1/dashboard/", "/"),
        ("/api/v1/rft/runs/run-1/dashboard", "/"),
        ("/api/v1/rft/runs/run-1/dashboard/a?b=1", "/a?b=1"),
        # Platform-shaped but out of scope for this run: reject.
        ("/api/v1/rft/runs/run-2/dashboard/x", None),
        ("/api/v1/rft/runs/run-1/logs", None),
        # Double slash after the prefix: single joining slash, never "//".
        ("/api/v1/rft/runs/run-1/dashboard//static/x", "/static/x"),
        (f"{BASE_URL}/api/v1/rft/runs/run-1/dashboard//static/x", "/static/x"),
        # Remainders with a leading "//" (triple slash) would build a
        # protocol-relative Location: reject as out-of-scope.
        ("/api/v1/rft/runs/run-1/dashboard///evil.example.com/x", None),
        (f"{BASE_URL}/api/v1/rft/runs/run-1/dashboard///evil.example.com/x", None),
        # A bare protocol-relative Location (netloc form): reject — the
        # browser would otherwise leave the loopback origin entirely.
        ("//evil.example.com/x", None),
        ("//127.0.0.1:9999/x", None),
        # Fragments are SPA routing state: they must survive every rewrite.
        ("/static/x#/metrics", "/static/x#/metrics"),
        (f"{BASE_URL}/api/v1/rft/runs/run-1/dashboard/#/metrics", "/#/metrics"),
        ("/api/v1/rft/runs/run-1/dashboard/static/x?q=1#/metrics", "/static/x?q=1#/metrics"),
        (f"{BASE_URL}/api/v1/rft/runs/run-1/dashboard/a/b#/metrics", "/a/b#/metrics"),
        # Cross-origin/rejected locations with fragments stay rejected.
        ("https://evil.example.com/dashboard/#/metrics", None),
        ("/api/v1/rft/runs/run-2/dashboard/x#/metrics", None),
    ],
)
def test_map_dashboard_redirect(location: str, expected: Optional[str]) -> None:
    from prime_cli.dashboard_proxy import map_dashboard_redirect

    assert map_dashboard_redirect(location, BASE_URL, "run-1") == expected


# --- Idle watchdog -----------------------------------------------------------


def test_server_shuts_down_after_idle_timeout(monkeypatch) -> None:
    from prime_cli.dashboard_proxy import _idle_watchdog

    server, _ = make_dashboard_proxy_server("run-1", base_url=BASE_URL, api_key="test-key")
    serve_thread = threading.Thread(target=server.serve_forever, daemon=True)
    serve_thread.start()

    # With a short timeout and poll interval the watchdog must call
    # server.shutdown(), which ends serve_forever().
    _idle_watchdog(server, idle_timeout_seconds=0.1, poll_interval=0.05)

    serve_thread.join(timeout=5)
    assert not serve_thread.is_alive()
    server.server_close()


# --- Detached spawner --------------------------------------------------------


class _FakeChildProcess:
    def __init__(self, ready_line: Optional[str]) -> None:
        import io

        self.stdout = io.StringIO(ready_line or "")
        self.killed = False
        self.stdout_closed = False
        self.pid = 424242

    def kill(self) -> None:
        self.killed = True

    def wait(self, timeout: float = 0) -> Optional[int]:
        return 0

    def close_stdout(self) -> None:
        self.stdout_closed = True


def test_proxy_fingerprint_is_stable_and_context_sensitive() -> None:
    """The reuse key must be stable across calls but distinct per context."""
    fp_a = _proxy_fingerprint(BASE_URL, "run-1", "token-A")
    assert _proxy_fingerprint(BASE_URL, "run-1", "token-A") == fp_a
    assert _proxy_fingerprint(BASE_URL, "run-1", "token-B") != fp_a
    assert _proxy_fingerprint("https://other.example.com", "run-1", "token-A") != fp_a
    assert _proxy_fingerprint(BASE_URL, "run-2", "token-A") != fp_a
    # The token itself must never appear in the digest.
    assert "token-A" not in fp_a


def test_start_detached_spawns_child_and_returns_ready_port(monkeypatch, tmp_path) -> None:
    spawn_calls: list[Any] = []

    def fake_popen(command, **kwargs):
        spawn_calls.append({"command": command, **kwargs})
        return _FakeChildProcess("PORT 51234 fake-capability-token\n")

    monkeypatch.setattr("prime_cli.dashboard_proxy.subprocess.Popen", fake_popen)

    url = start_detached_dashboard_proxy(
        "run-1",
        base_url=BASE_URL,
        api_key="test-key",
        state_dir=tmp_path,
    )

    assert url == "http://127.0.0.1:51234/fake-capability-token"
    assert len(spawn_calls) == 1
    call = spawn_calls[0]
    assert call["command"][:3] == [sys.executable, "-m", "prime_cli.dashboard_proxy"]
    # The API token must travel via the environment, never the command line
    # (process tables are world-readable).
    assert "test-key" not in " ".join(call["command"])
    # The capability token also never travels on the command line: it
    # crosses the private ready pipe and the 0600 state file only.
    assert "fake-capability-token" not in " ".join(call["command"])
    assert call["env"]["PRIME_DASHBOARD_PROXY_API_KEY"] == "test-key"
    assert call["start_new_session"] is True


def test_start_detached_fails_cleanly_when_child_never_becomes_ready(monkeypatch, tmp_path) -> None:
    child = _FakeChildProcess(None)  # emits no PORT line

    def fake_popen(command, **kwargs):
        return child

    monkeypatch.setattr("prime_cli.dashboard_proxy.subprocess.Popen", fake_popen)

    with pytest.raises(RuntimeError, match="failed to start"):
        start_detached_dashboard_proxy(
            "run-1",
            base_url=BASE_URL,
            api_key="test-key",
            state_dir=tmp_path,
            ready_timeout_seconds=0.2,
        )


def test_start_detached_reuses_live_proxy_from_state_file(monkeypatch, tmp_path) -> None:
    # A real loopback server counts as a live proxy: its port accepts
    # connections, the recorded pid is this (running) process, and its
    # upstream answers the credentials probe with a 2xx.
    healthy_upstream = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, content=b"<html>ok</html>")
        )
    )
    server, url = make_dashboard_proxy_server(
        "run-1", base_url=BASE_URL, api_key="test-key", upstream=healthy_upstream
    )
    serve_thread = threading.Thread(target=server.serve_forever, daemon=True)
    serve_thread.start()
    try:
        fingerprint = _proxy_fingerprint(BASE_URL, "run-1", "test-key")
        state_path = proxy_state_path("run-1", state_dir=tmp_path, fingerprint=fingerprint)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(
            json.dumps(
                {
                    "run_id": "run-1",
                    "pid": os.getpid(),
                    "port": server.server_address[1],
                    "token": server.capability_token,
                }
            )
        )

        def no_spawn(command, **kwargs):
            raise AssertionError("a live proxy must be reused, not respawned")

        monkeypatch.setattr("prime_cli.dashboard_proxy.subprocess.Popen", no_spawn)

        reused = start_detached_dashboard_proxy(
            "run-1", base_url=BASE_URL, api_key="test-key", state_dir=tmp_path
        )

        assert reused == url
    finally:
        server.shutdown()
        server.server_close()


def test_start_detached_ignores_stale_state_file(monkeypatch, tmp_path) -> None:
    fingerprint = _proxy_fingerprint(BASE_URL, "run-1", "test-key")
    state_path = proxy_state_path("run-1", state_dir=tmp_path, fingerprint=fingerprint)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    # A dead pid and a closed port: the starter must clean up the stale
    # state and respawn.
    state_path.write_text(json.dumps({"run_id": "run-1", "pid": 999999999, "port": 59999}))

    def fake_popen(command, **kwargs):
        return _FakeChildProcess("PORT 51235 stale-capability-token\n")

    monkeypatch.setattr("prime_cli.dashboard_proxy.subprocess.Popen", fake_popen)

    url = start_detached_dashboard_proxy(
        "run-1", base_url=BASE_URL, api_key="test-key", state_dir=tmp_path
    )

    assert url == "http://127.0.0.1:51235/stale-capability-token"


def test_detached_child_end_to_end(tmp_path) -> None:
    """Real child process: ready line, state file, clean SIGTERM shutdown."""
    pytest.importorskip("signal")
    state_dir = tmp_path / "state"
    url = start_detached_dashboard_proxy(
        "run-1",
        # Nothing listens on this local port: upstream requests fail fast,
        # which the proxy must surface as a clean local 502.
        base_url="http://127.0.0.1:1",
        api_key="test-key",
        state_dir=state_dir,
        idle_timeout_seconds=600,
    )

    parsed_url = urllib.parse.urlsplit(url)
    port = parsed_url.port
    assert parsed_url.path.startswith("/")
    token = parsed_url.path.strip("/")
    # The parent picks a fingerprinted state file path and passes it to the child.
    state_paths = list(state_dir.glob("train-dashboard-run-1-*.json"))
    assert len(state_paths) == 1
    state_path = state_paths[0]
    state = json.loads(state_path.read_text())
    assert state["port"] == port
    assert state["pid"] > 0
    assert state["token"] == token
    # The state file carries the capability token: it must be user-only.
    assert state_path.stat().st_mode & 0o777 == 0o600

    response = _get(port, token, "/")
    body = response.read()
    assert response.status == 502
    assert b"unreachable" in body

    os.kill(state["pid"], signal.SIGTERM)
    deadline = time.monotonic() + 10
    while state_path.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not state_path.exists(), "child must clean up its state file on SIGTERM"


# --- Platform-rewritten redirect composition (astra t018/t020) ---------------


def test_proxy_composes_platform_rewritten_redirect_through_loopback(proxy_factory) -> None:
    """Chain: platform-shaped Location -> CLI strip -> loopback follow-up.

    The platform proxy rewrites upstream redirects onto its own
    /api/v1/rft/runs/{id}/dashboard/ prefix so a browser re-enters
    platform authZ. The loopback must strip that prefix and answer the
    follow-up request with a SINGLE prefix, not a double-rewrite.
    """
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        if len(seen) == 1:
            return httpx.Response(
                302,
                headers={"Location": "/api/v1/rft/runs/run-1/dashboard/static/index.html?next=1"},
                content=b"",
            )
        return httpx.Response(200, content=b"<html>index</html>")

    _, port, token = proxy_factory(httpx.Client(transport=httpx.MockTransport(handler)))

    first = _get(port, token, "/")
    assert first.status == 302
    assert first.getheader("Location") == "/static/index.html?next=1"

    # The browser follows the rewritten Location on the loopback root and
    # the proxy must map it back to exactly one platform prefix.
    second = _get(port, token, "/static/index.html?next=1")
    assert second.status == 200
    assert second.read() == b"<html>index</html>"
    assert seen == [
        f"{BASE_URL}/api/v1/rft/runs/run-1/dashboard/",
        f"{BASE_URL}/api/v1/rft/runs/run-1/dashboard/static/index.html?next=1",
    ]


def test_proxy_still_passes_ordinary_relative_redirect_unchanged(proxy_factory) -> None:
    """Reverse composition: an ordinary dashboard-relative Location passes
    through unchanged (no prefix stripping, no double-rewrite)."""
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        if len(seen) == 1:
            return httpx.Response(308, headers={"Location": "/static/app.js"}, content=b"")
        return httpx.Response(200, content=b"js")

    _, port, token = proxy_factory(httpx.Client(transport=httpx.MockTransport(handler)))

    first = _get(port, token, "/")
    assert first.status == 308
    assert first.getheader("Location") == "/static/app.js"

    second = _get(port, token, "/static/app.js")
    assert second.status == 200
    assert second.read() == b"js"
    assert seen == [
        f"{BASE_URL}/api/v1/rft/runs/run-1/dashboard/",
        f"{BASE_URL}/api/v1/rft/runs/run-1/dashboard/static/app.js",
    ]


def test_proxy_rejects_platform_relative_redirect_for_other_run(proxy_factory) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            302,
            headers={"Location": "/api/v1/rft/runs/run-2/dashboard/static/index.html"},
            content=b"",
        )

    _, port, token = proxy_factory(httpx.Client(transport=httpx.MockTransport(handler)))

    response = _get(port, token, "/")

    assert response.status == 502
    assert response.read().decode("utf-8").startswith("Dashboard backend redirect rejected")


# --- DNS-rebinding guard: Host/Origin validation -----------------------------


def _raw_get(
    port: int, host: str, origin: Optional[str] = None, token: Optional[str] = None
) -> tuple[int, bytes]:
    request = f"GET / HTTP/1.1\r\nHost: {host}\r\n"
    if token:
        request += f"Cookie: t={token}\r\n"
    if origin:
        request += f"Origin: {origin}\r\n"
    request += "Connection: close\r\n\r\n"
    with socket.create_connection(("127.0.0.1", port), timeout=5) as sock:
        sock.sendall(request.encode())
        data = b""
        while True:
            chunk = sock.recv(65536)
            if not chunk:
                break
            data += chunk
    head, _, body = data.partition(b"\r\n\r\n")
    status = int(head.split(b" ")[1])
    return status, body


def test_proxy_rejects_forged_host_header(proxy_factory) -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, content=b"secret dashboard")

    _, port, token = proxy_factory(httpx.Client(transport=httpx.MockTransport(handler)))

    # A rebound DNS name (attacker.example.com -> 127.0.0.1) must be
    # rejected BEFORE any upstream work: the private data never leaves.
    status, body = _raw_get(port, f"attacker.example.com:{port}", token=token)
    assert status == 403
    assert seen == []
    assert b"secret" not in body


def test_proxy_rejects_loopback_host_with_wrong_port(proxy_factory) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"secret dashboard")

    _, port, token = proxy_factory(httpx.Client(transport=httpx.MockTransport(handler)))

    status, _ = _raw_get(port, "127.0.0.1:1", token=token)
    assert status == 403


def test_proxy_allows_localhost_and_loopback_hosts(proxy_factory) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"dashboard")

    _, port, token = proxy_factory(httpx.Client(transport=httpx.MockTransport(handler)))

    assert _raw_get(port, f"localhost:{port}", token=token)[0] == 200
    assert _raw_get(port, f"127.0.0.1:{port}", token=token)[0] == 200


def test_proxy_rejects_forged_origin(proxy_factory) -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, content=b"secret dashboard")

    _, port, token = proxy_factory(httpx.Client(transport=httpx.MockTransport(handler)))

    # A hostile page on another origin fetching the loopback port: reject.
    status, _ = _raw_get(port, f"127.0.0.1:{port}", origin="http://evil.example.com", token=token)
    assert status == 403
    assert seen == []


def test_proxy_allows_loopback_origin(proxy_factory) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"dashboard")

    _, port, token = proxy_factory(httpx.Client(transport=httpx.MockTransport(handler)))

    assert (
        _raw_get(port, f"127.0.0.1:{port}", origin=f"http://127.0.0.1:{port}", token=token)[0]
        == 200
    )


# --- Traversal guard ---------------------------------------------------------


def _raw_path_get(port: int, path: str, token: Optional[str] = None) -> int:
    request = f"GET {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n"
    if token:
        request += f"Cookie: t={token}\r\n"
    request += "Connection: close\r\n\r\n"
    with socket.create_connection(("127.0.0.1", port), timeout=5) as sock:
        sock.sendall(request.encode())
        data = b""
        while True:
            chunk = sock.recv(65536)
            if not chunk:
                break
            data += chunk
    return int(data.split(b" ")[1])


def test_proxy_rejects_dot_segment_traversal_before_upstream(proxy_factory) -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, content=b"x")

    _, port, token = proxy_factory(httpx.Client(transport=httpx.MockTransport(handler)))

    # Raw and percent-encoded dot segments must be rejected locally:
    # httpx would normalize them and attach the bearer token to paths
    # outside the run-scoped dashboard route.
    assert _raw_path_get(port, "/static/../../rft/runs/run-2", token=token) == 400
    assert _raw_path_get(port, "/static/%2e%2e/%2e%2e/rft/runs/run-2", token=token) == 400
    assert _raw_path_get(port, "/a/b/../c", token=token) == 400
    # The upstream was never contacted.
    assert seen == []


# --- Pre-compressed body passthrough ------------------------------------------


class _RawByteStream(httpx.SyncByteStream):
    """Wire-format stream: yields the RAW (still compressed) bytes."""

    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def __iter__(self):
        yield self.payload


def test_proxy_passes_precompressed_bodies_verbatim(proxy_factory) -> None:
    import gzip

    payload = gzip.compress(b"<html>precompressed</html>" * 100)

    def handler(request: httpx.Request) -> httpx.Response:
        # httpx.Response(content=...) treats the body as DECODED and would
        # try to decompress it again; mirror the wire with a raw stream.
        return httpx.Response(
            200,
            headers={
                "Content-Type": "text/html",
                "Content-Encoding": "gzip",
                "Content-Length": str(len(payload)),
            },
            stream=_RawByteStream(payload),
        )

    _, port, token = proxy_factory(httpx.Client(transport=httpx.MockTransport(handler)))

    response = _get(port, token, "/")
    body = response.read()

    # The compressed bytes, encoding header and compressed length must all
    # agree; httpx decompresses in iter_bytes, so raw passthrough is used.
    assert response.status == 200
    assert response.getheader("Content-Encoding") == "gzip"
    assert int(response.getheader("Content-Length")) == len(payload)
    assert body == payload
    assert gzip.decompress(body).startswith(b"<html>")


# --- Detached reuse: context fingerprints and credentials --------------------


def test_start_detached_never_reuses_across_contexts(monkeypatch, tmp_path) -> None:
    """Same run, different token: a fresh proxy must start, not be reused."""
    healthy_upstream = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, content=b"<html>ok</html>")
        )
    )
    server, _ = make_dashboard_proxy_server(
        "run-1", base_url=BASE_URL, api_key="token-A", upstream=healthy_upstream
    )
    serve_thread = threading.Thread(target=server.serve_forever, daemon=True)
    serve_thread.start()
    try:
        fp_a = _proxy_fingerprint(BASE_URL, "run-1", "token-A")
        state_a = proxy_state_path("run-1", state_dir=tmp_path, fingerprint=fp_a)
        state_a.parent.mkdir(parents=True, exist_ok=True)
        state_a.write_text(
            json.dumps(
                {
                    "run_id": "run-1",
                    "pid": os.getpid(),
                    "port": server.server_address[1],
                    "token": server.capability_token,
                }
            )
        )

        def fake_popen(command, **kwargs):
            return _FakeChildProcess("PORT 51236 other-context-token\n")

        monkeypatch.setattr("prime_cli.dashboard_proxy.subprocess.Popen", fake_popen)

        # token-B invocation: different fingerprint, no reuse, fresh spawn.
        url_b = start_detached_dashboard_proxy(
            "run-1", base_url=BASE_URL, api_key="token-B", state_dir=tmp_path
        )

        assert url_b == "http://127.0.0.1:51236/other-context-token"
        # The other context's state file is untouched (live proxy left to
        # its own idle exit — it serves a different, valid context).
        assert state_a.exists()
        assert not _proxy_fingerprint(BASE_URL, "run-1", "token-B") == fp_a
    finally:
        server.shutdown()
        server.server_close()


def test_start_detached_respawns_when_live_proxy_credentials_are_stale(
    monkeypatch, tmp_path
) -> None:
    """A live proxy whose upstream rejects its token must not be reused."""
    rejected_upstream = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(401, json={"detail": "token revoked"})
        )
    )
    server, stale_url = make_dashboard_proxy_server(
        "run-1", base_url=BASE_URL, api_key="revoked-token", upstream=rejected_upstream
    )
    serve_thread = threading.Thread(target=server.serve_forever, daemon=True)
    serve_thread.start()
    try:
        fingerprint = _proxy_fingerprint(BASE_URL, "run-1", "revoked-token")
        state_path = proxy_state_path("run-1", state_dir=tmp_path, fingerprint=fingerprint)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(
            json.dumps(
                {
                    "run_id": "run-1",
                    "pid": os.getpid(),
                    "port": server.server_address[1],
                    "token": server.capability_token,
                }
            )
        )

        def fake_popen(command, **kwargs):
            return _FakeChildProcess("PORT 51237 fresh-capability-token\n")

        monkeypatch.setattr("prime_cli.dashboard_proxy.subprocess.Popen", fake_popen)

        url = start_detached_dashboard_proxy(
            "run-1", base_url=BASE_URL, api_key="revoked-token", state_dir=tmp_path
        )

        # The stale proxy answered the credentials probe with a 502
        # (platform rejects the bearer), so a fresh proxy started.
        assert url == "http://127.0.0.1:51237/fresh-capability-token"
        assert url != stale_url
    finally:
        server.shutdown()
        server.server_close()


def test_start_detached_kills_orphan_and_cleans_state_on_failure(monkeypatch, tmp_path) -> None:
    """A half-started child (no PORT line) must be killed, not orphaned."""
    child = _FakeChildProcess(None)  # never emits the ready line
    state_path_passed: list[Path] = []

    def fake_popen(command, **kwargs):
        state_file = Path(command[command.index("--state-file") + 1])
        state_path_passed.append(state_file)
        # A real half-started orphan may already have written its state
        # file before failing to print the ready line.
        state_file.write_text(json.dumps({"run_id": "run-1", "pid": 424242, "port": 1}))
        return child

    monkeypatch.setattr("prime_cli.dashboard_proxy.subprocess.Popen", fake_popen)

    with pytest.raises(RuntimeError, match="failed to start"):
        start_detached_dashboard_proxy(
            "run-1",
            base_url=BASE_URL,
            api_key="test-key",
            state_dir=tmp_path,
            ready_timeout_seconds=0.2,
        )

    # The orphan was terminated (it holds the API token in its env), its
    # pipe was closed only after the kill, and the state file the orphan
    # may have written is gone, so it can never be mistaken for a live
    # reusable proxy.
    assert child.killed is True
    assert child.stdout.closed is True
    assert state_path_passed[0].exists() is False


def test_detached_child_leaves_replaced_state_file_alone(tmp_path) -> None:
    """A newer proxy's state file must survive the old child's exit."""
    pytest.importorskip("signal")
    state_dir = tmp_path / "state"
    start_detached_dashboard_proxy(
        "run-1",
        base_url="http://127.0.0.1:1",
        api_key="test-key",
        state_dir=state_dir,
        idle_timeout_seconds=600,
    )
    state_paths = list(state_dir.glob("train-dashboard-run-1-*.json"))
    assert len(state_paths) == 1
    state_path = state_paths[0]
    state = json.loads(state_path.read_text())

    # Simulate a replacement proxy taking over the state file.
    state_path.write_text(
        json.dumps({"run_id": "run-1", "pid": os.getpid() + 1, "port": state["port"]})
    )
    os.kill(state["pid"], signal.SIGTERM)
    deadline = time.monotonic() + 10
    while state_path.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    # The child exited (SIGTERM accepted) but did NOT unlink the file: the
    # recorded pid no longer names its own process.
    assert (
        not state_path.exists() or json.loads(state_path.read_text()).get("pid") == os.getpid() + 1
    ), "child must not delete a newer proxy's state file"


# --- t024: strict Host/Origin parser hardening (astra raw-socket matrix) ------


def _raw_request(
    port: int, header_lines: list[str], path: str = "/", token: Optional[str] = None
) -> tuple[int, bytes]:
    if token:
        header_lines = [*header_lines, f"Cookie: t={token}"]
    request = (
        f"GET {path} HTTP/1.1\r\n" + "\r\n".join(header_lines) + "\r\nConnection: close\r\n\r\n"
    )
    with socket.create_connection(("127.0.0.1", port), timeout=5) as sock:
        sock.sendall(request.encode())
        data = b""
        while True:
            chunk = sock.recv(65536)
            if not chunk:
                break
            data += chunk
    head, _, body = data.partition(b"\r\n\r\n")
    status = int(head.split(b" ")[1])
    return status, body


@pytest.mark.parametrize(
    "host_line",
    [
        "[::1]attacker.example",
        "[::1]:notaport",
        "[::1]:",
        "[::1]garbage:123",
        "127.0.0.1:123x",
    ],
)
def test_proxy_rejects_malformed_host_authorities(proxy_factory, host_line: str) -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, content=b"secret dashboard")

    _, port, token = proxy_factory(httpx.Client(transport=httpx.MockTransport(handler)))

    status, body = _raw_request(port, [f"Host: {host_line}"], token=token)
    assert status == 403, host_line
    assert seen == []
    assert b"secret" not in body


def test_proxy_accepts_well_formed_ipv6_bracket_host(proxy_factory) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"dashboard")

    _, port, token = proxy_factory(httpx.Client(transport=httpx.MockTransport(handler)))

    assert _raw_request(port, [f"Host: [::1]:{port}"], token=token)[0] == 200
    assert _raw_request(port, ["Host: [::1]"], token=token)[0] == 200


def test_proxy_rejects_duplicate_host_headers(proxy_factory) -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, content=b"secret dashboard")

    _, port, token = proxy_factory(httpx.Client(transport=httpx.MockTransport(handler)))

    # Good Host first, hostile second: duplicated Host is rejected outright.
    status, body = _raw_request(
        port, [f"Host: 127.0.0.1:{port}", "Host: attacker.example:80"], token=token
    )
    assert status == 403
    assert seen == []
    assert b"secret" not in body


@pytest.mark.parametrize(
    "origin_line",
    [
        "http://127.0.0.1:1",  # wrong local port
        "ftp://{placeholder}",  # non-http scheme
        "http://127.0.0.1:notaport",  # malformed port
        "http://[::1",  # malformed bracket: must 403, never raise
        "http://localhost:1",
    ],
)
def test_proxy_rejects_invalid_origins(proxy_factory, origin_line: str) -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, content=b"secret dashboard")

    _, port, token = proxy_factory(httpx.Client(transport=httpx.MockTransport(handler)))

    line = origin_line.replace("{placeholder}", f"127.0.0.1:{port}")
    status, body = _raw_request(port, [f"Host: 127.0.0.1:{port}", f"Origin: {line}"], token=token)
    assert status == 403, line
    assert seen == []
    assert b"secret" not in body


def test_proxy_still_accepts_same_origin_with_bound_port(proxy_factory) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"dashboard")

    _, port, token = proxy_factory(httpx.Client(transport=httpx.MockTransport(handler)))

    status, _ = _raw_request(
        port, [f"Host: 127.0.0.1:{port}", f"Origin: http://127.0.0.1:{port}"], token=token
    )
    assert status == 200


# --- t024 fold-ins: failed-start unlink ownership + platform detach flags ----


def test_failed_start_leaves_replaced_state_file_alone(monkeypatch, tmp_path) -> None:
    """A failed start must not delete a replacement proxy's state file."""
    child = _FakeChildProcess(None)  # never emits the ready line
    child.pid = 424242

    def fake_popen(command, **kwargs):
        state_file = Path(command[command.index("--state-file") + 1])
        state_file.parent.mkdir(parents=True, exist_ok=True)
        # A replacement proxy already claimed the state file.
        state_file.write_text(json.dumps({"run_id": "run-1", "pid": 999999, "port": 51299}))
        return child

    monkeypatch.setattr("prime_cli.dashboard_proxy.subprocess.Popen", fake_popen)

    with pytest.raises(RuntimeError, match="failed to start"):
        start_detached_dashboard_proxy(
            "run-1",
            base_url=BASE_URL,
            api_key="test-key",
            state_dir=tmp_path,
            ready_timeout_seconds=0.2,
        )

    # The child was still killed, but the replacement's state file survives.
    assert child.killed is True
    state_paths = list(tmp_path.glob("train-dashboard-run-1-*.json"))
    assert len(state_paths) == 1
    assert json.loads(state_paths[0].read_text())["pid"] == 999999


def test_failed_start_still_removes_orphans_own_state(monkeypatch, tmp_path) -> None:
    """When the state file still names the dead child, it IS cleaned up."""
    child = _FakeChildProcess(None)
    child.pid = 424242

    def fake_popen(command, **kwargs):
        state_file = Path(command[command.index("--state-file") + 1])
        state_file.parent.mkdir(parents=True, exist_ok=True)
        # The orphan wrote its own pid before failing to report the port.
        state_file.write_text(json.dumps({"run_id": "run-1", "pid": child.pid, "port": 1}))
        return child

    monkeypatch.setattr("prime_cli.dashboard_proxy.subprocess.Popen", fake_popen)

    with pytest.raises(RuntimeError, match="failed to start"):
        start_detached_dashboard_proxy(
            "run-1",
            base_url=BASE_URL,
            api_key="test-key",
            state_dir=tmp_path,
            ready_timeout_seconds=0.2,
        )

    assert child.killed is True
    assert list(tmp_path.glob("train-dashboard-run-1-*.json")) == []


def test_detached_popen_uses_windows_flags_on_windows(monkeypatch, tmp_path) -> None:
    seen: dict[str, Any] = {}

    def fake_popen(command, **kwargs):
        seen.update(kwargs)
        return _FakeChildProcess("PORT 51234 fake-capability-token\n")

    monkeypatch.setattr("prime_cli.dashboard_proxy.subprocess.Popen", fake_popen)
    monkeypatch.setattr("prime_cli.dashboard_proxy.sys.platform", "win32")

    url = start_detached_dashboard_proxy(
        "run-1", base_url=BASE_URL, api_key="test-key", state_dir=tmp_path
    )

    assert url == "http://127.0.0.1:51234/fake-capability-token"
    # POSIX-only kwarg must not be sent on Windows; detach via creation flags.
    assert "start_new_session" not in seen
    expected_flags = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(
        subprocess, "CREATE_NEW_PROCESS_GROUP", 0
    )
    assert seen.get("creationflags") == expected_flags


def test_detached_popen_uses_start_new_session_on_posix(monkeypatch, tmp_path) -> None:
    seen: dict[str, Any] = {}

    def fake_popen(command, **kwargs):
        seen.update(kwargs)
        return _FakeChildProcess("PORT 51234 fake-capability-token\n")

    monkeypatch.setattr("prime_cli.dashboard_proxy.subprocess.Popen", fake_popen)
    monkeypatch.setattr("prime_cli.dashboard_proxy.sys.platform", "darwin")

    url = start_detached_dashboard_proxy(
        "run-1", base_url=BASE_URL, api_key="test-key", state_dir=tmp_path
    )

    assert url == "http://127.0.0.1:51234/fake-capability-token"
    assert seen.get("start_new_session") is True
    assert "creationflags" not in seen
