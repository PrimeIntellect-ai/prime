"""Loopback dashboard proxy: request mapping, SSE passthrough, error hygiene."""

import http.client
import json
import os
import signal
import socket
import sys
import threading
import time
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

    def start(upstream: httpx.Client, run_id: str = "run-1") -> tuple[str, int]:
        server, url = make_dashboard_proxy_server(
            run_id,
            base_url=BASE_URL,
            api_key="test-key",
            upstream=upstream,
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        servers.append(server)
        return url, server.server_address[1]

    yield start

    for server in servers:
        server.shutdown()
        server.server_close()


def _get(port: int, path: str) -> http.client.HTTPResponse:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    conn.request("GET", path)
    return conn.getresponse()


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

    _, port = proxy_factory(httpx.Client(transport=httpx.MockTransport(handler)))

    response = _get(port, "/")

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

    _, port = proxy_factory(httpx.Client(transport=httpx.MockTransport(handler)))

    response = _get(port, "/static/app.js?v=2")

    assert response.status == 200
    assert response.getheader("Content-Type") == "text/javascript"
    assert response.read() == b"console.log(1)"
    assert seen["url"] == f"{BASE_URL}/api/v1/rft/runs/run-1/dashboard/static/app.js?v=2"


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

    _, port = proxy_factory(httpx.Client(transport=httpx.MockTransport(handler)))

    response = _get(port, "/api/view/events")

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


def test_proxy_surfaces_upstream_http_errors_as_clean_local_errors(proxy_factory) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        # Raw upstream failures must not leak internal hosts to the browser.
        return httpx.Response(
            503,
            headers={"Content-Type": "application/json"},
            content=b'{"detail": "upstream http://rl-dashboard.tail-9.ts.net:7788 down"}',
        )

    _, port = proxy_factory(httpx.Client(transport=httpx.MockTransport(handler)))

    response = _get(port, "/")

    body = response.read()
    assert response.status == 502
    assert b"Dashboard backend returned HTTP 503." in body
    assert b"tail-" not in body  # no internal tailnet URL leaks


def test_proxy_surfaces_transport_failures_as_clean_local_errors(proxy_factory) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused to 10.0.0.1:7788")

    _, port = proxy_factory(httpx.Client(transport=httpx.MockTransport(handler)))

    response = _get(port, "/")

    body = response.read()
    assert response.status == 502
    assert b"Dashboard backend is unreachable." in body
    assert b"10.0.0.1" not in body  # no internal address leaks


# --- Redirect handling ------------------------------------------------------


def test_proxy_rewrites_same_origin_redirect_to_loopback_root(proxy_factory) -> None:
    redirect_target = f"{BASE_URL}/api/v1/rft/runs/run-1/dashboard/static/index.html?v=2"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"Location": redirect_target}, content=b"")

    _, port = proxy_factory(httpx.Client(transport=httpx.MockTransport(handler)))

    response = _get(port, "/")

    # Absolute in-prefix Location must become a root-relative loopback path
    # (scheme, host, run prefix and query preserved on the loopback root).
    assert response.status == 302
    assert response.getheader("Location") == "/static/index.html?v=2"


def test_proxy_forwards_relative_redirect_unchanged(proxy_factory) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(308, headers={"Location": "/static/app.js"}, content=b"")

    _, port = proxy_factory(httpx.Client(transport=httpx.MockTransport(handler)))

    response = _get(port, "/")

    assert response.status == 308
    assert response.getheader("Location") == "/static/app.js"


def test_proxy_rejects_cross_origin_redirects(proxy_factory) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            302,
            headers={"Location": "https://evil.example.com/dashboard/index.html"},
            content=b"",
        )

    _, port = proxy_factory(httpx.Client(transport=httpx.MockTransport(handler)))

    response = _get(port, "/")

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

    _, port = proxy_factory(httpx.Client(transport=httpx.MockTransport(handler)))

    response = _get(port, "/")

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
        return _FakeChildProcess("PORT 51234\n")

    monkeypatch.setattr("prime_cli.dashboard_proxy.subprocess.Popen", fake_popen)

    url = start_detached_dashboard_proxy(
        "run-1",
        base_url=BASE_URL,
        api_key="test-key",
        state_dir=tmp_path,
    )

    assert url == "http://127.0.0.1:51234/"
    assert len(spawn_calls) == 1
    call = spawn_calls[0]
    assert call["command"][:3] == [sys.executable, "-m", "prime_cli.dashboard_proxy"]
    # The API token must travel via the environment, never the command line
    # (process tables are world-readable).
    assert "test-key" not in " ".join(call["command"])
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
            json.dumps({"run_id": "run-1", "pid": os.getpid(), "port": server.server_address[1]})
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
        return _FakeChildProcess("PORT 51235\n")

    monkeypatch.setattr("prime_cli.dashboard_proxy.subprocess.Popen", fake_popen)

    url = start_detached_dashboard_proxy(
        "run-1", base_url=BASE_URL, api_key="test-key", state_dir=tmp_path
    )

    assert url == "http://127.0.0.1:51235/"


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

    port = int(url.rstrip("/").rsplit(":", 1)[1])
    # The parent picks a fingerprinted state file path and passes it to the child.
    state_paths = list(state_dir.glob("train-dashboard-run-1-*.json"))
    assert len(state_paths) == 1
    state_path = state_paths[0]
    state = json.loads(state_path.read_text())
    assert state["port"] == port
    assert state["pid"] > 0

    response = _get(port, "/")
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

    _, port = proxy_factory(httpx.Client(transport=httpx.MockTransport(handler)))

    first = _get(port, "/")
    assert first.status == 302
    assert first.getheader("Location") == "/static/index.html?next=1"

    # The browser follows the rewritten Location on the loopback root and
    # the proxy must map it back to exactly one platform prefix.
    second = _get(port, "/static/index.html?next=1")
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

    _, port = proxy_factory(httpx.Client(transport=httpx.MockTransport(handler)))

    first = _get(port, "/")
    assert first.status == 308
    assert first.getheader("Location") == "/static/app.js"

    second = _get(port, "/static/app.js")
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

    _, port = proxy_factory(httpx.Client(transport=httpx.MockTransport(handler)))

    response = _get(port, "/")

    assert response.status == 502
    assert response.read().decode("utf-8").startswith("Dashboard backend redirect rejected")


# --- DNS-rebinding guard: Host/Origin validation -----------------------------


def _raw_get(port: int, host: str, origin: Optional[str] = None) -> tuple[int, bytes]:
    request = f"GET / HTTP/1.1\r\nHost: {host}\r\n"
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

    _, port = proxy_factory(httpx.Client(transport=httpx.MockTransport(handler)))

    # A rebound DNS name (attacker.example.com -> 127.0.0.1) must be
    # rejected BEFORE any upstream work: the private data never leaves.
    status, body = _raw_get(port, f"attacker.example.com:{port}")
    assert status == 403
    assert seen == []
    assert b"secret" not in body


def test_proxy_rejects_loopback_host_with_wrong_port(proxy_factory) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"secret dashboard")

    _, port = proxy_factory(httpx.Client(transport=httpx.MockTransport(handler)))

    status, _ = _raw_get(port, "127.0.0.1:1")
    assert status == 403


def test_proxy_allows_localhost_and_loopback_hosts(proxy_factory) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"dashboard")

    _, port = proxy_factory(httpx.Client(transport=httpx.MockTransport(handler)))

    assert _raw_get(port, f"localhost:{port}")[0] == 200
    assert _raw_get(port, f"127.0.0.1:{port}")[0] == 200


def test_proxy_rejects_forged_origin(proxy_factory) -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, content=b"secret dashboard")

    _, port = proxy_factory(httpx.Client(transport=httpx.MockTransport(handler)))

    # A hostile page on another origin fetching the loopback port: reject.
    status, _ = _raw_get(port, f"127.0.0.1:{port}", origin="http://evil.example.com")
    assert status == 403
    assert seen == []


def test_proxy_allows_loopback_origin(proxy_factory) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"dashboard")

    _, port = proxy_factory(httpx.Client(transport=httpx.MockTransport(handler)))

    assert _raw_get(port, f"127.0.0.1:{port}", origin=f"http://127.0.0.1:{port}")[0] == 200


# --- Traversal guard ---------------------------------------------------------


def _raw_path_get(port: int, path: str) -> int:
    request = f"GET {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n\r\n"
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

    _, port = proxy_factory(httpx.Client(transport=httpx.MockTransport(handler)))

    # Raw and percent-encoded dot segments must be rejected locally:
    # httpx would normalize them and attach the bearer token to paths
    # outside the run-scoped dashboard route.
    assert _raw_path_get(port, "/static/../../rft/runs/run-2") == 400
    assert _raw_path_get(port, "/static/%2e%2e/%2e%2e/rft/runs/run-2") == 400
    assert _raw_path_get(port, "/a/b/../c") == 400
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

    _, port = proxy_factory(httpx.Client(transport=httpx.MockTransport(handler)))

    response = _get(port, "/")
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
            json.dumps({"run_id": "run-1", "pid": os.getpid(), "port": server.server_address[1]})
        )

        def fake_popen(command, **kwargs):
            return _FakeChildProcess("PORT 51236\n")

        monkeypatch.setattr("prime_cli.dashboard_proxy.subprocess.Popen", fake_popen)

        # token-B invocation: different fingerprint, no reuse, fresh spawn.
        url_b = start_detached_dashboard_proxy(
            "run-1", base_url=BASE_URL, api_key="token-B", state_dir=tmp_path
        )

        assert url_b == "http://127.0.0.1:51236/"
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
            json.dumps({"run_id": "run-1", "pid": os.getpid(), "port": server.server_address[1]})
        )

        def fake_popen(command, **kwargs):
            return _FakeChildProcess("PORT 51237\n")

        monkeypatch.setattr("prime_cli.dashboard_proxy.subprocess.Popen", fake_popen)

        url = start_detached_dashboard_proxy(
            "run-1", base_url=BASE_URL, api_key="revoked-token", state_dir=tmp_path
        )

        # The stale proxy answered the credentials probe with a 502
        # (platform rejects the bearer), so a fresh proxy started.
        assert url == "http://127.0.0.1:51237/"
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
