"""Loopback dashboard proxy: request mapping, SSE passthrough, error hygiene."""

import http.client
import json
import os
import signal
import sys
import threading
import time
from typing import Any, Optional

import httpx
import pytest
from prime_cli.dashboard_proxy import (
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
        self.pid = 424242

    def kill(self) -> None:
        self.killed = True


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
    # connections and the recorded pid is this (running) process.
    server, url = make_dashboard_proxy_server("run-1", base_url=BASE_URL, api_key="test-key")
    serve_thread = threading.Thread(target=server.serve_forever, daemon=True)
    serve_thread.start()
    try:
        state_path = proxy_state_path("run-1", state_dir=tmp_path)
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
    state_path = proxy_state_path("run-1", state_dir=tmp_path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    # A dead pid and a closed port: the starter must respawn.
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
    state_path = proxy_state_path("run-1", state_dir=state_dir)
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
