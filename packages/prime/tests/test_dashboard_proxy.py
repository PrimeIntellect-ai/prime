"""Loopback dashboard proxy: request mapping, SSE passthrough, error hygiene."""

import http.client
import threading
import time
from typing import Any

import httpx
import pytest
from prime_cli.dashboard_proxy import make_dashboard_proxy_server

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
