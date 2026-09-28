"""Loopback proxy that serves a Hosted Training run dashboard locally.

The platform serves a run's dashboard behind a bearer-authenticated proxy
route (``GET /api/v1/rft/runs/{id}/dashboard/{path}``). A browser cannot
send that bearer token, and the underlying dashboard URL is internal to
the tailnet. ``prime train dashboard`` therefore starts a loopback HTTP
server on 127.0.0.1 that forwards every root-relative path to the
run-scoped platform route using the CLI's stored API token (sent as an
``Authorization`` header, never in a URL). Root-relative assets, API
calls and SSE streams (``text/event-stream``) work unmodified because
the loopback server serves the dashboard at its own root.
"""

from __future__ import annotations

import http.server
from typing import Any, Optional

import httpx

from .core.client import _default_user_agent

_UPSTREAM_REQUEST_TIMEOUT = httpx.Timeout(30.0, connect=10.0, read=None)
"""No read timeout: SSE event streams must not be cut off between events."""


def dashboard_upstream_path(run_id: str, path: str) -> str:
    """Map a loopback request path onto the run-scoped platform proxy route."""
    return f"/api/v1/rft/runs/{run_id}/dashboard/{path.lstrip('/')}"


class DashboardProxyHandler(http.server.BaseHTTPRequestHandler):
    """Forward every loopback GET to the platform dashboard proxy route.

    Class attributes ``run_id``, ``base_url``, ``api_key`` and ``upstream``
    are bound by :func:`make_dashboard_proxy_server`.
    """

    protocol_version = "HTTP/1.1"
    run_id: str = ""
    base_url: str = ""
    api_key: str = ""
    upstream: Optional[httpx.Client] = None

    def do_GET(self) -> None:
        if self.upstream is None:  # pragma: no cover - guarded by factory
            self._send_plain_error(500, "Dashboard proxy is not configured.")
            return
        # Absolute URL built like APIClient does (base_url + /api/v1 route),
        # so the proxy does not depend on upstream client defaults.
        url = f"{self.base_url.rstrip('/')}{dashboard_upstream_path(self.run_id, self.path)}"
        try:
            with self.upstream.stream(
                "GET",
                url,
                # The API token travels in the Authorization header only —
                # never in the URL — so it stays out of access logs.
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Accept-Encoding": "identity",
                },
            ) as response:
                if response.status_code >= 400:
                    # Relay a clean local error: upstream bodies may name
                    # internal hosts that must not reach the user's browser.
                    self._send_plain_error(
                        502,
                        f"Dashboard backend returned HTTP {response.status_code}.",
                    )
                    return
                try:
                    self._relay(response)
                except httpx.HTTPError:
                    # Headers are already sent; a mid-stream failure can
                    # only be signalled by dropping the connection.
                    self.close_connection = True
        except httpx.HTTPError:
            self._send_plain_error(502, "Dashboard backend is unreachable.")

    def _relay(self, response: httpx.Response) -> None:
        self.send_response(response.status_code)
        content_type = response.headers.get("content-type")
        if content_type:
            self.send_header("Content-Type", content_type)
        cache_control = response.headers.get("cache-control")
        if cache_control:
            self.send_header("Cache-Control", cache_control)
        length = response.headers.get("content-length")
        if length:
            self.send_header("Content-Length", length)
        else:
            # Unknown length (SSE streams, chunked responses): close the
            # connection to delimit the body instead of buffering it.
            self.close_connection = True
            self.send_header("Connection", "close")
        self.end_headers()
        try:
            for chunk in response.iter_bytes():
                self.wfile.write(chunk)
                self.wfile.flush()  # SSE events must reach the browser immediately
        except (BrokenPipeError, ConnectionResetError):
            # The browser tab was closed mid-stream; drop the connection.
            pass

    def _send_plain_error(self, status: int, message: str) -> None:
        body = (message + "\n").encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        # Per-request proxy logs would spam the terminal while serving.
        return


class DashboardProxyServer(http.server.ThreadingHTTPServer):
    """Loopback server bound to its run id and upstream client."""

    daemon_threads = True
    run_id: str = ""
    upstream_client: Optional[httpx.Client] = None


def make_dashboard_proxy_server(
    run_id: str,
    *,
    base_url: str,
    api_key: str,
    host: str = "127.0.0.1",
    upstream: Optional[httpx.Client] = None,
    user_agent: Optional[str] = None,
) -> tuple[DashboardProxyServer, str]:
    """Start a loopback dashboard proxy on an ephemeral port.

    Returns ``(server, url)``. The caller runs ``server.serve_forever()``
    and, on shutdown, closes the server and ``server.upstream_client``.
    ``upstream`` may be injected for tests.
    """
    upstream_client = upstream or httpx.Client(
        headers={"User-Agent": user_agent or _default_user_agent()},
        timeout=_UPSTREAM_REQUEST_TIMEOUT,
    )
    handler_cls = type(
        "BoundDashboardProxyHandler",
        (DashboardProxyHandler,),
        {
            "run_id": run_id,
            "base_url": base_url,
            "api_key": api_key,
            "upstream": upstream_client,
        },
    )
    server = DashboardProxyServer((host, 0), handler_cls)
    server.run_id = run_id
    server.upstream_client = upstream_client
    url = f"http://{host}:{server.server_address[1]}/"
    return server, url
