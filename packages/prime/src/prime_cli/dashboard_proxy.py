"""Loopback proxy that serves a Hosted Training run dashboard locally.

The platform serves a run's dashboard behind a bearer-authenticated proxy
route (``GET /api/v1/rft/runs/{id}/dashboard/{path}``). A browser cannot
send that bearer token, and the underlying dashboard URL is internal to
the tailnet. ``prime train dashboard`` therefore serves the dashboard
from a loopback HTTP server on 127.0.0.1 that forwards every
root-relative path to the run-scoped platform route using the CLI's
stored API token (sent as an ``Authorization`` header, never in a URL).
Root-relative assets, API calls and SSE streams (``text/event-stream``)
work unmodified because the loopback server serves the dashboard at its
own root.

The proxy normally runs in a detached child process
(:func:`start_detached_dashboard_proxy`) so the CLI can print the
loopback URL and exit — command substitution like
``open $(prime train dashboard <run_id> --no-browser)`` completes
immediately. The child writes a state file (pid + port) under the user
cache dir and self-exits after an idle timeout; repeat invocations reuse
a still-running proxy instead of spawning another one.
"""

from __future__ import annotations

import argparse
import http.server
import json
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.parse
from pathlib import Path
from typing import Any, Optional

import httpx

from .core.client import _default_user_agent

_UPSTREAM_REQUEST_TIMEOUT = httpx.Timeout(30.0, connect=10.0, read=None)
"""No read timeout: SSE event streams must not be cut off between events."""

DEFAULT_IDLE_TIMEOUT_SECONDS = 30 * 60
"""Detached proxies self-exit after this long without any live request."""

_CHILD_READY_TIMEOUT_SECONDS = 15.0
"""How long the parent waits for the detached proxy to report its port."""

_CHILD_API_KEY_ENV = "PRIME_DASHBOARD_PROXY_API_KEY"
"""Environment variable used to hand the API token to the detached child.

The token never appears in command-line arguments (visible to all users
via the process table) or in URLs; child environments are private to the
owning user."""

_PROXY_STATE_DIR = Path.home() / ".prime" / "dashboard_proxies"

_REDIRECT_STATUSES = {301, 302, 303, 307, 308}


def dashboard_upstream_path(run_id: str, path: str) -> str:
    """Map a loopback request path onto the run-scoped platform proxy route."""
    return f"/api/v1/rft/runs/{run_id}/dashboard/{path.lstrip('/')}"


def _origin_of(url: urllib.parse.SplitResult) -> Optional[tuple[str, str, int]]:
    if not url.scheme or not url.hostname:
        return None
    scheme = url.scheme.lower()
    port = url.port
    if port is None:
        port = 443 if scheme == "https" else 80
    return (scheme, url.hostname.lower(), port)


def map_dashboard_redirect(location: str, base_url: str, run_id: str) -> Optional[str]:
    """Map an upstream ``Location`` header onto a loopback-root-relative path.

    Only redirects that stay inside the run-scoped dashboard route can be
    served safely: same-origin absolute URLs below
    ``/api/v1/rft/runs/{run_id}/dashboard/`` are rewritten to loopback
    paths, and already-relative locations resolve identically against the
    loopback root and pass through unchanged. Anything else (off-origin
    URLs, or same-origin paths outside the dashboard scope) returns
    ``None`` so the caller can reject the redirect without leaking the
    upstream origin to the browser.
    """
    parsed = urllib.parse.urlsplit(location)
    if not parsed.scheme and not parsed.netloc:
        return location

    base = urllib.parse.urlsplit(base_url)
    if _origin_of(parsed) != _origin_of(base):
        return None

    prefix = dashboard_upstream_path(run_id, "")
    path = parsed.path or "/"
    if path == prefix.rstrip("/"):
        path = "/"
    elif path.startswith(prefix):
        path = "/" + path[len(prefix) :]
    else:
        return None
    if parsed.query:
        path += "?" + parsed.query
    return path


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
        server = self.server
        assert isinstance(server, DashboardProxyServer)
        server.request_started()
        try:
            self._proxy_get()
        finally:
            server.request_finished()

    def _proxy_get(self) -> None:
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
        # Validate redirects BEFORE sending any status line: the stdlib
        # buffers headers, and a rejected redirect must not flush a
        # half-written response.
        redirect_location: Optional[str] = None
        if response.status_code in _REDIRECT_STATUSES:
            location = response.headers.get("location")
            if location is not None:
                redirect_location = map_dashboard_redirect(location, self.base_url, self.run_id)
                if redirect_location is None:
                    self._send_plain_error(502, "Dashboard backend redirect rejected.")
                    return

        self.send_response(response.status_code)
        content_type = response.headers.get("content-type")
        if content_type:
            self.send_header("Content-Type", content_type)
        cache_control = response.headers.get("cache-control")
        if cache_control:
            self.send_header("Cache-Control", cache_control)
        if redirect_location is not None:
            self.send_header("Location", redirect_location)
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
    """Loopback server that tracks activity so it can idle-exit."""

    daemon_threads = True
    run_id: str = ""
    upstream_client: Optional[httpx.Client] = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._state_lock = threading.Lock()
        self._active_requests = 0
        self._last_activity = time.monotonic()

    def request_started(self) -> None:
        with self._state_lock:
            self._active_requests += 1
            self._last_activity = time.monotonic()

    def request_finished(self) -> None:
        with self._state_lock:
            self._active_requests -= 1
            self._last_activity = time.monotonic()

    def is_idle_for(self, seconds: float) -> bool:
        with self._state_lock:
            return (
                self._active_requests == 0 and (time.monotonic() - self._last_activity) >= seconds
            )


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


def _idle_watchdog(
    server: DashboardProxyServer,
    idle_timeout_seconds: float,
    poll_interval: float = 15.0,
) -> None:
    """Shut the server down once it has been idle for ``idle_timeout_seconds``.

    Open requests (e.g. a live SSE stream) count as activity, so a browser
    tab that keeps the dashboard connected keeps the proxy alive.
    """
    while not server.is_idle_for(idle_timeout_seconds):
        time.sleep(poll_interval)
    server.shutdown()


def proxy_state_path(run_id: str, state_dir: Optional[Path] = None) -> Path:
    """Per-run state file holding the detached proxy's pid and port."""
    directory = state_dir or _PROXY_STATE_DIR
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", run_id)
    return directory / f"train-dashboard-{safe}.json"


def _read_proxy_state(state_path: Path) -> Optional[dict[str, Any]]:
    try:
        data = json.loads(state_path.read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _pid_is_alive(pid: int) -> bool:
    if sys.platform == "win32":
        # os.kill(pid, 0) would *terminate* the target process on Windows;
        # query it through the Win32 API instead.
        try:
            import ctypes

            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            handle = ctypes.windll.kernel32.OpenProcess(  # type: ignore[attr-defined]
                PROCESS_QUERY_LIMITED_INFORMATION, False, pid
            )
            if not handle:
                return False
            ctypes.windll.kernel32.CloseHandle(handle)  # type: ignore[attr-defined]
            return True
        except Exception:  # pragma: no cover - defensive on exotic setups
            return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _live_proxy_url(state: Optional[dict[str, Any]]) -> Optional[str]:
    """Return the loopback URL of a still-running proxy, if any."""
    if not state:
        return None
    pid = state.get("pid")
    port = state.get("port")
    if not isinstance(pid, int) or not isinstance(port, int):
        return None
    if not _pid_is_alive(pid):
        return None
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.25):
            pass
    except OSError:
        return None
    return f"http://127.0.0.1:{port}/"


def _read_child_ready_line(process: "subprocess.Popen[bytes]", timeout: float) -> Optional[str]:
    """Read the child's ``PORT <n>`` line with a timeout (pipes cannot select)."""
    line: Optional[str] = None

    def reader() -> None:
        nonlocal line
        try:
            if process.stdout is not None:
                line = process.stdout.readline()
                if isinstance(line, bytes):
                    line = line.decode("utf-8", "replace")
            else:
                line = None
        except OSError:
            line = None

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    thread.join(timeout)
    return line


def start_detached_dashboard_proxy(
    run_id: str,
    *,
    base_url: str,
    api_key: str,
    idle_timeout_seconds: float = DEFAULT_IDLE_TIMEOUT_SECONDS,
    state_dir: Optional[Path] = None,
    ready_timeout_seconds: float = _CHILD_READY_TIMEOUT_SECONDS,
) -> str:
    """Start (or reuse) a detached loopback proxy and return its URL.

    The proxy runs in its own session (``start_new_session=True``) so it
    survives the CLI exiting; it writes a pid/port state file under the
    user cache dir and exits itself once idle. If a healthy proxy for this
    run is already running, its URL is returned without spawning another.
    """
    state_path = proxy_state_path(run_id, state_dir)
    existing = _live_proxy_url(_read_proxy_state(state_path))
    if existing is not None:
        return existing

    state_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "prime_cli.dashboard_proxy",
        "--run-id",
        run_id,
        "--base-url",
        base_url,
        "--state-file",
        str(state_path),
        "--idle-timeout-seconds",
        str(idle_timeout_seconds),
    ]
    process = subprocess.Popen(  # noqa: S603 - fixed module command
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        env={**os.environ, _CHILD_API_KEY_ENV: api_key},
    )
    try:
        ready_line = _read_child_ready_line(process, ready_timeout_seconds)
        port_token = (ready_line or "").split()
        if len(port_token) != 2 or port_token[0] != "PORT" or not port_token[1].isdigit():
            raise RuntimeError("The dashboard proxy failed to start.")
        port = int(port_token[1])
    finally:
        if process.stdout is not None:
            process.stdout.close()
    return f"http://127.0.0.1:{port}/"


def _main(argv: Optional[list[str]] = None) -> int:
    """Entry point of the detached child (``python -m prime_cli.dashboard_proxy``)."""
    parser = argparse.ArgumentParser(
        prog="python -m prime_cli.dashboard_proxy",
        description="Serve one Hosted Training run dashboard on a loopback proxy.",
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--state-file", required=True)
    parser.add_argument("--idle-timeout-seconds", type=float, default=DEFAULT_IDLE_TIMEOUT_SECONDS)
    args = parser.parse_args(argv)

    api_key = os.environ.get(_CHILD_API_KEY_ENV)
    if not api_key:
        # Literals only: CodeQL flags any clear-text output that mentions
        # credential-named variables, and the child never logs the token.
        print("the API token environment variable is not set", file=sys.stderr)
        return 2

    server, _url = make_dashboard_proxy_server(
        args.run_id,
        base_url=args.base_url,
        api_key=api_key,
    )
    port = server.server_address[1]
    state_path = Path(args.state_file)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps({"run_id": args.run_id, "pid": os.getpid(), "port": port}))
    # Report the port, then detach stdout: the parent may exit (closing the
    # pipe) at any time and this process must never block writing to it.
    print(f"PORT {port}", flush=True)
    os.dup2(os.open(os.devnull, os.O_WRONLY), 1)

    if hasattr(signal, "SIGTERM"):

        def _stop_on_sigterm(signum: int, frame: Any) -> None:  # noqa: ARG001
            # server.shutdown() blocks until serve_forever() exits; calling it
            # directly in the signal handler would deadlock the main thread,
            # which is exactly where serve_forever() is running.
            threading.Thread(target=server.shutdown, daemon=True).start()

        signal.signal(signal.SIGTERM, _stop_on_sigterm)

    threading.Thread(
        target=_idle_watchdog,
        args=(server, args.idle_timeout_seconds),
        daemon=True,
    ).start()
    try:
        server.serve_forever()
    finally:
        state_path.unlink(missing_ok=True)
        server.server_close()
        if server.upstream_client is not None:
            server.upstream_client.close()
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess in tests
    raise SystemExit(_main())
