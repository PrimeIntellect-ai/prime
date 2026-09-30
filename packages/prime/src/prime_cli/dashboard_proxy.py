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

Binding to 127.0.0.1 alone does not restrict access to the launching OS
account: another local user can port-scan the proxy and send valid
loopback Host headers. Every spawn therefore mints an unguessable
capability token (``secrets.token_urlsafe``). The printed loopback URL
carries it ONCE as a ``?t=`` query parameter — the one-time browser
handoff — and the entry response sets it as a loopback cookie
(``Path=/; HttpOnly; SameSite=Strict``). EVERY request must present that
cookie (constant-time validation BEFORE the Host/Origin checks), which
keeps root-relative assets, API calls and SSE streams working exactly
as before: the browser attaches the cookie to every path on the loopback
host. A DNS-rebound origin never receives the 127.0.0.1-host-keyed
cookie, and SameSite=Strict blocks cross-site sends; the Host/Origin
checks stay on as the second layer. The token is never logged or put in
command lines — it reaches the detached child only via the private
ready pipe and a 0600 state file.

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
import hashlib
import hmac
import http.cookies
import http.server
import json
import os
import queue
import re
import secrets
import select
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

_STREAM_LIVENESS_POLL_SECONDS = 5.0
"""How long a quiet streamed body waits before re-checking the browser.

Upstream reads have no timeout (a quiet SSE stream must not be cut
off), so the handler polls the client socket for a half-close instead
of waiting for the next write to fail: without this, a browser tab
closed mid-stream would leave the request "in flight" forever and the
detached proxy (holding the API credential) would never idle-exit."""

_CHILD_READY_TIMEOUT_SECONDS = 15.0
"""How long the parent waits for the detached proxy to report its port."""

_CHILD_API_KEY_ENV = "PRIME_DASHBOARD_PROXY_API_KEY"
"""Environment variable used to hand the API token to the detached child.

The token never appears in command-line arguments (visible to all users
via the process table) or in URLs; child environments are private to the
owning user."""

_PROXY_STATE_DIR = Path.home() / ".prime" / "dashboard_proxies"

_REDIRECT_STATUSES = {301, 302, 303, 307, 308}

_LOOPBACK_HOSTNAMES = {"127.0.0.1", "localhost", "::1"}
"""Host header names the loopback proxy may serve (DNS-rebinding guard)."""

_CAPABILITY_TOKEN_BYTES = 32
"""Entropy (bytes) of the per-spawn loopback capability token.

Binding to 127.0.0.1 does not restrict access to the launching OS
account — another local user can port-scan the loopback port and send
valid loopback Host headers with no Origin, reading the dashboard
through the victim's bearer token. The unguessable capability token
closes that hole: the printed URL hands it to the browser once
(``?t=<token>``), the entry response exchanges it for a loopback
cookie, and every request without that cookie is rejected (403) before
any Host/Origin or upstream work."""

_CAPABILITY_COOKIE_NAME = "t"
_CAPABILITY_QUERY_PARAM = "t"


def _origin_is_loopback(origin: str, bound_port: int) -> bool:
    """Strictly validate an ``Origin`` header against the loopback proxy.

    Accepted only when the scheme is http/https, the host is a loopback
    name, and the port matches the bound port (browser origins always
    carry an explicit port). Any malformed value — including brackets or
    ports that make the parser raise — is a clean rejection, never an
    exception.
    """
    try:
        parsed = urllib.parse.urlsplit(origin.strip())
        port = parsed.port
    except ValueError:
        return False
    if (parsed.scheme or "").lower() not in ("http", "https"):
        return False
    if (parsed.hostname or "").lower() not in _LOOPBACK_HOSTNAMES:
        return False
    return port is not None and port == bound_port


def _parse_host_header(value: str) -> tuple[Optional[str], Optional[int], bool]:
    """Strictly parse a ``Host`` header into (hostname, port, well_formed).

    Bracketed IPv6 authorities must end exactly at the closing bracket,
    optionally followed by ``:<int port>``: any trailing garbage
    (``[::1]attacker.example``, ``[::1]:notaport``, ``[::1]:``) is
    malformed and must be rejected, not tolerated. Non-bracket forms
    allow at most one ``:<int port>`` suffix.
    """
    host = value.strip()
    if not host:
        return None, None, False
    if host.startswith("["):
        end = host.find("]")
        if end == -1:
            return None, None, False
        hostname = host[1:end]
        rest = host[end + 1 :]
        if not rest:
            return hostname.lower(), None, True
        if rest.startswith(":") and rest[1:].isdigit():
            return hostname.lower(), int(rest[1:]), True
        return None, None, False
    before, sep, after = host.rpartition(":")
    if sep:
        if not after.isdigit():
            return None, None, False
        return before.lower(), int(after), True
    return host.lower(), None, True


def dashboard_upstream_path(run_id: str, path: str) -> str:
    """Map a loopback request path onto the run-scoped platform proxy route."""
    return f"/api/v1/rft/runs/{run_id}/dashboard/{path.lstrip('/')}"


def _has_traversal(path_with_query: str) -> bool:
    """Detect "." / ".." segments (encoded or backslash-separated).

    The loopback must never forward traversal paths: httpx normalizes dot
    segments BEFORE the bearer token is attached, which would turn the
    proxy into an authenticated GET of endpoints outside the run-scoped
    dashboard route.
    """
    parsed = urllib.parse.urlsplit(path_with_query)
    normalized = urllib.parse.unquote(parsed.path).replace("\\", "/")
    return any(segment in (".", "..") for segment in normalized.split("/"))


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
    upstream origin to the browser. Query strings and fragments are both
    preserved on rewritten locations — fragments carry SPA routing
    state that the browser never re-sends to the server.

    Relative locations may themselves be platform-shaped: the platform
    proxy rewrites upstream redirects onto the fixed
    ``/api/v1/rft/runs/{run_id}/dashboard/...`` prefix so a browser
    re-enters platform authZ. The loopback must strip that prefix here —
    otherwise it would prepend the run prefix a second time and the
    upstream request would 404 (double-rewrite composition).
    """
    parsed = urllib.parse.urlsplit(location)
    prefix = dashboard_upstream_path(run_id, "")
    query = f"?{parsed.query}" if parsed.query else ""
    # Fragments are client-side routing state (SPA routes like
    # ``.../dashboard/#/metrics``): a rewritten Location that drops them
    # strands the browser on the app root.
    fragment = f"#{parsed.fragment}" if parsed.fragment else ""

    if not parsed.scheme and not parsed.netloc:
        if parsed.path == prefix.rstrip("/"):
            return "/" + query + fragment
        if parsed.path.startswith(prefix):
            remainder = parsed.path[len(prefix) :]
            if remainder.startswith("//"):
                # Protocol-relative escape: the browser would leave the
                # loopback origin for an attacker-controlled host.
                return None
            if remainder.startswith("/"):
                # Double slash after the prefix: keep a single joining slash.
                return remainder + query + fragment
            return "/" + remainder + query + fragment
        if parsed.path.startswith("/api/v1/"):
            # Platform-shaped but outside this run's dashboard scope: the
            # loopback cannot serve it (its own prefix would compound).
            return None
        # Ordinary dashboard-relative location: resolves identically on
        # the loopback root (its fragment travels with it unchanged).
        return location

    base = urllib.parse.urlsplit(base_url)
    if _origin_of(parsed) != _origin_of(base):
        return None

    path = parsed.path or "/"
    if path == prefix.rstrip("/"):
        path = "/"
    elif path.startswith(prefix):
        remainder = path[len(prefix) :]
        if remainder.startswith("//"):
            return None
        path = remainder if remainder.startswith("/") else "/" + remainder
    else:
        return None
    return path + query + fragment


class DashboardProxyHandler(http.server.BaseHTTPRequestHandler):
    """Forward every loopback GET to the platform dashboard proxy route.

    Class attributes ``run_id``, ``base_url``, ``api_key``,
    ``capability_token`` and ``upstream`` are bound by
    :func:`make_dashboard_proxy_server`.
    """

    protocol_version = "HTTP/1.1"
    run_id: str = ""
    base_url: str = ""
    api_key: str = ""
    capability_token: str = ""
    _issue_capability_cookie = False
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
        # The capability cookie is validated BEFORE the Host/Origin checks:
        # another local user who port-scans the loopback port must never
        # reach any upstream work without the unguessable token.
        self._issue_capability_cookie = False
        remaining_path = self._authorize_request()
        if remaining_path is None:
            self._send_plain_error(403, "Forbidden: unknown dashboard proxy URL.")
            return
        if not self._request_targets_loopback():
            return
        if _has_traversal(remaining_path):
            # Rejected before URL construction: httpx would normalize dot
            # segments and attach the bearer token to paths outside the
            # run-scoped dashboard route.
            self._send_plain_error(400, "Bad request: path traversal is not allowed.")
            return
        if self.upstream is None:  # pragma: no cover - guarded by factory
            self._send_plain_error(500, "Dashboard proxy is not configured.")
            return
        # Absolute URL built like APIClient does (base_url + /api/v1 route),
        # so the proxy does not depend on upstream client defaults.
        url = f"{self.base_url.rstrip('/')}{dashboard_upstream_path(self.run_id, remaining_path)}"
        try:
            with self.upstream.stream(
                "GET",
                url,
                # The API token travels in the Authorization header only —
                # never in the URL — so it stays out of access logs.
                # The browser's Accept and Last-Event-ID are the only
                # caller headers forwarded: Accept negotiates SSE
                # (text/event-stream) correctly, and Last-Event-ID lets an
                # EventSource RESUME a stream after reconnecting instead
                # of restarting (which would duplicate events).
                headers=self._upstream_headers(),
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

    def _authorize_request(self) -> Optional[str]:
        """Validate the capability cookie (or the one-time ``?t=`` handoff).

        Every request must present the per-spawn capability token:
        normally as the loopback cookie the entry response sets, or — for
        the printed URL's first hit — as the ``?t=<token>`` query
        parameter, which additionally (re)issues the cookie. The
        comparison is constant-time and runs BEFORE any Host/Origin or
        upstream work, so another local user who port-scans the port
        cannot read the dashboard through the victim's bearer token.
        Returns the dashboard-relative path (handoff parameter stripped)
        when authorized, ``None`` when the token is absent or wrong.
        """
        try:
            parsed = urllib.parse.urlsplit(self.path)
            query_pairs = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
        except ValueError:
            # Malformed request targets (e.g. a bogus bracketed netloc) are
            # a clean rejection, never an exception.
            return None
        # compare_digest requires ASCII; bytes comparison accepts any
        # request target, including non-UTF-8 values.
        expected = self.capability_token.encode("utf-8")
        authorized = False
        remaining: list[tuple[str, str]] = []
        for key, value in query_pairs:
            if key == _CAPABILITY_QUERY_PARAM and hmac.compare_digest(
                value.encode("utf-8", "surrogateescape"), expected
            ):
                # The one-time handoff: authorize and (re)issue the cookie.
                authorized = True
                self._issue_capability_cookie = True
                continue
            remaining.append((key, value))
        cookie_header = self.headers.get("Cookie")
        if cookie_header:
            jar = http.cookies.SimpleCookie()
            try:
                jar.load(cookie_header)
            except http.cookies.CookieError:  # pragma: no cover - tolerant parser
                pass
            morsel = jar.get(_CAPABILITY_COOKIE_NAME)
            if morsel is not None and hmac.compare_digest(
                morsel.value.encode("utf-8", "surrogateescape"), expected
            ):
                authorized = True
        if not authorized:
            return None
        path = parsed.path or "/"
        if remaining:
            path += "?" + urllib.parse.urlencode(remaining)
        return path

    def _capability_set_cookie(self) -> str:
        """The entry response's Set-Cookie value for the capability token.

        ``Path=/`` covers every root-relative asset/API/SSE request the
        dashboard makes; ``HttpOnly`` keeps the token out of JavaScript;
        ``SameSite=Strict`` blocks cross-site sends, so a hostile page on
        another origin can never present the cookie. Cookies are also
        host-keyed: a DNS-rebound name pointing at 127.0.0.1 never
        receives the 127.0.0.1 cookie in the first place. ``Secure`` is
        omitted deliberately — the loopback origin is plain http.
        """
        return (
            f"{_CAPABILITY_COOKIE_NAME}={self.capability_token}; Path=/; HttpOnly; SameSite=Strict"
        )

    def _upstream_headers(self) -> dict[str, str]:
        """Headers forwarded upstream: auth, transport, SSE negotiation.

        Only ``Accept`` and ``Last-Event-ID`` are taken from the browser
        request — everything else (cookies, host-specific headers) stays
        local and is never relayed.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept-Encoding": "identity",
        }
        accept = self.headers.get("Accept")
        if accept:
            headers["Accept"] = accept
        last_event_id = self.headers.get("Last-Event-ID")
        if last_event_id:
            headers["Last-Event-ID"] = last_event_id
        return headers

    def _request_targets_loopback(self) -> bool:
        """Reject non-loopback Host/Origin headers (DNS-rebinding guard).

        The proxy serves private run data with the CLI's bearer token
        injected; a hostile page must not be able to read it through a
        rebound DNS name pointing at 127.0.0.1. Reject before any
        upstream work, so forged requests never touch the platform.
        Malformed or duplicated headers are rejected, never tolerated.
        """
        bound_port = self.server.server_address[1]
        hosts = self.headers.get_all("Host") or []
        if len(hosts) != 1:
            # Duplicate Host headers (good first, hostile second) are a
            # smuggling vector: reject outright.
            self._send_plain_error(403, "Forbidden: loopback requests only.")
            return False
        hostname, port, well_formed = _parse_host_header(hosts[0])
        if (
            not well_formed
            or hostname not in _LOOPBACK_HOSTNAMES
            or (port is not None and port != bound_port)
        ):
            self._send_plain_error(403, "Forbidden: loopback requests only.")
            return False
        origin = self.headers.get("Origin")
        if origin is not None and not _origin_is_loopback(origin, bound_port):
            self._send_plain_error(403, "Forbidden: loopback requests only.")
            return False
        return True

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
            # Rewritten locations are loopback-root-relative; the browser
            # sends the capability cookie with every follow-up regardless
            # of path, so no token prefixing is needed here.
            self.send_header("Location", redirect_location)
        if self._issue_capability_cookie:
            # The one-time ?t= handoff: exchange the query token for the
            # cookie that authorizes every subsequent request.
            self.send_header("Set-Cookie", self._capability_set_cookie())
        # Pre-compressed assets arrive with Content-Encoding even though we
        # requested identity. httpx DECODES in iter_bytes, so relaying the
        # compressed Content-Length with decoded bytes would break the
        # browser. Forward the encoding header and stream RAW bytes so the
        # body matches the advertised length.
        content_encoding = response.headers.get("content-encoding")
        body_iter = response.iter_raw() if content_encoding else response.iter_bytes()
        if content_encoding:
            self.send_header("Content-Encoding", content_encoding)
        length = response.headers.get("content-length")
        if length:
            self.send_header("Content-Length", length)
        else:
            # Unknown length (SSE streams, chunked responses): close the
            # connection to delimit the body instead of buffering it.
            self.close_connection = True
            self.send_header("Connection", "close")
        self.end_headers()
        self._pump_body(body_iter)

    def _pump_body(self, body_iter: Any) -> None:
        """Stream the upstream body to the browser, noticing disconnects.

        The upstream read has no timeout (a quiet SSE stream must not be
        cut off between events), so discovering a closed tab only on the
        next write is not enough: a quiet stream would block forever and
        the request would never finish, keeping the idle watchdog's
        in-flight counter nonzero and pinning the detached proxy (and
        its API credential) indefinitely. The upstream iterator is
        therefore consumed on a helper thread; while the main loop waits
        for chunks it polls the client socket for a half-close and drops
        out as soon as the browser is gone.
        """
        chunks: queue.Queue[Any] = queue.Queue()
        finished = object()

        def _read_upstream() -> None:
            try:
                for chunk in body_iter:
                    chunks.put(chunk)
            except Exception as error:  # noqa: BLE001 - relayed to the pump
                chunks.put(error)
            finally:
                chunks.put(finished)

        threading.Thread(target=_read_upstream, daemon=True).start()
        while True:
            try:
                item = chunks.get(timeout=_STREAM_LIVENESS_POLL_SECONDS)
            except queue.Empty:
                if self._downstream_disconnected():
                    # The browser closed the tab mid-stream: stop pumping
                    # and let the response context manager close the
                    # upstream so the reader thread unblocks and this
                    # request finally finishes (idle shutdown stays armed).
                    self.close_connection = True
                    return
                continue
            if item is finished:
                return
            if isinstance(item, Exception):
                raise item
            try:
                self.wfile.write(item)
                self.wfile.flush()  # SSE events must reach the browser immediately
            except (BrokenPipeError, ConnectionResetError):
                # The browser tab was closed mid-stream; drop the connection.
                return

    def _downstream_disconnected(self) -> bool:
        """Peek the client socket for a half-close without consuming bytes.

        ``select`` first so the peek never blocks (Windows has no
        ``MSG_DONTWAIT``); a readable socket that peeks back EOF means
        the browser closed the connection. A readable socket with data
        stays "connected" — the browser cannot legally send another
        request mid-response, so we do not try to interpret it.
        """
        conn = self.connection
        try:
            readable, _, _ = select.select([conn], [], [], 0)
            if not readable:
                return False
            return conn.recv(1, socket.MSG_PEEK) == b""
        except (OSError, ValueError):
            # Reset/invalid socket: treat as disconnected.
            return True

    def _send_plain_error(self, status: int, message: str) -> None:
        body = (message + "\n").encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        if self._issue_capability_cookie:
            # Even a failing entry response exchanges the handoff token for
            # the cookie, so the browser is not stranded after a retry.
            self.send_header("Set-Cookie", self._capability_set_cookie())
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        # Per-request proxy logs would spam the terminal while serving.
        return


class DashboardProxyServer(http.server.ThreadingHTTPServer):
    """Loopback server that tracks activity so it can idle-exit."""

    daemon_threads = True
    run_id: str = ""
    capability_token: str = ""
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

    Returns ``(server, url)`` where ``url`` carries the per-spawn
    capability token once as ``?t=<token>`` — the one-time browser
    handoff that the entry response exchanges for a loopback cookie;
    ``server.capability_token`` exposes it for callers that need the raw
    value. The caller runs
    ``server.serve_forever()`` and, on shutdown, closes the server and
    ``server.upstream_client``. ``upstream`` may be injected for tests.
    """
    upstream_client = upstream or httpx.Client(
        headers={"User-Agent": user_agent or _default_user_agent()},
        timeout=_UPSTREAM_REQUEST_TIMEOUT,
    )
    capability_token = secrets.token_urlsafe(_CAPABILITY_TOKEN_BYTES)
    handler_cls = type(
        "BoundDashboardProxyHandler",
        (DashboardProxyHandler,),
        {
            "run_id": run_id,
            "base_url": base_url,
            "api_key": api_key,
            "capability_token": capability_token,
            "upstream": upstream_client,
        },
    )
    server = DashboardProxyServer((host, 0), handler_cls)
    server.run_id = run_id
    server.capability_token = capability_token
    server.upstream_client = upstream_client
    # The URL carries the capability token ONCE as a query parameter — the
    # one-time browser handoff. The entry response exchanges it for a
    # loopback cookie that authorizes every subsequent request, so the
    # dashboard's root-relative asset/API/SSE URLs keep working.
    url = f"http://{host}:{server.server_address[1]}/?t={capability_token}"
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


_FINGERPRINT_SALT = b"prime-cli.dashboard_proxy.fingerprint-v1"
"""Fixed, NON-secret domain-separation salt for the reuse fingerprint."""


def _proxy_fingerprint(base_url: str, run_id: str, api_key: str) -> str:
    """Context fingerprint keying a detached proxy's state file.

    Combines the backend origin, the run id and a NON-REVERSIBLE digest
    of the API token (never the token itself), so switching
    PRIME_CONTEXT / base URL / API key starts a fresh proxy instead of
    reusing one authenticated for a different context.

    scrypt is used because CodeQL's password-hash policy applies to the
    token: fast hashes (sha256, even via HMAC) are flagged as insecure for
    password-class secrets. This is a KEYING fingerprint, not credential
    storage, so deliberately cheap fixed parameters are fine; the digest
    stays stable across restarts and derives only once per CLI invocation.
    """
    message = f"{base_url}\n{run_id}\n{api_key}".encode("utf-8")
    return hashlib.scrypt(message, salt=_FINGERPRINT_SALT, n=2**14, r=8, p=1, dklen=16).hex()


def proxy_state_path(
    run_id: str, state_dir: Optional[Path] = None, fingerprint: Optional[str] = None
) -> Path:
    """Per-run state file holding the detached proxy's pid and port."""
    directory = state_dir or _PROXY_STATE_DIR
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", run_id)
    suffix = f"-{fingerprint}" if fingerprint else ""
    return directory / f"train-dashboard-{safe}{suffix}.json"


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


def _proxy_still_serves(url: str, timeout: float = 10.0) -> bool:
    """Probe a live proxy to confirm its credentials still work upstream.

    A proxy whose stored token was revoked or whose backend moved answers
    with a clean local 502 (the platform rejects the bearer with 4xx).
    Only such healthy proxies are reused; otherwise a fresh proxy starts.
    """
    try:
        response = httpx.get(url, timeout=timeout, follow_redirects=False)
    except httpx.HTTPError:
        return False
    return response.status_code < 400


def _live_proxy_url(state: Optional[dict[str, Any]]) -> Optional[str]:
    """Return the loopback URL of a still-running proxy, if any.

    The URL carries the proxy's capability token as the one-time ``?t=``
    handoff; a state file without a token (pre-capability format) is not
    reusable.
    """
    if not state:
        return None
    pid = state.get("pid")
    port = state.get("port")
    token = state.get("token")
    if not isinstance(pid, int) or not isinstance(port, int) or not isinstance(token, str):
        return None
    if not token:
        return None
    if not _pid_is_alive(pid):
        return None
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.25):
            pass
    except OSError:
        return None
    return f"http://127.0.0.1:{port}/?t={token}"


def _unlink_state_if_owned(state_path: Path, pid: int) -> None:
    """Unlink a proxy state file only if it still records the given pid.

    A newer/replacement proxy may have overwritten the file; removing its
    record would break that proxy's reuse.
    """
    try:
        recorded = json.loads(state_path.read_text())
    except (OSError, ValueError):
        return
    if isinstance(recorded, dict) and recorded.get("pid") == pid:
        state_path.unlink(missing_ok=True)


def _cleanup_stale_proxy_states(
    run_id: str, *, keep_fingerprint: str, state_dir: Optional[Path] = None
) -> None:
    """Drop state files of this run that belong to a different context.

    A live proxy under a different fingerprint is NOT terminated here:
    it may be actively serving a dashboard that another terminal (with a
    different PRIME_CONTEXT / token) opened, and its bearer token is
    scoped to exactly the context that started it. The idle timeout bounds
    its lifetime. Dead orphans have their stale state files removed so
    they can never be mistaken for a reusable proxy.
    """
    directory = state_dir or _PROXY_STATE_DIR
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", run_id)
    pattern = f"train-dashboard-{safe}-*.json"
    for candidate in sorted(directory.glob(pattern)):
        suffix = candidate.name[len(f"train-dashboard-{safe}-") : -len(".json")]
        if suffix == keep_fingerprint:
            continue
        state = _read_proxy_state(candidate)
        pid = state.get("pid") if state else None
        if isinstance(pid, int) and _pid_is_alive(pid):
            continue  # live proxy of another context: leave it to its idle exit
        candidate.unlink(missing_ok=True)


def _terminate_child_process(process: "subprocess.Popen[Any]") -> None:
    """Kill a detached child and reap it, killing the whole process group.

    The child is its own session leader (``start_new_session=True``), so
    the group id equals its pid; killing the group also clears anything
    it may have spawned. On platforms without ``killpg`` fall back to
    ``Popen.kill()``.
    """
    killed = False
    try:
        if hasattr(os, "killpg") and hasattr(signal, "SIGKILL"):
            os.killpg(process.pid, signal.SIGKILL)
            killed = True
    except OSError:
        pass
    if not killed:
        try:
            process.kill()
        except OSError:
            pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:  # pragma: no cover - stubborn child
        pass


def _read_child_ready_line(process: "subprocess.Popen[Any]", timeout: float) -> Optional[str]:
    """Read the child's ``PORT <n> <token>`` line with a timeout (pipes cannot select)."""
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
    survives the CLI exiting; it writes a pid/port/token state file
    (0600) under the user cache dir and exits itself once idle. If a
    healthy proxy for this run is already running, its URL is returned
    without spawning another. The returned URL carries the proxy's
    capability token once as ``?t=<token>`` (the one-time browser
    handoff) — the token reaches this process only through the child's
    private ready pipe or the 0600 state file, never through logs or
    command lines.
    """
    fingerprint = _proxy_fingerprint(base_url, run_id, api_key)
    state_path = proxy_state_path(run_id, state_dir, fingerprint)
    existing = _live_proxy_url(_read_proxy_state(state_path))
    if existing is not None and _proxy_still_serves(existing):
        return existing
    _cleanup_stale_proxy_states(run_id, keep_fingerprint=fingerprint, state_dir=state_dir)

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
    detach_kwargs: dict[str, Any] = {}
    if sys.platform == "win32":
        # start_new_session is POSIX-only; without a new process group a
        # terminal Ctrl-C would kill the dashboard proxy early on Windows.
        detach_kwargs["creationflags"] = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(
            subprocess, "CREATE_NEW_PROCESS_GROUP", 0
        )
    else:
        detach_kwargs["start_new_session"] = True
    process = subprocess.Popen(  # noqa: S603 - fixed module command
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        env={**os.environ, _CHILD_API_KEY_ENV: api_key},
        **detach_kwargs,
    )
    ready_ok = False
    ready_parts: list[str] = []
    try:
        ready_line = _read_child_ready_line(process, ready_timeout_seconds)
        ready_parts = str(ready_line or "").split()
        # ``PORT <port> <capability-token>``: the private pipe hands the
        # per-spawn token to the parent without ever putting it on a
        # command line (process tables are world-readable) or in a log.
        if (
            len(ready_parts) == 3
            and ready_parts[0] == "PORT"
            and ready_parts[1].isdigit()
            and ready_parts[2]
        ):
            ready_ok = True
    finally:
        if not ready_ok:
            # A half-started child still holds the API token in its
            # environment: never leave it running (orphan-on-failure). Kill
            # FIRST — closing the pipe while the ready-line reader thread is
            # blocked in readline would wait on the buffered-reader lock
            # and hang the CLI despite the timeout.
            _terminate_child_process(process)
            # The orphan may have written the state file before failing,
            # but only remove it if it still names THIS child: a
            # concurrent/replacement proxy may already have claimed it.
            _unlink_state_if_owned(state_path, process.pid)
        if process.stdout is not None:
            process.stdout.close()
    if not ready_ok:
        raise RuntimeError("The dashboard proxy failed to start.")
    return f"http://127.0.0.1:{int(ready_parts[1])}/?t={ready_parts[2]}"


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
        # No output at all: CodeQL flags clear-text output that mentions
        # credential-named variables, and the token must never be logged.
        # The parent surfaces this as a generic "failed to start" error.
        return 2

    server, _url = make_dashboard_proxy_server(
        args.run_id,
        base_url=args.base_url,
        api_key=api_key,
    )
    port = server.server_address[1]
    state_path = Path(args.state_file)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    # The state file carries the capability token (needed to reuse the
    # proxy), so it is created with user-only permissions: another local
    # user must not be able to read it.
    state_fd = os.open(state_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(state_fd, "w") as state_file_obj:
        json.dump(
            {
                "run_id": args.run_id,
                "pid": os.getpid(),
                "port": port,
                "token": server.capability_token,
            },
            state_file_obj,
        )
    # Report the port and capability token, then detach stdout: the parent
    # may exit (closing the pipe) at any time and this process must never
    # block writing to it.
    print(f"PORT {port} {server.capability_token}", flush=True)
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
        server.server_close()
        if server.upstream_client is not None:
            server.upstream_client.close()
        # Unlink the state file ONLY if it still names THIS process: a
        # newer proxy for the same run/context may have replaced it, and
        # removing its state would break that proxy's reuse.
        _unlink_state_if_owned(state_path, os.getpid())
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess in tests
    raise SystemExit(_main())
