"""Keeping a started tunnel serving: watch it, restart frpc, give up when it is gone."""

import asyncio
import logging
import random
import time
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Optional

import httpx

from prime_tunnel.exceptions import TunnelError, TunnelGoneError
from prime_tunnel.frpc import FrpcEvent

if TYPE_CHECKING:
    from prime_tunnel.tunnel import Tunnel

logger = logging.getLogger(__name__)

# The edge in front of the tunnel service blocks default Python user agents.
_USER_AGENT = "Mozilla/5.0 (compatible; prime-tunnel-supervisor)"


@dataclass(frozen=True)
class Supervision:
    """What a supervised tunnel requests through its URL to check that it serves."""

    # Path of the request. It should answer quickly and without side effects.
    probe_path: str = "/"
    # Name of a response header the local service sets on every response. If
    # given, a response without it is taken as not coming from the local
    # service; otherwise the supervisor goes by the status code.
    probe_header: Optional[str] = None


class _Verdict(Enum):
    # A request through the public URL reached the local service.
    SERVING = "serving"
    # The public URL fails while the local service answers: the tunnel is at fault.
    NOT_SERVING = "not_serving"
    # The local service does not answer either, so the tunnel cannot be judged.
    UNKNOWN = "unknown"


class _Response:
    """The parts of a probe response the supervisor looks at."""

    def __init__(self, status_code: int, headers: httpx.Headers):
        self.status_code = status_code
        self.headers = headers


class Supervisor:
    """Watches one started Tunnel and repairs it.

    Signals, cheapest first: the frpc process exiting, what frpc logs, and a
    GET through the public URL. A failed GET is only blamed on the tunnel if
    the same GET sent straight to the local service succeeds.

    Recovery: one more look after a short grace period, in which frpc may
    reconnect by itself, then restart() with backoff and jitter for as long as
    it takes. A tunnel whose registration is gone is marked dead and left
    alone; only a new tunnel helps then.
    """

    # Seconds between probes of a tunnel that is serving.
    probe_interval = 15.0
    # Seconds one probe request may take.
    probe_timeout = 10.0
    # Seconds between a failed probe and the one that confirms it.
    grace = 5.0
    # Seconds between a restart and the probe that checks it.
    settle = 2.0
    # Seconds between checks of the frpc process.
    tick = 1.0
    # Delay before the second restart in a row; doubles up to backoff_max.
    backoff_base = 1.0
    backoff_max = 60.0

    def __init__(self, tunnel: "Tunnel", options: Optional[Supervision] = None):
        options = options or Supervision()
        path = options.probe_path
        self._tunnel = tunnel
        self._probe_path = path if path.startswith("/") else f"/{path}"
        self._probe_header = options.probe_header
        self._task: Optional[asyncio.Task] = None
        self._wake = asyncio.Event()
        self._probe_now = False
        self._warned_missing_header = False

    def start(self) -> None:
        """Start watching. Must be called from the tunnel's event loop."""
        if self._task is None:
            self._task = asyncio.create_task(self._run())

    def cancel(self) -> None:
        """Stop watching without waiting for the task to finish."""
        if self._task is not None:
            try:
                self._task.cancel()
            except RuntimeError:
                pass  # Event loop already closed

    async def stop(self) -> None:
        """Stop watching and wait for the task to finish."""
        task = self._task
        if task is None or task is asyncio.current_task():
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            if not task.cancelled():
                raise  # The caller was cancelled, not the task
        except Exception:
            pass

    def on_frpc_event(self, event: FrpcEvent) -> None:
        """Take one frpc event. Must be called on the event loop."""
        if event is FrpcEvent.CONNECTED:
            return
        # GONE is already recorded on the tunnel; anything else means frpc
        # gave up on something it retries slowly or never.
        if event is not FrpcEvent.GONE:
            self._probe_now = True
        self._wake.set()

    def _backoff(self, attempt: int) -> float:
        """Seconds to wait before restart number `attempt` in a row (from 0)."""
        if attempt <= 0:
            return 0.0
        # The exponent is capped so that a long outage cannot overflow it.
        delay = min(self.backoff_max, self.backoff_base * 2 ** min(attempt - 1, 32))
        return delay * random.uniform(0.5, 1.0)

    async def _run(self) -> None:
        tunnel = self._tunnel
        timeout = httpx.Timeout(self.probe_timeout)
        headers = {"User-Agent": _USER_AGENT}
        try:
            async with (
                httpx.AsyncClient(timeout=timeout, headers=headers) as public,
                # The local service is never behind a proxy.
                httpx.AsyncClient(timeout=timeout, headers=headers, trust_env=False) as local,
            ):
                await self._watch(public, local)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Supervisor of tunnel %s failed; it is no longer watched", tunnel.url)

    async def _watch(self, public: httpx.AsyncClient, local: httpx.AsyncClient) -> None:
        tunnel = self._tunnel
        next_probe = time.monotonic() + self.settle
        strikes = 0
        attempts = 0

        while True:
            self._wake.clear()
            wait = max(0.0, min(self.tick, next_probe - time.monotonic()))
            if not (self._probe_now or tunnel._gone):
                try:
                    await asyncio.wait_for(self._wake.wait(), timeout=wait)
                except asyncio.TimeoutError:
                    pass

            if not tunnel._started or tunnel._stopping:
                return
            if tunnel._gone:
                await self._give_up()
                return
            if tunnel._process_lock.locked():
                # Someone else is restarting it
                await asyncio.sleep(self.tick)
                continue

            if not tunnel.is_running:
                reason = "frpc exited"
            else:
                if not self._probe_now and time.monotonic() < next_probe:
                    continue
                self._probe_now = False
                verdict, reason = await self._check(public, local)
                if verdict is not _Verdict.NOT_SERVING:
                    strikes = 0
                    if verdict is _Verdict.SERVING:
                        attempts = 0
                    tunnel._set_recovering(False)
                    next_probe = time.monotonic() + self.probe_interval
                    continue
                strikes += 1
                if strikes < 2:
                    logger.debug("Tunnel %s is not serving (%s); rechecking", tunnel.url, reason)
                    next_probe = time.monotonic() + self.grace
                    continue

            strikes = 0
            tunnel._set_recovering(True)
            delay = self._backoff(attempts)
            attempts += 1
            logger.warning(
                "Tunnel %s is down (%s); restarting frpc in %.1fs (attempt %d)",
                tunnel.url,
                reason,
                delay,
                attempts,
            )
            if delay:
                process = tunnel._process
                await asyncio.sleep(delay)
                if tunnel._stopping or tunnel._gone or tunnel._process_lock.locked():
                    continue
                if tunnel._process is not process:
                    continue  # Someone else restarted it meanwhile

            try:
                await tunnel.restart()
            except TunnelGoneError:
                tunnel._mark_gone()
                continue
            except TunnelError as e:
                if tunnel._stopping:
                    return
                logger.warning("Restart of tunnel %s failed: %s", tunnel.url, e)
                # Only the service can tell a registration that is gone from
                # one that is merely unreachable right now.
                try:
                    await tunnel.status()
                except Exception:
                    pass
                continue

            logger.info("Restarted frpc of tunnel %s", tunnel.url)
            # Whatever the replaced frpc reported no longer applies.
            self._probe_now = False
            next_probe = time.monotonic() + self.settle

    async def _give_up(self) -> None:
        """Leave a tunnel whose registration is gone."""
        tunnel = self._tunnel
        logger.error(
            "Tunnel %s is dead: its registration is gone. Stop it and start a new one.",
            tunnel.url,
        )
        # A frpc the service rejected serves nothing and would retry forever.
        # One that never reported it may still be serving: leave that alone.
        process = tunnel._process
        if tunnel._frpc_gone and process is not None:
            await asyncio.to_thread(tunnel._end_process, process)

    async def _check(
        self, public: httpx.AsyncClient, local: httpx.AsyncClient
    ) -> tuple[_Verdict, str]:
        """Probe the public URL, and the local service if that fails."""
        tunnel = self._tunnel
        header = self._probe_header
        auth = None
        if tunnel.http_user and tunnel.http_password:
            auth = (tunnel.http_user, tunnel.http_password)

        through = await self._get(public, f"{tunnel.url}{self._probe_path}", auth=auth)
        if through is not None:
            if header is not None:
                suspect = header not in through.headers
            else:
                # What the tunnel service answers for a tunnel it cannot reach.
                suspect = through.status_code == 404 or through.status_code >= 500
            if not suspect:
                return _Verdict.SERVING, ""

        host = tunnel.local_addr
        if ":" in host:
            host = f"[{host}]"
        direct = await self._get(
            local,
            f"http://{host}:{tunnel.local_port}{self._probe_path}",
            headers={"Host": tunnel.hostname or host},
        )
        if direct is None:
            return _Verdict.UNKNOWN, "local service is not answering"
        if header is not None:
            if header not in direct.headers:
                if not self._warned_missing_header:
                    self._warned_missing_header = True
                    logger.warning(
                        "Local service of tunnel %s does not set the %s header on %s; "
                        "the tunnel cannot be checked",
                        tunnel.url,
                        header,
                        self._probe_path,
                    )
                return _Verdict.UNKNOWN, f"local service does not set {header}"
        elif through is not None and through.status_code == direct.status_code:
            # The local service itself answers this way.
            return _Verdict.SERVING, ""

        if through is None:
            return _Verdict.NOT_SERVING, "no response through the URL"
        if header is not None:
            return _Verdict.NOT_SERVING, f"HTTP {through.status_code} without {header}"
        return _Verdict.NOT_SERVING, f"HTTP {through.status_code} through the URL"

    async def _get(self, client: httpx.AsyncClient, url: str, **kwargs) -> Optional[_Response]:
        """GET `url` and return its status and headers, or None if it fails."""

        async def fetch() -> _Response:
            async with client.stream("GET", url, **kwargs) as response:
                return _Response(response.status_code, response.headers)

        try:
            return await asyncio.wait_for(fetch(), timeout=self.probe_timeout)
        except (httpx.HTTPError, asyncio.TimeoutError, OSError):
            return None
