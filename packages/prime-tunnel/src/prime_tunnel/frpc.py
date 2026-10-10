"""Reading frpc's output: logging it, keeping recent lines, and classifying events."""

import logging
import re
import subprocess
import threading
from collections import deque
from enum import Enum
from typing import Callable, Optional

# timestamp + level + caller prefix + message
_LOG_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}\.\d{3}\s"
    r"\[([EWIDT])\]\s"
    r"\[.*?\]\s"
    r"(?:\[.*?\]\s)*"
    r"(.+)"
)
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

frpc_logger = logging.getLogger("prime_tunnel.frpc")

_FRPC_LEVELS = {
    "E": logging.ERROR,
    "W": logging.WARNING,
    "I": logging.INFO,
    "D": logging.DEBUG,
    "T": logging.DEBUG,
}


def _log_frpc_line(raw_line: str, tunnel_id: str | None) -> None:
    """Forward one frpc output line to the prime_tunnel.frpc logger at its own level."""
    line = _ANSI_RE.sub("", raw_line)
    m = _LOG_RE.match(line)
    if m:
        level, msg = _FRPC_LEVELS.get(m.group(1), logging.INFO), m.group(2)
    else:
        level, msg = logging.INFO, line
    frpc_logger.log(level, "frpc %s: %s", tunnel_id or "-", msg)


class FrpcEvent(Enum):
    """What an frpc log line says about the tunnel."""

    # The proxy is registered with the server.
    CONNECTED = "connected"
    # The server rejected the registration itself: deleted, expired, or
    # disconnected for too long. Retrying can never fix it.
    GONE = "gone"
    # The server rejected the credentials. Retrying can never fix it.
    REJECTED = "rejected"
    # The server refused to start the proxy. frpc retries, but only after 30s.
    PROXY_REFUSED = "proxy_refused"


# Rejection reasons that mean the registration itself can no longer be used.
_GONE_REASONS = (
    "tunnel is inactive",
    "tunnel not registered",
)

# Rejection reasons that mean the credentials are wrong.
_REJECTED_REASONS = (
    "invalid binding secret",
    "invalid authentication token",
    "token in login doesn't match",
)


def classify_line(line: str) -> Optional[FrpcEvent]:
    """Classify one frpc log line.

    Returns None for anything that calls for no action, including the
    transient login and connect failures frpc retries by itself.
    """
    lowered = line.lower()
    if "start proxy success" in lowered:
        return FrpcEvent.CONNECTED
    proxy_refused = "start error" in lowered
    if not proxy_refused and "connect to server error" not in lowered:
        return None
    if any(reason in lowered for reason in _GONE_REASONS):
        return FrpcEvent.GONE
    if proxy_refused:
        return FrpcEvent.PROXY_REFUSED
    if any(reason in lowered for reason in _REJECTED_REASONS):
        return FrpcEvent.REJECTED
    return None


def failure_message(output_lines: list[str], return_code: int | None = None) -> str:
    """Pick the message that explains a failed frpc out of its output."""
    error_messages: list[str] = []
    for raw_line in output_lines:
        m = _LOG_RE.match(_ANSI_RE.sub("", raw_line))
        if m and m.group(1) in ("E", "W"):
            error_messages.append(m.group(2))

    if error_messages:
        return error_messages[-1]
    output_text = "\n".join(output_lines) if output_lines else "(no output captured)"
    exit_info = f" (exit code {return_code})" if return_code is not None else ""
    return f"frpc process failed{exit_info}: {output_text}"


class FrpcOutput:
    """Reads one frpc process's stdout and stderr, from launch until they close.

    Every line is forwarded to the ``prime_tunnel.frpc`` logger, so reconnects
    and dropped control connections show up in the caller's logs, and the
    last ``max_lines`` are kept for diagnostics. Reading also keeps the pipe
    buffers from filling up and blocking frpc.

    Lines that mean something for the tunnel are reported as FrpcEvents:
    ``first_event`` holds the first one, and ``on_event`` is called for each,
    on a reader thread.
    """

    def __init__(
        self,
        process: subprocess.Popen,
        tunnel_id: str | None,
        on_event: Optional[Callable[[FrpcEvent], None]] = None,
        max_lines: int = 50,
    ):
        self._tunnel_id = tunnel_id
        self._on_event = on_event
        self._lock = threading.Lock()
        self._lines: deque[str] = deque(maxlen=max_lines)
        self._first_event: Optional[FrpcEvent] = None
        self._threads: list[threading.Thread] = []
        for pipe in (process.stdout, process.stderr):
            if pipe is None:
                continue
            t = threading.Thread(target=self._read, args=(pipe,), daemon=True)
            t.start()
            self._threads.append(t)

    @property
    def first_event(self) -> Optional[FrpcEvent]:
        """The first event frpc reported, or None if it has reported none."""
        with self._lock:
            return self._first_event

    def lines(self) -> list[str]:
        """The most recent lines of output."""
        with self._lock:
            return list(self._lines)

    def join(self, timeout: float = 2.0) -> None:
        """Wait for the readers to reach the end of the output of an exited frpc."""
        for t in self._threads:
            t.join(timeout=timeout)

    def _read(self, pipe) -> None:
        try:
            for line in pipe:
                line = line.strip()
                if not line:
                    continue
                with self._lock:
                    self._lines.append(line)
                _log_frpc_line(line, self._tunnel_id)
                event = classify_line(line)
                if event is None:
                    continue
                # Notify before recording, so whoever acts on first_event
                # finds the callback's effects already in place.
                if self._on_event is not None:
                    try:
                        self._on_event(event)
                    except Exception:
                        pass  # A failing callback must not stop the reading
                with self._lock:
                    if self._first_event is None:
                        self._first_event = event
        except (OSError, ValueError):
            pass  # Pipe closed
