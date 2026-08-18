"""Shared reliability policy for the sandbox Connect-RPC surface.

One classifier and one set of tunables, used by every remote-control RPC (exec, background-job
launch, live-process stream + control) so a transient blip on the client <-> sandbox link is
retried/reconnected instead of killing the caller's work, while a permanent fault still fails fast.

All timeouts and retry budgets are overridable via ``PRIME_SANDBOX_*`` env vars.
"""

import os

from connectrpc.code import Code
from connectrpc.errors import ConnectError

from .core import APIError
from .exceptions import CommandTimeoutError


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


# Retry budget for the idempotent RPCs (background-job launch, live-process control).
RPC_MAX_ATTEMPTS = _env_int("PRIME_SANDBOX_RPC_MAX_ATTEMPTS", 3)
RPC_BACKOFF_BASE = _env_float("PRIME_SANDBOX_RPC_BACKOFF_BASE", 0.5)

# Background-job launch is fire-and-forget, so its exec should return in well under a second;
# a timeout is a transport blip, not a slow command.
BG_LAUNCH_TIMEOUT = _env_int("PRIME_SANDBOX_BG_LAUNCH_TIMEOUT", 30)

# Live-process control RPC deadlines (ms).
PROCESS_INPUT_TIMEOUT_MS = _env_int("PRIME_SANDBOX_PROCESS_INPUT_TIMEOUT_MS", 30_000)
PROCESS_SIGNAL_TIMEOUT_MS = _env_int("PRIME_SANDBOX_PROCESS_SIGNAL_TIMEOUT_MS", 10_000)

# The live-process output stream can be re-attached to (Connect RPC) after a transient drop, since
# the process keeps running in the sandbox. Bound the reconnects so a genuinely dead process/sandbox
# still surfaces.
STREAM_MAX_RECONNECTS = _env_int("PRIME_SANDBOX_STREAM_MAX_RECONNECTS", 5)
STREAM_RECONNECT_BACKOFF_BASE = _env_float("PRIME_SANDBOX_STREAM_RECONNECT_BACKOFF_BASE", 0.5)

# Live-process transport tuning: keep the long-lived stream connection warm so a brief idle stall
# does not get torn down (read_timeout None = no per-read deadline; the stream's own deadline and
# the server's keepalive events bound it).
STREAM_TCP_KEEPALIVE = _env_float("PRIME_SANDBOX_STREAM_TCP_KEEPALIVE", 15.0)
STREAM_POOL_IDLE_TIMEOUT = _env_float("PRIME_SANDBOX_STREAM_POOL_IDLE_TIMEOUT", 300.0)

# Connect codes that mean "the link hiccuped", not "the request is wrong". UNAUTHENTICATED is
# excluded on purpose: it is handled by the token-refresh retry, not this backoff. INTERNAL is
# included because a broken output stream surfaces as INTERNAL "Error reading content" (observed
# in production), the same transient stream-break class as UNAVAILABLE "... timed out".
_TRANSIENT_CODES = frozenset(
    {Code.DEADLINE_EXCEEDED, Code.UNAVAILABLE, Code.ABORTED, Code.INTERNAL}
)

# Substrings of a transport error that mean the same, seen on both ConnectError and APIError.
_TRANSIENT_MARKERS = (
    "timed out",
    "reading a body",
    "reading content",
    "connection reset",
    "connection closed",
    "broken pipe",
    "unavailable",
    "deadline_exceeded",
)


def is_transient_rpc_error(error: BaseException) -> bool:
    """Whether ``error`` is a transient sandbox-transport fault safe to retry/reconnect.

    Transient: a stalled/reset Connect-RPC (DEADLINE_EXCEEDED / UNAVAILABLE / ABORTED, or a
    ``reading a body ... timed out`` / connection-reset body error). Permanent (returns False):
    a 404 sandbox-not-found, other 4xx, or any non-transport error — those must fail fast.
    """
    if isinstance(error, CommandTimeoutError):
        return True
    if isinstance(error, ConnectError):
        if error.code in _TRANSIENT_CODES:
            return True
        message = (error.message or "").lower()
        return any(marker in message for marker in _TRANSIENT_MARKERS)
    if isinstance(error, APIError):
        message = str(error).lower()
        if "not found" in message or "sandbox is no longer" in message:
            return False
        return any(marker in message for marker in _TRANSIENT_MARKERS)
    return False
