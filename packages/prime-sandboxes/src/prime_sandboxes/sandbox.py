"""Sandbox client implementations."""

import asyncio
import functools
import json
import os
import random
import re
import shlex
import sys
import threading
import time
import uuid
from concurrent.futures import Future
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Generic,
    Hashable,
    List,
    Literal,
    NoReturn,
    Optional,
    TypeVar,
)

import aiofiles
import certifi
import httpx
from connectrpc.client import ConnectClient, ConnectClientSync
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from connectrpc.method import MethodInfo
from google.protobuf.message import Message
from pyqwest import Client as HTTPClient
from pyqwest import HTTPTransport
from tenacity import (
    retry,
    retry_if_exception,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .core import APIClient, APIError, AsyncAPIClient
from .exceptions import (
    BatchStatusUnsupportedError,
    CommandTimeoutError,
    DownloadTimeoutError,
    SandboxFileNotFoundError,
    SandboxFileTooLargeError,
    SandboxImagePullError,
    SandboxNotRunningError,
    SandboxOOMError,
    SandboxTimeoutError,
    UploadTimeoutError,
)
from .models import (
    BackgroundJob,
    BackgroundJobStatus,
    BatchBackgroundJobStatusResponse,
    BatchSandboxStatusResponse,
    BulkDeleteSandboxRequest,
    BulkDeleteSandboxResponse,
    CommandResponse,
    CreateSandboxRequest,
    DockerImageCheckResponse,
    EgressPolicyStatus,
    ExposedPort,
    ExposePortRequest,
    FileUploadResponse,
    ListExposedPortsResponse,
    ReadFileResponse,
    RegistryCredentialSummary,
    Sandbox,
    SandboxListResponse,
    SandboxLogsResponse,
    SandboxStatusSnapshot,
    SSHSession,
    validate_egress_lists,
)
from .process import AsyncSandboxProcess
from .rpc_command_session import (
    COMMAND_SESSION_SEND_INPUT_RPC_METHOD,
    COMMAND_SESSION_SEND_SIGNAL_RPC_METHOD,
    COMMAND_SESSION_START_RPC_METHOD,
    build_command_session_send_input_request,
    build_command_session_send_signal_request,
    build_command_session_start_request,
    collect_command_session_start_event,
)

# Connection-level errors: request never reached the server, so retry is safe
# for any HTTP method (including non-idempotent POSTs).
# Note: ReadTimeout / ReadError are NOT in this tuple because they occur after
# the server has (at least partially) started responding — meaning the request
# was already processed and a retry would duplicate side effects on POST.
GATEWAY_CONNECTION_RETRYABLE_EXCEPTIONS = (
    httpx.RemoteProtocolError,  # Server disconnected unexpectedly
    httpx.ConnectError,  # Connection refused/failed
    httpx.PoolTimeout,  # No connection available in pool
)

# connectrpc-python cancels a server stream before its first event when no
# timeout is supplied. A live process cannot outlast the sandbox's 24-hour
# maximum lifetime, so use that lifetime as the transport bound.
_LIVE_PROCESS_TIMEOUT_MS = 24 * 60 * 60 * 1000
_PROCESS_INPUT_TIMEOUT_MS = 30_000
_PROCESS_SIGNAL_TIMEOUT_MS = 10_000

_RequestMessage = TypeVar("_RequestMessage", bound=Message)
_ResponseMessage = TypeVar("_ResponseMessage", bound=Message)
_BatchKey = TypeVar("_BatchKey", bound=Hashable)
_BatchValue = TypeVar("_BatchValue")


@dataclass(frozen=True)
class _BatchItemError:
    """An error for one key in an otherwise successful transport batch."""

    error: Exception


@functools.lru_cache(maxsize=1)
def _ca_bundle() -> bytes:
    with open(certifi.where(), "rb") as ca_file:
        return ca_file.read()


def _live_process_transport() -> HTTPTransport:
    # A bare HTTPTransport carries no trust roots on some pyqwest versions
    # (only the default singleton does), so pass certifi's bundle explicitly.
    return HTTPTransport(tls_ca_cert=_ca_bundle())


def _network_update_payload(
    allow: Optional[List[str]], deny: Optional[List[str]]
) -> Dict[str, Optional[List[str]]]:
    if (allow is None) == (deny is None):
        raise ValueError("exactly one of allow or deny must be provided")

    entries = list(allow if allow is not None else deny or [])
    if not entries:
        raise ValueError("allow or deny must contain at least one destination")
    if "*" in entries:
        if entries != ["*"]:
            raise ValueError("'*' must be the only destination")
        return (
            {"allowlist": None, "denylist": []}
            if allow is not None
            else {"allowlist": [], "denylist": None}
        )

    allowlist = entries if allow is not None else None
    denylist = entries if deny is not None else None
    validate_egress_lists(allowlist, denylist)
    return {"allowlist": allowlist, "denylist": denylist}


# Response-read failures: the server may have already processed the request
# (TCP connection dropped mid-response body). Only safe to retry on idempotent
# methods (GET/HEAD/PUT/DELETE), where a duplicate request is a no-op.
# ReadTimeout is STILL excluded even for idempotent methods — we can't
# distinguish "server hanging" from "server mid-processing", and retrying a
# hung request compounds load on an already-stressed server instead of
# recovering. ReadError (TCP reset / connection drop during read) is different:
# the connection is definitively gone, so retrying doesn't pile on.
GATEWAY_IDEMPOTENT_RETRYABLE_EXCEPTIONS = GATEWAY_CONNECTION_RETRYABLE_EXCEPTIONS + (
    httpx.ReadError,  # TCP connection broken while reading response
)

READ_FILE_RETRYABLE_EXCEPTIONS = GATEWAY_IDEMPOTENT_RETRYABLE_EXCEPTIONS + (
    httpx.ConnectTimeout,  # Timed out before the request reached the gateway
    httpx.ReadTimeout,  # Timed out waiting for an idempotent read response
)

# Retryable HTTP 5xx status codes (e.g. Cloudflare 524 timeout, server errors)
RETRYABLE_5XX_STATUSES = frozenset({500, 502, 503, 504, 524})

# Max retries for transient 409 errors
MAX_409_RETRIES = 4
RETRY_409_BASE_DELAY = 0.25  # 250ms, 500ms, 1000ms, 2000ms with exponential backoff
MAX_GATEWAY_ATTEMPTS = MAX_409_RETRIES + 1

# Refresh cached gateway auth this many seconds before its reported expiry.
AUTH_REFRESH_MARGIN_SECONDS = 60

# Max bytes of stdout/stderr returned per background-job status check
JOB_OUTPUT_TAIL_BYTES = 10 * 1024 * 1024

# Platform batch-status contracts cap one request at 100 identifiers. Concurrent
# single-item waits are collected briefly so callers share a request without
# adding a persistent worker to the client lifecycle.
MAX_STATUS_BATCH_SIZE = 100
STATUS_BATCH_WINDOW_SECONDS = 0.025

# Creation status-poll pacing. Sandbox creation is polled with exponential
# backoff plus jitter rather than at a fixed interval
CREATION_POLL_INITIAL_DELAY = 1.0
CREATION_POLL_MAX_DELAY = 10.0
CREATION_POLL_BACKOFF_FACTOR = 1.5
# Fraction of the computed delay applied as +/- jitter.
CREATION_POLL_JITTER = 0.25
# Legacy fixed schedule (1s for the first 5 polls, then 2s) that `max_attempts`
# used to describe. Retained only to derive the same wall-clock budget.
_LEGACY_FAST_POLLS = 5


class _SyncRequestBatcher(Generic[_BatchKey, _BatchValue]):
    """Coalesce concurrent sync lookups into bounded calls."""

    def __init__(
        self,
        fetch: Callable[[List[_BatchKey]], Dict[_BatchKey, _BatchValue | _BatchItemError]],
    ) -> None:
        self._fetch = fetch
        self._lock = threading.Lock()
        self._pending: Dict[_BatchKey, List[Future[_BatchValue]]] = {}
        self._dispatching = False

    def get(self, key: _BatchKey) -> _BatchValue:
        """Return one result, sharing a batch with concurrent callers."""
        future: Future[_BatchValue] = Future()
        with self._lock:
            self._pending.setdefault(key, []).append(future)
            leader = not self._dispatching
            if leader:
                self._dispatching = True

        if leader:
            time.sleep(STATUS_BATCH_WINDOW_SECONDS)
            self._dispatch()

        return future.result()

    def _dispatch(self) -> None:
        """Drain all currently pending lookups in bounded chunks."""
        while True:
            with self._lock:
                keys = list(self._pending)[:MAX_STATUS_BATCH_SIZE]
                if not keys:
                    self._dispatching = False
                    return
                waiters = {key: self._pending.pop(key) for key in keys}

            try:
                results = self._fetch(keys)
            except Exception as exc:
                for key_waiters in waiters.values():
                    for waiter in key_waiters:
                        waiter.set_exception(exc)
                continue

            for key, key_waiters in waiters.items():
                if key not in results:
                    exc = APIError(f"Batch status response omitted {key!r}")
                    for waiter in key_waiters:
                        waiter.set_exception(exc)
                    continue
                result = results[key]
                if isinstance(result, _BatchItemError):
                    for waiter in key_waiters:
                        waiter.set_exception(result.error)
                    continue
                for waiter in key_waiters:
                    waiter.set_result(result)


class _AsyncRequestBatcher(Generic[_BatchKey, _BatchValue]):
    """Coalesce concurrent async lookups into bounded calls."""

    def __init__(
        self,
        fetch: Callable[
            [List[_BatchKey]],
            Awaitable[Dict[_BatchKey, _BatchValue | _BatchItemError]],
        ],
    ) -> None:
        self._fetch = fetch
        self._pending: Dict[_BatchKey, List[asyncio.Future[_BatchValue]]] = {}
        self._dispatch_task: Optional[asyncio.Task[None]] = None

    async def get(self, key: _BatchKey) -> _BatchValue:
        """Return one result, sharing a batch with concurrent callers."""
        future = asyncio.get_running_loop().create_future()
        self._pending.setdefault(key, []).append(future)
        if self._dispatch_task is None:
            self._dispatch_task = asyncio.create_task(self._dispatch())
        return await future

    async def _dispatch(self) -> None:
        """Drain all currently pending lookups in bounded chunks."""
        await asyncio.sleep(STATUS_BATCH_WINDOW_SECONDS)
        try:
            while self._pending:
                keys = list(self._pending)[:MAX_STATUS_BATCH_SIZE]
                waiters = {key: self._pending.pop(key) for key in keys}
                try:
                    results = await self._fetch(keys)
                except Exception as exc:
                    for key_waiters in waiters.values():
                        for waiter in key_waiters:
                            if not waiter.done():
                                waiter.set_exception(exc)
                    continue

                for key, key_waiters in waiters.items():
                    if key not in results:
                        exc = APIError(f"Batch status response omitted {key!r}")
                        for waiter in key_waiters:
                            if not waiter.done():
                                waiter.set_exception(exc)
                        continue
                    result = results[key]
                    if isinstance(result, _BatchItemError):
                        for waiter in key_waiters:
                            if not waiter.done():
                                waiter.set_exception(result.error)
                        continue
                    for waiter in key_waiters:
                        if not waiter.done():
                            waiter.set_result(result)
        finally:
            self._dispatch_task = None
            if self._pending:
                self._dispatch_task = asyncio.create_task(self._dispatch())


def _creation_poll_delay(poll_index: int) -> float:
    """Jittered exponential backoff delay for the Nth creation status poll."""
    delay = min(
        CREATION_POLL_INITIAL_DELAY * (CREATION_POLL_BACKOFF_FACTOR**poll_index),
        CREATION_POLL_MAX_DELAY,
    )
    jitter = delay * CREATION_POLL_JITTER
    return max(0.0, delay + random.uniform(-jitter, jitter))


def _creation_timeout_seconds(max_attempts: int) -> float:
    """Wall-clock budget the legacy fixed-interval schedule gave `max_attempts`.

    Backoff means attempts no longer map 1:1 to elapsed time, so the budget is
    derived once here and enforced as a deadline. This keeps the effective
    timeout of every existing caller unchanged.
    """
    if max_attempts <= _LEGACY_FAST_POLLS:
        return float(max_attempts)
    return float(_LEGACY_FAST_POLLS + (max_attempts - _LEGACY_FAST_POLLS) * 2)


def _is_retryable_gateway_error(exc: BaseException) -> bool:
    """Check if an exception is retryable for idempotent gateway requests."""
    if isinstance(exc, GATEWAY_IDEMPOTENT_RETRYABLE_EXCEPTIONS):
        return True
    if (
        isinstance(exc, httpx.HTTPStatusError)
        and exc.response.status_code in RETRYABLE_5XX_STATUSES
    ):
        if _is_gateway_sandbox_not_found(exc.response):
            return False
        return True
    return False


def _is_retryable_read_file_error(exc: BaseException) -> bool:
    """Check if an exception is retryable for read-file gateway requests."""
    if isinstance(exc, READ_FILE_RETRYABLE_EXCEPTIONS):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        if status == 408 or status in RETRYABLE_5XX_STATUSES:
            if _is_gateway_sandbox_not_found(exc.response):
                return False
            return True
    return False


# Retry decorator for idempotent gateway requests (connection errors, ReadError,
# and 5xx responses). Safe for GET/HEAD/PUT/DELETE since duplicate requests are no-ops.
_gateway_retry = retry(
    retry=retry_if_exception(_is_retryable_gateway_error),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=1, max=30),
    reraise=True,
)

# Retry decorator for non-idempotent gateway requests (connection errors only —
# ReadError and 5xx both imply the server received/processed the request, so
# retrying POSTs on those risks duplicate side effects).
_gateway_post_retry = retry(
    retry=retry_if_exception_type(GATEWAY_CONNECTION_RETRYABLE_EXCEPTIONS),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=1, max=30),
    reraise=True,
)

# Retry decorator for read-file requests. read_file is idempotent and often used
# as a polling primitive, so transient read/connect/pool timeouts should not fail
# the operation on the first missed gateway response.
_read_file_retry = retry(
    retry=retry_if_exception(_is_retryable_read_file_error),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=1, max=30),
    reraise=True,
)


_ENV_VAR_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _build_user_agent() -> str:
    """Build User-Agent string for prime-sandboxes"""
    from prime_sandboxes import __version__

    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    return f"prime-sandboxes/{__version__} python/{python_version}"


def _validate_env_key(key: str) -> str:
    """Ensure environment variable keys are valid shell identifiers."""
    if not _ENV_VAR_PATTERN.fullmatch(key):
        raise ValueError(f"Invalid environment variable name: {key!r}")
    return key


def _validate_unique_batch_values(values: List[str], field_name: str) -> None:
    """Validate the shared bounded, non-empty, unique batch contract."""
    if not values or len(values) > MAX_STATUS_BATCH_SIZE:
        raise ValueError(f"{field_name} must contain between 1 and {MAX_STATUS_BATCH_SIZE} entries")
    if len(values) != len(set(values)):
        raise ValueError(f"{field_name} must be unique")


def _canonical_background_job(sandbox_id: str, job_id: str) -> BackgroundJob:
    """Build the canonical SDK background-job handle for a VM batch lookup."""
    return BackgroundJob(
        job_id=job_id,
        sandbox_id=sandbox_id,
        stdout_log_file=f"/tmp/job_{job_id}.stdout.log",
        stderr_log_file=f"/tmp/job_{job_id}.stderr.log",
        exit_file=f"/tmp/job_{job_id}.exit",
    )


def _validate_background_job_batch(jobs: List[BackgroundJob]) -> None:
    """Validate VM batching is limited to canonical SDK job handles."""
    if not jobs or len(jobs) > MAX_STATUS_BATCH_SIZE:
        raise ValueError(f"jobs must contain between 1 and {MAX_STATUS_BATCH_SIZE} entries")

    keys = []
    for job in jobs:
        if not re.fullmatch(r"[0-9A-Fa-f]{8}", job.job_id):
            raise ValueError(f"Invalid background job ID: {job.job_id}")
        if job != _canonical_background_job(job.sandbox_id, job.job_id):
            raise ValueError(
                "Batch status requires an unmodified BackgroundJob returned by "
                "start_background_job()."
            )
        keys.append((job.sandbox_id, job.job_id))
    if len(keys) != len(set(keys)):
        raise ValueError("jobs must be unique")


def _sandbox_to_status_snapshot(sandbox: Sandbox) -> SandboxStatusSnapshot:
    """Convert a legacy full-sandbox lookup into the lightweight batch shape."""
    return SandboxStatusSnapshot(
        sandbox_id=sandbox.id,
        status=sandbox.status,
        error_type=sandbox.error_type,
        error_message=sandbox.error_message,
        pending_image_build_id=sandbox.pending_image_build_id,
    )


def _build_terminated_message(command: str, ctx: dict) -> str:
    """Build helpful error message for terminated sandbox."""
    cmd_preview = command[:50] + "..." if len(command) > 50 else command
    parts = [f"Command '{cmd_preview}' failed: sandbox is no longer running."]

    error_type = ctx.get("error_type")
    error_message = ctx.get("error_message")
    status = ctx.get("status")

    if error_type == "OOM_KILLED":
        parts.append("The sandbox was terminated due to out-of-memory (OOM).")
        parts.append("Consider requesting more memory or optimizing memory usage.")
    elif error_type == "TIMEOUT":
        parts.append("The sandbox exceeded its maximum runtime and was terminated.")
    elif error_type == "IMAGE_PULL_FAILED":
        parts.append("The sandbox failed to start due to image pull failure.")
    elif status == "TERMINATED":
        parts.append("The sandbox was terminated.")

    if error_message:
        parts.append(f"Details: {error_message}")

    return " ".join(parts)


def _is_gateway_sandbox_not_found(response: Optional[httpx.Response]) -> bool:
    """Return True when gateway indicates target sandbox no longer exists."""
    if response is None or response.status_code != 502:
        return False

    try:
        body = response.json()
    except Exception:
        return False

    if not isinstance(body, dict):
        return False

    return body.get("error") == "sandbox_not_found"


def _raise_not_running_error(
    sandbox_id: str,
    ctx: dict,
    command: str | None = None,
    cause: BaseException | None = None,
) -> NoReturn:
    """Raise appropriate SandboxNotRunningError subclass based on error_type."""
    error_type = ctx.get("error_type")
    status = ctx.get("status")

    if command:
        message = _build_terminated_message(command, ctx)
    elif ctx.get("error_message"):
        message = f"Sandbox {sandbox_id} failed ({error_type}): {ctx['error_message']}"
    else:
        message = None

    kwargs = {"command": command, "message": message}
    if error_type == "OOM_KILLED":
        exc = SandboxOOMError(sandbox_id, status, error_type, **kwargs)
    elif error_type == "TIMEOUT":
        exc = SandboxTimeoutError(sandbox_id, status, error_type, **kwargs)
    elif error_type == "IMAGE_PULL_FAILED":
        exc = SandboxImagePullError(sandbox_id, status, error_type, **kwargs)
    else:
        exc = SandboxNotRunningError(sandbox_id, status, error_type, **kwargs)

    if cause:
        raise exc from cause
    raise exc


def _auth_refresh_cutoff(auth_info: Dict[str, Any]) -> datetime:
    """Moment at which cached auth stops being usable (expiry minus margin)."""
    expires_at_str = auth_info["expires_at"].replace("Z", "+00:00")
    expires_at = datetime.fromisoformat(expires_at_str)
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    return expires_at - timedelta(seconds=AUTH_REFRESH_MARGIN_SECONDS)


def _load_auth_cache(cache_file: Any) -> tuple[Dict[str, Any], bool]:
    """Load auth cache from file and clean expired entries."""
    try:
        if cache_file.exists():
            with open(cache_file, "r") as f:
                cache = json.load(f)
            cleaned_cache = {}
            now = datetime.now(timezone.utc)
            for sandbox_id, auth_info in cache.items():
                try:
                    if now < _auth_refresh_cutoff(auth_info):
                        cleaned_cache[sandbox_id] = auth_info
                except Exception:
                    pass

            return cleaned_cache, len(cleaned_cache) != len(cache)
    except Exception:
        pass
    return {}, False


def _check_cached_auth(cache: Dict[str, Any], sandbox_id: str) -> Optional[Dict[str, Any]]:
    """Return a copy of cached auth if still valid, else evict and return None."""
    if sandbox_id in cache:
        auth_info = cache[sandbox_id]
        if datetime.now(timezone.utc) < _auth_refresh_cutoff(auth_info):
            return dict(auth_info)
        del cache[sandbox_id]
    return None


class SandboxAuthCache:
    """Thread-safe auth cache for sync SandboxClient."""

    def __init__(self, cache_file_path: Any, client: Any) -> None:
        self._cache_file = cache_file_path
        self.client = client
        self._lock = threading.Lock()
        self._inflight: Dict[str, threading.Event] = {}
        self._auth_cache, needs_save = _load_auth_cache(self._cache_file)
        if needs_save:
            self._save_cache()

    def _save_cache(self) -> None:
        """Write current in-memory cache to disk. Must be called under self._lock."""
        try:
            self._cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self._cache_file, "w") as f:
                json.dump(self._auth_cache, f)
        except Exception:
            pass

    def get_or_refresh(self, sandbox_id: str) -> Dict[str, Any]:
        """Get cached auth or fetch a new token.

        Coalesces concurrent requests for the same sandbox_id so only one
        auth POST is issued while others wait for the result.
        """
        while True:
            with self._lock:
                cached = _check_cached_auth(self._auth_cache, sandbox_id)
                if cached:
                    return cached

                if sandbox_id in self._inflight:
                    event = self._inflight[sandbox_id]
                else:
                    event = None
                    self._inflight[sandbox_id] = threading.Event()

            if event is not None:
                event.wait()
                with self._lock:
                    cached = _check_cached_auth(self._auth_cache, sandbox_id)
                    if cached:
                        return cached
                continue

            try:
                response = self.client.request(
                    "POST",
                    f"/sandbox/{sandbox_id}/auth",
                    idempotent_post=True,
                )
                with self._lock:
                    self._auth_cache[sandbox_id] = response
                    self._save_cache()
                return dict(response)
            finally:
                with self._lock:
                    ev = self._inflight.pop(sandbox_id, None)
                if ev is not None:
                    ev.set()

    def is_vm(self, sandbox_id: str) -> bool:
        """Return True if sandbox is VM-backed, cached alongside auth token data."""
        with self._lock:
            cached = _check_cached_auth(self._auth_cache, sandbox_id)
            if cached and isinstance(cached.get("is_vm"), bool):
                return bool(cached["is_vm"])

        sandbox_data = self.client.request("GET", f"/sandbox/{sandbox_id}")
        sandbox = Sandbox.model_validate(sandbox_data)
        is_vm = sandbox.vm

        with self._lock:
            if sandbox_id in self._auth_cache:
                self._auth_cache[sandbox_id]["is_vm"] = is_vm
                self._save_cache()

        return is_vm

    def set(self, sandbox_id: str, auth_info: Dict[str, Any]) -> None:
        with self._lock:
            self._auth_cache[sandbox_id] = auth_info
            self._save_cache()

    def invalidate(self, sandbox_id: str) -> None:
        """Drop cached auth for one sandbox (e.g. after a gateway 401)."""
        with self._lock:
            if self._auth_cache.pop(sandbox_id, None) is not None:
                self._save_cache()

    def clear(self) -> None:
        with self._lock:
            self._auth_cache = {}
            self._save_cache()


class AsyncSandboxAuthCache:
    """Async auth cache for AsyncSandboxClient."""

    def __init__(self, cache_file_path: Any, client: Any) -> None:
        self._cache_file = cache_file_path
        self.client = client
        self._lock = asyncio.Lock()
        self._inflight: Dict[str, asyncio.Event] = {}
        self._auth_cache: Dict[str, Any] = {}
        self._loaded = False

    async def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._auth_cache, needs_save = await asyncio.to_thread(_load_auth_cache, self._cache_file)
        self._loaded = True
        if needs_save:
            await self._save_cache()

    async def _save_cache(self) -> None:
        """Write current in-memory cache to disk. Must be called under self._lock."""
        data = json.dumps(self._auth_cache)

        def _write() -> None:
            self._cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self._cache_file, "w") as f:
                f.write(data)

        try:
            await asyncio.to_thread(_write)
        except Exception:
            pass

    async def get_or_refresh(self, sandbox_id: str) -> Dict[str, Any]:
        """Get cached auth or fetch a new token.

        Coalesces concurrent requests for the same sandbox_id so only one
        auth POST is issued while others await the result.
        """
        while True:
            async with self._lock:
                await self._ensure_loaded()
                cached = _check_cached_auth(self._auth_cache, sandbox_id)
                if cached:
                    return cached

                if sandbox_id in self._inflight:
                    event = self._inflight[sandbox_id]
                else:
                    event = None
                    self._inflight[sandbox_id] = asyncio.Event()

            if event is not None:
                await event.wait()
                async with self._lock:
                    cached = _check_cached_auth(self._auth_cache, sandbox_id)
                    if cached:
                        return cached
                continue

            try:
                response = await self.client.request(
                    "POST",
                    f"/sandbox/{sandbox_id}/auth",
                    idempotent_post=True,
                )
                async with self._lock:
                    self._auth_cache[sandbox_id] = response
                    await self._save_cache()
                return dict(response)
            finally:
                async with self._lock:
                    ev = self._inflight.pop(sandbox_id, None)
                if ev is not None:
                    ev.set()

    async def is_vm(self, sandbox_id: str) -> bool:
        """Return True if sandbox is VM-backed, cached alongside auth token data."""
        async with self._lock:
            await self._ensure_loaded()
            cached = _check_cached_auth(self._auth_cache, sandbox_id)
            if cached and isinstance(cached.get("is_vm"), bool):
                return bool(cached["is_vm"])

        sandbox_data = await self.client.request("GET", f"/sandbox/{sandbox_id}")
        sandbox = Sandbox.model_validate(sandbox_data)
        is_vm = sandbox.vm

        async with self._lock:
            if sandbox_id in self._auth_cache:
                self._auth_cache[sandbox_id]["is_vm"] = is_vm
                await self._save_cache()

        return is_vm

    async def set(self, sandbox_id: str, auth_info: Dict[str, Any]) -> None:
        async with self._lock:
            await self._ensure_loaded()
            self._auth_cache[sandbox_id] = auth_info
            await self._save_cache()

    async def invalidate(self, sandbox_id: str) -> None:
        """Drop cached auth for one sandbox (e.g. after a gateway 401)."""
        async with self._lock:
            await self._ensure_loaded()
            if self._auth_cache.pop(sandbox_id, None) is not None:
                await self._save_cache()

    async def clear(self) -> None:
        async with self._lock:
            self._auth_cache = {}
            self._loaded = True
            await self._save_cache()


def _is_waiting_for_image_build(sandbox: Sandbox | SandboxStatusSnapshot) -> bool:
    return sandbox.status == "PENDING" and bool(getattr(sandbox, "pending_image_build_id", None))


class SandboxClient:
    """Client for sandbox API operations"""

    def __init__(self, api_client: APIClient):
        self.client = api_client
        self._auth_cache = SandboxAuthCache(
            self.client.config.config_dir / "sandbox_auth_cache.json",
            self.client,
        )
        self._sandbox_status_batcher = _SyncRequestBatcher(self._fetch_sandbox_statuses)
        self._background_job_status_batcher = _SyncRequestBatcher(
            self._fetch_background_job_statuses
        )
        self._sandbox_status_batch_supported: Optional[bool] = None
        self._background_job_status_batch_supported: Optional[bool] = None

    @staticmethod
    @_gateway_post_retry
    def _gateway_post(
        url: str,
        headers: Dict[str, str],
        timeout: float,
        json: Optional[Dict[str, Any]] = None,
        files: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> httpx.Response:
        """Make a POST request to the gateway with retry on connection errors only."""
        with httpx.Client(timeout=timeout) as client:
            return client.post(url, json=json, files=files, params=params, headers=headers)

    @staticmethod
    @_gateway_retry
    def _gateway_get(
        url: str,
        headers: Dict[str, str],
        params: Dict[str, Any],
        timeout: float,
    ) -> httpx.Response:
        """Make a GET request to the gateway with retry on transient errors."""
        with httpx.Client(timeout=timeout) as client:
            response = client.get(url, params=params, headers=headers)
        if response.status_code in RETRYABLE_5XX_STATUSES:
            response.raise_for_status()
        return response

    @staticmethod
    @_read_file_retry
    def _gateway_read_file_get(
        url: str,
        headers: Dict[str, str],
        params: Dict[str, Any],
        timeout: float,
    ) -> httpx.Response:
        """Make a read-file GET request to the gateway with read-timeout retries."""
        with httpx.Client(timeout=timeout) as client:
            response = client.get(url, params=params, headers=headers)
        if response.status_code == 408 or response.status_code in RETRYABLE_5XX_STATUSES:
            response.raise_for_status()
        return response

    def _is_sandbox_reachable(self, sandbox_id: str, timeout: int = 10) -> bool:
        """Test if a sandbox is reachable by executing a simple echo command"""
        try:
            self.execute_command(sandbox_id, "echo 'sandbox ready'", timeout=timeout)
            return True
        except Exception:
            return False

    def _get_sandbox_error_context(self, sandbox_id: str) -> dict:
        """Fetch sandbox error context from the lightweight server endpoint."""
        try:
            response = self.client.request("GET", f"/sandbox/{sandbox_id}/error-context")
            return {
                "status": response.get("status"),
                "error_type": response.get("errorType") or response.get("error_type"),
                "error_message": response.get("errorMessage") or response.get("error_message"),
            }
        except Exception:
            return {"status": None, "error_type": None, "error_message": None}

    def _should_retry_409(
        self,
        sandbox_id: str,
        error: httpx.HTTPStatusError,
        attempt: int,
        command: Optional[str] = None,
    ) -> bool:
        """Check if a 409 error should be retried.

        Returns True and sleeps if should retry, raises appropriate error otherwise.
        """
        ctx = self._get_sandbox_error_context(sandbox_id)
        if ctx["status"] == "RUNNING":
            if attempt < MAX_409_RETRIES - 1:
                time.sleep(RETRY_409_BASE_DELAY * (2**attempt))
                return True
            raise APIError(
                f"Sandbox {sandbox_id} returned 409 after {MAX_409_RETRIES} retries. "
                "This may be a transient DNS or gateway issue. Please retry."
            ) from error
        # Sandbox is not running
        _raise_not_running_error(sandbox_id, ctx, command=command, cause=error)

    def _should_retry_upload_error(self, error: httpx.HTTPStatusError, attempt: int) -> bool:
        """Check if a transient error (408/5xx) on an idempotent upload should be retried."""
        status = error.response.status_code
        if status != 408 and status not in RETRYABLE_5XX_STATUSES:
            return False
        if _is_gateway_sandbox_not_found(error.response):
            return False
        if attempt < MAX_409_RETRIES - 1:
            time.sleep(RETRY_409_BASE_DELAY * (2**attempt))
            return True
        return False

    def _should_retry_401(self, sandbox_id: str, reauthed: bool) -> bool:
        """Check if a gateway 401 should be retried with fresh auth."""
        if reauthed:
            return False
        self._auth_cache.invalidate(sandbox_id)
        return True

    def clear_auth_cache(self) -> None:
        """Clear all cached auth tokens"""
        self._auth_cache.clear()

    def is_vm(self, sandbox_id: str) -> bool:
        """Return True if the sandbox is VM-backed.

        Uses the internal auth cache when available and falls back to a
        ``GET /sandbox/<id>`` lookup on a cold cache. The result is cached
        alongside the auth token so subsequent calls are essentially free.
        """
        return self._auth_cache.is_vm(sandbox_id)

    def _guard_vm_unsupported(self, sandbox_id: str, feature_name: str) -> None:
        """Raise APIError if the operation is not supported on VM sandboxes.

        Mirrors the CLI behavior of short-circuiting operations the backend
        does not currently support for VM-backed sandboxes, so callers fail
        fast with a clear message instead of an opaque gateway error.
        """
        if self._auth_cache.is_vm(sandbox_id):
            raise APIError(f"{feature_name} is not yet supported for VM sandboxes.")

    def create(self, request: CreateSandboxRequest) -> Sandbox:
        """Create a new sandbox"""
        payload = request.model_dump(by_alias=False, exclude_none=True)
        # Auto-populate team_id from config if not specified
        if request.team_id is None and self.client.config.team_id is not None:
            payload["team_id"] = self.client.config.team_id
        payload["idempotency_key"] = request.idempotency_key or uuid.uuid4().hex

        response = self.client.request(
            "POST",
            "/sandbox",
            json=payload,
            idempotent_post=True,
        )
        return Sandbox.model_validate(response)

    def list(
        self,
        team_id: Optional[str] = None,
        status: Optional[str] = None,
        labels: Optional[List[str]] = None,
        page: int = 1,
        per_page: int = 50,
        exclude_terminated: Optional[bool] = None,
        user_id: Optional[str] = None,
    ) -> SandboxListResponse:
        """List sandboxes"""
        # Auto-populate team_id from config if not specified
        if team_id is None:
            team_id = self.client.config.team_id

        params: Dict[str, Any] = {"page": page, "per_page": per_page}
        if team_id:
            params["team_id"] = team_id
        if user_id:
            params["user_id"] = user_id
        if status:
            params["status"] = status
        if labels:
            params["labels"] = labels
        if exclude_terminated is not None:
            params["is_active"] = exclude_terminated

        response = self.client.request("GET", "/sandbox", params=params)
        return SandboxListResponse.model_validate(response)

    def get(self, sandbox_id: str) -> Sandbox:
        """Get a specific sandbox"""
        response = self.client.request("GET", f"/sandbox/{sandbox_id}")
        return Sandbox.model_validate(response)

    def get_sandbox_statuses(self, sandbox_ids: List[str]) -> BatchSandboxStatusResponse:
        """Get lightweight lifecycle state for up to 100 sandboxes."""
        _validate_unique_batch_values(sandbox_ids, "sandbox_ids")
        if self._sandbox_status_batch_supported is False:
            raise BatchStatusUnsupportedError(
                "The platform does not support batch sandbox status lookups."
            )
        try:
            response = self.client.request(
                "POST",
                "/sandbox/status:batchGet",
                json={"sandbox_ids": sandbox_ids},
                idempotent_post=True,
            )
        except APIError as exc:
            if "HTTP 404" in str(exc) or "HTTP 405" in str(exc):
                self._sandbox_status_batch_supported = False
                raise BatchStatusUnsupportedError(
                    "The platform does not support batch sandbox status lookups."
                ) from exc
            raise
        self._sandbox_status_batch_supported = True
        return BatchSandboxStatusResponse.model_validate(response)

    def _fetch_sandbox_statuses(
        self, sandbox_ids: List[str]
    ) -> Dict[str, SandboxStatusSnapshot | _BatchItemError]:
        """Fetch one coalesced lifecycle batch for concurrent waiters."""
        try:
            response = self.get_sandbox_statuses(sandbox_ids)
        except BatchStatusUnsupportedError:
            results: Dict[str, SandboxStatusSnapshot | _BatchItemError] = {}
            for sandbox_id in sandbox_ids:
                try:
                    results[sandbox_id] = _sandbox_to_status_snapshot(self.get(sandbox_id))
                except Exception as exc:
                    results[sandbox_id] = _BatchItemError(exc)
            return results

        results: Dict[str, SandboxStatusSnapshot | _BatchItemError] = {
            snapshot.sandbox_id: snapshot for snapshot in response.statuses
        }
        for error in response.errors:
            results[error.sandbox_id] = _BatchItemError(
                APIError(
                    f"Sandbox status lookup failed for {error.sandbox_id}: "
                    f"{error.code}: {error.message}"
                )
            )
        return results

    def delete(self, sandbox_id: str) -> Dict[str, Any]:
        """Delete a sandbox"""
        response = self.client.request("DELETE", f"/sandbox/{sandbox_id}")
        return response

    def get_network(self, sandbox_id: str) -> EgressPolicyStatus:
        """Get the desired and applied network rules of a VM sandbox."""
        response = self.client.request("GET", f"/sandbox/{sandbox_id}/egress-policy")
        return EgressPolicyStatus.model_validate(response)

    def set_network(
        self,
        sandbox_id: str,
        *,
        allow: Optional[List[str]] = None,
        deny: Optional[List[str]] = None,
    ) -> EgressPolicyStatus:
        """Replace the network rules of a running VM sandbox.

        Exactly one of ``allow`` or ``deny`` must be provided. Each call is a
        complete replacement, never a merge; use ``["*"]`` to allow or deny
        all destinations. New connections follow the replacement once
        ``applied`` is true; established flows are not revoked.
        """
        payload = _network_update_payload(allow, deny)

        response = self.client.request(
            "PUT",
            f"/sandbox/{sandbox_id}/egress-policy",
            json=payload,
        )
        return EgressPolicyStatus.model_validate(response)

    def bulk_delete(
        self,
        sandbox_ids: Optional[List[str]] = None,
        labels: Optional[List[str]] = None,
        team_id: Optional[str] = None,
        user_id: Optional[str] = None,
        all_users: bool = False,
    ) -> BulkDeleteSandboxResponse:
        """Bulk delete multiple sandboxes."""
        request = BulkDeleteSandboxRequest(
            sandbox_ids=sandbox_ids,
            labels=labels,
            team_id=team_id,
            user_id=user_id,
            all_users=all_users,
        )
        payload = request.model_dump(by_alias=False, exclude_none=True)
        response = self.client.request(
            "DELETE",
            "/sandbox",
            json=payload,
        )
        return BulkDeleteSandboxResponse.model_validate(response)

    def get_logs(self, sandbox_id: str) -> str:
        """Get sandbox logs via backend"""
        response = self.client.request("GET", f"/sandbox/{sandbox_id}/logs")
        logs_response = SandboxLogsResponse.model_validate(response)
        return logs_response.logs

    def execute_command(
        self,
        sandbox_id: str,
        command: str,
        working_dir: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None,
        user: Optional[str] = None,
    ) -> CommandResponse:
        """Execute command directly via gateway."""
        self._auth_cache.get_or_refresh(sandbox_id)

        if self._auth_cache.is_vm(sandbox_id):
            if user is not None:
                raise ValueError(
                    "The 'user' parameter is only supported for container sandboxes, "
                    "not VM sandboxes."
                )
            return self._execute_command_connect_rpc(
                sandbox_id=sandbox_id,
                command=command,
                working_dir=working_dir,
                env=env,
                timeout=timeout,
            )

        return self._execute_command_rest(
            sandbox_id=sandbox_id,
            command=command,
            working_dir=working_dir,
            env=env,
            timeout=timeout,
            user=user,
        )

    def _execute_command_connect_rpc(
        self,
        sandbox_id: str,
        command: str,
        working_dir: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None,
    ) -> CommandResponse:
        effective_timeout = timeout if timeout is not None else 300
        request = build_command_session_start_request(command, working_dir, env)

        reauthed = False
        while True:
            auth = self._auth_cache.get_or_refresh(sandbox_id)
            gateway_url = auth["gateway_url"].rstrip("/")
            base_url = f"{gateway_url}/{auth['user_ns']}/{auth['job_id']}"
            headers = {"Authorization": f"Bearer {auth['token']}"}
            stdout_parts: List[str] = []
            stderr_parts: List[str] = []
            exit_code: Optional[int] = None
            stream_started = False

            rpc_client = ConnectClientSync(base_url)
            try:
                stream = rpc_client.execute_server_stream(
                    request=request,
                    method=COMMAND_SESSION_START_RPC_METHOD,
                    headers=headers,
                    timeout_ms=effective_timeout * 1000,
                )
                for event in stream:
                    stream_started = True
                    event_exit_code = collect_command_session_start_event(
                        event,
                        stdout_parts,
                        stderr_parts,
                    )
                    if event_exit_code is not None:
                        exit_code = event_exit_code

                if exit_code is None:
                    raise APIError("Command stream ended without exit code")

                return CommandResponse(
                    stdout="".join(stdout_parts),
                    stderr="".join(stderr_parts),
                    exit_code=exit_code,
                )
            except ConnectError as e:
                # Only re-auth before any stream event.
                if (
                    e.code == Code.UNAUTHENTICATED
                    and not stream_started
                    and self._should_retry_401(sandbox_id, reauthed)
                ):
                    reauthed = True
                    continue

                if e.code == Code.DEADLINE_EXCEEDED:
                    ctx = self._get_sandbox_error_context(sandbox_id)
                    if ctx["status"] in ("TERMINATED", "ERROR", "TIMEOUT"):
                        _raise_not_running_error(sandbox_id, ctx, command=command, cause=e)
                    raise CommandTimeoutError(sandbox_id, command, effective_timeout) from e

                if e.code == Code.NOT_FOUND:
                    ctx = self._get_sandbox_error_context(sandbox_id)
                    ctx["status"] = "TERMINATED"
                    if not ctx.get("error_type"):
                        ctx["error_type"] = "SANDBOX_NOT_FOUND"
                    if not ctx.get("error_message"):
                        ctx["error_message"] = (
                            "Sandbox is no longer present on the runtime node. "
                            "Please create a new sandbox."
                        )
                    _raise_not_running_error(sandbox_id, ctx, command=command, cause=e)

                raise APIError(f"Connect RPC failed ({e.code.value}): {e.message}") from e
            except APIError:
                raise
            except Exception as e:
                raise APIError(f"Request failed: {e.__class__.__name__}: {e}") from e
            finally:
                rpc_client.close()

    def _execute_command_rest(
        self,
        sandbox_id: str,
        command: str,
        working_dir: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None,
        user: Optional[str] = None,
    ) -> CommandResponse:
        effective_timeout = timeout if timeout is not None else 300

        payload = {
            "command": command,
            "working_dir": working_dir,
            "env": env or {},
            "sandbox_id": sandbox_id,
            "timeout": effective_timeout,
        }
        if user is not None:
            payload["user"] = user

        reauthed = False
        attempt = 0
        for _ in range(MAX_GATEWAY_ATTEMPTS):
            auth = self._auth_cache.get_or_refresh(sandbox_id)
            gateway_url = auth["gateway_url"].rstrip("/")
            url = f"{gateway_url}/{auth['user_ns']}/{auth['job_id']}/exec"
            headers = {"Authorization": f"Bearer {auth['token']}"}
            try:
                # The + 5 accounts for connection creation and closing. Prevents any command
                # running close to its `effective_timeout` from being killed prematurely
                client_timeout = effective_timeout + 5
                response = self._gateway_post(
                    url, headers=headers, timeout=client_timeout, json=payload
                )
                response.raise_for_status()
                return CommandResponse.model_validate(response.json())
            except httpx.TimeoutException as e:
                ctx = self._get_sandbox_error_context(sandbox_id)
                if ctx["status"] in ("TERMINATED", "ERROR", "TIMEOUT"):
                    _raise_not_running_error(sandbox_id, ctx, command=command, cause=e)
                raise CommandTimeoutError(sandbox_id, command, effective_timeout) from e
            except httpx.HTTPStatusError as e:
                resp = getattr(e, "response", None)
                status = getattr(resp, "status_code", "?")

                if status == 401 and self._should_retry_401(sandbox_id, reauthed):
                    reauthed = True
                    continue

                if status == 502 and _is_gateway_sandbox_not_found(resp):
                    ctx = self._get_sandbox_error_context(sandbox_id)
                    ctx["status"] = "TERMINATED"
                    if not ctx.get("error_type"):
                        ctx["error_type"] = "SANDBOX_NOT_FOUND"
                    if not ctx.get("error_message"):
                        ctx["error_message"] = (
                            "Sandbox is no longer present on the runtime node. "
                            "Please create a new sandbox."
                        )
                    _raise_not_running_error(sandbox_id, ctx, command=command, cause=e)

                if status == 409:
                    if self._should_retry_409(sandbox_id, e, attempt, command=command):
                        attempt += 1
                        continue

                if status == 408:
                    ctx = self._get_sandbox_error_context(sandbox_id)
                    if ctx["status"] in ("TERMINATED", "ERROR", "TIMEOUT"):
                        _raise_not_running_error(sandbox_id, ctx, command=command, cause=e)
                    raise CommandTimeoutError(sandbox_id, command, effective_timeout) from e

                req = getattr(e, "request", None)
                method = getattr(req, "method", "?")
                u = getattr(req, "url", "?")
                text = getattr(resp, "text", "")
                raise APIError(f"HTTP {status} {method} {u}: {text}") from e
            except httpx.RequestError as e:
                req = getattr(e, "request", None)
                method = getattr(req, "method", "?")
                u = getattr(req, "url", "?")
                raise APIError(
                    f"Request failed: {e.__class__.__name__} at {method} {u}: {e}"
                ) from e
            except Exception as e:
                raise APIError(f"Request failed: {e.__class__.__name__}: {e}") from e

        raise APIError("Command execution failed after retries")

    def start_background_job(
        self,
        sandbox_id: str,
        command: str,
        working_dir: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        user: Optional[str] = None,
    ) -> BackgroundJob:
        """Start a long-running command in the background.

        Returns immediately with a job handle. Use get_background_job() to check
        status and retrieve results.

        Args:
            sandbox_id: The sandbox ID
            command: Command to execute
            working_dir: Working directory for command execution
            env: Environment variables
            user: Run the job as this user, like ``docker exec -u`` (username or
                numeric UID, optionally USER:GROUP). Container sandboxes only.

        Returns:
            BackgroundJob with job_id and file paths for polling
        """
        job_id = uuid.uuid4().hex[:8]
        stdout_log_file = f"/tmp/job_{job_id}.stdout.log"
        stderr_log_file = f"/tmp/job_{job_id}.stderr.log"
        exit_file = f"/tmp/job_{job_id}.exit"

        env_prefix = ""
        if env:
            exports = []
            for k, v in env.items():
                _validate_env_key(k)
                exports.append(f"export {k}={shlex.quote(v)}")
            env_prefix = "; ".join(exports)
            if env_prefix:
                env_prefix += "; "

        dir_prefix = f"cd {shlex.quote(working_dir)} && " if working_dir else ""
        command_body = f"{env_prefix}{dir_prefix}{command}"
        exit_file_quoted = shlex.quote(exit_file)
        stdout_log_file_quoted = shlex.quote(stdout_log_file)
        stderr_log_file_quoted = shlex.quote(stderr_log_file)
        # Wrap command in subshell so 'exit' terminates the subshell, not the outer shell.
        # This ensures 'echo $?' always runs to capture the exit code.
        sh_command = (
            f"({command_body}) > {stdout_log_file_quoted} 2> {stderr_log_file_quoted}; "
            f"echo $? > {exit_file_quoted}"
        )
        quoted_sh_command = shlex.quote(sh_command)

        # Outer nohup redirects to /dev/null since output goes to log files inside sh -c
        bg_cmd = f"nohup sh -c {quoted_sh_command} < /dev/null > /dev/null 2>&1 &"
        self.execute_command(sandbox_id, bg_cmd, timeout=30, user=user)

        return BackgroundJob(
            job_id=job_id,
            sandbox_id=sandbox_id,
            stdout_log_file=stdout_log_file,
            stderr_log_file=stderr_log_file,
            exit_file=exit_file,
        )

    def get_background_job(
        self,
        sandbox_id: str,
        job: BackgroundJob,
        timeout: Optional[int] = None,
    ) -> BackgroundJobStatus:
        """Check the status of a background job.

        Args:
            sandbox_id: The sandbox ID
            job: The BackgroundJob handle from start_background_job()
            timeout: Optional per-call timeout (in seconds) forwarded to the
                underlying read_file calls. When None, the APIClient default
                applies.

        Returns:
            BackgroundJobStatus with completed flag, and exit_code/stdout if
            done. stdout/stderr hold at most the last JOB_OUTPUT_TAIL_BYTES
            of each stream; the *_truncated flags report dropped output.
        """

        def read_or_empty(path: str) -> str:
            try:
                return self.read_file(sandbox_id, path, timeout=timeout).content
            except SandboxFileNotFoundError:
                return ""

        def read_output_tail(path: str) -> "tuple[str, bool]":
            try:
                response = self.read_file(
                    sandbox_id,
                    path,
                    timeout=timeout,
                    offset=-JOB_OUTPUT_TAIL_BYTES,
                    length=JOB_OUTPUT_TAIL_BYTES,
                )
                # Servers without windowed-read support omit `truncated`.
                return response.content, bool(response.truncated)
            except SandboxFileNotFoundError:
                return "", False

        exit_content = read_or_empty(job.exit_file)
        if not exit_content.strip():
            return BackgroundJobStatus(job_id=job.job_id, completed=False)

        try:
            exit_code = int(exit_content.strip())
        except ValueError:
            return BackgroundJobStatus(job_id=job.job_id, completed=False)

        stdout, stdout_truncated = read_output_tail(job.stdout_log_file)
        stderr, stderr_truncated = read_output_tail(job.stderr_log_file)
        return BackgroundJobStatus(
            job_id=job.job_id,
            completed=True,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            stdout_truncated=stdout_truncated,
            stderr_truncated=stderr_truncated,
        )

    def _request_background_job_status_batch(
        self,
        jobs: List[BackgroundJob],
        timeout: Optional[int] = None,
    ) -> Optional[BatchBackgroundJobStatusResponse]:
        """Return one raw platform batch, or None when the endpoint is unavailable."""
        if self._background_job_status_batch_supported is False:
            return None
        try:
            response = self.client.request(
                "POST",
                "/sandbox/background-jobs/status:batchGet",
                json={
                    "jobs": [{"sandbox_id": job.sandbox_id, "job_id": job.job_id} for job in jobs]
                },
                timeout=timeout if timeout is not None else 30,
                idempotent_post=True,
            )
        except APIError as exc:
            if "HTTP 404" in str(exc) or "HTTP 405" in str(exc):
                self._background_job_status_batch_supported = False
                return None
            raise
        self._background_job_status_batch_supported = True
        return BatchBackgroundJobStatusResponse.model_validate(response)

    def get_background_jobs(
        self,
        jobs: List[BackgroundJob],
        timeout: Optional[int] = None,
    ) -> List[BackgroundJobStatus]:
        """Get ordered status for up to 100 jobs across VM sandboxes.

        One platform request checks all canonical exit-code files. Output is
        fetched with the existing bounded-tail behavior once a job completes.
        Container sandboxes intentionally continue to use get_background_job().
        """
        _validate_background_job_batch(jobs)
        body = self._request_background_job_status_batch(jobs, timeout)
        if body is None:
            return self._get_background_jobs_legacy(jobs, timeout)
        if body.errors:
            details = "; ".join(
                f"{error.sandbox_id}/{error.job_id}: {error.message}" for error in body.errors
            )
            if any(error.code == "NOT_VM" for error in body.errors):
                raise BatchStatusUnsupportedError(details)
            raise APIError(f"Background job batch status failed: {details}")
        runtime_statuses = {(status.sandbox_id, status.job_id): status for status in body.statuses}

        results = []
        for job in jobs:
            runtime_status = runtime_statuses.get((job.sandbox_id, job.job_id))
            if runtime_status is None:
                raise APIError(f"VM batch status response omitted job {job.job_id}")
            if not runtime_status.completed:
                results.append(BackgroundJobStatus(job_id=job.job_id, completed=False))
                continue
            if runtime_status.exit_code is None:
                raise APIError(f"Completed VM background job {job.job_id} omitted exit_code")
            results.append(
                self._get_completed_background_job_output(
                    job.sandbox_id,
                    job,
                    runtime_status.exit_code,
                    timeout,
                )
            )
        return results

    def _get_background_jobs_legacy(
        self,
        jobs: List[BackgroundJob],
        timeout: Optional[int],
    ) -> List[BackgroundJobStatus]:
        """Use VM-only per-job reads when the platform batch API is unavailable."""
        for sandbox_id in dict.fromkeys(job.sandbox_id for job in jobs):
            self._auth_cache.get_or_refresh(sandbox_id)
            if not self._auth_cache.is_vm(sandbox_id):
                raise BatchStatusUnsupportedError(
                    "Batched background job status is only supported for VM sandboxes."
                )
        return [self.get_background_job(job.sandbox_id, job, timeout=timeout) for job in jobs]

    def _get_completed_background_job_output(
        self,
        sandbox_id: str,
        job: BackgroundJob,
        exit_code: int,
        timeout: Optional[int],
    ) -> BackgroundJobStatus:
        """Read bounded stdout and stderr tails for one completed job."""

        def read_output_tail(path: str) -> tuple[str, bool]:
            try:
                response = self.read_file(
                    sandbox_id,
                    path,
                    timeout=timeout,
                    offset=-JOB_OUTPUT_TAIL_BYTES,
                    length=JOB_OUTPUT_TAIL_BYTES,
                )
                return response.content, bool(response.truncated)
            except SandboxFileNotFoundError:
                return "", False

        stdout, stdout_truncated = read_output_tail(job.stdout_log_file)
        stderr, stderr_truncated = read_output_tail(job.stderr_log_file)
        return BackgroundJobStatus(
            job_id=job.job_id,
            completed=True,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            stdout_truncated=stdout_truncated,
            stderr_truncated=stderr_truncated,
        )

    def _fetch_background_job_statuses(
        self, keys: List[tuple[str, str]]
    ) -> Dict[tuple[str, str], BackgroundJobStatus | _BatchItemError]:
        """Fetch one coalesced VM job batch for concurrent run waiters."""
        jobs = [_canonical_background_job(sandbox_id, job_id) for sandbox_id, job_id in keys]
        body = self._request_background_job_status_batch(jobs)
        if body is None:
            results: Dict[tuple[str, str], BackgroundJobStatus | _BatchItemError] = {}
            for job in jobs:
                key = (job.sandbox_id, job.job_id)
                try:
                    results[key] = self._get_background_jobs_legacy([job], None)[0]
                except Exception as exc:
                    results[key] = _BatchItemError(exc)
            return results

        results: Dict[tuple[str, str], BackgroundJobStatus | _BatchItemError] = {}
        for error in body.errors:
            key = (error.sandbox_id, error.job_id)
            details = f"{error.sandbox_id}/{error.job_id}: {error.message}"
            exc = (
                BatchStatusUnsupportedError(details)
                if error.code == "NOT_VM"
                else APIError(f"Background job batch status failed: {details}")
            )
            results[key] = _BatchItemError(exc)

        runtime_statuses = {(status.sandbox_id, status.job_id): status for status in body.statuses}
        for job in jobs:
            key = (job.sandbox_id, job.job_id)
            if key in results:
                continue
            runtime_status = runtime_statuses.get(key)
            if runtime_status is None:
                continue
            if not runtime_status.completed:
                results[key] = BackgroundJobStatus(job_id=job.job_id, completed=False)
                continue
            if runtime_status.exit_code is None:
                results[key] = _BatchItemError(
                    APIError(f"Completed VM background job {job.job_id} omitted exit_code")
                )
                continue
            try:
                results[key] = self._get_completed_background_job_output(
                    job.sandbox_id,
                    job,
                    runtime_status.exit_code,
                    None,
                )
            except Exception as exc:
                results[key] = _BatchItemError(exc)
        return results

    def run_background_job(
        self,
        sandbox_id: str,
        command: str,
        timeout: int = 900,
        working_dir: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        poll_interval: int = 3,
    ) -> BackgroundJobStatus:
        """Run a command in the background and wait for completion.

        Combines start_background_job() + polling into a single call.
        Use this for long-running commands that would exceed HTTP timeouts
        with execute_command().

        Args:
            sandbox_id: The sandbox ID
            command: Command to execute
            timeout: Maximum seconds to wait for completion
            working_dir: Working directory for command execution
            env: Environment variables
            poll_interval: Seconds between status polls

        Returns:
            BackgroundJobStatus with exit_code, stdout, stderr

        Raises:
            CommandTimeoutError: If command doesn't complete within timeout
        """
        job = self.start_background_job(sandbox_id, command, working_dir=working_dir, env=env)
        use_batch_status = self._auth_cache.is_vm(sandbox_id)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if use_batch_status:
                status = self._background_job_status_batcher.get((sandbox_id, job.job_id))
            else:
                status = self.get_background_job(sandbox_id, job)
            if status.completed:
                return status
            time.sleep(poll_interval)
        raise CommandTimeoutError(sandbox_id, command, timeout)

    def wait_for_creation(
        self,
        sandbox_id: str,
        max_attempts: int = 60,
        stability_checks: int = 1,
        image_build_timeout_seconds: int = 3000,
    ) -> None:
        """Wait for sandbox to be running and stable.

        Args:
            sandbox_id: The sandbox ID to wait for
            max_attempts: Defines the wall-clock budget for the wait, expressed
                in polls of the legacy fixed-interval schedule (see
                `_creation_timeout_seconds`). Status polls now back off, so this
                bounds elapsed time rather than the literal number of requests.
                Reaching RUNNING starts a fresh budget of the same size for the
                reachability phase, so a wait that gets that far can take up to
                twice this long.
            stability_checks: Number of consecutive successful reachability checks required
            image_build_timeout_seconds: Separate wall-clock budget while the
                platform auto-builds the VM image for a first-use image (the
                sandbox stays PENDING with pending_image_build_id set). That
                phase polls slowly and does not consume the creation budget.
        """
        consecutive_successes = 0
        image_build_deadline: Optional[float] = None
        deadline = time.monotonic() + _creation_timeout_seconds(max_attempts)
        poll_index = 0
        reachability_phase = False
        while time.monotonic() < deadline:
            sandbox = self._sandbox_status_batcher.get(sandbox_id)
            if sandbox.status == "RUNNING":
                if not reachability_phase:
                    reachability_phase = True
                    deadline = time.monotonic() + _creation_timeout_seconds(max_attempts)
                    poll_index = 0
                if self._is_sandbox_reachable(sandbox_id):
                    consecutive_successes += 1
                    if consecutive_successes >= stability_checks:
                        return
                    # Small delay between stability checks
                    time.sleep(0.5)
                    continue
                else:
                    # Reset counter if check fails
                    consecutive_successes = 0
            elif sandbox.status in ["ERROR", "TERMINATED", "TIMEOUT"]:
                ctx = {
                    "status": sandbox.status,
                    "error_type": sandbox.error_type,
                    "error_message": sandbox.error_message,
                }
                _raise_not_running_error(sandbox.sandbox_id, ctx)
            elif _is_waiting_for_image_build(sandbox):
                # The platform is building the VM image for this sandbox; it
                # starts on its own once the build completes. This phase runs on
                # its own budget, so hold the creation deadline back while it
                # lasts and reset the backoff for when the sandbox starts.
                if image_build_deadline is None:
                    image_build_deadline = time.monotonic() + image_build_timeout_seconds
                if time.monotonic() >= image_build_deadline:
                    raise SandboxNotRunningError(
                        sandbox_id, "Timeout waiting for the VM image build"
                    )
                time.sleep(10)
                deadline = time.monotonic() + _creation_timeout_seconds(max_attempts)
                poll_index = 0
                continue

            # Never sleep past the deadline. The loop only re-checks it on the
            # next iteration, so an uncapped backoff delay would let the wait
            # run up to CREATION_POLL_MAX_DELAY (plus jitter) beyond the budget.
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(_creation_poll_delay(poll_index), remaining))
            poll_index += 1
        raise SandboxNotRunningError(sandbox_id, "Timeout during sandbox creation")

    def bulk_wait_for_creation(
        self,
        sandbox_ids: List[str],
        max_attempts: int = 60,
        image_build_timeout_seconds: int = 3000,
    ) -> Dict[str, str]:
        """Wait for up to 100 sandboxes using the batch lifecycle endpoint.

        Sandboxes PENDING on an automatic VM image build (first use of an
        image) are waited on a separate slower budget bounded by
        image_build_timeout_seconds instead of consuming max_attempts.
        """
        _validate_unique_batch_values(sandbox_ids, "sandbox_ids")
        final_statuses: Dict[str, str] = {}
        image_build_deadline: Optional[float] = None

        attempt = 0
        while attempt < max_attempts:
            try:
                response = self.get_sandbox_statuses(sandbox_ids)
            except BatchStatusUnsupportedError:
                outcomes = self._fetch_sandbox_statuses(sandbox_ids)
                snapshots = []
                for sandbox_id in sandbox_ids:
                    outcome = outcomes[sandbox_id]
                    if isinstance(outcome, _BatchItemError):
                        raise outcome.error
                    snapshots.append(outcome)
                response = BatchSandboxStatusResponse(
                    statuses=snapshots,
                    errors=[],
                )
            except Exception as exc:
                if "429" in str(exc) or "Too Many Requests" in str(exc):
                    time.sleep(min(2**attempt, 60))
                    continue
                raise

            if response.errors:
                failures = [(error.sandbox_id, error.code) for error in response.errors]
                raise RuntimeError(f"Sandboxes unavailable: {failures}")

            total_running = 0
            all_failed = []
            total_image_build_waiting = 0
            for snapshot in response.statuses:
                status_value = snapshot.status.value
                if status_value == "RUNNING":
                    total_running += 1
                    final_statuses[snapshot.sandbox_id] = status_value
                elif status_value in ["ERROR", "TERMINATED", "TIMEOUT"]:
                    all_failed.append((snapshot.sandbox_id, status_value))
                    final_statuses[snapshot.sandbox_id] = status_value
                elif _is_waiting_for_image_build(snapshot):
                    total_image_build_waiting += 1

            if all_failed:
                raise RuntimeError(f"Sandboxes failed: {all_failed}")

            if total_running == len(sandbox_ids):
                all_reachable = True
                for sandbox_id in sandbox_ids:
                    if final_statuses.get(sandbox_id) == "RUNNING":
                        if not self._is_sandbox_reachable(sandbox_id):
                            all_reachable = False
                            final_statuses.pop(sandbox_id, None)

                if all_reachable:
                    return final_statuses

            if total_image_build_waiting:
                # At least one sandbox is waiting on an automatic VM image
                # build; poll slowly on the image-build budget instead of
                # consuming the normal attempts.
                if image_build_deadline is None:
                    image_build_deadline = time.monotonic() + image_build_timeout_seconds
                if time.monotonic() < image_build_deadline:
                    time.sleep(10)
                    continue
                break

            attempt += 1
            sleep_time = 1 if attempt <= 5 else 2
            time.sleep(sleep_time)

        for sandbox_id in sandbox_ids:
            if sandbox_id not in final_statuses:
                final_statuses[sandbox_id] = "TIMEOUT"

        raise RuntimeError(f"Timeout waiting for sandboxes to be ready. Status: {final_statuses}")

    def upload_file(
        self,
        sandbox_id: str,
        file_path: str,
        local_file_path: str,
        timeout: Optional[int] = None,
    ) -> FileUploadResponse:
        """Upload file directly via gateway"""
        if not os.path.exists(local_file_path):
            raise FileNotFoundError(f"Local file not found: {local_file_path}")

        effective_timeout = timeout if timeout is not None else 300

        with open(local_file_path, "rb") as f:
            file_content = f.read()

        reauthed = False
        attempt = 0
        for _ in range(MAX_GATEWAY_ATTEMPTS):
            auth = self._auth_cache.get_or_refresh(sandbox_id)
            url = f"{auth['gateway_url']}/{auth['user_ns']}/{auth['job_id']}/upload"
            headers = {"Authorization": f"Bearer {auth['token']}"}
            try:
                files = {"file": (os.path.basename(local_file_path), file_content)}
                params = {"path": file_path, "sandbox_id": sandbox_id}
                response = self._gateway_post(
                    url, headers=headers, timeout=effective_timeout, files=files, params=params
                )
                response.raise_for_status()
                return FileUploadResponse.model_validate(response.json())
            except httpx.TimeoutException as e:
                raise UploadTimeoutError(sandbox_id, file_path, effective_timeout) from e
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 401 and self._should_retry_401(sandbox_id, reauthed):
                    reauthed = True
                    continue
                if e.response.status_code == 409:
                    if self._should_retry_409(sandbox_id, e, attempt):
                        attempt += 1
                        continue
                elif self._should_retry_upload_error(e, attempt):
                    attempt += 1
                    continue
                error_details = (
                    f"HTTP {e.response.status_code} {e.request.method} "
                    f"{e.request.url}: {e.response.text}"
                )
                raise APIError(f"Upload failed: {error_details}") from e
            except httpx.RequestError as e:
                req = getattr(e, "request", None)
                method = getattr(req, "method", "?")
                u = getattr(req, "url", "?")
                raise APIError(f"Upload failed: {e.__class__.__name__} at {method} {u}: {e}") from e
            except Exception as e:
                raise APIError(f"Upload failed: {e.__class__.__name__}: {e}") from e

        raise APIError("Upload failed after retries")

    def upload_bytes(
        self,
        sandbox_id: str,
        file_path: str,
        file_bytes: bytes,
        filename: str,
        timeout: Optional[int] = None,
    ) -> FileUploadResponse:
        """Upload bytes directly to sandbox via gateway without writing to disk

        Args:
            sandbox_id: The sandbox ID
            file_path: Remote path in the sandbox where the file will be saved
            file_bytes: The bytes content to upload
            filename: Name for the file (used in multipart form)
            timeout: Optional timeout in seconds
        """
        effective_timeout = timeout if timeout is not None else 300

        reauthed = False
        # `attempt` counts only transient (409/5xx/408) retries, capped by the
        # helpers at MAX_409_RETRIES; the single 401 re-auth is bounded by
        # `reauthed`. The loop bound is a backstop sized for both budgets.
        attempt = 0
        for _ in range(MAX_GATEWAY_ATTEMPTS):
            auth = self._auth_cache.get_or_refresh(sandbox_id)
            url = f"{auth['gateway_url']}/{auth['user_ns']}/{auth['job_id']}/upload"
            headers = {"Authorization": f"Bearer {auth['token']}"}
            try:
                files = {"file": (filename, file_bytes)}
                params = {"path": file_path, "sandbox_id": sandbox_id}
                response = self._gateway_post(
                    url, headers=headers, timeout=effective_timeout, files=files, params=params
                )
                response.raise_for_status()
                return FileUploadResponse.model_validate(response.json())
            except httpx.TimeoutException:
                raise UploadTimeoutError(sandbox_id, file_path, effective_timeout)
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 401 and self._should_retry_401(sandbox_id, reauthed):
                    reauthed = True
                    continue
                if e.response.status_code == 409:
                    if self._should_retry_409(sandbox_id, e, attempt):
                        attempt += 1
                        continue
                elif self._should_retry_upload_error(e, attempt):
                    attempt += 1
                    continue
                error_details = f"HTTP {e.response.status_code}: {e.response.text}"
                raise APIError(f"Upload failed: {error_details}")
            except Exception as e:
                raise APIError(f"Upload failed: {str(e)}")

        raise APIError("Upload failed after retries")

    def download_file(
        self,
        sandbox_id: str,
        file_path: str,
        local_file_path: str,
        timeout: Optional[int] = None,
    ) -> None:
        """Download file directly via gateway"""
        params = {"path": file_path, "sandbox_id": sandbox_id}

        effective_timeout = timeout if timeout is not None else 300

        reauthed = False
        attempt = 0
        for _ in range(MAX_GATEWAY_ATTEMPTS):
            auth = self._auth_cache.get_or_refresh(sandbox_id)
            url = f"{auth['gateway_url']}/{auth['user_ns']}/{auth['job_id']}/download"
            headers = {"Authorization": f"Bearer {auth['token']}"}
            try:
                response = self._gateway_get(
                    url, headers=headers, params=params, timeout=effective_timeout
                )
                response.raise_for_status()

                dir_path = os.path.dirname(local_file_path)
                if dir_path:
                    os.makedirs(dir_path, exist_ok=True)

                with open(local_file_path, "wb") as f:
                    f.write(response.content)
                return
            except httpx.TimeoutException as e:
                raise DownloadTimeoutError(sandbox_id, file_path, effective_timeout) from e
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 401 and self._should_retry_401(sandbox_id, reauthed):
                    reauthed = True
                    continue
                if e.response.status_code == 409:
                    if self._should_retry_409(sandbox_id, e, attempt):
                        attempt += 1
                        continue
                error_details = (
                    f"HTTP {e.response.status_code} {e.request.method} "
                    f"{e.request.url}: {e.response.text}"
                )
                raise APIError(f"Download failed: {error_details}") from e
            except httpx.RequestError as e:
                req = getattr(e, "request", None)
                method = getattr(req, "method", "?")
                u = getattr(req, "url", "?")
                raise APIError(
                    f"Download failed: {e.__class__.__name__} at {method} {u}: {e}"
                ) from e
            except Exception as e:
                raise APIError(f"Download failed: {e.__class__.__name__}: {e}") from e

        raise APIError("Download failed after retries")

    def read_file(
        self,
        sandbox_id: str,
        file_path: str,
        timeout: Optional[int] = None,
        offset: Optional[int] = None,
        length: Optional[int] = None,
    ) -> ReadFileResponse:
        """Read a file (or a byte window of it) from a sandbox via gateway.

        offset/length require server-side windowed-read support. VM sandboxes
        don't support it yet: they ignore both params, return the whole file
        (subject to the read size limit), and omit total_size/offset/truncated
        from the response (detectable via ``response.offset is None``).
        """
        params: Dict[str, Any] = {"path": file_path}
        if offset is not None:
            params["offset"] = offset
        if length is not None:
            params["length"] = length

        effective_timeout = timeout if timeout is not None else 30

        reauthed = False
        # `attempt` counts only transient (409/5xx/408) retries, capped by the
        # helpers at MAX_409_RETRIES; the single 401 re-auth is bounded by
        # `reauthed`. The loop bound is a backstop sized for both budgets.
        attempt = 0
        for _ in range(MAX_GATEWAY_ATTEMPTS):
            auth = self._auth_cache.get_or_refresh(sandbox_id)
            gateway_url = auth["gateway_url"].rstrip("/")
            url = f"{gateway_url}/{auth['user_ns']}/{auth['job_id']}/read-file"
            headers = {"Authorization": f"Bearer {auth['token']}"}
            try:
                response = self._gateway_read_file_get(
                    url, headers=headers, params=params, timeout=effective_timeout
                )
                response.raise_for_status()
                return ReadFileResponse.model_validate(response.json())
            except httpx.TimeoutException as e:
                raise APIError(
                    f"Read file timed out after {effective_timeout}s "
                    f"({e.__class__.__name__}): {file_path}"
                ) from e
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 401 and self._should_retry_401(sandbox_id, reauthed):
                    reauthed = True
                    continue
                if e.response.status_code == 404:
                    raise SandboxFileNotFoundError(f"File not found: {file_path}") from e
                if e.response.status_code == 413:
                    raise SandboxFileTooLargeError(
                        f"File too large to read: {file_path}: {e.response.text}"
                    ) from e
                if e.response.status_code == 409:
                    if self._should_retry_409(sandbox_id, e, attempt):
                        attempt += 1
                        continue
                error_details = (
                    f"HTTP {e.response.status_code} {e.request.method} "
                    f"{e.request.url}: {e.response.text}"
                )
                raise APIError(f"Read file failed: {error_details}") from e
            except httpx.RequestError as e:
                req = getattr(e, "request", None)
                method = getattr(req, "method", "?")
                u = getattr(req, "url", "?")
                raise APIError(
                    f"Read file failed: {e.__class__.__name__} at {method} {u}: {e}"
                ) from e
            except Exception as e:
                raise APIError(f"Read file failed: {e.__class__.__name__}: {e}") from e

        raise APIError("Read file failed after retries")

    def expose(
        self,
        sandbox_id: str,
        port: int,
        name: Optional[str] = None,
        protocol: str = "HTTP",
    ) -> ExposedPort:
        """Expose a port from a sandbox."""
        self._guard_vm_unsupported(sandbox_id, "Port exposure")
        request = ExposePortRequest(port=port, name=name, protocol=protocol)
        response = self.client.request(
            "POST",
            f"/sandbox/{sandbox_id}/expose",
            json=request.model_dump(by_alias=False, exclude_none=True),
        )
        return ExposedPort.model_validate(response)

    def unexpose(self, sandbox_id: str, exposure_id: str) -> None:
        """Unexpose a port from a sandbox."""
        self._guard_vm_unsupported(sandbox_id, "Port unexpose")
        self.client.request("DELETE", f"/sandbox/{sandbox_id}/expose/{exposure_id}")

    def list_exposed_ports(self, sandbox_id: str) -> ListExposedPortsResponse:
        """List all exposed ports for a sandbox"""
        self._guard_vm_unsupported(sandbox_id, "Port listing")
        response = self.client.request("GET", f"/sandbox/{sandbox_id}/expose")
        return ListExposedPortsResponse.model_validate(response)

    def list_all_exposed_ports(self) -> ListExposedPortsResponse:
        """List all exposed ports across all sandboxes for the current user"""
        response = self.client.request("GET", "/sandbox/expose/all")
        return ListExposedPortsResponse.model_validate(response)

    def create_ssh_session(
        self,
        sandbox_id: str,
        ttl_seconds: Optional[int] = None,
    ) -> SSHSession:
        """Create an SSH session"""
        self._guard_vm_unsupported(sandbox_id, "SSH")
        payload: Dict[str, Any] = {}
        if ttl_seconds is not None:
            payload["ttl_seconds"] = ttl_seconds
        response = self.client.request(
            "POST",
            f"/sandbox/{sandbox_id}/ssh-session",
            json=payload,
        )
        return SSHSession.model_validate(response)

    def close_ssh_session(self, sandbox_id: str, session_id: str) -> None:
        """Close an SSH session and remove its exposure"""
        self._guard_vm_unsupported(sandbox_id, "SSH")
        self.client.request("DELETE", f"/sandbox/{sandbox_id}/ssh-session/{session_id}")


class AsyncSandboxClient:
    """Async client for sandbox API operations"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        max_connections: int = 1000,
        max_keepalive_connections: int = 200,
    ):
        """Initialize async sandbox client

        Args:
            api_key: Optional API key (reads from config if not provided)
            max_connections: Maximum number of concurrent connections (default: 1000)
            max_keepalive_connections: Maximum keep-alive connections (default: 200)
        """
        self.client = AsyncAPIClient(api_key=api_key, user_agent=_build_user_agent())
        self._auth_cache = AsyncSandboxAuthCache(
            self.client.config.config_dir / "sandbox_auth_cache.json",
            self.client,
        )
        # Connection pool configuration
        self._max_connections = max_connections
        self._max_keepalive_connections = max_keepalive_connections
        # Shared httpx client for gateway operations (upload/download/execute)
        # Initialized lazily to allow connection pooling and reuse
        self._gateway_client: Optional[httpx.AsyncClient] = None
        self._sandbox_status_batcher = _AsyncRequestBatcher(self._fetch_sandbox_statuses)
        self._background_job_status_batcher = _AsyncRequestBatcher(
            self._fetch_background_job_statuses
        )
        self._sandbox_status_batch_supported: Optional[bool] = None
        self._background_job_status_batch_supported: Optional[bool] = None

    def _get_gateway_client(self) -> httpx.AsyncClient:
        """Get or create the shared gateway client for connection pooling

        Note: Timeout is set per-request, not on the client, to allow
        different operations to have different timeout values.
        """
        if self._gateway_client is None:
            self._gateway_client = httpx.AsyncClient(
                timeout=None,  # No default timeout - set per request
                limits=httpx.Limits(
                    max_connections=self._max_connections,
                    max_keepalive_connections=self._max_keepalive_connections,
                ),
            )
        return self._gateway_client

    @_gateway_post_retry
    async def _gateway_post(
        self,
        url: str,
        headers: Dict[str, str],
        timeout: float,
        json: Optional[Dict[str, Any]] = None,
        files: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> httpx.Response:
        """Make a POST request to the gateway with retry on connection errors only."""
        gateway_client = self._get_gateway_client()
        return await gateway_client.post(
            url, json=json, files=files, params=params, headers=headers, timeout=timeout
        )

    @_gateway_retry
    async def _gateway_get(
        self,
        url: str,
        headers: Dict[str, str],
        params: Dict[str, Any],
        timeout: float,
    ) -> httpx.Response:
        """Make a GET request to the gateway with retry on transient errors."""
        gateway_client = self._get_gateway_client()
        response = await gateway_client.get(url, params=params, headers=headers, timeout=timeout)
        if response.status_code in RETRYABLE_5XX_STATUSES:
            response.raise_for_status()
        return response

    @_read_file_retry
    async def _gateway_read_file_get(
        self,
        url: str,
        headers: Dict[str, str],
        params: Dict[str, Any],
        timeout: float,
    ) -> httpx.Response:
        """Make a read-file GET request to the gateway with read-timeout retries."""
        gateway_client = self._get_gateway_client()
        response = await gateway_client.get(url, params=params, headers=headers, timeout=timeout)
        if response.status_code == 408 or response.status_code in RETRYABLE_5XX_STATUSES:
            response.raise_for_status()
        return response

    async def _is_sandbox_reachable(self, sandbox_id: str, timeout: int = 10) -> bool:
        """Test if a sandbox is reachable by executing a simple echo command"""
        try:
            await self.execute_command(sandbox_id, "echo 'sandbox ready'", timeout=timeout)
            return True
        except Exception:
            return False

    async def _get_sandbox_error_context(self, sandbox_id: str) -> dict:
        """Fetch sandbox error context from the lightweight server endpoint."""
        try:
            response = await self.client.request("GET", f"/sandbox/{sandbox_id}/error-context")
            return {
                "status": response.get("status"),
                "error_type": response.get("errorType") or response.get("error_type"),
                "error_message": response.get("errorMessage") or response.get("error_message"),
            }
        except Exception:
            return {"status": None, "error_type": None, "error_message": None}

    async def _should_retry_409(
        self,
        sandbox_id: str,
        error: httpx.HTTPStatusError,
        attempt: int,
        command: Optional[str] = None,
    ) -> bool:
        """Check if a 409 error should be retried (async).

        Returns True and sleeps if should retry, raises appropriate error otherwise.
        """
        ctx = await self._get_sandbox_error_context(sandbox_id)
        if ctx["status"] == "RUNNING":
            if attempt < MAX_409_RETRIES - 1:
                await asyncio.sleep(RETRY_409_BASE_DELAY * (2**attempt))
                return True
            raise APIError(
                f"Sandbox {sandbox_id} returned 409 after {MAX_409_RETRIES} retries. "
                "This may be a transient DNS or gateway issue. Please retry."
            ) from error
        # Sandbox is not running
        _raise_not_running_error(sandbox_id, ctx, command=command, cause=error)

    async def _should_retry_upload_error(self, error: httpx.HTTPStatusError, attempt: int) -> bool:
        """Check if a transient 408/5xx on an idempotent upload should be retried (async)."""
        status = error.response.status_code
        if status != 408 and status not in RETRYABLE_5XX_STATUSES:
            return False
        if _is_gateway_sandbox_not_found(error.response):
            return False
        if attempt < MAX_409_RETRIES - 1:
            await asyncio.sleep(RETRY_409_BASE_DELAY * (2**attempt))
            return True
        return False

    async def _should_retry_401(self, sandbox_id: str, reauthed: bool) -> bool:
        """Check if a gateway 401 should be retried with fresh auth (async)."""
        if reauthed:
            return False
        await self._auth_cache.invalidate(sandbox_id)
        return True

    async def clear_auth_cache(self) -> None:
        """Clear all cached auth tokens."""
        await self._auth_cache.clear()

    async def is_vm(self, sandbox_id: str) -> bool:
        """Return True if the sandbox is VM-backed.

        Uses the internal auth cache when available and falls back to a
        ``GET /sandbox/<id>`` lookup on a cold cache. The result is cached
        alongside the auth token so subsequent calls are essentially free.
        """
        return await self._auth_cache.is_vm(sandbox_id)

    async def _guard_vm_unsupported(self, sandbox_id: str, feature_name: str) -> None:
        """Raise APIError if the operation is not supported on VM sandboxes.

        Mirrors the CLI behavior of short-circuiting operations the backend
        does not currently support for VM-backed sandboxes, so callers fail
        fast with a clear message instead of an opaque gateway error.
        """
        if await self._auth_cache.is_vm(sandbox_id):
            raise APIError(f"{feature_name} is not yet supported for VM sandboxes.")

    async def create(self, request: CreateSandboxRequest) -> Sandbox:
        """Create a new sandbox"""
        payload = request.model_dump(by_alias=False, exclude_none=True)
        if request.team_id is None and self.client.config.team_id is not None:
            payload["team_id"] = self.client.config.team_id
        payload["idempotency_key"] = request.idempotency_key or uuid.uuid4().hex

        response = await self.client.request(
            "POST",
            "/sandbox",
            json=payload,
            idempotent_post=True,
        )
        return Sandbox.model_validate(response)

    async def list(
        self,
        team_id: Optional[str] = None,
        status: Optional[str] = None,
        labels: Optional[List[str]] = None,
        page: int = 1,
        per_page: int = 50,
        exclude_terminated: Optional[bool] = None,
        user_id: Optional[str] = None,
    ) -> SandboxListResponse:
        """List sandboxes"""
        if team_id is None:
            team_id = self.client.config.team_id

        params: Dict[str, Any] = {"page": page, "per_page": per_page}
        if team_id:
            params["team_id"] = team_id
        if user_id:
            params["user_id"] = user_id
        if status:
            params["status"] = status
        if labels:
            params["labels"] = labels
        if exclude_terminated is not None:
            params["is_active"] = exclude_terminated

        response = await self.client.request("GET", "/sandbox", params=params)
        return SandboxListResponse.model_validate(response)

    async def get(self, sandbox_id: str) -> Sandbox:
        """Get a specific sandbox"""
        response = await self.client.request("GET", f"/sandbox/{sandbox_id}")
        return Sandbox.model_validate(response)

    async def get_sandbox_statuses(self, sandbox_ids: List[str]) -> BatchSandboxStatusResponse:
        """Get lightweight lifecycle state for up to 100 sandboxes."""
        _validate_unique_batch_values(sandbox_ids, "sandbox_ids")
        if self._sandbox_status_batch_supported is False:
            raise BatchStatusUnsupportedError(
                "The platform does not support batch sandbox status lookups."
            )
        try:
            response = await self.client.request(
                "POST",
                "/sandbox/status:batchGet",
                json={"sandbox_ids": sandbox_ids},
                idempotent_post=True,
            )
        except APIError as exc:
            if "HTTP 404" in str(exc) or "HTTP 405" in str(exc):
                self._sandbox_status_batch_supported = False
                raise BatchStatusUnsupportedError(
                    "The platform does not support batch sandbox status lookups."
                ) from exc
            raise
        self._sandbox_status_batch_supported = True
        return BatchSandboxStatusResponse.model_validate(response)

    async def _fetch_sandbox_statuses(
        self, sandbox_ids: List[str]
    ) -> Dict[str, SandboxStatusSnapshot | _BatchItemError]:
        """Fetch one coalesced lifecycle batch for concurrent waiters."""
        try:
            response = await self.get_sandbox_statuses(sandbox_ids)
        except BatchStatusUnsupportedError:
            sandboxes = await asyncio.gather(
                *(self.get(sandbox_id) for sandbox_id in sandbox_ids),
                return_exceptions=True,
            )
            results: Dict[str, SandboxStatusSnapshot | _BatchItemError] = {}
            for sandbox_id, sandbox in zip(sandbox_ids, sandboxes):
                if isinstance(sandbox, Exception):
                    results[sandbox_id] = _BatchItemError(sandbox)
                else:
                    results[sandbox_id] = _sandbox_to_status_snapshot(sandbox)
            return results

        results: Dict[str, SandboxStatusSnapshot | _BatchItemError] = {
            snapshot.sandbox_id: snapshot for snapshot in response.statuses
        }
        for error in response.errors:
            results[error.sandbox_id] = _BatchItemError(
                APIError(
                    f"Sandbox status lookup failed for {error.sandbox_id}: "
                    f"{error.code}: {error.message}"
                )
            )
        return results

    async def delete(self, sandbox_id: str) -> Dict[str, Any]:
        """Delete a sandbox"""
        response = await self.client.request("DELETE", f"/sandbox/{sandbox_id}")
        return response

    async def get_network(self, sandbox_id: str) -> EgressPolicyStatus:
        """Get the desired and applied network rules of a VM sandbox."""
        response = await self.client.request("GET", f"/sandbox/{sandbox_id}/egress-policy")
        return EgressPolicyStatus.model_validate(response)

    async def set_network(
        self,
        sandbox_id: str,
        *,
        allow: Optional[List[str]] = None,
        deny: Optional[List[str]] = None,
    ) -> EgressPolicyStatus:
        """Replace the network rules of a running VM sandbox.

        Exactly one of ``allow`` or ``deny`` must be provided. Each call is a
        complete replacement, never a merge; use ``["*"]`` to allow or deny
        all destinations. New connections follow the replacement once
        ``applied`` is true; established flows are not revoked.
        """
        payload = _network_update_payload(allow, deny)

        response = await self.client.request(
            "PUT",
            f"/sandbox/{sandbox_id}/egress-policy",
            json=payload,
        )
        return EgressPolicyStatus.model_validate(response)

    async def bulk_delete(
        self,
        sandbox_ids: Optional[List[str]] = None,
        labels: Optional[List[str]] = None,
        team_id: Optional[str] = None,
        user_id: Optional[str] = None,
        all_users: bool = False,
    ) -> BulkDeleteSandboxResponse:
        """Bulk delete multiple sandboxes."""
        request = BulkDeleteSandboxRequest(
            sandbox_ids=sandbox_ids,
            labels=labels,
            team_id=team_id,
            user_id=user_id,
            all_users=all_users,
        )
        payload = request.model_dump(by_alias=False, exclude_none=True)
        response = await self.client.request(
            "DELETE",
            "/sandbox",
            json=payload,
        )
        return BulkDeleteSandboxResponse.model_validate(response)

    async def get_logs(self, sandbox_id: str) -> str:
        """Get sandbox logs"""
        response = await self.client.request("GET", f"/sandbox/{sandbox_id}/logs")
        logs_response = SandboxLogsResponse.model_validate(response)
        return logs_response.logs

    async def execute_command(
        self,
        sandbox_id: str,
        command: str,
        working_dir: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None,
        user: Optional[str] = None,
    ) -> CommandResponse:
        """Execute command directly via gateway (async)."""
        await self._auth_cache.get_or_refresh(sandbox_id)

        if await self._auth_cache.is_vm(sandbox_id):
            if user is not None:
                raise ValueError(
                    "The 'user' parameter is only supported for container sandboxes, "
                    "not VM sandboxes."
                )
            return await self._execute_command_connect_rpc(
                sandbox_id=sandbox_id,
                command=command,
                working_dir=working_dir,
                env=env,
                timeout=timeout,
            )

        return await self._execute_command_rest(
            sandbox_id=sandbox_id,
            command=command,
            working_dir=working_dir,
            env=env,
            timeout=timeout,
            user=user,
        )

    async def open_process(
        self,
        sandbox_id: str,
        command: str,
        working_dir: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        user: Optional[str] = None,
    ) -> AsyncSandboxProcess:
        """Start a live process in a VM sandbox.

        The returned handle streams stdout and stderr, accepts stdin writes,
        waits for the exit code, and can signal the process. Container sandboxes
        do not expose this transport and fail fast.
        """
        await self._auth_cache.get_or_refresh(sandbox_id)
        if not await self._auth_cache.is_vm(sandbox_id):
            raise APIError("Live processes are only supported for VM sandboxes.")
        if user is not None:
            raise ValueError("The 'user' parameter is not supported for VM sandbox processes.")

        auth = await self._auth_cache.get_or_refresh(sandbox_id)
        gateway_url = auth["gateway_url"].rstrip("/")
        base_url = f"{gateway_url}/{auth['user_ns']}/{auth['job_id']}"
        headers = {"Authorization": f"Bearer {auth['token']}"}
        # Each live process gets its own transport: the session stream occupies one
        # HTTP/2 stream for the process's whole lifetime, and the gateway caps
        # concurrent streams per connection. On the shared default transport, enough
        # concurrent live processes exhaust that cap and every later gateway RPC
        # (stdin, signals, new sessions) queues until its deadline. The process's
        # control RPCs reuse the same transport so they cannot starve either.
        transport = _live_process_transport()
        http_client = HTTPClient(transport=transport)
        rpc_client = ConnectClient(base_url, http_client=http_client)
        request = build_command_session_start_request(
            command,
            working_dir,
            env,
            stdin=True,
        )
        stream = rpc_client.execute_server_stream(
            request=request,
            method=COMMAND_SESSION_START_RPC_METHOD,
            headers=headers,
            timeout_ms=_LIVE_PROCESS_TIMEOUT_MS,
        )

        async def write_stdin(pid: int, data: bytes) -> None:
            await self._execute_process_control_rpc(
                sandbox_id,
                build_command_session_send_input_request(pid, data),
                COMMAND_SESSION_SEND_INPUT_RPC_METHOD,
                _PROCESS_INPUT_TIMEOUT_MS,
                "stdin",
                http_client=http_client,
            )

        async def send_signal(pid: int, signal: Literal["terminate", "kill"]) -> None:
            await self._execute_process_control_rpc(
                sandbox_id,
                build_command_session_send_signal_request(pid, signal),
                COMMAND_SESSION_SEND_SIGNAL_RPC_METHOD,
                _PROCESS_SIGNAL_TIMEOUT_MS,
                "signal",
                http_client=http_client,
            )

        return await AsyncSandboxProcess._create(
            rpc_client,
            stream,
            write_stdin,
            send_signal,
            transport=transport,
        )

    async def _execute_process_control_rpc(
        self,
        sandbox_id: str,
        request: _RequestMessage,
        method: MethodInfo[_RequestMessage, _ResponseMessage],
        timeout_ms: int,
        operation: str,
        http_client: Optional[HTTPClient] = None,
    ) -> None:
        """Run one live-process control RPC with current sandbox auth."""
        reauthed = False
        while True:
            auth = await self._auth_cache.get_or_refresh(sandbox_id)
            gateway_url = auth["gateway_url"].rstrip("/")
            base_url = f"{gateway_url}/{auth['user_ns']}/{auth['job_id']}"
            headers = {"Authorization": f"Bearer {auth['token']}"}
            rpc_client = ConnectClient(base_url, http_client=http_client)
            try:
                await rpc_client.execute_unary(
                    request=request,
                    method=method,
                    headers=headers,
                    timeout_ms=timeout_ms,
                )
                return
            except ConnectError as error:
                if error.code == Code.UNAUTHENTICATED and await self._should_retry_401(
                    sandbox_id, reauthed
                ):
                    reauthed = True
                    continue
                raise APIError(
                    f"process {operation} RPC failed ({error.code.value}): {error.message}"
                ) from error
            finally:
                await rpc_client.close()

    async def _execute_command_connect_rpc(
        self,
        sandbox_id: str,
        command: str,
        working_dir: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None,
    ) -> CommandResponse:
        effective_timeout = timeout if timeout is not None else 300
        request = build_command_session_start_request(command, working_dir, env)

        reauthed = False
        while True:
            auth = await self._auth_cache.get_or_refresh(sandbox_id)
            gateway_url = auth["gateway_url"].rstrip("/")
            base_url = f"{gateway_url}/{auth['user_ns']}/{auth['job_id']}"
            headers = {"Authorization": f"Bearer {auth['token']}"}
            stdout_parts: List[str] = []
            stderr_parts: List[str] = []
            exit_code: Optional[int] = None
            stream_started = False

            rpc_client = ConnectClient(base_url)
            try:
                stream = rpc_client.execute_server_stream(
                    request=request,
                    method=COMMAND_SESSION_START_RPC_METHOD,
                    headers=headers,
                    timeout_ms=effective_timeout * 1000,
                )
                async for event in stream:
                    stream_started = True
                    event_exit_code = collect_command_session_start_event(
                        event,
                        stdout_parts,
                        stderr_parts,
                    )
                    if event_exit_code is not None:
                        exit_code = event_exit_code

                if exit_code is None:
                    raise APIError("Command stream ended without exit code")

                return CommandResponse(
                    stdout="".join(stdout_parts),
                    stderr="".join(stderr_parts),
                    exit_code=exit_code,
                )
            except ConnectError as e:
                # Only re-auth before any stream event.
                if (
                    e.code == Code.UNAUTHENTICATED
                    and not stream_started
                    and await self._should_retry_401(sandbox_id, reauthed)
                ):
                    reauthed = True
                    continue

                if e.code == Code.DEADLINE_EXCEEDED:
                    ctx = await self._get_sandbox_error_context(sandbox_id)
                    if ctx["status"] in ("TERMINATED", "ERROR", "TIMEOUT"):
                        _raise_not_running_error(sandbox_id, ctx, command=command, cause=e)
                    raise CommandTimeoutError(sandbox_id, command, effective_timeout) from e

                if e.code == Code.NOT_FOUND:
                    ctx = await self._get_sandbox_error_context(sandbox_id)
                    ctx["status"] = "TERMINATED"
                    if not ctx.get("error_type"):
                        ctx["error_type"] = "SANDBOX_NOT_FOUND"
                    if not ctx.get("error_message"):
                        ctx["error_message"] = (
                            "Sandbox is no longer present on the runtime node. "
                            "Please create a new sandbox."
                        )
                    _raise_not_running_error(sandbox_id, ctx, command=command, cause=e)

                raise APIError(f"Connect RPC failed ({e.code.value}): {e.message}") from e
            except APIError:
                raise
            except Exception as e:
                raise APIError(f"Request failed: {e.__class__.__name__}: {e}") from e
            finally:
                await rpc_client.close()

    async def _execute_command_rest(
        self,
        sandbox_id: str,
        command: str,
        working_dir: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None,
        user: Optional[str] = None,
    ) -> CommandResponse:
        effective_timeout = timeout if timeout is not None else 300

        payload = {
            "command": command,
            "working_dir": working_dir,
            "env": env or {},
            "sandbox_id": sandbox_id,
            "timeout": effective_timeout,
        }
        if user is not None:
            payload["user"] = user

        reauthed = False
        attempt = 0
        for _ in range(MAX_GATEWAY_ATTEMPTS):
            auth = await self._auth_cache.get_or_refresh(sandbox_id)
            gateway_url = auth["gateway_url"].rstrip("/")
            url = f"{gateway_url}/{auth['user_ns']}/{auth['job_id']}/exec"
            headers = {"Authorization": f"Bearer {auth['token']}"}
            try:
                # The + 5 accounts for connection creation and closing. Prevents any command
                # running close to its `effective_timeout` from being killed prematurely
                client_timeout = effective_timeout + 5
                response = await self._gateway_post(
                    url, headers=headers, timeout=client_timeout, json=payload
                )
                response.raise_for_status()
                return CommandResponse.model_validate(response.json())
            except httpx.TimeoutException as e:
                ctx = await self._get_sandbox_error_context(sandbox_id)
                if ctx["status"] in ("TERMINATED", "ERROR", "TIMEOUT"):
                    _raise_not_running_error(sandbox_id, ctx, command=command, cause=e)
                raise CommandTimeoutError(sandbox_id, command, effective_timeout) from e
            except httpx.HTTPStatusError as e:
                resp = getattr(e, "response", None)
                status = getattr(resp, "status_code", "?")

                if status == 401 and await self._should_retry_401(sandbox_id, reauthed):
                    reauthed = True
                    continue

                if status == 502 and _is_gateway_sandbox_not_found(resp):
                    ctx = await self._get_sandbox_error_context(sandbox_id)
                    ctx["status"] = "TERMINATED"
                    if not ctx.get("error_type"):
                        ctx["error_type"] = "SANDBOX_NOT_FOUND"
                    if not ctx.get("error_message"):
                        ctx["error_message"] = (
                            "Sandbox is no longer present on the runtime node. "
                            "Please create a new sandbox."
                        )
                    _raise_not_running_error(sandbox_id, ctx, command=command, cause=e)

                if status == 409:
                    if await self._should_retry_409(sandbox_id, e, attempt, command=command):
                        attempt += 1
                        continue

                if status == 408:
                    ctx = await self._get_sandbox_error_context(sandbox_id)
                    if ctx["status"] in ("TERMINATED", "ERROR", "TIMEOUT"):
                        _raise_not_running_error(sandbox_id, ctx, command=command, cause=e)
                    raise CommandTimeoutError(sandbox_id, command, effective_timeout) from e

                req = getattr(e, "request", None)
                method = getattr(req, "method", "?")
                u = getattr(req, "url", "?")
                text = getattr(resp, "text", "")
                raise APIError(f"HTTP {status} {method} {u}: {text}") from e
            except httpx.RequestError as e:
                req = getattr(e, "request", None)
                method = getattr(req, "method", "?")
                u = getattr(req, "url", "?")
                raise APIError(
                    f"Request failed: {e.__class__.__name__} at {method} {u}: {e}"
                ) from e
            except Exception as e:
                raise APIError(f"Request failed: {e.__class__.__name__}: {e}") from e

        raise APIError("Command execution failed after retries")

    async def start_background_job(
        self,
        sandbox_id: str,
        command: str,
        working_dir: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        user: Optional[str] = None,
    ) -> BackgroundJob:
        """Start a long-running command in the background (async).

        Returns immediately with a job handle. Use get_background_job() to check
        status and retrieve results.

        Args:
            sandbox_id: The sandbox ID
            command: Command to execute
            working_dir: Working directory for command execution
            env: Environment variables
            user: Run the job as this user, like ``docker exec -u`` (username or
                numeric UID, optionally USER:GROUP). Container sandboxes only.

        Returns:
            BackgroundJob with job_id and file paths for polling
        """
        job_id = uuid.uuid4().hex[:8]
        stdout_log_file = f"/tmp/job_{job_id}.stdout.log"
        stderr_log_file = f"/tmp/job_{job_id}.stderr.log"
        exit_file = f"/tmp/job_{job_id}.exit"

        env_prefix = ""
        if env:
            exports = []
            for k, v in env.items():
                _validate_env_key(k)
                exports.append(f"export {k}={shlex.quote(v)}")
            env_prefix = "; ".join(exports)
            if env_prefix:
                env_prefix += "; "

        dir_prefix = f"cd {shlex.quote(working_dir)} && " if working_dir else ""
        command_body = f"{env_prefix}{dir_prefix}{command}"
        exit_file_quoted = shlex.quote(exit_file)
        stdout_log_file_quoted = shlex.quote(stdout_log_file)
        stderr_log_file_quoted = shlex.quote(stderr_log_file)
        # Wrap command in subshell so 'exit' terminates the subshell, not the outer shell.
        # This ensures 'echo $?' always runs to capture the exit code.
        sh_command = (
            f"({command_body}) > {stdout_log_file_quoted} 2> {stderr_log_file_quoted}; "
            f"echo $? > {exit_file_quoted}"
        )
        quoted_sh_command = shlex.quote(sh_command)

        # Outer nohup redirects to /dev/null since output goes to log files inside sh -c
        bg_cmd = f"nohup sh -c {quoted_sh_command} < /dev/null > /dev/null 2>&1 &"
        await self.execute_command(sandbox_id, bg_cmd, timeout=30, user=user)

        return BackgroundJob(
            job_id=job_id,
            sandbox_id=sandbox_id,
            stdout_log_file=stdout_log_file,
            stderr_log_file=stderr_log_file,
            exit_file=exit_file,
        )

    async def get_background_job(
        self,
        sandbox_id: str,
        job: BackgroundJob,
        timeout: Optional[int] = None,
    ) -> BackgroundJobStatus:
        """Check the status of a background job (async).

        Args:
            sandbox_id: The sandbox ID
            job: The BackgroundJob handle from start_background_job()
            timeout: Optional per-call timeout (in seconds) forwarded to the
                underlying read_file calls. When None, the APIClient default
                applies.

        Returns:
            BackgroundJobStatus with completed flag, and exit_code/stdout if
            done. stdout/stderr hold at most the last JOB_OUTPUT_TAIL_BYTES
            of each stream; the *_truncated flags report dropped output.
        """

        async def read_or_empty(path: str) -> str:
            try:
                return (await self.read_file(sandbox_id, path, timeout=timeout)).content
            except SandboxFileNotFoundError:
                return ""

        async def read_output_tail(path: str) -> "tuple[str, bool]":
            try:
                response = await self.read_file(
                    sandbox_id,
                    path,
                    timeout=timeout,
                    offset=-JOB_OUTPUT_TAIL_BYTES,
                    length=JOB_OUTPUT_TAIL_BYTES,
                )
                # Servers without windowed-read support omit `truncated`.
                return response.content, bool(response.truncated)
            except SandboxFileNotFoundError:
                return "", False

        exit_content = await read_or_empty(job.exit_file)
        if not exit_content.strip():
            return BackgroundJobStatus(job_id=job.job_id, completed=False)

        try:
            exit_code = int(exit_content.strip())
        except ValueError:
            return BackgroundJobStatus(job_id=job.job_id, completed=False)

        stdout, stdout_truncated = await read_output_tail(job.stdout_log_file)
        stderr, stderr_truncated = await read_output_tail(job.stderr_log_file)
        return BackgroundJobStatus(
            job_id=job.job_id,
            completed=True,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            stdout_truncated=stdout_truncated,
            stderr_truncated=stderr_truncated,
        )

    async def _request_background_job_status_batch(
        self,
        jobs: List[BackgroundJob],
        timeout: Optional[int] = None,
    ) -> Optional[BatchBackgroundJobStatusResponse]:
        """Return one raw platform batch, or None when the endpoint is unavailable."""
        if self._background_job_status_batch_supported is False:
            return None
        try:
            response = await self.client.request(
                "POST",
                "/sandbox/background-jobs/status:batchGet",
                json={
                    "jobs": [{"sandbox_id": job.sandbox_id, "job_id": job.job_id} for job in jobs]
                },
                timeout=timeout if timeout is not None else 30,
                idempotent_post=True,
            )
        except APIError as exc:
            if "HTTP 404" in str(exc) or "HTTP 405" in str(exc):
                self._background_job_status_batch_supported = False
                return None
            raise
        self._background_job_status_batch_supported = True
        return BatchBackgroundJobStatusResponse.model_validate(response)

    async def get_background_jobs(
        self,
        jobs: List[BackgroundJob],
        timeout: Optional[int] = None,
    ) -> List[BackgroundJobStatus]:
        """Get ordered status for up to 100 jobs across VM sandboxes."""
        _validate_background_job_batch(jobs)
        body = await self._request_background_job_status_batch(jobs, timeout)
        if body is None:
            return await self._get_background_jobs_legacy(jobs, timeout)
        if body.errors:
            details = "; ".join(
                f"{error.sandbox_id}/{error.job_id}: {error.message}" for error in body.errors
            )
            if any(error.code == "NOT_VM" for error in body.errors):
                raise BatchStatusUnsupportedError(details)
            raise APIError(f"Background job batch status failed: {details}")
        runtime_statuses = {(status.sandbox_id, status.job_id): status for status in body.statuses}

        async def build_status(job: BackgroundJob) -> BackgroundJobStatus:
            runtime_status = runtime_statuses.get((job.sandbox_id, job.job_id))
            if runtime_status is None:
                raise APIError(f"VM batch status response omitted job {job.job_id}")
            if not runtime_status.completed:
                return BackgroundJobStatus(job_id=job.job_id, completed=False)
            if runtime_status.exit_code is None:
                raise APIError(f"Completed VM background job {job.job_id} omitted exit_code")
            return await self._get_completed_background_job_output(
                job.sandbox_id,
                job,
                runtime_status.exit_code,
                timeout,
            )

        return list(await asyncio.gather(*(build_status(job) for job in jobs)))

    async def _get_background_jobs_legacy(
        self,
        jobs: List[BackgroundJob],
        timeout: Optional[int],
    ) -> List[BackgroundJobStatus]:
        """Use VM-only per-job reads when the platform batch API is unavailable."""
        for sandbox_id in dict.fromkeys(job.sandbox_id for job in jobs):
            await self._auth_cache.get_or_refresh(sandbox_id)
            if not await self._auth_cache.is_vm(sandbox_id):
                raise BatchStatusUnsupportedError(
                    "Batched background job status is only supported for VM sandboxes."
                )
        return list(
            await asyncio.gather(
                *(self.get_background_job(job.sandbox_id, job, timeout=timeout) for job in jobs)
            )
        )

    async def _get_completed_background_job_output(
        self,
        sandbox_id: str,
        job: BackgroundJob,
        exit_code: int,
        timeout: Optional[int],
    ) -> BackgroundJobStatus:
        """Read bounded stdout and stderr tails for one completed job."""

        async def read_output_tail(path: str) -> tuple[str, bool]:
            try:
                response = await self.read_file(
                    sandbox_id,
                    path,
                    timeout=timeout,
                    offset=-JOB_OUTPUT_TAIL_BYTES,
                    length=JOB_OUTPUT_TAIL_BYTES,
                )
                return response.content, bool(response.truncated)
            except SandboxFileNotFoundError:
                return "", False

        stdout, stderr = await asyncio.gather(
            read_output_tail(job.stdout_log_file),
            read_output_tail(job.stderr_log_file),
        )
        return BackgroundJobStatus(
            job_id=job.job_id,
            completed=True,
            exit_code=exit_code,
            stdout=stdout[0],
            stderr=stderr[0],
            stdout_truncated=stdout[1],
            stderr_truncated=stderr[1],
        )

    async def _fetch_background_job_statuses(
        self, keys: List[tuple[str, str]]
    ) -> Dict[tuple[str, str], BackgroundJobStatus | _BatchItemError]:
        """Fetch one coalesced VM job batch for concurrent run waiters."""
        jobs = [_canonical_background_job(sandbox_id, job_id) for sandbox_id, job_id in keys]
        body = await self._request_background_job_status_batch(jobs)
        if body is None:

            async def get_legacy_status(
                job: BackgroundJob,
            ) -> BackgroundJobStatus | _BatchItemError:
                try:
                    return (await self._get_background_jobs_legacy([job], None))[0]
                except Exception as exc:
                    return _BatchItemError(exc)

            statuses = await asyncio.gather(*(get_legacy_status(job) for job in jobs))
            return {(job.sandbox_id, job.job_id): status for job, status in zip(jobs, statuses)}

        results: Dict[tuple[str, str], BackgroundJobStatus | _BatchItemError] = {}
        for error in body.errors:
            key = (error.sandbox_id, error.job_id)
            details = f"{error.sandbox_id}/{error.job_id}: {error.message}"
            exc = (
                BatchStatusUnsupportedError(details)
                if error.code == "NOT_VM"
                else APIError(f"Background job batch status failed: {details}")
            )
            results[key] = _BatchItemError(exc)

        runtime_statuses = {(status.sandbox_id, status.job_id): status for status in body.statuses}

        async def build_status(job: BackgroundJob) -> BackgroundJobStatus | _BatchItemError | None:
            key = (job.sandbox_id, job.job_id)
            if key in results:
                return None
            runtime_status = runtime_statuses.get(key)
            if runtime_status is None:
                return None
            if not runtime_status.completed:
                return BackgroundJobStatus(job_id=job.job_id, completed=False)
            if runtime_status.exit_code is None:
                return _BatchItemError(
                    APIError(f"Completed VM background job {job.job_id} omitted exit_code")
                )
            try:
                return await self._get_completed_background_job_output(
                    job.sandbox_id,
                    job,
                    runtime_status.exit_code,
                    None,
                )
            except Exception as exc:
                return _BatchItemError(exc)

        statuses = await asyncio.gather(*(build_status(job) for job in jobs))
        for job, status in zip(jobs, statuses):
            if status is not None:
                results[(job.sandbox_id, job.job_id)] = status
        return results

    async def run_background_job(
        self,
        sandbox_id: str,
        command: str,
        timeout: int = 900,
        working_dir: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        poll_interval: int = 3,
    ) -> BackgroundJobStatus:
        """Run a command in the background and wait for completion (async).

        Combines start_background_job() + polling into a single call.
        Use this for long-running commands that would exceed HTTP timeouts
        with execute_command().

        Args:
            sandbox_id: The sandbox ID
            command: Command to execute
            timeout: Maximum seconds to wait for completion
            working_dir: Working directory for command execution
            env: Environment variables
            poll_interval: Seconds between status polls

        Returns:
            BackgroundJobStatus with exit_code, stdout, stderr

        Raises:
            CommandTimeoutError: If command doesn't complete within timeout
        """
        job = await self.start_background_job(sandbox_id, command, working_dir=working_dir, env=env)
        use_batch_status = await self._auth_cache.is_vm(sandbox_id)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if use_batch_status:
                status = await self._background_job_status_batcher.get((sandbox_id, job.job_id))
            else:
                status = await self.get_background_job(sandbox_id, job)
            if status.completed:
                return status
            await asyncio.sleep(poll_interval)
        raise CommandTimeoutError(sandbox_id, command, timeout)

    async def wait_for_creation(
        self,
        sandbox_id: str,
        max_attempts: int = 60,
        stability_checks: int = 1,
        image_build_timeout_seconds: int = 3000,
    ) -> None:
        """Wait for sandbox to be running and stable (async version).

        Args:
            sandbox_id: The sandbox ID to wait for
            max_attempts: Defines the wall-clock budget for the wait, expressed
                in polls of the legacy fixed-interval schedule (see
                `_creation_timeout_seconds`). Status polls now back off, so this
                bounds elapsed time rather than the literal number of requests.
                Reaching RUNNING starts a fresh budget of the same size for the
                reachability phase, so a wait that gets that far can take up to
                twice this long.
            stability_checks: Number of consecutive successful reachability checks required
            image_build_timeout_seconds: Separate wall-clock budget while the
                platform auto-builds the VM image for a first-use image (the
                sandbox stays PENDING with pending_image_build_id set). That
                phase polls slowly and does not consume the creation budget.
        """
        consecutive_successes = 0
        image_build_deadline: Optional[float] = None
        deadline = time.monotonic() + _creation_timeout_seconds(max_attempts)
        poll_index = 0
        reachability_phase = False
        while time.monotonic() < deadline:
            sandbox = await self._sandbox_status_batcher.get(sandbox_id)
            if sandbox.status == "RUNNING":
                if not reachability_phase:
                    reachability_phase = True
                    deadline = time.monotonic() + _creation_timeout_seconds(max_attempts)
                    poll_index = 0
                if await self._is_sandbox_reachable(sandbox_id):
                    consecutive_successes += 1
                    if consecutive_successes >= stability_checks:
                        return
                    # Small delay between stability checks
                    await asyncio.sleep(0.5)
                    continue
                else:
                    # Reset counter if check fails
                    consecutive_successes = 0
            elif sandbox.status in ["ERROR", "TERMINATED", "TIMEOUT"]:
                ctx = {
                    "status": sandbox.status,
                    "error_type": sandbox.error_type,
                    "error_message": sandbox.error_message,
                }
                _raise_not_running_error(sandbox.sandbox_id, ctx)
            elif _is_waiting_for_image_build(sandbox):
                # The platform is building the VM image for this sandbox; it
                # starts on its own once the build completes. This phase runs on
                # its own budget, so hold the creation deadline back while it
                # lasts and reset the backoff for when the sandbox starts.
                if image_build_deadline is None:
                    image_build_deadline = time.monotonic() + image_build_timeout_seconds
                if time.monotonic() >= image_build_deadline:
                    raise SandboxNotRunningError(
                        sandbox_id, "Timeout waiting for the VM image build"
                    )
                await asyncio.sleep(10)
                deadline = time.monotonic() + _creation_timeout_seconds(max_attempts)
                poll_index = 0
                continue

            # Never sleep past the deadline. The loop only re-checks it on the
            # next iteration, so an uncapped backoff delay would let the wait
            # run up to CREATION_POLL_MAX_DELAY (plus jitter) beyond the budget.
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            await asyncio.sleep(min(_creation_poll_delay(poll_index), remaining))
            poll_index += 1
        raise SandboxNotRunningError(sandbox_id, "Timeout during sandbox creation")

    async def bulk_wait_for_creation(
        self,
        sandbox_ids: List[str],
        max_attempts: int = 60,
        image_build_timeout_seconds: int = 3000,
    ) -> Dict[str, str]:
        """Wait for up to 100 sandboxes using the batch lifecycle endpoint.

        Sandboxes PENDING on an automatic VM image build (first use of an
        image) are waited on a separate slower budget bounded by
        image_build_timeout_seconds instead of consuming max_attempts.
        """

        _validate_unique_batch_values(sandbox_ids, "sandbox_ids")
        final_statuses: Dict[str, str] = {}
        image_build_deadline: Optional[float] = None

        attempt = 0
        while attempt < max_attempts:
            try:
                response = await self.get_sandbox_statuses(sandbox_ids)
            except BatchStatusUnsupportedError:
                outcomes = await self._fetch_sandbox_statuses(sandbox_ids)
                snapshots = []
                for sandbox_id in sandbox_ids:
                    outcome = outcomes[sandbox_id]
                    if isinstance(outcome, _BatchItemError):
                        raise outcome.error
                    snapshots.append(outcome)
                response = BatchSandboxStatusResponse(
                    statuses=snapshots,
                    errors=[],
                )
            except Exception as exc:
                if "429" in str(exc) or "Too Many Requests" in str(exc):
                    await asyncio.sleep(min(2**attempt, 60))
                    continue
                raise

            if response.errors:
                failures = [(error.sandbox_id, error.code) for error in response.errors]
                raise RuntimeError(f"Sandboxes unavailable: {failures}")

            total_running = 0
            all_failed = []
            total_image_build_waiting = 0
            for snapshot in response.statuses:
                status_value = snapshot.status.value
                if status_value == "RUNNING":
                    total_running += 1
                    final_statuses[snapshot.sandbox_id] = status_value
                elif status_value in ["ERROR", "TERMINATED", "TIMEOUT"]:
                    all_failed.append((snapshot.sandbox_id, status_value))
                    final_statuses[snapshot.sandbox_id] = status_value
                elif _is_waiting_for_image_build(snapshot):
                    total_image_build_waiting += 1

            if all_failed:
                raise RuntimeError(f"Sandboxes failed: {all_failed}")

            if total_running == len(sandbox_ids):
                all_reachable = True
                for sandbox_id in sandbox_ids:
                    if final_statuses.get(sandbox_id) == "RUNNING":
                        if not await self._is_sandbox_reachable(sandbox_id):
                            all_reachable = False
                            final_statuses.pop(sandbox_id, None)

                if all_reachable:
                    return final_statuses

            if total_image_build_waiting:
                # At least one sandbox is waiting on an automatic VM image
                # build; poll slowly on the image-build budget instead of
                # consuming the normal attempts.
                if image_build_deadline is None:
                    image_build_deadline = time.monotonic() + image_build_timeout_seconds
                if time.monotonic() < image_build_deadline:
                    await asyncio.sleep(10)
                    continue
                break

            attempt += 1
            sleep_time = 1 if attempt <= 5 else 2
            await asyncio.sleep(sleep_time)

        for sandbox_id in sandbox_ids:
            if sandbox_id not in final_statuses:
                final_statuses[sandbox_id] = "TIMEOUT"

        raise RuntimeError(f"Timeout waiting for sandboxes to be ready. Status: {final_statuses}")

    async def upload_file(
        self,
        sandbox_id: str,
        file_path: str,
        local_file_path: str,
        timeout: Optional[int] = None,
    ) -> FileUploadResponse:
        """Upload a file to a sandbox via gateway (async)

        Uses aiofiles for non-blocking file I/O, then passes content to httpx.
        File content is loaded into memory, suitable for typical sandbox files.

        Args:
            sandbox_id: The sandbox ID
            file_path: Remote path in the sandbox
            local_file_path: Local file path to upload
            timeout: Optional timeout in seconds
        """
        if not await asyncio.to_thread(os.path.exists, local_file_path):
            raise FileNotFoundError(f"Local file not found: {local_file_path}")

        params = {"path": file_path, "sandbox_id": sandbox_id}

        effective_timeout = timeout if timeout is not None else 300

        # Read file asynchronously (non-blocking I/O)
        async with aiofiles.open(local_file_path, "rb") as f:
            file_content = await f.read()

        reauthed = False
        attempt = 0
        for _ in range(MAX_GATEWAY_ATTEMPTS):
            auth = await self._auth_cache.get_or_refresh(sandbox_id)
            gateway_url = auth["gateway_url"].rstrip("/")
            url = f"{gateway_url}/{auth['user_ns']}/{auth['job_id']}/upload"
            headers = {"Authorization": f"Bearer {auth['token']}"}
            try:
                files = {"file": (os.path.basename(local_file_path), file_content)}
                response = await self._gateway_post(
                    url, headers=headers, timeout=effective_timeout, files=files, params=params
                )
                response.raise_for_status()
                return FileUploadResponse.model_validate(response.json())
            except httpx.TimeoutException as e:
                raise UploadTimeoutError(sandbox_id, file_path, effective_timeout) from e
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 401 and await self._should_retry_401(
                    sandbox_id, reauthed
                ):
                    reauthed = True
                    continue
                if e.response.status_code == 409:
                    if await self._should_retry_409(sandbox_id, e, attempt):
                        attempt += 1
                        continue
                elif await self._should_retry_upload_error(e, attempt):
                    attempt += 1
                    continue
                error_details = (
                    f"HTTP {e.response.status_code} {e.request.method} "
                    f"{e.request.url}: {e.response.text}"
                )
                raise APIError(f"Upload failed: {error_details}") from e
            except httpx.RequestError as e:
                req = getattr(e, "request", None)
                method = getattr(req, "method", "?")
                u = getattr(req, "url", "?")
                raise APIError(f"Upload failed: {e.__class__.__name__} at {method} {u}: {e}") from e
            except Exception as e:
                raise APIError(f"Upload failed: {e.__class__.__name__}: {e}") from e

        raise APIError("Upload failed after retries")

    async def upload_bytes(
        self,
        sandbox_id: str,
        file_path: str,
        file_bytes: bytes,
        filename: str,
        timeout: Optional[int] = None,
    ) -> FileUploadResponse:
        """Upload bytes directly to sandbox via gateway without writing to disk (async)

        Args:
            sandbox_id: The sandbox ID
            file_path: Remote path in the sandbox where the file will be saved
            file_bytes: The bytes content to upload
            filename: Name for the file (used in multipart form)
            timeout: Optional timeout in seconds
        """
        params = {"path": file_path, "sandbox_id": sandbox_id}

        effective_timeout = timeout if timeout is not None else 300

        reauthed = False
        attempt = 0
        for _ in range(MAX_GATEWAY_ATTEMPTS):
            auth = await self._auth_cache.get_or_refresh(sandbox_id)
            gateway_url = auth["gateway_url"].rstrip("/")
            url = f"{gateway_url}/{auth['user_ns']}/{auth['job_id']}/upload"
            headers = {"Authorization": f"Bearer {auth['token']}"}
            try:
                files = {"file": (filename, file_bytes)}
                response = await self._gateway_post(
                    url, headers=headers, timeout=effective_timeout, files=files, params=params
                )
                response.raise_for_status()
                return FileUploadResponse.model_validate(response.json())
            except httpx.TimeoutException:
                raise UploadTimeoutError(sandbox_id, file_path, effective_timeout)
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 401 and await self._should_retry_401(
                    sandbox_id, reauthed
                ):
                    reauthed = True
                    continue
                if e.response.status_code == 409:
                    if await self._should_retry_409(sandbox_id, e, attempt):
                        attempt += 1
                        continue
                elif await self._should_retry_upload_error(e, attempt):
                    attempt += 1
                    continue
                error_details = f"HTTP {e.response.status_code}: {e.response.text}"
                raise APIError(f"Upload failed: {error_details}")
            except Exception as e:
                raise APIError(f"Upload failed: {str(e)}")

        raise APIError("Upload failed after retries")

    async def download_file(
        self,
        sandbox_id: str,
        file_path: str,
        local_file_path: str,
        timeout: Optional[int] = None,
    ) -> None:
        """Download a file from a sandbox via gateway (async)"""
        params = {"path": file_path, "sandbox_id": sandbox_id}

        effective_timeout = timeout if timeout is not None else 300

        reauthed = False
        attempt = 0
        for _ in range(MAX_GATEWAY_ATTEMPTS):
            auth = await self._auth_cache.get_or_refresh(sandbox_id)
            gateway_url = auth["gateway_url"].rstrip("/")
            url = f"{gateway_url}/{auth['user_ns']}/{auth['job_id']}/download"
            headers = {"Authorization": f"Bearer {auth['token']}"}
            try:
                response = await self._gateway_get(
                    url, headers=headers, params=params, timeout=effective_timeout
                )
                response.raise_for_status()
                content = response.content

                dir_path = os.path.dirname(local_file_path)
                if dir_path:
                    await asyncio.to_thread(os.makedirs, dir_path, exist_ok=True)

                # Write file asynchronously (non-blocking I/O)
                async with aiofiles.open(local_file_path, "wb") as f:
                    await f.write(content)
                return
            except httpx.TimeoutException as e:
                raise DownloadTimeoutError(sandbox_id, file_path, effective_timeout) from e
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 401 and await self._should_retry_401(
                    sandbox_id, reauthed
                ):
                    reauthed = True
                    continue
                if e.response.status_code == 409:
                    if await self._should_retry_409(sandbox_id, e, attempt):
                        attempt += 1
                        continue
                error_details = (
                    f"HTTP {e.response.status_code} {e.request.method} "
                    f"{e.request.url}: {e.response.text}"
                )
                raise APIError(f"Download failed: {error_details}") from e
            except httpx.RequestError as e:
                req = getattr(e, "request", None)
                method = getattr(req, "method", "?")
                u = getattr(req, "url", "?")
                raise APIError(
                    f"Download failed: {e.__class__.__name__} at {method} {u}: {e}"
                ) from e
            except Exception as e:
                raise APIError(f"Download failed: {e.__class__.__name__}: {e}") from e

        raise APIError("Download failed after retries")

    async def read_file(
        self,
        sandbox_id: str,
        file_path: str,
        timeout: Optional[int] = None,
        offset: Optional[int] = None,
        length: Optional[int] = None,
    ) -> ReadFileResponse:
        """Read a file (or a byte window of it) from a sandbox via gateway (async).

        offset/length require server-side windowed-read support. VM sandboxes
        don't support it yet: they ignore both params, return the whole file
        (subject to the read size limit), and omit total_size/offset/truncated
        from the response (detectable via ``response.offset is None``).
        """
        params: Dict[str, Any] = {"path": file_path}
        if offset is not None:
            params["offset"] = offset
        if length is not None:
            params["length"] = length

        effective_timeout = timeout if timeout is not None else 30

        reauthed = False
        attempt = 0
        for _ in range(MAX_GATEWAY_ATTEMPTS):
            auth = await self._auth_cache.get_or_refresh(sandbox_id)
            gateway_url = auth["gateway_url"].rstrip("/")
            url = f"{gateway_url}/{auth['user_ns']}/{auth['job_id']}/read-file"
            headers = {"Authorization": f"Bearer {auth['token']}"}
            try:
                response = await self._gateway_read_file_get(
                    url, headers=headers, params=params, timeout=effective_timeout
                )
                response.raise_for_status()
                return ReadFileResponse.model_validate(response.json())
            except httpx.TimeoutException as e:
                raise APIError(
                    f"Read file timed out after {effective_timeout}s "
                    f"({e.__class__.__name__}): {file_path}"
                ) from e
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 401 and await self._should_retry_401(
                    sandbox_id, reauthed
                ):
                    reauthed = True
                    continue
                if e.response.status_code == 404:
                    raise SandboxFileNotFoundError(f"File not found: {file_path}") from e
                if e.response.status_code == 413:
                    raise SandboxFileTooLargeError(
                        f"File too large to read: {file_path}: {e.response.text}"
                    ) from e
                if e.response.status_code == 409:
                    if await self._should_retry_409(sandbox_id, e, attempt):
                        attempt += 1
                        continue
                error_details = (
                    f"HTTP {e.response.status_code} {e.request.method} "
                    f"{e.request.url}: {e.response.text}"
                )
                raise APIError(f"Read file failed: {error_details}") from e
            except httpx.RequestError as e:
                req = getattr(e, "request", None)
                method = getattr(req, "method", "?")
                u = getattr(req, "url", "?")
                raise APIError(
                    f"Read file failed: {e.__class__.__name__} at {method} {u}: {e}"
                ) from e
            except Exception as e:
                raise APIError(f"Read file failed: {e.__class__.__name__}: {e}") from e

        raise APIError("Read file failed after retries")

    async def aclose(self) -> None:
        """Close the async client and gateway client"""
        if self._gateway_client is not None:
            await self._gateway_client.aclose()
        await self.client.aclose()

    async def __aenter__(self) -> "AsyncSandboxClient":
        """Async context manager entry"""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit"""
        await self.aclose()

    async def expose(
        self,
        sandbox_id: str,
        port: int,
        name: Optional[str] = None,
        protocol: str = "HTTP",
    ) -> ExposedPort:
        """Expose a port from a sandbox."""
        await self._guard_vm_unsupported(sandbox_id, "Port exposure")
        request = ExposePortRequest(port=port, name=name, protocol=protocol)
        response = await self.client.request(
            "POST",
            f"/sandbox/{sandbox_id}/expose",
            json=request.model_dump(by_alias=False, exclude_none=True),
        )
        return ExposedPort.model_validate(response)

    async def unexpose(self, sandbox_id: str, exposure_id: str) -> None:
        """Unexpose a port from a sandbox."""
        await self._guard_vm_unsupported(sandbox_id, "Port unexpose")
        await self.client.request("DELETE", f"/sandbox/{sandbox_id}/expose/{exposure_id}")

    async def list_exposed_ports(self, sandbox_id: str) -> ListExposedPortsResponse:
        """List all exposed ports for a sandbox"""
        await self._guard_vm_unsupported(sandbox_id, "Port listing")
        response = await self.client.request("GET", f"/sandbox/{sandbox_id}/expose")
        return ListExposedPortsResponse.model_validate(response)

    async def list_all_exposed_ports(self) -> ListExposedPortsResponse:
        """List all exposed ports across all sandboxes for the current user"""
        response = await self.client.request("GET", "/sandbox/expose/all")
        return ListExposedPortsResponse.model_validate(response)

    async def create_ssh_session(
        self,
        sandbox_id: str,
        ttl_seconds: Optional[int] = None,
    ) -> SSHSession:
        """Create an SSH session"""
        await self._guard_vm_unsupported(sandbox_id, "SSH")
        payload: Dict[str, Any] = {}
        if ttl_seconds is not None:
            payload["ttl_seconds"] = ttl_seconds
        response = await self.client.request(
            "POST",
            f"/sandbox/{sandbox_id}/ssh-session",
            json=payload,
        )
        return SSHSession.model_validate(response)

    async def close_ssh_session(self, sandbox_id: str, session_id: str) -> None:
        """Close an SSH session and remove its exposure"""
        await self._guard_vm_unsupported(sandbox_id, "SSH")
        await self.client.request("DELETE", f"/sandbox/{sandbox_id}/ssh-session/{session_id}")


class TemplateClient:
    """Client for template/registry helper APIs."""

    def __init__(self, api_client: Optional[APIClient] = None):
        self.client = api_client or APIClient()

    def list_registry_credentials(self) -> List[RegistryCredentialSummary]:
        response = self.client.request("GET", "/template/registry-credentials")
        credentials = response.get("credentials", [])
        return [RegistryCredentialSummary.model_validate(item) for item in credentials]

    def check_docker_image(
        self, image: str, registry_credentials_id: Optional[str] = None
    ) -> DockerImageCheckResponse:
        payload: Dict[str, Any] = {"image": image}
        if registry_credentials_id:
            payload["registry_credentials_id"] = registry_credentials_id
        response = self.client.request(
            "POST",
            "/template/check-docker-image",
            json=payload,
        )
        return DockerImageCheckResponse.model_validate(response)


class AsyncTemplateClient:
    """Async client for template/registry helper APIs."""

    def __init__(self, api_client: Optional[AsyncAPIClient] = None):
        self.client = api_client or AsyncAPIClient()

    async def list_registry_credentials(self) -> List[RegistryCredentialSummary]:
        response = await self.client.request("GET", "/template/registry-credentials")
        credentials = response.get("credentials", [])
        return [RegistryCredentialSummary.model_validate(item) for item in credentials]

    async def check_docker_image(
        self, image: str, registry_credentials_id: Optional[str] = None
    ) -> DockerImageCheckResponse:
        payload: Dict[str, Any] = {"image": image}
        if registry_credentials_id:
            payload["registry_credentials_id"] = registry_credentials_id
        response = await self.client.request(
            "POST",
            "/template/check-docker-image",
            json=payload,
        )
        return DockerImageCheckResponse.model_validate(response)

    async def aclose(self) -> None:
        """Close the async client"""
        await self.client.aclose()

    async def __aenter__(self) -> "AsyncTemplateClient":
        """Async context manager entry"""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit"""
        await self.aclose()
