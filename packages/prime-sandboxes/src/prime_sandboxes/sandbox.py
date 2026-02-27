"""Sandbox client implementations."""

import asyncio
import json
import os
import re
import shlex
import sys
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, NoReturn, Optional

import aiofiles
import httpx
from connectrpc.client import ConnectClient, ConnectClientSync
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from tenacity import (
    retry,
    retry_if_exception,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .core import APIClient, APIError, AsyncAPIClient
from .exceptions import (
    CommandTimeoutError,
    DownloadTimeoutError,
    SandboxImagePullError,
    SandboxNotRunningError,
    SandboxOOMError,
    SandboxTimeoutError,
    UploadTimeoutError,
)
from .models import (
    BackgroundJob,
    BackgroundJobStatus,
    BulkDeleteSandboxRequest,
    BulkDeleteSandboxResponse,
    CommandResponse,
    CreateSandboxRequest,
    DockerImageCheckResponse,
    ExposedPort,
    ExposePortRequest,
    FileUploadResponse,
    ListExposedPortsResponse,
    RegistryCredentialSummary,
    Sandbox,
    SandboxListResponse,
    SandboxLogsResponse,
    SSHSession,
)
from .rpc_command_session import (
    COMMAND_SESSION_START_RPC_METHOD,
    build_command_session_start_request,
    collect_command_session_start_event,
)

# Retry configuration for transient connection errors on gateway requests
# Note: ReadTimeout is NOT included because the request may have been processed
GATEWAY_RETRYABLE_EXCEPTIONS = (
    httpx.RemoteProtocolError,  # Server disconnected unexpectedly
    httpx.ConnectError,  # Connection refused/failed
    httpx.PoolTimeout,  # No connection available in pool
)

# Retryable HTTP 5xx status codes (e.g. Cloudflare 524 timeout, server errors)
RETRYABLE_5XX_STATUSES = frozenset({500, 502, 503, 504, 524})

# Max retries for transient 409 errors
MAX_409_RETRIES = 4
RETRY_409_BASE_DELAY = 0.25  # 250ms, 500ms, 1000ms, 2000ms with exponential backoff


def _is_retryable_gateway_error(exc: BaseException) -> bool:
    """Check if an exception is retryable for gateway requests."""
    if isinstance(exc, GATEWAY_RETRYABLE_EXCEPTIONS):
        return True
    if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code in RETRYABLE_5XX_STATUSES:
        return True
    return False


# Retry decorator for gateway requests (connection errors + 5xx responses)
_gateway_retry = retry(
    retry=retry_if_exception(_is_retryable_gateway_error),
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


class SandboxAuthCache:
    """Shared auth cache management for sandbox clients"""

    def __init__(self, cache_file_path: Any, client: Any) -> None:
        self._cache_file = cache_file_path
        self._auth_cache = self._load_cache()
        self.client = client

    def _load_cache(self) -> Dict[str, Any]:
        """Load auth cache from file and clean expired entries"""
        try:
            if self._cache_file.exists():
                with open(self._cache_file, "r") as f:
                    cache = json.load(f)
                cleaned_cache = {}
                for sandbox_id, auth_info in cache.items():
                    try:
                        expires_at_str = auth_info["expires_at"].replace("Z", "+00:00")
                        expires_at = datetime.fromisoformat(expires_at_str)
                        if expires_at.tzinfo is None:
                            expires_at = expires_at.replace(tzinfo=timezone.utc)
                        now = datetime.now(timezone.utc)
                        if now < expires_at:
                            cleaned_cache[sandbox_id] = auth_info
                    except Exception:
                        pass

                if len(cleaned_cache) != len(cache):
                    self._auth_cache = cleaned_cache
                    self._save_cache()

                return cleaned_cache
        except Exception:
            pass
        return {}

    def _save_cache(self) -> None:
        """Save auth cache to file"""
        try:
            self._cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self._cache_file, "w") as f:
                json.dump(self._auth_cache, f)
        except Exception:
            pass

    async def _save_cache_async(self) -> None:
        """Save auth cache to file (async version)"""
        try:
            self._cache_file.parent.mkdir(parents=True, exist_ok=True)
            async with aiofiles.open(self._cache_file, "w") as f:
                await f.write(json.dumps(self._auth_cache))
        except Exception:
            pass

    def _check_cached_auth(self, sandbox_id: str) -> Optional[Dict[str, Any]]:
        """Check if cached auth info exists and is valid"""
        if sandbox_id in self._auth_cache:
            auth_info = self._auth_cache[sandbox_id]
            expires_at_str = auth_info["expires_at"].replace("Z", "+00:00")
            expires_at = datetime.fromisoformat(expires_at_str)
            if expires_at.tzinfo is None:
                expires_at = expires_at.replace(tzinfo=timezone.utc)
            if datetime.now(timezone.utc) < expires_at:
                return dict(auth_info)
            else:
                del self._auth_cache[sandbox_id]
                self._save_cache()
        return None

    async def _check_cached_auth_async(self, sandbox_id: str) -> Optional[Dict[str, Any]]:
        """Check if cached auth info exists and is valid (async version)"""
        if sandbox_id in self._auth_cache:
            auth_info = self._auth_cache[sandbox_id]
            expires_at_str = auth_info["expires_at"].replace("Z", "+00:00")
            expires_at = datetime.fromisoformat(expires_at_str)
            if expires_at.tzinfo is None:
                expires_at = expires_at.replace(tzinfo=timezone.utc)
            if datetime.now(timezone.utc) < expires_at:
                return dict(auth_info)
            else:
                del self._auth_cache[sandbox_id]
                await self._save_cache_async()
        return None

    def get_or_refresh(self, sandbox_id: str) -> Dict[str, Any]:
        """Get cached auth info or fetch new token if expired/missing"""
        cached_auth = self._check_cached_auth(sandbox_id)
        if cached_auth:
            return cached_auth

        response = self.client.request("POST", f"/sandbox/{sandbox_id}/auth")
        self.set(sandbox_id, response)
        self._save_cache()
        return dict(response)

    async def get_or_refresh_async(self, sandbox_id: str) -> Dict[str, Any]:
        """Get cached auth info or fetch new token if expired/missing (async)"""
        cached_auth = await self._check_cached_auth_async(sandbox_id)
        if cached_auth:
            return cached_auth
        response = await self.client.request("POST", f"/sandbox/{sandbox_id}/auth")
        self._auth_cache[sandbox_id] = response
        await self._save_cache_async()
        return dict(response)

    def is_gpu(self, sandbox_id: str) -> bool:
        """Return True if sandbox is GPU-backed, cached alongside auth token data."""
        cached_auth = self._check_cached_auth(sandbox_id)
        if cached_auth and isinstance(cached_auth.get("is_gpu"), bool):
            return bool(cached_auth["is_gpu"])

        sandbox_data = self.client.request("GET", f"/sandbox/{sandbox_id}")
        sandbox = Sandbox.model_validate(sandbox_data)
        is_gpu = sandbox.gpu_count > 0

        if sandbox_id in self._auth_cache:
            self._auth_cache[sandbox_id]["is_gpu"] = is_gpu
            self._save_cache()

        return is_gpu

    async def is_gpu_async(self, sandbox_id: str) -> bool:
        """Return True if sandbox is GPU-backed, cached alongside auth token data."""
        cached_auth = await self._check_cached_auth_async(sandbox_id)
        if cached_auth and isinstance(cached_auth.get("is_gpu"), bool):
            return bool(cached_auth["is_gpu"])

        sandbox_data = await self.client.request("GET", f"/sandbox/{sandbox_id}")
        sandbox = Sandbox.model_validate(sandbox_data)
        is_gpu = sandbox.gpu_count > 0

        if sandbox_id in self._auth_cache:
            self._auth_cache[sandbox_id]["is_gpu"] = is_gpu
            await self._save_cache_async()

        return is_gpu

    def set(self, sandbox_id: str, auth_info: Dict[str, Any]) -> None:
        """Cache auth info"""
        self._auth_cache[sandbox_id] = auth_info
        self._save_cache()

    def clear(self) -> None:
        """Clear all cached auth tokens"""
        self._auth_cache = {}
        try:
            if self._cache_file.exists():
                self._cache_file.unlink()
        except Exception:
            pass


def _check_sandbox_statuses(
    sandboxes: List[Sandbox], target_ids: set
) -> tuple[int, List[tuple], Dict[str, str]]:
    """Helper function to check sandbox statuses

    Returns:
        tuple of (running_count, failed_sandboxes, final_statuses)
    """
    running_count = 0
    failed_sandboxes = []
    final_statuses = {}

    for sandbox in sandboxes:
        if sandbox.id in target_ids:
            if sandbox.status == "RUNNING":
                running_count += 1
                final_statuses[sandbox.id] = sandbox.status
            elif sandbox.status in ["ERROR", "TERMINATED", "TIMEOUT"]:
                failed_sandboxes.append((sandbox.id, sandbox.status))
                final_statuses[sandbox.id] = sandbox.status

    return running_count, failed_sandboxes, final_statuses


class SandboxClient:
    """Client for sandbox API operations"""

    def __init__(self, api_client: APIClient):
        self.client = api_client
        self._auth_cache = SandboxAuthCache(
            self.client.config.config_dir / "sandbox_auth_cache.json",
            self.client,
        )

    @staticmethod
    @_gateway_retry
    def _gateway_post(
        url: str,
        headers: Dict[str, str],
        timeout: float,
        json: Optional[Dict[str, Any]] = None,
        files: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> httpx.Response:
        """Make a POST request to the gateway with retry on transient errors."""
        with httpx.Client(timeout=timeout) as client:
            response = client.post(url, json=json, files=files, params=params, headers=headers)
        if response.status_code in RETRYABLE_5XX_STATUSES:
            response.raise_for_status()  # let _gateway_retry handle it
        return response

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

    def clear_auth_cache(self) -> None:
        """Clear all cached auth tokens"""
        self._auth_cache.clear()

    def create(self, request: CreateSandboxRequest) -> Sandbox:
        """Create a new sandbox"""
        # Auto-populate team_id from config if not specified
        if request.team_id is None:
            request.team_id = self.client.config.team_id

        response = self.client.request(
            "POST",
            "/sandbox",
            json=request.model_dump(by_alias=False, exclude_none=True),
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
    ) -> SandboxListResponse:
        """List sandboxes"""
        # Auto-populate team_id from config if not specified
        if team_id is None:
            team_id = self.client.config.team_id

        params: Dict[str, Any] = {"page": page, "per_page": per_page}
        if team_id:
            params["team_id"] = team_id
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

    def delete(self, sandbox_id: str) -> Dict[str, Any]:
        """Delete a sandbox"""
        response = self.client.request("DELETE", f"/sandbox/{sandbox_id}")
        return response

    def bulk_delete(
        self,
        sandbox_ids: Optional[List[str]] = None,
        labels: Optional[List[str]] = None,
    ) -> BulkDeleteSandboxResponse:
        """Bulk delete multiple sandboxes by IDs or labels (must specify one, not both)"""
        request = BulkDeleteSandboxRequest(sandbox_ids=sandbox_ids, labels=labels)
        response = self.client.request(
            "DELETE",
            "/sandbox",
            json=request.model_dump(by_alias=False, exclude_none=True),
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
    ) -> CommandResponse:
        """Execute command directly via gateway."""
        auth = self._auth_cache.get_or_refresh(sandbox_id)

        if self._auth_cache.is_gpu(sandbox_id):
            return self._execute_command_connect_rpc(
                sandbox_id=sandbox_id,
                command=command,
                auth=auth,
                working_dir=working_dir,
                env=env,
                timeout=timeout,
            )

        return self._execute_command_rest(
            sandbox_id=sandbox_id,
            command=command,
            auth=auth,
            working_dir=working_dir,
            env=env,
            timeout=timeout,
        )

    def _execute_command_connect_rpc(
        self,
        sandbox_id: str,
        command: str,
        auth: Dict[str, Any],
        working_dir: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None,
    ) -> CommandResponse:
        gateway_url = auth["gateway_url"].rstrip("/")
        base_url = f"{gateway_url}/{auth['user_ns']}/{auth['job_id']}"
        headers = {"Authorization": f"Bearer {auth['token']}"}
        effective_timeout = timeout if timeout is not None else 300
        request = build_command_session_start_request(command, working_dir, env)
        stdout_parts: List[str] = []
        stderr_parts: List[str] = []
        exit_code: Optional[int] = None

        rpc_client = ConnectClientSync(base_url)
        try:
            stream = rpc_client.execute_server_stream(
                request=request,
                method=COMMAND_SESSION_START_RPC_METHOD,
                headers=headers,
                timeout_ms=effective_timeout * 1000,
            )
            for event in stream:
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
        auth: Dict[str, Any],
        working_dir: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None,
    ) -> CommandResponse:
        gateway_url = auth["gateway_url"].rstrip("/")
        url = f"{gateway_url}/{auth['user_ns']}/{auth['job_id']}/exec"
        headers = {"Authorization": f"Bearer {auth['token']}"}
        effective_timeout = timeout if timeout is not None else 300

        payload = {
            "command": command,
            "working_dir": working_dir,
            "env": env or {},
            "sandbox_id": sandbox_id,
            "timeout": effective_timeout,
        }

        for attempt in range(MAX_409_RETRIES):
            try:
                # The + 2 accounts for connection creation and closing. Prevents any command
                # running close to its `effective_timeout` from being killed prematurely
                client_timeout = effective_timeout + 2
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
    ) -> BackgroundJob:
        """Start a long-running command in the background.

        Returns immediately with a job handle. Use get_background_job() to check
        status and retrieve results.

        Args:
            sandbox_id: The sandbox ID
            command: Command to execute
            working_dir: Working directory for command execution
            env: Environment variables

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
        self.execute_command(sandbox_id, bg_cmd, timeout=10)

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
    ) -> BackgroundJobStatus:
        """Check the status of a background job.

        Args:
            sandbox_id: The sandbox ID
            job: The BackgroundJob handle from start_background_job()

        Returns:
            BackgroundJobStatus with completed flag, and exit_code/stdout if done
        """
        exit_file_quoted = shlex.quote(job.exit_file)
        stdout_log_file_quoted = shlex.quote(job.stdout_log_file)
        stderr_log_file_quoted = shlex.quote(job.stderr_log_file)

        check = self.execute_command(sandbox_id, f"cat {exit_file_quoted} 2>/dev/null", timeout=30)

        if not check.stdout.strip():
            return BackgroundJobStatus(job_id=job.job_id, completed=False)

        exit_code = int(check.stdout.strip())
        stdout_logs = self.execute_command(sandbox_id, f"cat {stdout_log_file_quoted}", timeout=60)
        stderr_logs = self.execute_command(sandbox_id, f"cat {stderr_log_file_quoted}", timeout=60)

        return BackgroundJobStatus(
            job_id=job.job_id,
            completed=True,
            exit_code=exit_code,
            stdout=stdout_logs.stdout,
            stderr=stderr_logs.stdout,
        )

    def wait_for_creation(
        self, sandbox_id: str, max_attempts: int = 60, stability_checks: int = 1
    ) -> None:
        """Wait for sandbox to be running and stable.

        Args:
            sandbox_id: The sandbox ID to wait for
            max_attempts: Maximum polling attempts
            stability_checks: Number of consecutive successful reachability checks required
        """
        consecutive_successes = 0
        for attempt in range(max_attempts):
            sandbox = self.get(sandbox_id)
            if sandbox.status == "RUNNING":
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
                _raise_not_running_error(sandbox.id, ctx)

            # Aggressive polling for first 5 attempts (5 seconds), then back off
            sleep_time = 1 if attempt < 5 else 2
            time.sleep(sleep_time)
        raise SandboxNotRunningError(sandbox_id, "Timeout during sandbox creation")

    def bulk_wait_for_creation(
        self, sandbox_ids: List[str], max_attempts: int = 60
    ) -> Dict[str, str]:
        """Wait for multiple sandboxes to be running using list endpoint to avoid rate limits"""
        sandbox_id_set = set(sandbox_ids)
        final_statuses = {}

        for attempt in range(max_attempts):
            total_running = 0
            all_failed = []
            page = 1

            while True:
                try:
                    list_response = self.list(per_page=100, page=page)
                except Exception as e:
                    if "429" in str(e) or "Too Many Requests" in str(e):
                        wait_time = min(2**attempt, 60)
                        time.sleep(wait_time)
                        continue
                    raise

                running_count, failed_sandboxes, page_statuses = _check_sandbox_statuses(
                    list_response.sandboxes, sandbox_id_set
                )

                total_running += running_count
                all_failed.extend(failed_sandboxes)
                final_statuses.update(page_statuses)

                if len(final_statuses) == len(sandbox_ids) or not list_response.has_next:
                    break

                page += 1

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

            sleep_time = 1 if attempt < 5 else 2
            time.sleep(sleep_time)

        for sandbox_id in sandbox_id_set:
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

        auth = self._auth_cache.get_or_refresh(sandbox_id)

        url = f"{auth['gateway_url']}/{auth['user_ns']}/{auth['job_id']}/upload"
        headers = {"Authorization": f"Bearer {auth['token']}"}

        effective_timeout = timeout if timeout is not None else 300

        with open(local_file_path, "rb") as f:
            file_content = f.read()

        for attempt in range(MAX_409_RETRIES):
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
                if e.response.status_code == 409:
                    if self._should_retry_409(sandbox_id, e, attempt):
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
        auth = self._auth_cache.get_or_refresh(sandbox_id)

        url = f"{auth['gateway_url']}/{auth['user_ns']}/{auth['job_id']}/upload"
        headers = {"Authorization": f"Bearer {auth['token']}"}

        effective_timeout = timeout if timeout is not None else 300

        for attempt in range(MAX_409_RETRIES):
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
                if e.response.status_code == 409:
                    if self._should_retry_409(sandbox_id, e, attempt):
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
        auth = self._auth_cache.get_or_refresh(sandbox_id)

        url = f"{auth['gateway_url']}/{auth['user_ns']}/{auth['job_id']}/download"
        headers = {"Authorization": f"Bearer {auth['token']}"}
        params = {"path": file_path, "sandbox_id": sandbox_id}

        effective_timeout = timeout if timeout is not None else 300

        for attempt in range(MAX_409_RETRIES):
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
                if e.response.status_code == 409:
                    if self._should_retry_409(sandbox_id, e, attempt):
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

    def expose(
        self,
        sandbox_id: str,
        port: int,
        name: Optional[str] = None,
        protocol: str = "HTTP",
    ) -> ExposedPort:
        """Expose a port from a sandbox."""
        request = ExposePortRequest(port=port, name=name, protocol=protocol)
        response = self.client.request(
            "POST",
            f"/sandbox/{sandbox_id}/expose",
            json=request.model_dump(by_alias=False, exclude_none=True),
        )
        return ExposedPort.model_validate(response)

    def unexpose(self, sandbox_id: str, exposure_id: str) -> None:
        """Unexpose a port from a sandbox."""
        self.client.request("DELETE", f"/sandbox/{sandbox_id}/expose/{exposure_id}")

    def list_exposed_ports(self, sandbox_id: str) -> ListExposedPortsResponse:
        """List all exposed ports for a sandbox"""
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
        self._auth_cache = SandboxAuthCache(
            self.client.config.config_dir / "sandbox_auth_cache.json",
            self.client,
        )
        # Connection pool configuration
        self._max_connections = max_connections
        self._max_keepalive_connections = max_keepalive_connections
        # Shared httpx client for gateway operations (upload/download/execute)
        # Initialized lazily to allow connection pooling and reuse
        self._gateway_client: Optional[httpx.AsyncClient] = None

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

    @_gateway_retry
    async def _gateway_post(
        self,
        url: str,
        headers: Dict[str, str],
        timeout: float,
        json: Optional[Dict[str, Any]] = None,
        files: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> httpx.Response:
        """Make a POST request to the gateway with retry on transient errors."""
        gateway_client = self._get_gateway_client()
        response = await gateway_client.post(
            url, json=json, files=files, params=params, headers=headers, timeout=timeout
        )
        if response.status_code in RETRYABLE_5XX_STATUSES:
            response.raise_for_status()  # let _gateway_retry handle it
        return response

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

    def clear_auth_cache(self) -> None:
        """Clear all cached auth tokens"""
        self._auth_cache.clear()

    async def create(self, request: CreateSandboxRequest) -> Sandbox:
        """Create a new sandbox"""
        if request.team_id is None:
            request.team_id = self.client.config.team_id

        response = await self.client.request(
            "POST",
            "/sandbox",
            json=request.model_dump(by_alias=False, exclude_none=True),
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
    ) -> SandboxListResponse:
        """List sandboxes"""
        if team_id is None:
            team_id = self.client.config.team_id

        params: Dict[str, Any] = {"page": page, "per_page": per_page}
        if team_id:
            params["team_id"] = team_id
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

    async def delete(self, sandbox_id: str) -> Dict[str, Any]:
        """Delete a sandbox"""
        response = await self.client.request("DELETE", f"/sandbox/{sandbox_id}")
        return response

    async def bulk_delete(
        self,
        sandbox_ids: Optional[List[str]] = None,
        labels: Optional[List[str]] = None,
    ) -> BulkDeleteSandboxResponse:
        """Bulk delete multiple sandboxes by IDs or labels"""
        request = BulkDeleteSandboxRequest(sandbox_ids=sandbox_ids, labels=labels)
        response = await self.client.request(
            "DELETE",
            "/sandbox",
            json=request.model_dump(by_alias=False, exclude_none=True),
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
    ) -> CommandResponse:
        """Execute command directly via gateway (async)."""
        auth = await self._auth_cache.get_or_refresh_async(sandbox_id)

        if await self._auth_cache.is_gpu_async(sandbox_id):
            return await self._execute_command_connect_rpc(
                sandbox_id=sandbox_id,
                command=command,
                auth=auth,
                working_dir=working_dir,
                env=env,
                timeout=timeout,
            )

        return await self._execute_command_rest(
            sandbox_id=sandbox_id,
            command=command,
            auth=auth,
            working_dir=working_dir,
            env=env,
            timeout=timeout,
        )

    async def _execute_command_connect_rpc(
        self,
        sandbox_id: str,
        command: str,
        auth: Dict[str, Any],
        working_dir: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None,
    ) -> CommandResponse:
        gateway_url = auth["gateway_url"].rstrip("/")
        base_url = f"{gateway_url}/{auth['user_ns']}/{auth['job_id']}"
        headers = {"Authorization": f"Bearer {auth['token']}"}
        effective_timeout = timeout if timeout is not None else 300
        request = build_command_session_start_request(command, working_dir, env)
        stdout_parts: List[str] = []
        stderr_parts: List[str] = []
        exit_code: Optional[int] = None

        rpc_client = ConnectClient(base_url)
        try:
            stream = rpc_client.execute_server_stream(
                request=request,
                method=COMMAND_SESSION_START_RPC_METHOD,
                headers=headers,
                timeout_ms=effective_timeout * 1000,
            )
            async for event in stream:
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
        auth: Dict[str, Any],
        working_dir: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None,
    ) -> CommandResponse:
        gateway_url = auth["gateway_url"].rstrip("/")
        url = f"{gateway_url}/{auth['user_ns']}/{auth['job_id']}/exec"
        headers = {"Authorization": f"Bearer {auth['token']}"}
        effective_timeout = timeout if timeout is not None else 300

        payload = {
            "command": command,
            "working_dir": working_dir,
            "env": env or {},
            "sandbox_id": sandbox_id,
            "timeout": effective_timeout,
        }

        for attempt in range(MAX_409_RETRIES):
            try:
                # The + 2 accounts for connection creation and closing. Prevents any command
                # running close to its `effective_timeout` from being killed prematurely
                client_timeout = effective_timeout + 2
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
    ) -> BackgroundJob:
        """Start a long-running command in the background (async).

        Returns immediately with a job handle. Use get_background_job() to check
        status and retrieve results.

        Args:
            sandbox_id: The sandbox ID
            command: Command to execute
            working_dir: Working directory for command execution
            env: Environment variables

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
        await self.execute_command(sandbox_id, bg_cmd, timeout=10)

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
    ) -> BackgroundJobStatus:
        """Check the status of a background job (async).

        Args:
            sandbox_id: The sandbox ID
            job: The BackgroundJob handle from start_background_job()

        Returns:
            BackgroundJobStatus with completed flag, and exit_code/stdout if done
        """
        exit_file_quoted = shlex.quote(job.exit_file)
        stdout_log_file_quoted = shlex.quote(job.stdout_log_file)
        stderr_log_file_quoted = shlex.quote(job.stderr_log_file)

        check = await self.execute_command(
            sandbox_id, f"cat {exit_file_quoted} 2>/dev/null", timeout=30
        )

        if not check.stdout.strip():
            return BackgroundJobStatus(job_id=job.job_id, completed=False)

        exit_code = int(check.stdout.strip())
        stdout_logs = await self.execute_command(
            sandbox_id, f"cat {stdout_log_file_quoted}", timeout=60
        )
        stderr_logs = await self.execute_command(
            sandbox_id, f"cat {stderr_log_file_quoted}", timeout=60
        )

        return BackgroundJobStatus(
            job_id=job.job_id,
            completed=True,
            exit_code=exit_code,
            stdout=stdout_logs.stdout,
            stderr=stderr_logs.stdout,
        )

    async def wait_for_creation(
        self, sandbox_id: str, max_attempts: int = 60, stability_checks: int = 1
    ) -> None:
        """Wait for sandbox to be running and stable (async version).

        Args:
            sandbox_id: The sandbox ID to wait for
            max_attempts: Maximum polling attempts
            stability_checks: Number of consecutive successful reachability checks required
        """
        consecutive_successes = 0
        for attempt in range(max_attempts):
            sandbox = await self.get(sandbox_id)
            if sandbox.status == "RUNNING":
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
                _raise_not_running_error(sandbox.id, ctx)

            sleep_time = 1 if attempt < 5 else 2
            await asyncio.sleep(sleep_time)
        raise SandboxNotRunningError(sandbox_id, "Timeout during sandbox creation")

    async def bulk_wait_for_creation(
        self, sandbox_ids: List[str], max_attempts: int = 60
    ) -> Dict[str, str]:
        """Wait for multiple sandboxes to be running using list endpoint"""

        sandbox_id_set = set(sandbox_ids)
        final_statuses = {}

        for attempt in range(max_attempts):
            total_running = 0
            all_failed = []
            page = 1

            while True:
                try:
                    list_response = await self.list(per_page=100, page=page)
                except Exception as e:
                    if "429" in str(e) or "Too Many Requests" in str(e):
                        wait_time = min(2**attempt, 60)
                        await asyncio.sleep(wait_time)
                        continue
                    raise

                running_count, failed_sandboxes, page_statuses = _check_sandbox_statuses(
                    list_response.sandboxes, sandbox_id_set
                )

                total_running += running_count
                all_failed.extend(failed_sandboxes)
                final_statuses.update(page_statuses)

                if len(final_statuses) == len(sandbox_ids) or not list_response.has_next:
                    break

                page += 1

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

            sleep_time = 1 if attempt < 5 else 2
            await asyncio.sleep(sleep_time)

        for sandbox_id in sandbox_id_set:
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
        if not os.path.exists(local_file_path):
            raise FileNotFoundError(f"Local file not found: {local_file_path}")

        auth = await self._auth_cache.get_or_refresh_async(sandbox_id)

        gateway_url = auth["gateway_url"].rstrip("/")
        url = f"{gateway_url}/{auth['user_ns']}/{auth['job_id']}/upload"
        headers = {"Authorization": f"Bearer {auth['token']}"}
        params = {"path": file_path, "sandbox_id": sandbox_id}

        effective_timeout = timeout if timeout is not None else 300

        # Read file asynchronously (non-blocking I/O)
        async with aiofiles.open(local_file_path, "rb") as f:
            file_content = await f.read()

        for attempt in range(MAX_409_RETRIES):
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
                if e.response.status_code == 409:
                    if await self._should_retry_409(sandbox_id, e, attempt):
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
        auth = await self._auth_cache.get_or_refresh_async(sandbox_id)

        gateway_url = auth["gateway_url"].rstrip("/")
        url = f"{gateway_url}/{auth['user_ns']}/{auth['job_id']}/upload"
        headers = {"Authorization": f"Bearer {auth['token']}"}
        params = {"path": file_path, "sandbox_id": sandbox_id}

        effective_timeout = timeout if timeout is not None else 300

        for attempt in range(MAX_409_RETRIES):
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
                if e.response.status_code == 409:
                    if await self._should_retry_409(sandbox_id, e, attempt):
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
        auth = await self._auth_cache.get_or_refresh_async(sandbox_id)

        gateway_url = auth["gateway_url"].rstrip("/")
        url = f"{gateway_url}/{auth['user_ns']}/{auth['job_id']}/download"
        headers = {"Authorization": f"Bearer {auth['token']}"}
        params = {"path": file_path, "sandbox_id": sandbox_id}

        effective_timeout = timeout if timeout is not None else 300

        for attempt in range(MAX_409_RETRIES):
            try:
                response = await self._gateway_get(
                    url, headers=headers, params=params, timeout=effective_timeout
                )
                response.raise_for_status()
                content = response.content

                dir_path = os.path.dirname(local_file_path)
                if dir_path:
                    os.makedirs(dir_path, exist_ok=True)

                # Write file asynchronously (non-blocking I/O)
                async with aiofiles.open(local_file_path, "wb") as f:
                    await f.write(content)
                return
            except httpx.TimeoutException as e:
                raise DownloadTimeoutError(sandbox_id, file_path, effective_timeout) from e
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 409:
                    if await self._should_retry_409(sandbox_id, e, attempt):
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
        request = ExposePortRequest(port=port, name=name, protocol=protocol)
        response = await self.client.request(
            "POST",
            f"/sandbox/{sandbox_id}/expose",
            json=request.model_dump(by_alias=False, exclude_none=True),
        )
        return ExposedPort.model_validate(response)

    async def unexpose(self, sandbox_id: str, exposure_id: str) -> None:
        """Unexpose a port from a sandbox."""
        await self.client.request("DELETE", f"/sandbox/{sandbox_id}/expose/{exposure_id}")

    async def list_exposed_ports(self, sandbox_id: str) -> ListExposedPortsResponse:
        """List all exposed ports for a sandbox"""
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
