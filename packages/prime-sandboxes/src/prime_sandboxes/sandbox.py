"""Sandbox client implementations."""

import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import aiofiles
import httpx
from prime_core import APIClient, APIError, AsyncAPIClient

from .exceptions import (
    CommandTimeoutError,
    DownloadTimeoutError,
    SandboxNotRunningError,
    UploadTimeoutError,
)
from .models import (
    BulkDeleteSandboxRequest,
    BulkDeleteSandboxResponse,
    CommandResponse,
    CreateSandboxRequest,
    FileUploadResponse,
    Sandbox,
    SandboxListResponse,
    SandboxLogsResponse,
)


def _build_user_agent() -> str:
    """Build User-Agent string for prime-sandboxes"""
    from prime_sandboxes import __version__

    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    return f"prime-sandboxes/{__version__} python/{python_version}"


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

    def _is_sandbox_reachable(self, sandbox_id: str, timeout: int = 10) -> bool:
        """Test if a sandbox is reachable by executing a simple echo command"""
        try:
            self.execute_command(sandbox_id, "echo 'sandbox ready'", timeout=timeout)
            return True
        except Exception:
            return False

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
        """Execute command directly via gateway"""
        auth = self._auth_cache.get_or_refresh(sandbox_id)
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

        try:
            client_timeout = effective_timeout + 5
            with httpx.Client(timeout=client_timeout) as client:
                response = client.post(
                    url,
                    json=payload,
                    headers=headers,
                )
                response.raise_for_status()
                return CommandResponse.model_validate(response.json())
        except httpx.TimeoutException as e:
            raise CommandTimeoutError(sandbox_id, command, effective_timeout) from e
        except httpx.HTTPStatusError as e:
            req = getattr(e, "request", None)
            resp = getattr(e, "response", None)
            method = getattr(req, "method", "?")
            u = getattr(req, "url", "?")
            status = getattr(resp, "status_code", "?")
            text = getattr(resp, "text", "")
            if status == 408:
                raise CommandTimeoutError(sandbox_id, command, effective_timeout) from e
            raise APIError(f"HTTP {status} {method} {u}: {text}") from e
        except httpx.RequestError as e:
            req = getattr(e, "request", None)
            method = getattr(req, "method", "?")
            u = getattr(req, "url", "?")
            raise APIError(f"Request failed: {e.__class__.__name__} at {method} {u}: {e}") from e
        except Exception as e:
            raise APIError(f"Request failed: {e.__class__.__name__}: {e}") from e

    def wait_for_creation(self, sandbox_id: str, max_attempts: int = 60) -> None:
        for attempt in range(max_attempts):
            sandbox = self.get(sandbox_id)
            if sandbox.status == "RUNNING":
                if self._is_sandbox_reachable(sandbox_id):
                    return
            elif sandbox.status in ["ERROR", "TERMINATED", "TIMEOUT"]:
                raise SandboxNotRunningError(sandbox_id, sandbox.status)

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
            files = {"file": (os.path.basename(local_file_path), f)}
            params = {"path": file_path, "sandbox_id": sandbox_id}

            try:
                with httpx.Client(timeout=effective_timeout) as client:
                    response = client.post(url, files=files, params=params, headers=headers)
                    response.raise_for_status()
                    return FileUploadResponse.model_validate(response.json())
            except httpx.TimeoutException as e:
                raise UploadTimeoutError(sandbox_id, file_path, effective_timeout) from e
            except httpx.HTTPStatusError as e:
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

        files = {"file": (filename, file_bytes)}
        params = {"path": file_path, "sandbox_id": sandbox_id}

        try:
            with httpx.Client(timeout=effective_timeout) as client:
                response = client.post(url, files=files, params=params, headers=headers)
                response.raise_for_status()
                return FileUploadResponse.model_validate(response.json())
        except httpx.TimeoutException:
            raise UploadTimeoutError(sandbox_id, file_path, effective_timeout)
        except httpx.HTTPStatusError as e:
            error_details = f"HTTP {e.response.status_code}: {e.response.text}"
            raise APIError(f"Upload failed: {error_details}")
        except Exception as e:
            raise APIError(f"Upload failed: {str(e)}")

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

        try:
            with httpx.Client(timeout=effective_timeout) as client:
                response = client.get(url, params=params, headers=headers)
                response.raise_for_status()

                dir_path = os.path.dirname(local_file_path)
                if dir_path:
                    os.makedirs(dir_path, exist_ok=True)

                with open(local_file_path, "wb") as f:
                    f.write(response.content)
        except httpx.TimeoutException as e:
            raise DownloadTimeoutError(sandbox_id, file_path, effective_timeout) from e
        except httpx.HTTPStatusError as e:
            error_details = (
                f"HTTP {e.response.status_code} {e.request.method} "
                f"{e.request.url}: {e.response.text}"
            )
            raise APIError(f"Download failed: {error_details}") from e
        except httpx.RequestError as e:
            req = getattr(e, "request", None)
            method = getattr(req, "method", "?")
            u = getattr(req, "url", "?")
            raise APIError(f"Download failed: {e.__class__.__name__} at {method} {u}: {e}") from e
        except Exception as e:
            raise APIError(f"Download failed: {e.__class__.__name__}: {e}") from e


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

    async def _is_sandbox_reachable(self, sandbox_id: str, timeout: int = 10) -> bool:
        """Test if a sandbox is reachable by executing a simple echo command"""
        try:
            await self.execute_command(sandbox_id, "echo 'sandbox ready'", timeout=timeout)
            return True
        except Exception:
            return False

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
        """Execute command directly via gateway (async)"""
        auth = await self._auth_cache.get_or_refresh_async(sandbox_id)

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

        try:
            client_timeout = effective_timeout + 5
            gateway_client = self._get_gateway_client()
            response = await gateway_client.post(
                url, json=payload, headers=headers, timeout=client_timeout
            )
            response.raise_for_status()
            return CommandResponse.model_validate(response.json())
        except httpx.TimeoutException as e:
            raise CommandTimeoutError(sandbox_id, command, effective_timeout) from e
        except httpx.HTTPStatusError as e:
            req = getattr(e, "request", None)
            resp = getattr(e, "response", None)
            method = getattr(req, "method", "?")
            u = getattr(req, "url", "?")
            status = getattr(resp, "status_code", "?")
            text = getattr(resp, "text", "")
            if status == 408:
                raise CommandTimeoutError(sandbox_id, command, effective_timeout) from e
            raise APIError(f"HTTP {status} {method} {u}: {text}") from e
        except httpx.RequestError as e:
            req = getattr(e, "request", None)
            method = getattr(req, "method", "?")
            u = getattr(req, "url", "?")
            raise APIError(f"Request failed: {e.__class__.__name__} at {method} {u}: {e}") from e
        except Exception as e:
            raise APIError(f"Request failed: {e.__class__.__name__}: {e}") from e

    async def wait_for_creation(self, sandbox_id: str, max_attempts: int = 60) -> None:
        """Wait for sandbox to be running (async version)"""
        import asyncio

        for attempt in range(max_attempts):
            sandbox = await self.get(sandbox_id)
            if sandbox.status == "RUNNING":
                if await self._is_sandbox_reachable(sandbox_id):
                    return
            elif sandbox.status in ["ERROR", "TERMINATED", "TIMEOUT"]:
                raise SandboxNotRunningError(sandbox_id, sandbox.status)

            sleep_time = 1 if attempt < 5 else 2
            await asyncio.sleep(sleep_time)
        raise SandboxNotRunningError(sandbox_id, "Timeout during sandbox creation")

    async def bulk_wait_for_creation(
        self, sandbox_ids: List[str], max_attempts: int = 60
    ) -> Dict[str, str]:
        """Wait for multiple sandboxes to be running using list endpoint"""
        import asyncio

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

        try:
            gateway_client = self._get_gateway_client()
            files = {"file": (os.path.basename(local_file_path), file_content)}
            response = await gateway_client.post(
                url, files=files, params=params, headers=headers, timeout=effective_timeout
            )
            response.raise_for_status()
            return FileUploadResponse.model_validate(response.json())
        except httpx.TimeoutException as e:
            raise UploadTimeoutError(sandbox_id, file_path, effective_timeout) from e
        except httpx.HTTPStatusError as e:
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

        try:
            gateway_client = self._get_gateway_client()
            files = {"file": (filename, file_bytes)}
            response = await gateway_client.post(
                url, files=files, params=params, headers=headers, timeout=effective_timeout
            )
            response.raise_for_status()
            return FileUploadResponse.model_validate(response.json())
        except httpx.TimeoutException:
            raise UploadTimeoutError(sandbox_id, file_path, effective_timeout)
        except httpx.HTTPStatusError as e:
            error_details = f"HTTP {e.response.status_code}: {e.response.text}"
            raise APIError(f"Upload failed: {error_details}")
        except Exception as e:
            raise APIError(f"Upload failed: {str(e)}")

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

        try:
            gateway_client = self._get_gateway_client()
            response = await gateway_client.get(
                url, params=params, headers=headers, timeout=effective_timeout
            )
            response.raise_for_status()
            content = response.content

            dir_path = os.path.dirname(local_file_path)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)

            # Write file asynchronously (non-blocking I/O)
            async with aiofiles.open(local_file_path, "wb") as f:
                await f.write(content)
        except httpx.TimeoutException as e:
            raise DownloadTimeoutError(sandbox_id, file_path, effective_timeout) from e
        except httpx.HTTPStatusError as e:
            error_details = (
                f"HTTP {e.response.status_code} {e.request.method} "
                f"{e.request.url}: {e.response.text}"
            )
            raise APIError(f"Download failed: {error_details}") from e
        except httpx.RequestError as e:
            req = getattr(e, "request", None)
            method = getattr(req, "method", "?")
            u = getattr(req, "url", "?")
            raise APIError(f"Download failed: {e.__class__.__name__} at {method} {u}: {e}") from e
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
