import json
import os
import time
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

import httpx
from pydantic import BaseModel, ConfigDict, Field

from prime_cli.api.client import APIClient, APIError, AsyncAPIClient


class SandboxStatus(str, Enum):
    """Sandbox status enum"""

    PENDING = "PENDING"
    PROVISIONING = "PROVISIONING"
    RUNNING = "RUNNING"
    STOPPED = "STOPPED"
    ERROR = "ERROR"
    TERMINATED = "TERMINATED"


class SandboxNotRunningError(RuntimeError):
    """Raised when an operation requires a RUNNING sandbox but it is not running."""

    def __init__(self, sandbox_id: str, status: Optional[str] = None):
        msg = f"Sandbox {sandbox_id} is not running" + (f" (status={status})" if status else ".")
        super().__init__(msg)


class CommandTimeoutError(RuntimeError):
    """Raised when a command execution times out."""

    def __init__(self, sandbox_id: str, command: str, timeout: int):
        msg = f"Command '{command}' timed out after {timeout}s in sandbox {sandbox_id}"
        super().__init__(msg)


class AdvancedConfigs(BaseModel):
    """Advanced configuration options for sandbox"""

    # Reserved for future advanced configuration options
    # Allow extra fields for backward compatibility with existing data
    model_config = ConfigDict(extra="allow")


class Sandbox(BaseModel):
    """Sandbox model"""

    id: str
    name: str
    docker_image: str = Field(..., alias="dockerImage")
    start_command: Optional[str] = Field(None, alias="startCommand")
    cpu_cores: int = Field(..., alias="cpuCores")
    memory_gb: int = Field(..., alias="memoryGB")
    disk_size_gb: int = Field(..., alias="diskSizeGB")
    disk_mount_path: str = Field(..., alias="diskMountPath")
    gpu_count: int = Field(..., alias="gpuCount")
    status: str
    timeout_minutes: int = Field(..., alias="timeoutMinutes")
    environment_vars: Optional[Dict[str, Any]] = Field(None, alias="environmentVars")
    advanced_configs: Optional[AdvancedConfigs] = Field(None, alias="advancedConfigs")
    created_at: datetime = Field(..., alias="createdAt")
    updated_at: datetime = Field(..., alias="updatedAt")
    started_at: Optional[datetime] = Field(None, alias="startedAt")
    terminated_at: Optional[datetime] = Field(None, alias="terminatedAt")
    exit_code: Optional[int] = Field(None, alias="exitCode")
    user_id: Optional[str] = Field(None, alias="userId")
    team_id: Optional[str] = Field(None, alias="teamId")
    kubernetes_job_id: Optional[str] = Field(None, alias="kubernetesJobId")

    model_config = ConfigDict(populate_by_name=True)


class SandboxListResponse(BaseModel):
    """Sandbox list response model"""

    sandboxes: List[Sandbox]
    total: int
    page: int
    per_page: int = Field(..., alias="perPage")
    has_next: bool = Field(..., alias="hasNext")

    model_config = ConfigDict(populate_by_name=True)


class SandboxLogsResponse(BaseModel):
    """Sandbox logs response model"""

    logs: str


class CreateSandboxRequest(BaseModel):
    """Create sandbox request model"""

    name: str
    docker_image: str
    start_command: Optional[str] = None
    cpu_cores: int = 1
    memory_gb: int = 2
    disk_size_gb: int = 5
    gpu_count: int = 0
    timeout_minutes: int = 60
    environment_vars: Optional[Dict[str, str]] = None
    team_id: Optional[str] = None
    advanced_configs: Optional[AdvancedConfigs] = None


class UpdateSandboxRequest(BaseModel):
    """Update sandbox request model"""

    name: Optional[str] = None
    docker_image: Optional[str] = None
    start_command: Optional[str] = None
    cpu_cores: Optional[int] = None
    memory_gb: Optional[int] = None
    disk_size_gb: Optional[int] = None
    gpu_count: Optional[int] = None
    timeout_minutes: Optional[int] = None
    environment_vars: Optional[Dict[str, str]] = None


class CommandRequest(BaseModel):
    """Execute command request model"""

    command: str
    working_dir: Optional[str] = None
    env: Optional[Dict[str, str]] = None


class CommandResponse(BaseModel):
    """Execute command response model"""

    stdout: str
    stderr: str
    exit_code: int


class FileUploadResponse(BaseModel):
    """File upload response model"""

    success: bool
    path: str
    size: int
    timestamp: datetime


class BulkDeleteSandboxRequest(BaseModel):
    """Bulk delete sandboxes request model"""

    sandbox_ids: List[str]


class BulkDeleteSandboxResponse(BaseModel):
    """Bulk delete sandboxes response model"""

    succeeded: List[str]
    failed: List[Dict[str, str]]
    message: str


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
        cached_auth = self._check_cached_auth(sandbox_id)
        if cached_auth:
            return cached_auth
        response = await self.client.request("POST", f"/sandbox/{sandbox_id}/auth")
        self.set(sandbox_id, response)
        self._save_cache()
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
            elif sandbox.status in ["ERROR", "TERMINATED"]:
                failed_sandboxes.append((sandbox.id, sandbox.status))
                final_statuses[sandbox.id] = sandbox.status

    return running_count, failed_sandboxes, final_statuses


class SandboxClient:
    """Client for sandbox API operations"""

    def __init__(self, api_client: APIClient):
        self.client = api_client
        self._auth_cache = SandboxAuthCache(
            self.client.config.config_dir / "sandbox_auth_cache.json", self.client
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
            "POST", "/sandbox", json=request.model_dump(by_alias=False, exclude_none=True)
        )
        return Sandbox(**response)

    def list(
        self,
        team_id: Optional[str] = None,
        status: Optional[str] = None,
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
        if exclude_terminated is not None:
            params["is_active"] = exclude_terminated

        response = self.client.request("GET", "/sandbox", params=params)
        return SandboxListResponse(**response)

    def get(self, sandbox_id: str) -> Sandbox:
        """Get a specific sandbox"""
        response = self.client.request("GET", f"/sandbox/{sandbox_id}")
        return Sandbox(**response)

    def delete(self, sandbox_id: str) -> Dict[str, Any]:
        """Delete a sandbox"""
        response = self.client.request("DELETE", f"/sandbox/{sandbox_id}")
        return response

    def bulk_delete(self, sandbox_ids: List[str]) -> BulkDeleteSandboxResponse:
        """Bulk delete multiple sandboxes"""
        request = BulkDeleteSandboxRequest(sandbox_ids=sandbox_ids)
        response = self.client.request(
            "DELETE", "/sandbox", json=request.model_dump(by_alias=False, exclude_none=True)
        )
        return BulkDeleteSandboxResponse(**response)

    def get_logs(self, sandbox_id: str) -> str:
        """Get sandbox logs via backend"""
        response = self.client.request("GET", f"/sandbox/{sandbox_id}/logs")
        logs_response = SandboxLogsResponse(**response)
        return logs_response.logs

    def update_status(self, sandbox_id: str) -> Sandbox:
        """Update sandbox status from Kubernetes"""
        response = self.client.request("POST", f"/sandbox/{sandbox_id}/status")
        return Sandbox(**response)

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
        payload = {
            "command": command,
            "working_dir": working_dir,
            "env": env or {},
            "sandbox_id": sandbox_id,
        }

        effective_timeout = timeout if timeout is not None else 300

        try:
            with httpx.Client(timeout=effective_timeout) as client:
                response = client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                return CommandResponse(**response.json())
        except httpx.TimeoutException:
            raise CommandTimeoutError(sandbox_id, command, effective_timeout)
        except httpx.HTTPStatusError as e:
            raise APIError(f"HTTP {e.response.status_code}: {e.response.text}")
        except Exception as e:
            raise APIError(f"Request failed: {str(e)}")

    def wait_for_creation(self, sandbox_id: str, max_attempts: int = 60) -> None:
        for attempt in range(max_attempts):
            sandbox = self.get(sandbox_id)
            if sandbox.status == "RUNNING":
                if self._is_sandbox_reachable(sandbox_id):
                    return
            elif sandbox.status in ["ERROR", "TERMINATED"]:
                raise SandboxNotRunningError(sandbox_id, sandbox.status)

            # Aggressive polling for first 5 attempts (5 seconds), then back off
            sleep_time = 1 if attempt < 5 else 2
            time.sleep(sleep_time)
        raise SandboxNotRunningError(sandbox_id, "Timeout during sandbox creation")

    def bulk_wait_for_creation(
        self, sandbox_ids: List[str], max_attempts: int = 60
    ) -> Dict[str, str]:
        """Wait for multiple sandboxes to be running using list endpoint to avoid rate limits

        Args:
            sandbox_ids: List of sandbox IDs to wait for
            max_attempts: Maximum number of polling attempts

        Returns:
            Dict mapping sandbox_id to final status

        Raises:
            RuntimeError: If any sandboxes fail or timeout
        """
        sandbox_id_set = set(sandbox_ids)
        final_statuses = {}

        for attempt in range(max_attempts):
            # Get all sandboxes with pagination
            total_running = 0
            all_failed = []
            page = 1

            while True:
                try:
                    list_response = self.list(per_page=100, page=page)
                except Exception as e:
                    if "429" in str(e) or "Too Many Requests" in str(e):
                        # Rate limited, wait with exponential backoff
                        wait_time = min(2**attempt, 60)  # Cap at 60 seconds
                        time.sleep(wait_time)
                        continue
                    raise

                # Check status of our sandboxes on this page
                running_count, failed_sandboxes, page_statuses = _check_sandbox_statuses(
                    list_response.sandboxes, sandbox_id_set
                )

                total_running += running_count
                all_failed.extend(failed_sandboxes)
                final_statuses.update(page_statuses)

                # If we found all our sandboxes or no more pages, break
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

            # Aggressive polling for first 5 attempts, then back off
            sleep_time = 1 if attempt < 5 else 2
            time.sleep(sleep_time)

        # Timeout - mark remaining as timeout
        for sandbox_id in sandbox_id_set:
            if sandbox_id not in final_statuses:
                final_statuses[sandbox_id] = "TIMEOUT"

        raise RuntimeError(f"Timeout waiting for sandboxes to be ready. Status: {final_statuses}")

    def upload_file(
        self, sandbox_id: str, file_path: str, local_file_path: str
    ) -> FileUploadResponse:
        """Upload file directly via gateway"""
        if not os.path.exists(local_file_path):
            raise FileNotFoundError(f"Local file not found: {local_file_path}")

        auth = self._auth_cache.get_or_refresh(sandbox_id)

        url = f"{auth['gateway_url']}/{auth['user_ns']}/{auth['job_id']}/upload"
        headers = {"Authorization": f"Bearer {auth['token']}"}

        with open(local_file_path, "rb") as f:
            files = {"file": (os.path.basename(local_file_path), f)}
            params = {"path": file_path, "sandbox_id": sandbox_id}

            try:
                with httpx.Client(timeout=300.0) as client:
                    response = client.post(url, files=files, params=params, headers=headers)
                    response.raise_for_status()
                    return FileUploadResponse(**response.json())
            except httpx.HTTPStatusError as e:
                error_details = f"HTTP {e.response.status_code}: {e.response.text}"
                raise APIError(f"Upload failed: {error_details}")
            except Exception as e:
                raise APIError(f"Upload failed: {str(e)}")

    def download_file(self, sandbox_id: str, file_path: str, local_file_path: str) -> None:
        """Download file directly via gateway"""
        auth = self._auth_cache.get_or_refresh(sandbox_id)

        url = f"{auth['gateway_url']}/{auth['user_ns']}/{auth['job_id']}/download"
        headers = {"Authorization": f"Bearer {auth['token']}"}
        params = {"path": file_path, "sandbox_id": sandbox_id}

        try:
            with httpx.Client(timeout=300.0) as client:
                response = client.get(url, params=params, headers=headers)
                response.raise_for_status()

                dir_path = os.path.dirname(local_file_path)
                if dir_path:
                    os.makedirs(dir_path, exist_ok=True)

                with open(local_file_path, "wb") as f:
                    f.write(response.content)
        except httpx.HTTPStatusError as e:
            error_details = f"HTTP {e.response.status_code}: {e.response.text}"
            raise APIError(f"Download failed: {error_details}")
        except Exception as e:
            raise APIError(f"Download failed: {str(e)}")


class AsyncSandboxClient:
    """Async client for sandbox API operations"""

    def __init__(self, api_key: Optional[str] = None):
        self.client = AsyncAPIClient(api_key=api_key)
        self._auth_cache = SandboxAuthCache(
            self.client.config.config_dir / "sandbox_auth_cache.json", self.client
        )

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
        # Auto-populate team_id from config if not specified
        if request.team_id is None:
            request.team_id = self.client.config.team_id

        response = await self.client.request(
            "POST", "/sandbox", json=request.model_dump(by_alias=False, exclude_none=True)
        )
        return Sandbox(**response)

    async def list(
        self,
        team_id: Optional[str] = None,
        status: Optional[str] = None,
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
        if exclude_terminated is not None:
            params["is_active"] = exclude_terminated

        response = await self.client.request("GET", "/sandbox", params=params)
        return SandboxListResponse(**response)

    async def get(self, sandbox_id: str) -> Sandbox:
        """Get a specific sandbox"""
        response = await self.client.request("GET", f"/sandbox/{sandbox_id}")
        return Sandbox(**response)

    async def delete(self, sandbox_id: str) -> Dict[str, Any]:
        """Delete a sandbox"""
        response = await self.client.request("DELETE", f"/sandbox/{sandbox_id}")
        return response

    async def bulk_delete(self, sandbox_ids: List[str]) -> BulkDeleteSandboxResponse:
        """Bulk delete multiple sandboxes"""
        request = BulkDeleteSandboxRequest(sandbox_ids=sandbox_ids)
        response = await self.client.request(
            "DELETE", "/sandbox", json=request.model_dump(by_alias=False, exclude_none=True)
        )
        return BulkDeleteSandboxResponse(**response)

    async def get_logs(self, sandbox_id: str) -> str:
        """Get sandbox logs"""
        response = await self.client.request("GET", f"/sandbox/{sandbox_id}/logs")
        logs_response = SandboxLogsResponse(**response)
        return logs_response.logs

    async def update_status(self, sandbox_id: str) -> Sandbox:
        """Update sandbox status from Kubernetes"""
        response = await self.client.request("POST", f"/sandbox/{sandbox_id}/status")
        return Sandbox(**response)

    async def execute_command(
        self,
        sandbox_id: str,
        command: str,
        working_dir: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None,
    ) -> CommandResponse:
        """Execute command directly via gateway (async)

        Args:
            sandbox_id: ID of the sandbox to execute the command in
            command: Command to execute
            working_dir: Working directory for the command
            env: Environment variables for the command
            timeout: Timeout in seconds for the command execution

        Raises:
            CommandTimeoutError: If the command execution times out
        """
        # Get auth for direct gateway access
        auth = await self._auth_cache.get_or_refresh_async(sandbox_id)

        gateway_url = auth["gateway_url"].rstrip("/")
        url = f"{gateway_url}/{auth['user_ns']}/{auth['job_id']}/exec"
        headers = {"Authorization": f"Bearer {auth['token']}"}
        payload = {
            "command": command,
            "working_dir": working_dir,
            "env": env or {},
            "sandbox_id": sandbox_id,
        }

        effective_timeout = timeout if timeout is not None else 300

        try:
            async with httpx.AsyncClient(timeout=effective_timeout) as client:
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                return CommandResponse(**response.json())
        except httpx.TimeoutException:
            raise CommandTimeoutError(sandbox_id, command, effective_timeout)
        except httpx.HTTPStatusError as e:
            raise APIError(f"HTTP {e.response.status_code}: {e.response.text}")
        except Exception as e:
            raise APIError(f"Request failed: {str(e)}")

    async def wait_for_creation(self, sandbox_id: str, max_attempts: int = 60) -> None:
        """Wait for sandbox to be running (async version)"""
        import asyncio

        for attempt in range(max_attempts):
            sandbox = await self.get(sandbox_id)
            if sandbox.status == "RUNNING":
                if await self._is_sandbox_reachable(sandbox_id):
                    return
            elif sandbox.status in ["ERROR", "TERMINATED"]:
                raise SandboxNotRunningError(sandbox_id, sandbox.status)

            # Aggressive polling for first 5 attempts (5 seconds), then back off
            sleep_time = 1 if attempt < 5 else 2
            await asyncio.sleep(sleep_time)
        raise SandboxNotRunningError(sandbox_id, "Timeout during sandbox creation")

    async def bulk_wait_for_creation(
        self, sandbox_ids: List[str], max_attempts: int = 60
    ) -> Dict[str, str]:
        """Wait for multiple sandboxes to be running using list endpoint to avoid rate limits

        Args:
            sandbox_ids: List of sandbox IDs to wait for
            max_attempts: Maximum number of polling attempts

        Returns:
            Dict mapping sandbox_id to final status

        Raises:
            RuntimeError: If any sandboxes fail or timeout
        """
        import asyncio

        sandbox_id_set = set(sandbox_ids)
        final_statuses = {}

        for attempt in range(max_attempts):
            # Get all sandboxes with pagination
            total_running = 0
            all_failed = []
            page = 1

            while True:
                try:
                    list_response = await self.list(per_page=100, page=page)
                except Exception as e:
                    if "429" in str(e) or "Too Many Requests" in str(e):
                        # Rate limited, wait with exponential backoff
                        wait_time = min(2**attempt, 60)  # Cap at 60 seconds
                        await asyncio.sleep(wait_time)
                        continue
                    raise

                # Check status of our sandboxes on this page
                running_count, failed_sandboxes, page_statuses = _check_sandbox_statuses(
                    list_response.sandboxes, sandbox_id_set
                )

                total_running += running_count
                all_failed.extend(failed_sandboxes)
                final_statuses.update(page_statuses)

                # If we found all our sandboxes or no more pages, break
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

            # Aggressive polling for first 5 attempts, then back off
            sleep_time = 1 if attempt < 5 else 2
            await asyncio.sleep(sleep_time)

        # Timeout - mark remaining as timeout
        for sandbox_id in sandbox_id_set:
            if sandbox_id not in final_statuses:
                final_statuses[sandbox_id] = "TIMEOUT"

        raise RuntimeError(f"Timeout waiting for sandboxes to be ready. Status: {final_statuses}")

    async def upload_file(
        self, sandbox_id: str, file_path: str, local_file_path: str
    ) -> FileUploadResponse:
        """Upload a file to a sandbox via gateway (async)

        Args:
            sandbox_id: ID of the sandbox to upload to
            file_path: Path where the file should be stored in the sandbox
            local_file_path: Path to the local file to upload

        Returns:
            FileUploadResponse with upload details
        """
        if not os.path.exists(local_file_path):
            raise FileNotFoundError(f"Local file not found: {local_file_path}")

        # Get auth for direct gateway access
        auth = await self._auth_cache.get_or_refresh_async(sandbox_id)

        gateway_url = auth["gateway_url"].rstrip("/")
        url = f"{gateway_url}/{auth['user_ns']}/{auth['job_id']}/upload"
        headers = {"Authorization": f"Bearer {auth['token']}"}
        params = {"path": file_path, "sandbox_id": sandbox_id}

        with open(local_file_path, "rb") as f:
            files = {"file": (os.path.basename(local_file_path), f)}

            try:
                async with httpx.AsyncClient(timeout=300.0) as upload_client:
                    response = await upload_client.post(
                        url, files=files, params=params, headers=headers
                    )
                    response.raise_for_status()
                    return FileUploadResponse(**response.json())
            except httpx.HTTPStatusError as e:
                error_details = f"HTTP {e.response.status_code}: {e.response.text}"
                raise APIError(f"Upload failed: {error_details}")
            except Exception as e:
                raise APIError(f"Upload failed: {str(e)}")

    async def download_file(self, sandbox_id: str, file_path: str, local_file_path: str) -> None:
        """Download a file from a sandbox via gateway (async)

        Args:
            sandbox_id: ID of the sandbox to download from
            file_path: Path to the file in the sandbox
            local_file_path: Path where to save the downloaded file locally
        """
        # Get auth for direct gateway access
        auth = await self._auth_cache.get_or_refresh_async(sandbox_id)

        gateway_url = auth["gateway_url"].rstrip("/")
        url = f"{gateway_url}/{auth['user_ns']}/{auth['job_id']}/download"
        headers = {"Authorization": f"Bearer {auth['token']}"}
        params = {"path": file_path, "sandbox_id": sandbox_id}

        try:
            async with httpx.AsyncClient(timeout=300.0) as download_client:
                response = await download_client.get(url, params=params, headers=headers)
                response.raise_for_status()
                content = response.content

            dir_path = os.path.dirname(local_file_path)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)

            with open(local_file_path, "wb") as f:
                f.write(content)
        except httpx.HTTPStatusError as e:
            error_details = f"HTTP {e.response.status_code}: {e.response.text}"
            raise APIError(f"Download failed: {error_details}")
        except Exception as e:
            raise APIError(f"Download failed: {str(e)}")

    async def aclose(self) -> None:
        """Close the async client"""
        await self.client.aclose()

    async def __aenter__(self) -> "AsyncSandboxClient":
        """Async context manager entry"""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit"""
        await self.aclose()
