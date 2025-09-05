import os
import time
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import httpx
from pydantic import BaseModel, ConfigDict, Field

from prime_cli.api.client import APIClient, APIError, AsyncAPIClient, TimeoutError


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
    disk_size_gb: int = 10
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


class SandboxClient:
    """Client for sandbox API operations"""

    def __init__(self, api_client: APIClient):
        self.client = api_client

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
        """Get sandbox logs"""
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
        """Execute a command in a sandbox

        Args:
            sandbox_id: ID of the sandbox to execute the command in
            command: Command to execute
            working_dir: Working directory for the command
            env: Environment variables for the command
            timeout: Timeout in seconds for the command execution

        Raises:
            CommandTimeoutError: If the command execution times out
        """
        request = CommandRequest(command=command, working_dir=working_dir, env=env)
        try:
            response = self.client.request(
                "POST",
                f"/sandbox/{sandbox_id}/command",
                json=request.model_dump(by_alias=False, exclude_none=True),
                timeout=timeout,
            )
            return CommandResponse(**response)
        except TimeoutError:
            raise CommandTimeoutError(sandbox_id, command, timeout or 0)

    def wait_for_creation(self, sandbox_id: str, max_attempts: int = 60) -> None:
        for attempt in range(max_attempts):
            sandbox = self.get(sandbox_id)
            if sandbox.status == "RUNNING":
                return
            elif sandbox.status in ["ERROR", "TERMINATED"]:
                raise SandboxNotRunningError(sandbox_id, sandbox.status)

            # Aggressive polling for first 5 attempts (5 seconds), then back off
            sleep_time = 1 if attempt < 5 else 2
            time.sleep(sleep_time)
        raise SandboxNotRunningError(sandbox_id, "Timeout during sandbox creation")

    def upload_file(
        self, sandbox_id: str, file_path: str, local_file_path: str
    ) -> FileUploadResponse:
        """Upload a file to a sandbox

        Args:
            sandbox_id: ID of the sandbox to upload to
            file_path: Path where the file should be stored in the sandbox
            local_file_path: Path to the local file to upload

        Returns:
            FileUploadResponse with upload details
        """
        if not os.path.exists(local_file_path):
            raise FileNotFoundError(f"Local file not found: {local_file_path}")

        filename = os.path.basename(local_file_path)

        endpoint = f"/sandbox/{sandbox_id}/upload"
        url = f"{self.client.base_url}/api/v1{endpoint}"
        params = {"file_path": file_path}

        with open(local_file_path, "rb") as f:
            files = {"file": (filename, f, "application/octet-stream")}

            try:
                headers = {}
                if self.client.api_key:
                    headers["Authorization"] = f"Bearer {self.client.api_key}"

                with httpx.Client(
                    headers=headers, follow_redirects=True, timeout=300.0
                ) as upload_client:
                    response = upload_client.post(url, files=files, params=params)
                    response.raise_for_status()
                    return FileUploadResponse(**response.json())
            except httpx.HTTPStatusError as e:
                error_details = f"HTTP {e.response.status_code}: {e.response.text}"
                raise APIError(f"Upload failed: {error_details}")
            except Exception as e:
                raise APIError(f"Upload failed: {str(e)}")

    def download_file(self, sandbox_id: str, file_path: str, local_file_path: str) -> None:
        """Download a file from a sandbox

        Args:
            sandbox_id: ID of the sandbox to download from
            file_path: Path to the file in the sandbox
            local_file_path: Path where to save the downloaded file locally
        """
        endpoint = f"/sandbox/{sandbox_id}/download"
        url = f"{self.client.base_url}/api/v1{endpoint}"
        params = {"file_path": file_path}

        try:
            response = self.client.client.get(url, params=params, timeout=300.0)
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
        """Execute a command in a sandbox

        Args:
            sandbox_id: ID of the sandbox to execute the command in
            command: Command to execute
            working_dir: Working directory for the command
            env: Environment variables for the command
            timeout: Timeout in seconds for the command execution

        Raises:
            CommandTimeoutError: If the command execution times out
        """
        request = CommandRequest(command=command, working_dir=working_dir, env=env)
        try:
            response = await self.client.request(
                "POST",
                f"/sandbox/{sandbox_id}/command",
                json=request.model_dump(by_alias=False, exclude_none=True),
                timeout=timeout,
            )
            return CommandResponse(**response)
        except TimeoutError:
            raise CommandTimeoutError(sandbox_id, command, timeout or 0)

    async def wait_for_creation(self, sandbox_id: str, max_attempts: int = 60) -> None:
        """Wait for sandbox to be running (async version)"""
        import asyncio

        for attempt in range(max_attempts):
            sandbox = await self.get(sandbox_id)
            if sandbox.status == "RUNNING":
                return
            elif sandbox.status in ["ERROR", "TERMINATED"]:
                raise SandboxNotRunningError(sandbox_id, sandbox.status)

            # Aggressive polling for first 5 attempts (5 seconds), then back off
            sleep_time = 1 if attempt < 5 else 2
            await asyncio.sleep(sleep_time)
        raise SandboxNotRunningError(sandbox_id, "Timeout during sandbox creation")

    async def upload_file(
        self, sandbox_id: str, file_path: str, local_file_path: str
    ) -> FileUploadResponse:
        """Upload a file to a sandbox (async)

        Args:
            sandbox_id: ID of the sandbox to upload to
            file_path: Path where the file should be stored in the sandbox
            local_file_path: Path to the local file to upload

        Returns:
            FileUploadResponse with upload details
        """
        if not os.path.exists(local_file_path):
            raise FileNotFoundError(f"Local file not found: {local_file_path}")

        filename = os.path.basename(local_file_path)

        endpoint = f"/sandbox/{sandbox_id}/upload"
        url = f"{self.client.base_url}/api/v1{endpoint}"
        params = {"file_path": file_path}

        with open(local_file_path, "rb") as f:
            files = {"file": (filename, f, "application/octet-stream")}

            try:
                headers = {}
                if self.client.api_key:
                    headers["Authorization"] = f"Bearer {self.client.api_key}"

                async with httpx.AsyncClient(
                    headers=headers, follow_redirects=True, timeout=300.0
                ) as upload_client:
                    response = await upload_client.post(url, files=files, params=params)
                    response.raise_for_status()
                    return FileUploadResponse(**response.json())
            except httpx.HTTPStatusError as e:
                error_details = f"HTTP {e.response.status_code}: {e.response.text}"
                raise APIError(f"Upload failed: {error_details}")
            except Exception as e:
                raise APIError(f"Upload failed: {str(e)}")

    async def download_file(self, sandbox_id: str, file_path: str, local_file_path: str) -> None:
        """Download a file from a sandbox (async)

        Args:
            sandbox_id: ID of the sandbox to download from
            file_path: Path to the file in the sandbox
            local_file_path: Path where to save the downloaded file locally
        """
        endpoint = f"/sandbox/{sandbox_id}/download"
        url = f"{self.client.base_url}/api/v1{endpoint}"
        params = {"file_path": file_path}

        headers = {}
        if self.client.api_key:
            headers["Authorization"] = f"Bearer {self.client.api_key}"

        try:
            async with httpx.AsyncClient(
                headers=headers, follow_redirects=True, timeout=300.0
            ) as download_client:
                response = await download_client.get(url, params=params)
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
