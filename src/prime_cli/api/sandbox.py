import os
import tarfile
import tempfile
import time
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import httpx
from pydantic import BaseModel, ConfigDict, Field

from prime_cli.api.client import APIClient, AsyncAPIClient, TimeoutError

from ..utils.debug import debug_log, debug_log_ascii, debug_log_hex


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


class SandboxUploadResponse(BaseModel):
    """Sandbox upload response model"""

    success: bool
    message: str
    files_uploaded: Optional[int] = Field(None, alias="filesUploaded")
    bytes_uploaded: Optional[int] = Field(None, alias="bytesUploaded")
    dest_path: str = Field(..., alias="destPath")

    model_config = ConfigDict(populate_by_name=True)


class SandboxDownloadStreamResponse(BaseModel):
    """Sandbox download stream response model"""

    stream: httpx.Response
    src_path: str = Field(..., alias="srcPath")
    content_type: Optional[str] = Field(None, alias="contentType")
    content_length: Optional[int] = Field(None, alias="contentLength")

    model_config = ConfigDict(populate_by_name=True, arbitrary_types_allowed=True)


class SandboxClient:
    """Client for sandbox API operations"""

    def __init__(self, api_client: APIClient):
        debug_log("SandboxClient constructor called")
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

    async def aclose(self) -> None:
        """Close the async client"""
        await self.client.aclose()

    async def __aenter__(self) -> "AsyncSandboxClient":
        """Async context manager entry"""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit"""
        await self.aclose()

    async def upload_file(
        self,
        sandbox_id: str,
        dest_path: str,
        file_path: str,
        strip_components: int = 0,
        working_dir: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> SandboxUploadResponse:
        """Upload a tar archive file directly to the sandbox (async low-level method).
        
        This is the async version of the low-level upload method that expects a tar archive file.
        Most users should use `upload_path()` instead, which handles tar creation automatically.
        """
        import aiofiles
        
        # Prepare form data
        form_data: Dict[str, Any] = {
            "dest_path": dest_path,
            "compressed": "false",
            "strip_components": str(strip_components),
        }
        if working_dir:
            form_data["working_dir"] = working_dir
        
        # Read file asynchronously
        async with aiofiles.open(file_path, "rb") as file_handle:
            file_content = await file_handle.read()
        
        # Prepare file data with appropriate content type
        content_type = "application/x-tar"
        files_data = {"file": (os.path.basename(file_path), file_content, content_type)}
        
        raw_response = await self.client.multipart_post(
            f"/sandbox/{sandbox_id}/upload",
            files=files_data,
            data=form_data,
            timeout=timeout,
        )
        
        return SandboxUploadResponse(
            success=raw_response.get("success", True),
            message=raw_response.get("message", "Upload completed successfully"),
            filesUploaded=raw_response.get("filesUploaded"),
            bytesUploaded=raw_response.get("bytesUploaded"),
            destPath=dest_path,
        )

    async def download_stream(
        self,
        sandbox_id: str,
        src_path: str,
        working_dir: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> SandboxDownloadStreamResponse:
        """Download a file/directory as a tar archive stream (async low-level method).
        
        This is the async version of the low-level download method that returns a raw tar archive stream.
        Most users should use `download_path()` instead, which handles tar extraction automatically.
        """
        params: Dict[str, Any] = {"src_path": src_path, "compress": "false"}
        if working_dir:
            params["working_dir"] = working_dir
        
        stream_response = await self.client.stream_get(
            f"/sandbox/{sandbox_id}/download", params=params, timeout=timeout
        )
        
        return SandboxDownloadStreamResponse(
            stream=stream_response,
            srcPath=src_path,
            contentType=stream_response.headers.get("content-type"),
            contentLength=int(stream_response.headers.get("content-length", 0)) or None,
        )

    async def upload_path(
        self,
        sandbox_id: str,
        local_path: str,
        sandbox_path: str,
        working_dir: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> SandboxUploadResponse:
        """Upload a local file or directory to a sandbox (async high-level method).
        
        This is the async version of the recommended method for uploading files and directories.
        It automatically handles tar archive creation and cleanup.
        """
        import asyncio
        import aiofiles
        
        abs_path = os.path.abspath(local_path)
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"Path does not exist: {local_path}")
        
        # Create tar archive for all uploads (files and directories)
        temp_file_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".tar") as tmp:
                temp_file_path = tmp.name
            
            # Run tar creation in executor to avoid blocking
            def create_tar():
                with tarfile.open(temp_file_path, mode="w:") as tf:
                    if os.path.isfile(local_path):
                        # For single files, add with just the filename
                        tf.add(local_path, arcname=os.path.basename(sandbox_path))
                    else:
                        # For directories, add the entire directory
                        base_name = os.path.basename(local_path.rstrip("/"))
                        tf.add(local_path, arcname=base_name)
            
            await asyncio.get_event_loop().run_in_executor(None, create_tar)
            
            # Upload the archive
            result = await self.upload_file(
                sandbox_id,
                sandbox_path,
                temp_file_path,
                working_dir=working_dir,
                timeout=timeout,
            )
            
            return result
            
        finally:
            # Clean up temporary file
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                except Exception:
                    pass

    async def download_path(
        self,
        sandbox_id: str,
        sandbox_path: str,
        local_path: str,
        working_dir: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> None:
        """Download a file or directory from a sandbox to local path (async high-level method).
        
        This is the async version of the recommended method for downloading files and directories.
        It automatically handles tar stream extraction and file saving.
        """
        import asyncio
        import aiofiles
        
        # Download from sandbox
        response = await self.download_stream(
            sandbox_id, sandbox_path, working_dir=working_dir, timeout=timeout
        )
        
        if response.content_length == 0:
            raise Exception("No data received from sandbox. The source path may not exist.")
        
        # Save stream to temp file first
        temp_file_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".tar") as tmp_file:
                temp_file_path = tmp_file.name
            
            # Write stream content asynchronously
            async with aiofiles.open(temp_file_path, "wb") as async_file:
                total_bytes = 0
                async for chunk in response.stream.aiter_content(chunk_size=1024 * 64):
                    if chunk:
                        await async_file.write(chunk)
                        total_bytes += len(chunk)
            
            # Extract the archive - run in executor to avoid blocking
            def extract_tar():
                with tarfile.open(temp_file_path, mode="r:") as tf:
                    members = list(tf.getmembers())
                    
                    if not members:
                        raise Exception("Tar archive is empty")
                    
                    # Determine if this is a single file or directory
                    is_single_file = len(members) == 1 and members[0].isfile()
                    
                    # Ensure destination directory exists
                    dst_abs = os.path.abspath(local_path)
                    if is_single_file:
                        # For single files, ensure parent directory exists
                        os.makedirs(os.path.dirname(dst_abs), exist_ok=True)
                        # Extract single file
                        for member in members:
                            if member.isfile():
                                extract_file = tf.extractfile(member)
                                if extract_file is not None:
                                    with open(dst_abs, "wb") as f:
                                        f.write(extract_file.read())
                                    break
                    else:
                        # For directories, ensure the directory itself exists
                        os.makedirs(dst_abs, exist_ok=True)
                        # Extract all members
                        for member in members:
                            # Strip first component if it exists
                            if "/" in member.name:
                                new_name = "/".join(member.name.split("/")[1:])
                                member.name = new_name
                            tf.extract(member, path=dst_abs)
            
            await asyncio.get_event_loop().run_in_executor(None, extract_tar)
            
        finally:
            # Clean up temporary file
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                except Exception:
                    pass
