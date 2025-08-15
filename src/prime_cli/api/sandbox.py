import os
import shutil
import tarfile
import tempfile
import time
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional

import requests
from pydantic import BaseModel, ConfigDict, Field

from prime_cli.api.client import APIClient, TimeoutError


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

    stream: requests.Response
    src_path: str = Field(..., alias="srcPath")
    compressed: bool
    content_type: Optional[str] = Field(None, alias="contentType")
    content_length: Optional[int] = Field(None, alias="contentLength")

    model_config = ConfigDict(populate_by_name=True, arbitrary_types_allowed=True)


class SandboxClient:
    """Client for sandbox API operations"""

    def __init__(self, api_client: APIClient):
        self.client = api_client

    def create(self, request: CreateSandboxRequest) -> Sandbox:
        """Create a new sandbox"""
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
        for _ in range(max_attempts):
            sandbox = self.get(sandbox_id)
            if sandbox.status == "RUNNING":
                # Give it a few extra seconds to be ready for commands
                time.sleep(10)
                return
            elif sandbox.status in ["ERROR", "TERMINATED"]:
                raise SandboxNotRunningError(sandbox_id, sandbox.status)
            time.sleep(2)
        raise SandboxNotRunningError(sandbox_id, "Timeout during sandbox creation")

    def upload_stream(
        self,
        sandbox_id: str,
        dest_path: str,
        data_iter: Iterable[bytes],
        compressed: bool = False,
        strip_components: int = 0,
        working_dir: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> SandboxUploadResponse:
        params: Dict[str, Any] = {
            "dest_path": dest_path,
            "compressed": str(compressed).lower(),
            "strip_components": strip_components,
        }
        if working_dir:
            params["working_dir"] = working_dir

        raw_response = self.client.stream_post(
            f"/sandbox/{sandbox_id}/upload", params=params, data=data_iter, timeout=timeout
        )

        return SandboxUploadResponse(
            success=raw_response.get("success", True),
            message=raw_response.get("message", "Upload completed successfully"),
            filesUploaded=raw_response.get("filesUploaded"),
            bytesUploaded=raw_response.get("bytesUploaded"),
            destPath=dest_path,
        )

    def upload_file(
        self,
        sandbox_id: str,
        dest_path: str,
        file_path: str,
        compressed: bool = False,
        strip_components: int = 0,
        working_dir: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> SandboxUploadResponse:
        """Upload a file by streaming it to the backend.

        Sends Content-Length when available to avoid chunked transfer, which may
        help some proxies/backends and aligns with the backend's temp-file approach.
        """
        params: Dict[str, Any] = {
            "dest_path": dest_path,
            "compressed": str(compressed).lower(),
            "strip_components": strip_components,
        }
        if working_dir:
            params["working_dir"] = working_dir

        file_size = os.path.getsize(file_path)
        headers = {"Content-Length": str(file_size)}
        # Use a real file object so requests/urllib3 can send a proper Content-Length body
        with open(file_path, "rb") as f:
            raw_response = self.client.stream_post(
                f"/sandbox/{sandbox_id}/upload",
                params=params,
                data=f,
                headers=headers,
                timeout=timeout,
            )

        return SandboxUploadResponse(
            success=raw_response.get("success", True),
            message=raw_response.get("message", "Upload completed successfully"),
            filesUploaded=raw_response.get("filesUploaded"),
            bytesUploaded=raw_response.get("bytesUploaded"),
            destPath=dest_path,
        )

    def download_stream(
        self,
        sandbox_id: str,
        src_path: str,
        compress: bool = True,
        working_dir: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> SandboxDownloadStreamResponse:
        params: Dict[str, Any] = {"src_path": src_path, "compress": str(compress).lower()}
        if working_dir:
            params["working_dir"] = working_dir

        stream_response = self.client.stream_get(
            f"/sandbox/{sandbox_id}/download", params=params, timeout=timeout
        )

        return SandboxDownloadStreamResponse(
            stream=stream_response,
            srcPath=src_path,
            compressed=compress,
            contentType=stream_response.headers.get("content-type"),
            contentLength=int(stream_response.headers.get("content-length", 0)) or None,
        )

    def upload_path(
        self,
        sandbox_id: str,
        local_path: str,
        sandbox_path: str,
        compress: bool = True,
        working_dir: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> SandboxUploadResponse:
        """Upload a local file or directory to a sandbox.

        This is a high-level method that handles:
        - Creating tar archives for directories
        - Compression decision based on size
        - Temporary file management
        - Cleanup

        Args:
            sandbox_id: ID of the target sandbox
            local_path: Local path to file or directory
            sandbox_path: Destination path in the sandbox
            compress: Whether to compress (auto-disabled for small files)
            working_dir: Working directory in the sandbox
            timeout: Request timeout

        Returns:
            SandboxUploadResponse with upload details
        """
        # Auto-disable compression for small files
        if compress:
            abs_path = os.path.abspath(local_path)
            if os.path.exists(abs_path):
                if os.path.isfile(abs_path):
                    size = os.path.getsize(abs_path)
                    if size < 100 * 1024 * 1024:  # 100MB
                        compress = False
                else:
                    # Calculate directory size
                    total_size = 0
                    for root, dirs, files in os.walk(abs_path):
                        for file in files:
                            try:
                                total_size += os.path.getsize(os.path.join(root, file))
                            except (OSError, IOError):
                                pass
                    if total_size < 100 * 1024 * 1024:  # 100MB
                        compress = False

        # Create tar archive
        temp_file_path = None
        try:
            mode = "w:gz" if compress else "w:"
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".tar.gz" if compress else ".tar"
            ) as tmp:
                temp_file_path = tmp.name

            with tarfile.open(temp_file_path, mode=mode) as tf:  # type: ignore[call-overload]
                if os.path.isfile(local_path):
                    # For single files, add with just the filename
                    tf.add(local_path, arcname=os.path.basename(sandbox_path))
                else:
                    # For directories, add the entire directory
                    base_name = os.path.basename(local_path.rstrip("/"))
                    tf.add(local_path, arcname=base_name)

            # Upload the archive
            result = self.upload_file(
                sandbox_id,
                sandbox_path,
                temp_file_path,
                compressed=compress,
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

    def download_path(
        self,
        sandbox_id: str,
        sandbox_path: str,
        local_path: str,
        compress: bool = True,
        working_dir: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> None:
        """Download a file or directory from a sandbox to local path.

        This is a high-level method that handles:
        - Downloading the stream
        - Extracting tar archives
        - Temporary file management
        - Cleanup

        Args:
            sandbox_id: ID of the source sandbox
            sandbox_path: Path in the sandbox to download
            local_path: Local destination path
            compress: Whether the download is compressed
            working_dir: Working directory in the sandbox
            timeout: Request timeout
        """
        # Download from sandbox
        response = self.download_stream(
            sandbox_id, sandbox_path, compress=compress, working_dir=working_dir, timeout=timeout
        )

        if response.content_length == 0:
            raise Exception("No data received from sandbox. The source path may not exist.")

        # Save stream to temp file first
        temp_file_path = None
        try:
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".tar.gz" if compress else ".tar"
            ) as tmp_file:
                temp_file_path = tmp_file.name

                total_bytes = 0
                for chunk in response.stream.iter_content(chunk_size=1024 * 64):
                    if chunk:
                        tmp_file.write(chunk)
                        total_bytes += len(chunk)

            # Ensure destination directory exists
            dst_abs = os.path.abspath(local_path)
            if os.path.splitext(sandbox_path)[1] or os.path.splitext(local_path)[1]:
                # Likely a file
                os.makedirs(os.path.dirname(dst_abs), exist_ok=True)
            else:
                # Likely a directory
                os.makedirs(dst_abs, exist_ok=True)

            # Extract the archive
            mode = "r:gz" if compress else "r:"
            with tarfile.open(temp_file_path, mode=mode) as tf:  # type: ignore[call-overload]
                members = list(tf.getmembers())

                if not members:
                    raise Exception("Tar archive is empty")

                # Detect if it's a single file or directory
                is_file = len(members) == 1 and members[0].isfile()

                if is_file:
                    # Handle single file extraction
                    if os.path.isdir(dst_abs):
                        shutil.rmtree(dst_abs)

                    for member in members:
                        if member.isfile():
                            with open(dst_abs, "wb") as f:
                                f.write(tf.extractfile(member).read())
                            break
                else:
                    # Handle directory extraction
                    for member in members:
                        # Remove the first component of the path if it exists
                        if "/" in member.name:
                            member.name = "/".join(member.name.split("/")[1:])
                        tf.extract(member, path=dst_abs)

        finally:
            # Clean up temporary file
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                except Exception:
                    pass
