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

    def upload_file(
        self,
        sandbox_id: str,
        dest_path: str,
        file_path: str,
        strip_components: int = 0,
        working_dir: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> SandboxUploadResponse:
        """Upload a tar archive file directly to the sandbox (low-level method).

        This is the low-level upload method that expects a tar archive file.
        Most users should use `upload_path()` instead, which handles tar creation
        automatically.

        Use this method when:
        - You already have a tar archive prepared
        - You need fine control over tar archive creation
        - You're implementing custom upload logic

        Use `upload_path()` instead when:
        - You want to upload regular files or directories (recommended)
        - You don't want to manage tar archive creation/cleanup

        Args:
            sandbox_id: ID of the target sandbox
            dest_path: Destination path in the sandbox
            file_path: Path to the tar archive file to upload
            strip_components: Number of leading path components to strip
            working_dir: Working directory in the sandbox
            timeout: Request timeout

        Returns:
            SandboxUploadResponse with upload details
        """
        import logging

        logger = logging.getLogger(__name__)

        logger.debug("🚀 Upload File Debug:")
        logger.debug(f"   Sandbox ID: {sandbox_id}")
        logger.debug(f"   Local file: {file_path}")
        logger.debug(f"   Dest path: {dest_path}")
        logger.debug(f"   Strip components: {strip_components}")
        logger.debug(f"   Working dir: {working_dir}")

        # Prepare form data
        form_data: Dict[str, Any] = {
            "dest_path": dest_path,
            "compressed": "false",
            "strip_components": str(strip_components),
        }
        if working_dir:
            form_data["working_dir"] = working_dir

        logger.debug(f"📝 Form data prepared: {form_data}")

        # Prepare file data with appropriate content type
        content_type = "application/x-tar"

        with open(file_path, "rb") as file_handle:
            files_data = {"file": (os.path.basename(file_path), file_handle, content_type)}
            logger.debug(
                f"📁 Files data prepared: {list(files_data.keys())} with content-type: {content_type}"
            )

            logger.debug("📤 Sending multipart request...")
            raw_response = self.client.multipart_post(
                f"/sandbox/{sandbox_id}/upload",
                files=files_data,
                data=form_data,
                timeout=timeout,
            )
            logger.debug(f"✅ Response received: {raw_response}")

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
        working_dir: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> SandboxDownloadStreamResponse:
        """Download a file/directory as a tar archive stream (low-level method).

        This is the low-level download method that returns a raw tar archive stream.
        Most users should use `download_path()` instead, which handles tar extraction
        automatically.

        Use this method when:
        - You need to process the tar stream directly
        - You want to implement custom tar extraction logic
        - You're streaming data to another service without saving to disk

        Use `download_path()` instead when:
        - You want to download and save files/directories to disk (recommended)
        - You don't want to handle tar extraction manually

        Args:
            sandbox_id: ID of the source sandbox
            src_path: Path in the sandbox to download
            working_dir: Working directory in the sandbox
            timeout: Request timeout

        Returns:
            SandboxDownloadStreamResponse with the tar archive stream
        """
        params: Dict[str, Any] = {"src_path": src_path, "compress": "false"}
        if working_dir:
            params["working_dir"] = working_dir

        debug_log(
            f"Making download request to /sandbox/{sandbox_id}/download with params: {params}"
        )
        stream_response = self.client.stream_get(
            f"/sandbox/{sandbox_id}/download", params=params, timeout=timeout
        )

        debug_log(f"Got response with content-type: {stream_response.headers.get('content-type')}")
        debug_log(
            f"Got response with content-length: {stream_response.headers.get('content-length')}"
        )

        return SandboxDownloadStreamResponse(
            stream=stream_response,
            srcPath=src_path,
            contentType=stream_response.headers.get("content-type"),
            contentLength=int(stream_response.headers.get("content-length", 0)) or None,
        )

    def upload_path(
        self,
        sandbox_id: str,
        local_path: str,
        sandbox_path: str,
        working_dir: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> SandboxUploadResponse:
        """Upload a local file or directory to a sandbox (high-level method).

        This is the recommended method for uploading files and directories.
        It automatically handles tar archive creation and cleanup.

        Use this method when:
        - You want to upload regular files or directories (recommended)
        - You want automatic tar archive handling
        - You don't need custom tar creation logic

        Use `upload_file()` instead when:
        - You already have a tar archive prepared
        - You need fine control over the tar archive format
        - You're implementing a custom upload pipeline

        This method:
        - Creates a temporary tar archive of your file/directory
        - Uploads it to the sandbox
        - Cleans up the temporary archive automatically
        - Preserves file metadata and directory structure

        Args:
            sandbox_id: ID of the target sandbox
            local_path: Local path to file or directory to upload
            sandbox_path: Destination path in the sandbox
            working_dir: Working directory in the sandbox
            timeout: Request timeout

        Returns:
            SandboxUploadResponse with upload details
        """
        abs_path = os.path.abspath(local_path)
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"Path does not exist: {local_path}")

        # Create tar archive for all uploads (files and directories)
        temp_file_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".tar") as tmp:
                temp_file_path = tmp.name

            with tarfile.open(temp_file_path, mode="w:") as tf:
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
        working_dir: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> None:
        """Download a file or directory from a sandbox to local path (high-level method).

        This is the recommended method for downloading files and directories.
        It automatically handles tar stream extraction and file saving.

        Use this method when:
        - You want to download and save files/directories to disk (recommended)
        - You want automatic tar extraction
        - You don't need to process the stream directly

        Use `download_stream()` instead when:
        - You need the raw tar archive stream
        - You want to implement custom extraction logic
        - You're streaming data to another service

        This method:
        - Downloads the tar stream from the sandbox
        - Extracts the tar archive automatically
        - Handles single files vs directories correctly
        - Manages temporary files and cleanup
        - Preserves file metadata and permissions

        Args:
            sandbox_id: ID of the source sandbox
            sandbox_path: Path in the sandbox to download
            local_path: Local destination path where files will be saved
            working_dir: Working directory in the sandbox
            timeout: Request timeout
        """
        debug_log(
            f"download_path called with sandbox_id={sandbox_id}, "
            f"sandbox_path={sandbox_path}, local_path={local_path}"
        )
        # Download from sandbox
        response = self.download_stream(
            sandbox_id, sandbox_path, working_dir=working_dir, timeout=timeout
        )

        if response.content_length == 0:
            raise Exception("No data received from sandbox. The source path may not exist.")

        # Save stream to temp file first
        temp_file_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".tar") as tmp_file:
                temp_file_path = tmp_file.name

                total_bytes = 0
                for chunk in response.stream.iter_content(chunk_size=1024 * 64):
                    if chunk:
                        tmp_file.write(chunk)
                        total_bytes += len(chunk)

                debug_log(f"Downloaded {total_bytes} bytes to {temp_file_path}")

                # Check the first few bytes to see what we got
                with open(temp_file_path, "rb") as f:
                    first_bytes = f.read(100)
                    debug_log_hex("First 100 bytes", first_bytes)
                    debug_log_ascii("First 100 bytes", first_bytes)

            # Extract the archive - use raw tar format
            debug_log(f"Attempting to open tar file: {temp_file_path}")
            tf = None
            members = None
            try:
                tf = tarfile.open(temp_file_path, mode="r:")
                members = list(tf.getmembers())
                debug_log(f"Successfully opened tar file with {len(members)} members")
            except Exception as e:
                debug_log(f"Failed to open tar file with mode 'r:': {e}")
                # Try with different mode - explicitly specify no compression
                tf = tarfile.open(temp_file_path, mode="r")
                members = list(tf.getmembers())
                debug_log(f"Successfully opened tar file with mode 'r' with {len(members)} members")

            if not members:
                raise Exception("Tar archive is empty")

            # Determine if this is a single file or directory based on the tar archive contents
            # If the archive contains multiple members or the first member is a directory,
            # treat as directory
            is_single_file = len(members) == 1 and members[0].isfile()
            debug_log(f"local_path='{local_path}', is_single_file={is_single_file}")
            debug_log(
                f"members count={len(members)}, "
                f"first member isfile={members[0].isfile() if members else 'N/A'}"
            )

            # Ensure destination directory exists
            dst_abs = os.path.abspath(local_path)
            if is_single_file:
                # For single files, ensure parent directory exists
                os.makedirs(os.path.dirname(dst_abs), exist_ok=True)
            else:
                # For directories, ensure the directory itself exists
                os.makedirs(dst_abs, exist_ok=True)

            try:
                debug_log(f"Starting extraction, is_single_file={is_single_file}")
                if is_single_file:
                    debug_log("Handling as single file")
                    # Handle single file extraction
                    # Find the first file in the archive and extract it to the exact destination
                    for member in members:
                        if member.isfile():
                            extract_file = tf.extractfile(member)
                            if extract_file is not None:
                                with open(dst_abs, "wb") as f:
                                    f.write(extract_file.read())
                                break
                            else:
                                raise Exception(f"Failed to extract file member: {member.name}")
                    else:
                        raise Exception("No file found in the archive")
                else:
                    debug_log("Handling as directory")
                    # Handle directory extraction
                    # Extract all members, removing the first path component if it exists
                    debug_log(f"Extracting directory with {len(members)} members")
                    for member in members:
                        debug_log(
                            f"Processing member: {member.name} "
                            f"(isfile: {member.isfile()}, isdir: {member.isdir()})"
                        )
                        # Create a copy of the member with modified name
                        if "/" in member.name:
                            new_name = "/".join(member.name.split("/")[1:])
                            debug_log(f"Renaming member from '{member.name}' to '{new_name}'")
                            # Create a new TarInfo object with the modified name
                            new_member = tarfile.TarInfo(name=new_name)
                            new_member.size = member.size
                            new_member.mode = member.mode
                            new_member.type = member.type
                            new_member.linkname = member.linkname
                            new_member.uid = member.uid
                            new_member.gid = member.gid
                            new_member.uname = member.uname
                            new_member.gname = member.gname
                            new_member.mtime = member.mtime
                            new_member.devmajor = member.devmajor
                            new_member.devminor = member.devminor
                            tf.extract(new_member, path=dst_abs)
                        else:
                            tf.extract(member, path=dst_abs)
            finally:
                if tf:
                    tf.close()

        finally:
            # Clean up temporary file
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                except Exception:
                    pass

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
