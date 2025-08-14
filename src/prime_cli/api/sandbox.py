import time
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

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

    container_user_uid: Optional[int] = Field(
        None,
        ge=1000,
        le=65535,
        description=(
            "Container user UID to overwrite default UID 1000 (must be non-root, minimum UID 1000)"
        ),
    )

    model_config = ConfigDict(extra="forbid")


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
