"""Pydantic models for Prime Sandboxes SDK."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class SandboxStatus(str, Enum):
    """Sandbox status enum"""

    PENDING = "PENDING"
    PROVISIONING = "PROVISIONING"
    RUNNING = "RUNNING"
    STOPPED = "STOPPED"
    ERROR = "ERROR"
    TERMINATED = "TERMINATED"
    TIMEOUT = "TIMEOUT"


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
    cpu_cores: float = Field(..., alias="cpuCores")
    memory_gb: float = Field(..., alias="memoryGB")
    disk_size_gb: float = Field(..., alias="diskSizeGB")
    disk_mount_path: str = Field(..., alias="diskMountPath")
    gpu_count: int = Field(..., alias="gpuCount")
    network_access: bool = Field(True, alias="networkAccess")
    status: str
    timeout_minutes: int = Field(..., alias="timeoutMinutes")
    environment_vars: Optional[Dict[str, Any]] = Field(None, alias="environmentVars")
    secrets: Optional[Dict[str, Any]] = Field(None, alias="secrets")
    advanced_configs: Optional[AdvancedConfigs] = Field(None, alias="advancedConfigs")
    labels: List[str] = Field(default_factory=list)
    created_at: datetime = Field(..., alias="createdAt")
    updated_at: datetime = Field(..., alias="updatedAt")
    started_at: Optional[datetime] = Field(None, alias="startedAt")
    terminated_at: Optional[datetime] = Field(None, alias="terminatedAt")
    exit_code: Optional[int] = Field(None, alias="exitCode")
    error_type: Optional[str] = Field(None, alias="errorType")
    error_message: Optional[str] = Field(None, alias="errorMessage")
    user_id: Optional[str] = Field(None, alias="userId")
    team_id: Optional[str] = Field(None, alias="teamId")
    kubernetes_job_id: Optional[str] = Field(None, alias="kubernetesJobId")
    registry_credentials_id: Optional[str] = Field(default=None, alias="registryCredentialsId")

    model_config = ConfigDict(populate_by_name=True)


class SandboxListResponse(BaseModel):
    """Sandbox list response model"""

    sandboxes: List[Sandbox]
    total: int
    page: int
    per_page: int = Field(..., alias="perPage")
    has_next: bool = Field(..., alias="hasNext")

    model_config = ConfigDict(populate_by_name=True)


class CreateSandboxRequest(BaseModel):
    """Create sandbox request model"""

    name: str
    docker_image: str
    start_command: Optional[str] = "tail -f /dev/null"
    cpu_cores: float = 1.0
    memory_gb: float = 2.0
    disk_size_gb: float = 5.0
    gpu_count: int = 0
    network_access: bool = True
    timeout_minutes: int = 60
    environment_vars: Optional[Dict[str, str]] = None
    secrets: Optional[Dict[str, str]] = None
    labels: List[str] = Field(default_factory=list)
    team_id: Optional[str] = None
    advanced_configs: Optional[AdvancedConfigs] = None
    registry_credentials_id: Optional[str] = None


class UpdateSandboxRequest(BaseModel):
    """Update sandbox request model"""

    name: Optional[str] = None
    docker_image: Optional[str] = None
    start_command: Optional[str] = None
    cpu_cores: Optional[float] = None
    memory_gb: Optional[float] = None
    disk_size_gb: Optional[float] = None
    gpu_count: Optional[int] = None
    timeout_minutes: Optional[int] = None
    environment_vars: Optional[Dict[str, str]] = None
    registry_credentials_id: Optional[str] = None
    secrets: Optional[Dict[str, str]] = None
    network_access: Optional[bool] = None


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


class SandboxLogsResponse(BaseModel):
    """Sandbox logs response model"""

    logs: str


class BulkDeleteSandboxRequest(BaseModel):
    """Bulk delete sandboxes request model"""

    sandbox_ids: Optional[List[str]] = None
    labels: Optional[List[str]] = None


class BulkDeleteSandboxResponse(BaseModel):
    """Bulk delete sandboxes response model"""

    succeeded: List[str]
    failed: List[Dict[str, str]]
    message: str


class RegistryCredentialSummary(BaseModel):
    """Summary of registry credential data (no secrets)."""

    id: str
    name: str
    server: str
    created_at: datetime = Field(..., alias="createdAt")
    updated_at: datetime = Field(..., alias="updatedAt")
    user_id: Optional[str] = Field(default=None, alias="userId")
    team_id: Optional[str] = Field(default=None, alias="teamId")

    model_config = ConfigDict(populate_by_name=True)


class DockerImageCheckResponse(BaseModel):
    accessible: bool
    details: str


class ExposePortRequest(BaseModel):
    """Request to expose a port"""

    port: int
    name: Optional[str] = None
    protocol: str = "HTTP"  # HTTP or TCP


class ExposedPort(BaseModel):
    """Information about an exposed port"""

    exposure_id: str
    sandbox_id: str
    port: int
    name: Optional[str]
    url: str
    tls_socket: str
    protocol: Optional[str] = None
    external_port: Optional[int] = None  # For TCP exposures
    external_endpoint: Optional[str] = None  # For TCP: host:port endpoint
    created_at: Optional[str] = None


class ListExposedPortsResponse(BaseModel):
    """Response for listing exposed ports"""

    exposures: List[ExposedPort]


class SSHSession(BaseModel):
    """SSH session details"""

    session_id: str
    exposure_id: str
    sandbox_id: str
    host: str
    port: int
    external_endpoint: str
    expires_at: datetime
    ttl_seconds: int
    gateway_url: str
    user_ns: str
    job_id: str
    token: str


class BackgroundJob(BaseModel):
    """Background job handle returned when starting a background job"""

    job_id: str
    sandbox_id: str
    stdout_log_file: str
    stderr_log_file: str
    exit_file: str


class BackgroundJobStatus(BaseModel):
    """Status of a background job"""

    job_id: str
    completed: bool
    exit_code: Optional[int] = None
    stdout: Optional[str] = None
    stderr: Optional[str] = None
