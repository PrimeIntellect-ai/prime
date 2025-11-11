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
    cpu_cores: int = Field(..., alias="cpuCores")
    memory_gb: int = Field(..., alias="memoryGB")
    disk_size_gb: int = Field(..., alias="diskSizeGB")
    disk_mount_path: str = Field(..., alias="diskMountPath")
    gpu_count: int = Field(..., alias="gpuCount")
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


class CreateSandboxRequest(BaseModel):
    """Create sandbox request model"""

    name: str
    docker_image: str
    start_command: Optional[str] = "tail -f /dev/null"
    cpu_cores: int = 1
    memory_gb: int = 2
    disk_size_gb: int = 5
    gpu_count: int = 0
    timeout_minutes: int = 60
    environment_vars: Optional[Dict[str, str]] = None
    secrets: Optional[Dict[str, str]] = None
    labels: List[str] = Field(default_factory=list)
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
    secrets: Optional[Dict[str, str]] = None


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
