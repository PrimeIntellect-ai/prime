"""Pydantic models for Prime Sandboxes SDK."""

from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator


class SandboxStatus(str, Enum):
    """Sandbox status enum"""

    PENDING = "PENDING"
    PROVISIONING = "PROVISIONING"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
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
    gpu_type: Optional[str] = Field(None, alias="gpuType")
    vm: bool = False
    network_access: bool = Field(True, alias="networkAccess")
    status: str
    timeout_minutes: int = Field(..., alias="timeoutMinutes")
    idle_timeout_minutes: Optional[int] = Field(None, alias="idleTimeoutMinutes")
    termination_reason: Optional[str] = Field(None, alias="terminationReason")
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
    region: Optional[str] = None
    registry_credentials_id: Optional[str] = Field(default=None, alias="registryCredentialsId")
    pending_image_build_id: Optional[str] = Field(default=None, alias="pendingImageBuildId")

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
    gpu_type: Optional[str] = None
    vm: bool = False
    network_access: bool = True
    timeout_minutes: int = 60
    idle_timeout_minutes: Optional[int] = None
    environment_vars: Optional[Dict[str, str]] = None
    secrets: Optional[Dict[str, str]] = None
    labels: List[str] = Field(default_factory=list)
    team_id: Optional[str] = None
    region: Optional[str] = None
    advanced_configs: Optional[AdvancedConfigs] = None
    registry_credentials_id: Optional[str] = None
    guaranteed: bool = False
    idempotency_key: Optional[str] = None

    @model_validator(mode="after")
    def validate_gpu_fields(self) -> "CreateSandboxRequest":
        if self.gpu_count > 0 and not self.gpu_type:
            raise ValueError("gpu_type is required when gpu_count is greater than 0")
        if self.gpu_count > 0 and not self.vm:
            raise ValueError("gpu_count is only supported when vm is true")
        if self.gpu_count == 0 and self.gpu_type is not None:
            raise ValueError("gpu_type requires gpu_count greater than 0")
        return self

    @model_validator(mode="after")
    def validate_guaranteed(self) -> "CreateSandboxRequest":
        if self.guaranteed and self.vm:
            raise ValueError("guaranteed is not supported for VM sandboxes")
        return self

    @model_validator(mode="after")
    def validate_idle_timeout(self) -> "CreateSandboxRequest":
        if self.idle_timeout_minutes is None:
            return self
        if self.idle_timeout_minutes < 1:
            raise ValueError("idle_timeout_minutes must be >= 1")
        if self.timeout_minutes > 0 and self.idle_timeout_minutes > self.timeout_minutes:
            raise ValueError(
                "idle_timeout_minutes must be <= timeout_minutes "
                f"(got idle={self.idle_timeout_minutes}, lifetime={self.timeout_minutes})"
            )
        return self


class UpdateSandboxRequest(BaseModel):
    """Update sandbox request model"""

    name: Optional[str] = None
    docker_image: Optional[str] = None
    start_command: Optional[str] = None
    cpu_cores: Optional[float] = None
    memory_gb: Optional[float] = None
    disk_size_gb: Optional[float] = None
    gpu_count: Optional[int] = None
    gpu_type: Optional[str] = None
    timeout_minutes: Optional[int] = None
    idle_timeout_minutes: Optional[int] = None
    environment_vars: Optional[Dict[str, str]] = None
    registry_credentials_id: Optional[str] = None
    secrets: Optional[Dict[str, str]] = None
    network_access: Optional[bool] = None

    @model_validator(mode="after")
    def validate_idle_timeout(self) -> "UpdateSandboxRequest":
        if self.idle_timeout_minutes is None:
            return self
        if self.idle_timeout_minutes < 1:
            raise ValueError("idle_timeout_minutes must be >= 1")
        if (
            self.timeout_minutes is not None
            and self.timeout_minutes > 0
            and self.idle_timeout_minutes > self.timeout_minutes
        ):
            raise ValueError(
                "idle_timeout_minutes must be <= timeout_minutes "
                f"(got idle={self.idle_timeout_minutes}, lifetime={self.timeout_minutes})"
            )
        return self


class CommandRequest(BaseModel):
    """Execute command request model"""

    command: str
    working_dir: Optional[str] = None
    env: Optional[Dict[str, str]] = None
    user: Optional[str] = None


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


class ReadFileResponse(BaseModel):
    """Read file response model"""

    content: str
    size: int
    # VM sandboxes don't support windowed reads yet and omit them.
    total_size: Optional[int] = None
    offset: Optional[int] = None
    truncated: Optional[bool] = None


class SandboxLogsResponse(BaseModel):
    """Sandbox logs response model"""

    logs: str


class BulkDeleteSandboxRequest(BaseModel):
    """Bulk delete sandboxes request model"""

    sandbox_ids: Optional[List[str]] = None
    labels: Optional[List[str]] = None
    team_id: Optional[str] = None
    user_id: Optional[str] = None
    all_users: bool = False


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


class ImageVisibility(str, Enum):
    PRIVATE = "PRIVATE"
    PUBLIC = "PUBLIC"


class BuildImageRequest(BaseModel):
    image_name: Optional[str] = None
    image_tag: Optional[str] = None
    dockerfile_path: str = "Dockerfile"
    source_image: Optional[str] = Field(default=None, alias="sourceImage")
    platform: str = "linux/amd64"
    team_id: Optional[str] = Field(default=None, alias="teamId")
    visibility: Optional[ImageVisibility] = None
    owner_scope: Optional[Literal["platform"]] = Field(default=None, alias="ownerScope")

    model_config = ConfigDict(populate_by_name=True)


class BuildImageResponse(BaseModel):
    build_id: str = Field(
        ...,
        alias="build_id",
        validation_alias=AliasChoices("build_id", "buildId"),
    )
    build_ids: List[str] = Field(default_factory=list, alias="buildIds")
    upload_url: Optional[str] = None
    expires_in: Optional[int] = None
    full_image_path: str = Field(..., alias="fullImagePath")
    visibility: Optional[ImageVisibility] = None

    model_config = ConfigDict(populate_by_name=True)


class TransferImageResult(BaseModel):
    """Per-source result returned by bulk image transfer requests."""

    source_image: str = Field(..., alias="sourceImage")
    success: bool
    build_id: Optional[str] = Field(default=None, alias="buildId")
    full_image_path: Optional[str] = Field(default=None, alias="fullImagePath")
    visibility: Optional[ImageVisibility] = None
    error: Optional[str] = None
    retryable: bool = False

    model_config = ConfigDict(populate_by_name=True)


class BulkImageTransferResponse(BaseModel):
    """Response returned for comma-separated image transfer requests."""

    results: List[TransferImageResult] = Field(default_factory=list)
    failed: List[TransferImageResult] = Field(default_factory=list)

    model_config = ConfigDict(populate_by_name=True)


MAX_IMAGE_UPDATES = 100


class PersonalImageOwner(BaseModel):
    """The authenticated caller's personal image scope."""

    type: Literal["personal"] = "personal"

    model_config = ConfigDict(populate_by_name=True)


class TeamImageOwner(BaseModel):
    """A team image scope."""

    type: Literal["team"] = "team"
    team_id: str = Field(..., alias="teamId")

    model_config = ConfigDict(populate_by_name=True)


class PlatformImageOwner(BaseModel):
    """Org-less platform image scope (admin-managed, always PUBLIC)."""

    type: Literal["platform"] = "platform"

    model_config = ConfigDict(populate_by_name=True)


ImageOwner = Annotated[
    Union[PersonalImageOwner, TeamImageOwner, PlatformImageOwner],
    Field(discriminator="type"),
]


class ImageUpdateSource(BaseModel):
    """Source selector for one logical-image update.

    Either the structured coordinate form (``owner`` + ``name`` + ``tag``) or
    the ``reference`` form (owner-prefixed / plain ``name:tag`` reference).
    The two forms are mutually exclusive.
    """

    owner: Optional[ImageOwner] = None
    name: Optional[str] = None
    tag: Optional[str] = None
    reference: Optional[str] = None

    model_config = ConfigDict(populate_by_name=True)

    @model_validator(mode="after")
    def _require_exactly_one_form(self) -> "ImageUpdateSource":
        coordinate_fields = (self.owner, self.name, self.tag)
        if self.reference is not None:
            if any(value is not None for value in coordinate_fields):
                raise ValueError("source accepts either reference or owner/name/tag, not both")
        elif any(value is None for value in coordinate_fields):
            raise ValueError("source requires owner, name, and tag (or a reference)")
        return self


class ImageUpdatePatch(BaseModel):
    """Partial patch for one logical image; omitted fields keep their value."""

    name: Optional[str] = None
    tag: Optional[str] = None
    owner: Optional[ImageOwner] = None
    visibility: Optional[ImageVisibility] = None

    model_config = ConfigDict(populate_by_name=True)

    @model_validator(mode="after")
    def _require_some_change(self) -> "ImageUpdatePatch":
        if (
            self.name is None
            and self.tag is None
            and self.owner is None
            and self.visibility is None
        ):
            raise ValueError("set must change at least one field")
        if (
            isinstance(self.owner, PlatformImageOwner)
            and self.visibility == ImageVisibility.PRIVATE
        ):
            raise ValueError("platform images are always PUBLIC")
        return self


class ImageUpdateItem(BaseModel):
    """One independent logical-image patch."""

    source: ImageUpdateSource
    set: ImageUpdatePatch

    model_config = ConfigDict(populate_by_name=True)


class UpdateImagesRequest(BaseModel):
    """Explicit list of independent logical-image patches.

    Every update names its source exactly; there is no server-side search
    selection for mutations.
    """

    mode: Literal["explicit"] = "explicit"
    dry_run: bool = Field(default=False, alias="dryRun")
    updates: List[ImageUpdateItem]

    model_config = ConfigDict(populate_by_name=True)


class ImageMutationError(BaseModel):
    """Machine-readable failure for one update item.

    ``code`` is intentionally a plain string so new server-side codes never
    break response parsing.
    """

    code: str
    message: str

    model_config = ConfigDict(populate_by_name=True)


class ImageCoordinateState(BaseModel):
    """Canonical logical-image coordinate plus effective visibility."""

    owner: ImageOwner
    name: str
    tag: str
    visibility: ImageVisibility

    model_config = ConfigDict(populate_by_name=True)


class ImageUpdateResult(BaseModel):
    """Per-item outcome, in deterministic request order."""

    source: ImageUpdateSource
    success: bool
    before: Optional[ImageCoordinateState] = None
    after: Optional[ImageCoordinateState] = None
    error: Optional[ImageMutationError] = None

    model_config = ConfigDict(populate_by_name=True)


class UpdateImagesResponse(BaseModel):
    """Response for both explicit and search update modes."""

    success: bool
    dry_run: bool = Field(default=False, alias="dryRun")
    results: List[ImageUpdateResult] = Field(default_factory=list)

    model_config = ConfigDict(populate_by_name=True)


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
    stdout_truncated: bool = False
    stderr_truncated: bool = False
