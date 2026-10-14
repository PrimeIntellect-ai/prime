"""Pydantic models for the dev-sandboxes API (/api/v1/dev-sandboxes).

Lifecycle states, operation states, and the structured error taxonomy are
ported verbatim from the dev-sandbox skill (the live-proven source of truth).
Wire fields follow the MVP contract: environments carry a lease/expiry, tunnel
URLs, and a resolved SHA; operations carry kind/state, progress, bounded
diagnostics, and a result payload. ``extra="allow"`` covers additive growth:
new response fields must not break an older SDK.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class EnvironmentState(str, Enum):
    """Dev environment lifecycle, ported verbatim from the skill."""

    ALLOCATING = "ALLOCATING"
    PREPARING = "PREPARING"
    STARTING = "STARTING"
    VERIFYING = "VERIFYING"
    READY = "READY"
    FAILED = "FAILED"
    DELETING = "DELETING"
    DELETED = "DELETED"


#: States where the environment still exists and may become/become-again usable.
ACTIVE_STATES: frozenset[EnvironmentState] = frozenset(
    {
        EnvironmentState.ALLOCATING,
        EnvironmentState.PREPARING,
        EnvironmentState.STARTING,
        EnvironmentState.VERIFYING,
        EnvironmentState.READY,
        EnvironmentState.DELETING,
    }
)

#: States counted against the per-user/per-team concurrent allocation cap.
CAPPED_STATES: frozenset[EnvironmentState] = frozenset(
    {
        EnvironmentState.ALLOCATING,
        EnvironmentState.PREPARING,
        EnvironmentState.STARTING,
        EnvironmentState.VERIFYING,
        EnvironmentState.READY,
    }
)


class OperationState(str, Enum):
    """Operation lifecycle states (ported verbatim)."""

    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


TERMINAL_OPERATION_STATES: frozenset[OperationState] = frozenset(
    {OperationState.SUCCEEDED, OperationState.FAILED, OperationState.INTERRUPTED}
)


class OperationKind(str, Enum):
    """Kinds of long-running dev environment operations."""

    CREATE = "create"
    EXEC = "exec"
    DELETE = "delete"


class ErrorDetail(BaseModel):
    """Structured error body (taxonomy ported from the skill's errors.py)."""

    model_config = ConfigDict(extra="allow")

    code: str
    stage: str = ""
    message: str = ""
    retryable: bool = False
    provider_code: Optional[str] = None
    operation_id: Optional[str] = None
    owned_resource_ids: List[str] = Field(default_factory=list)
    diagnostic_path: Optional[str] = None

    @property
    def group(self) -> Optional[str]:
        from .errors import ALL_CODES, GROUPS

        if self.code in ALL_CODES:
            return next(g for g, codes in GROUPS.items() if self.code in codes)
        return None


class CreateEnvironmentRequest(BaseModel):
    """POST /api/v1/dev-sandboxes request body.

    Actor and team identity come from the verified token; ``team_id`` selects
    the team context when the token carries more than one.
    """

    template: str
    ref: str
    ttl_minutes: int = 120
    team_id: Optional[str] = None


class CreateEnvironmentResponse(BaseModel):
    """202 response for environment creation."""

    model_config = ConfigDict(extra="allow")

    environment_id: str
    operation_id: str


class DeleteEnvironmentResponse(BaseModel):
    """202 response for environment deletion."""

    model_config = ConfigDict(extra="allow")

    operation_id: str


class ExecCommandRequest(BaseModel):
    """POST /api/v1/dev-sandboxes/{id}/exec request body."""

    command: str
    timeout_seconds: int = 300


class ExecCommandResponse(BaseModel):
    """202 response for command execution."""

    model_config = ConfigDict(extra="allow")

    operation_id: str


class DevEnvironment(BaseModel):
    """A dev environment record (GET /{id} and list items)."""

    model_config = ConfigDict(extra="allow")

    id: str
    state: EnvironmentState
    template: str
    template_version: Optional[str] = None
    ref: str
    resolved_sha: Optional[str] = None
    ttl_minutes: int = 120
    team_id: Optional[str] = None
    actor_user_id: Optional[str] = None
    sandbox_id: Optional[str] = None
    urls: Dict[str, str] = Field(default_factory=dict)
    tunnel_ids: List[str] = Field(default_factory=list)
    credential_ref: Optional[str] = None
    expiry: Optional[datetime] = None
    last_error: Optional[ErrorDetail] = None
    operation_id: Optional[str] = None
    boot_seconds: Optional[float] = None
    ready_at: Optional[datetime] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    @property
    def is_active(self) -> bool:
        return self.state in ACTIVE_STATES

    @property
    def is_ready(self) -> bool:
        return self.state == EnvironmentState.READY


class EnvironmentListResponse(BaseModel):
    """Paginated GET /api/v1/dev-sandboxes response."""

    model_config = ConfigDict(extra="allow")

    environments: List[DevEnvironment] = Field(default_factory=list)
    total: int = 0
    page: int = 1
    per_page: int = 50
    has_next: bool = False


class Operation(BaseModel):
    """A long-running operation record (GET /operations/{id})."""

    model_config = ConfigDict(extra="allow")

    operation_id: str
    kind: OperationKind
    state: OperationState
    environment_id: Optional[str] = None
    stage: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    progress: Dict[str, Any] = Field(default_factory=dict)
    diagnostics: List[str] = Field(default_factory=list)
    result: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[ErrorDetail] = None

    @property
    def is_terminal(self) -> bool:
        return self.state in TERMINAL_OPERATION_STATES


class Template(BaseModel):
    """A read-only environment template (GET /templates)."""

    model_config = ConfigDict(extra="allow")

    id: str
    image: str
    cpu: int
    memory_gb: int
    disk_gb: int
    boot: str = "baked_image"
    version: Optional[str] = None
    maintainer: Optional[str] = None
    description: Optional[str] = None
