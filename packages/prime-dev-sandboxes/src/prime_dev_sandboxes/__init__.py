"""Prime Intellect Dev Sandboxes SDK.

A typed client of the platform's dev-sandboxes API (/api/v1/dev-sandboxes):
long-lived per-task development environments booted from checked-in
templates. This package speaks the API only — it contains no local
lifecycle engine.
"""

from .client import AsyncDevSandboxClient, DevSandboxClient
from .core import (
    APIClient,
    APIError,
    APIStatusError,
    APITimeoutError,
    AsyncAPIClient,
    Config,
    PaymentRequiredError,
    UnauthorizedError,
)
from .errors import ALL_CODES, GROUPS, DevSandboxError, group_of
from .models import (
    ACTIVE_STATES,
    CAPPED_STATES,
    TERMINAL_OPERATION_STATES,
    CreateEnvironmentRequest,
    CreateEnvironmentResponse,
    DeleteEnvironmentResponse,
    DevEnvironment,
    EnvironmentListResponse,
    EnvironmentState,
    ErrorDetail,
    ExecCommandRequest,
    ExecCommandResponse,
    Operation,
    OperationKind,
    OperationState,
    Template,
)

__version__ = "0.1.0"

__all__ = [
    # Clients
    "DevSandboxClient",
    "AsyncDevSandboxClient",
    # Core HTTP Client & Config
    "APIClient",
    "AsyncAPIClient",
    "Config",
    # Models
    "DevEnvironment",
    "EnvironmentState",
    "EnvironmentListResponse",
    "CreateEnvironmentRequest",
    "CreateEnvironmentResponse",
    "DeleteEnvironmentResponse",
    "ExecCommandRequest",
    "ExecCommandResponse",
    "Operation",
    "OperationKind",
    "OperationState",
    "ErrorDetail",
    "Template",
    "ACTIVE_STATES",
    "CAPPED_STATES",
    "TERMINAL_OPERATION_STATES",
    # Errors
    "APIError",
    "APIStatusError",
    "APITimeoutError",
    "UnauthorizedError",
    "PaymentRequiredError",
    "DevSandboxError",
    "GROUPS",
    "ALL_CODES",
    "group_of",
]
