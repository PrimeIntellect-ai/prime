"""Prime Intellect Sandboxes SDK.

A standalone SDK for managing remote code execution environments (sandboxes).
Includes HTTP client, authentication, and full sandbox lifecycle management.
"""

from .client import (
    APIClient,
    APIError,
    APITimeoutError,
    AsyncAPIClient,
    PaymentRequiredError,
    TimeoutError,  # Deprecated alias for APITimeoutError
    UnauthorizedError,
)
from .config import Config
from .exceptions import CommandTimeoutError, SandboxNotRunningError
from .models import (
    AdvancedConfigs,
    BulkDeleteSandboxRequest,
    BulkDeleteSandboxResponse,
    CommandRequest,
    CommandResponse,
    CreateSandboxRequest,
    FileUploadResponse,
    Sandbox,
    SandboxListResponse,
    SandboxStatus,
    UpdateSandboxRequest,
)
from .sandbox import AsyncSandboxClient, SandboxClient

__version__ = "0.1.0"

__all__ = [
    # Core HTTP Client & Config
    "APIClient",
    "AsyncAPIClient",
    "Config",
    # Sandbox Clients
    "SandboxClient",
    "AsyncSandboxClient",
    # Models
    "Sandbox",
    "SandboxStatus",
    "SandboxListResponse",
    "CreateSandboxRequest",
    "UpdateSandboxRequest",
    "CommandRequest",
    "CommandResponse",
    "FileUploadResponse",
    "BulkDeleteSandboxRequest",
    "BulkDeleteSandboxResponse",
    "AdvancedConfigs",
    # Exceptions
    "APIError",
    "UnauthorizedError",
    "PaymentRequiredError",
    "APITimeoutError",
    "TimeoutError",  # Deprecated alias
    "SandboxNotRunningError",
    "CommandTimeoutError",
]
