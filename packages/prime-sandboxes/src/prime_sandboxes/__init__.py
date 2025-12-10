"""Prime Intellect Sandboxes SDK.

A standalone SDK for managing remote code execution environments (sandboxes).
Includes HTTP client, authentication, and full sandbox lifecycle management.
"""

from .core import (
    APIClient,
    APIError,
    APITimeoutError,
    AsyncAPIClient,
    Config,
    PaymentRequiredError,
    UnauthorizedError,
)
from .exceptions import (
    CommandTimeoutError,
    DownloadTimeoutError,
    SandboxNotRunningError,
    UploadTimeoutError,
)
from .models import (
    AdvancedConfigs,
    BackgroundJob,
    BackgroundJobStatus,
    BulkDeleteSandboxRequest,
    BulkDeleteSandboxResponse,
    CommandRequest,
    CommandResponse,
    CreateSandboxRequest,
    ExposedPort,
    ExposePortRequest,
    FileUploadResponse,
    ListExposedPortsResponse,
    Sandbox,
    SandboxListResponse,
    SandboxStatus,
    SSHSession,
    UpdateSandboxRequest,
)
from .sandbox import AsyncSandboxClient, SandboxClient

__version__ = "0.2.6"

# Deprecated alias for backward compatibility
TimeoutError = APITimeoutError

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
    "BackgroundJob",
    "BackgroundJobStatus",
    # Port Forwarding
    "ExposePortRequest",
    "ExposedPort",
    "ListExposedPortsResponse",
    "SSHSession",
    # Exceptions
    "APIError",
    "UnauthorizedError",
    "PaymentRequiredError",
    "APITimeoutError",
    "TimeoutError",  # Deprecated alias
    "SandboxNotRunningError",
    "CommandTimeoutError",
    "UploadTimeoutError",
    "DownloadTimeoutError",
]
