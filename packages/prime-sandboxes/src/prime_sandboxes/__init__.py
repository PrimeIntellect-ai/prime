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
    SandboxFileNotFoundError,
    SandboxImagePullError,
    SandboxNotRunningError,
    SandboxOOMError,
    SandboxTimeoutError,
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
    DockerImageCheckResponse,
    ExposedPort,
    ExposePortRequest,
    FileUploadResponse,
    ListExposedPortsResponse,
    ReadFileResponse,
    RegistryCredentialSummary,
    Sandbox,
    SandboxListResponse,
    SandboxStatus,
    SSHSession,
    UpdateSandboxRequest,
)
from .sandbox import AsyncSandboxClient, AsyncTemplateClient, SandboxClient, TemplateClient

__version__ = "0.2.22"

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
    "TemplateClient",
    "AsyncTemplateClient",
    # Models
    "Sandbox",
    "SandboxStatus",
    "SandboxListResponse",
    "CreateSandboxRequest",
    "UpdateSandboxRequest",
    "CommandRequest",
    "CommandResponse",
    "FileUploadResponse",
    "ReadFileResponse",
    "BulkDeleteSandboxRequest",
    "BulkDeleteSandboxResponse",
    "RegistryCredentialSummary",
    "DockerImageCheckResponse",
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
    "SandboxFileNotFoundError",
    "APITimeoutError",
    "TimeoutError",  # Deprecated alias
    "SandboxOOMError",
    "SandboxTimeoutError",
    "SandboxImagePullError",
    "SandboxNotRunningError",
    "CommandTimeoutError",
    "UploadTimeoutError",
    "DownloadTimeoutError",
]
