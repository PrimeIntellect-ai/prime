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
    RegistryCredentialSummary,
    Sandbox,
    SandboxListResponse,
    SandboxStatus,
    UpdateSandboxRequest,
)
from .sandbox import AsyncSandboxClient, AsyncTemplateClient, SandboxClient, TemplateClient

__version__ = "0.2.11"

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
    # Exceptions
    "APIError",
    "UnauthorizedError",
    "PaymentRequiredError",
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
