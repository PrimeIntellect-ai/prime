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
    SandboxFileTooLargeError,
    SandboxImagePullError,
    SandboxNotRunningError,
    SandboxOOMError,
    SandboxTimeoutError,
    UploadTimeoutError,
)
from .images import AsyncImageClient, ImageClient
from .models import (
    AdvancedConfigs,
    BackgroundJob,
    BackgroundJobStatus,
    BuildImageRequest,
    BuildImageResponse,
    BulkDeleteSandboxRequest,
    BulkDeleteSandboxResponse,
    BulkImageTransferResponse,
    CommandRequest,
    CommandResponse,
    CreateSandboxRequest,
    DockerImageCheckResponse,
    EgressPolicyStatus,
    ExposedPort,
    ExposePortRequest,
    FileUploadResponse,
    ImageCoordinateState,
    ImageMutationError,
    ImageOwner,
    ImageUpdateItem,
    ImageUpdatePatch,
    ImageUpdateResult,
    ImageUpdateSource,
    ImageVisibility,
    ListExposedPortsResponse,
    PersonalImageOwner,
    PlatformImageOwner,
    ReadFileResponse,
    RegistryCredentialSummary,
    Sandbox,
    SandboxEgressPolicy,
    SandboxListResponse,
    SandboxStatus,
    SSHSession,
    TeamImageOwner,
    TransferImageResult,
    UpdateImagesRequest,
    UpdateImagesResponse,
    UpdateSandboxRequest,
)
from .sandbox import AsyncSandboxClient, AsyncTemplateClient, SandboxClient, TemplateClient

__version__ = "0.2.32"

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
    "ImageClient",
    "AsyncImageClient",
    # Models
    "Sandbox",
    "SandboxEgressPolicy",
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
    "EgressPolicyStatus",
    "AdvancedConfigs",
    "BackgroundJob",
    "BackgroundJobStatus",
    "BuildImageRequest",
    "BuildImageResponse",
    "BulkImageTransferResponse",
    "TransferImageResult",
    "ImageVisibility",
    "ImageOwner",
    "PersonalImageOwner",
    "TeamImageOwner",
    "PlatformImageOwner",
    "ImageUpdateSource",
    "ImageUpdatePatch",
    "ImageUpdateItem",
    "UpdateImagesRequest",
    "UpdateImagesResponse",
    "ImageUpdateResult",
    "ImageCoordinateState",
    "ImageMutationError",
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
    "SandboxFileTooLargeError",
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
