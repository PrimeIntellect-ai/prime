"""Prime Intellect Evals SDK.

A standalone SDK for managing and pushing evaluations to Prime Intellect.
Includes HTTP client, authentication, and evaluation management
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
from .evals import AsyncEvalsClient, EvalsClient
from .exceptions import (
    EnvironmentNotFoundError,
    EvalsAPIError,
    EvaluationNotFoundError,
    InvalidEvaluationError,
    InvalidSampleError,
)
from .models import (
    CreateEvaluationRequest,
    Environment,
    EnvironmentReference,
    Evaluation,
    EvaluationListResponse,
    EvaluationStatus,
    FinalizeEvaluationRequest,
    PushSamplesRequest,
    Sample,
    SamplesResponse,
)

__version__ = "0.1.5"

__all__ = [
    # Core HTTP Client & Config
    "APIClient",
    "AsyncAPIClient",
    "Config",
    # Evals Clients
    "EvalsClient",
    "AsyncEvalsClient",
    # Models
    "Evaluation",
    "EvaluationStatus",
    "EvaluationListResponse",
    "CreateEvaluationRequest",
    "Sample",
    "SamplesResponse",
    "PushSamplesRequest",
    "FinalizeEvaluationRequest",
    "Environment",
    "EnvironmentReference",
    # Exceptions
    "APIError",
    "UnauthorizedError",
    "PaymentRequiredError",
    "APITimeoutError",
    "EvalsAPIError",
    "EnvironmentNotFoundError",
    "EvaluationNotFoundError",
    "InvalidEvaluationError",
    "InvalidSampleError",
]
