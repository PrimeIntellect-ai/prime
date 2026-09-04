"""Prime Intellect Runs SDK: track eval and training runs on the Prime platform.

A leaf package: the ``prime`` CLI depends on ``verifiers`` and verifiers depends
on this, so nothing here imports a producer package; records are duck-typed.
"""

from . import metrics, projection
from .exceptions import (
    APIError,
    ConfigurationError,
    EnvironmentResolutionError,
    ForbiddenError,
    NotFoundError,
    PaymentRequiredError,
    PrimeRunsError,
    RetryableAPIError,
    RunFinishedError,
    TransportError,
    UnauthorizedError,
)
from .models import (
    CONFIG_SOURCE_KEY,
    ConfigSource,
    EnvironmentRef,
    RunKind,
    RunStatus,
    TrainingSpec,
)
from .run import MODE_ENV, Run, init

__version__ = "0.1.1"

__all__ = [
    "init",
    "Run",
    "RunStatus",
    "RunKind",
    "TrainingSpec",
    "ConfigSource",
    "CONFIG_SOURCE_KEY",
    "EnvironmentRef",
    "MODE_ENV",
    "metrics",
    "projection",
    "PrimeRunsError",
    "ConfigurationError",
    "EnvironmentResolutionError",
    "APIError",
    "RunFinishedError",
    "ForbiddenError",
    "NotFoundError",
    "PaymentRequiredError",
    "RetryableAPIError",
    "TransportError",
    "UnauthorizedError",
]
