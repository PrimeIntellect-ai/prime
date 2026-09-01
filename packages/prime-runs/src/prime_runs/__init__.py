"""Prime Intellect Runs SDK.

Track eval and training runs on the Prime platform::

    import prime_runs as pr

    run = pr.init(name="gsm8k-qwen3-8b", environments=["gsm8k"], model=model)
    print(run.url)

    for episode in rollouts:              # stamp run.id onto the traces
        run.log_traces([episode])

    run.finish(summary=pr.metrics.from_episodes(episodes))

This is a leaf package: the ``prime`` CLI depends on ``verifiers``, so verifiers
can never depend on ``prime``. Nothing here imports a producer package; records
are duck-typed through ``to_record()``.
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

__version__ = "0.1.0"

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
