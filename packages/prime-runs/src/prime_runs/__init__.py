"""Prime Intellect Runs SDK.

Track eval and training runs on the Prime platform::

    import prime_runs as pr

    run = pr.init(name="gsm8k-qwen3-8b", environments=["gsm8k"], model=model)
    print(run.url)

    for episode in rollouts:              # stamp run.id onto the traces
        run.log_traces([episode])
        run.log({"reward": episode.reward}, step=step)

    run.finish(summary=pr.metrics.from_episodes(episodes))

``init()`` opens the run and returns a handle carrying its ID and dashboard
URL; records stream out on a background thread as the run proceeds; ``finish()``
closes it out. A ``with`` block does the last part for you, including on the
paths where the producer never gets to it.

This is a leaf package on purpose. The ``prime`` CLI depends on ``verifiers``,
so verifiers can never depend on ``prime`` — and verifiers is one of the two
producers this SDK exists to serve. Nothing here imports a producer package;
records are duck-typed through ``to_record()``.
"""

from . import metrics, projection
from .backends import Backend, EvalsBackend, OfflineBackend
from .config import Config
from .exceptions import (
    ConfigurationError,
    EnvironmentResolutionError,
    ForbiddenError,
    NotFoundError,
    PaymentRequiredError,
    PrimeRunsError,
    RetryableAPIError,
    RunAPIError,
    RunFinishedError,
    TransportError,
    UnauthorizedError,
)
from .models import (
    EnvironmentRef,
    Mode,
    OnError,
    RunHandle,
    RunKind,
    RunSpec,
    RunStatus,
)
from .projection import build_samples, trace_to_sample
from .run import MODE_ENV, RUN_ID_ENV, Run, init
from .sinks import EvalSamplesSink, OfflineSink, Sink, TracesSink

__version__ = "0.1.0"

__all__ = [
    # The surface almost every caller needs
    "init",
    "Run",
    "metrics",
    "projection",
    # Types
    "Config",
    "EnvironmentRef",
    "Mode",
    "OnError",
    "RunHandle",
    "RunKind",
    "RunSpec",
    "RunStatus",
    "MODE_ENV",
    "RUN_ID_ENV",
    # Backends & sinks, for callers assembling their own
    "Backend",
    "EvalsBackend",
    "OfflineBackend",
    "Sink",
    "EvalSamplesSink",
    "OfflineSink",
    "TracesSink",
    # The v0 sample projection, moved here from verifiers
    "build_samples",
    "trace_to_sample",
    # Exceptions
    "PrimeRunsError",
    "ConfigurationError",
    "EnvironmentResolutionError",
    "RunAPIError",
    "RunFinishedError",
    "ForbiddenError",
    "NotFoundError",
    "PaymentRequiredError",
    "RetryableAPIError",
    "TransportError",
    "UnauthorizedError",
]
