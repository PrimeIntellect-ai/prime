"""Exceptions for the Prime Runs SDK.

Two families. Errors raised here, before any request is made, derive from
:class:`PrimeRunsError`. Errors from the platform derive from
``prime_traces.APIError``: both SDKs talk to the same platform with the same
credential, and the uploader already handles the traces family, so one
vocabulary serves both. Nothing in either family escapes into a producer loop
under the default ``on_error="warn"``. To catch everything the SDK can raise
under ``on_error="raise"``::

    except (pr.PrimeRunsError, pr.APIError):
"""

from prime_traces.exceptions import (
    APIError,
    APITimeoutError,
    ForbiddenError,
    NotFoundError,
    PaymentRequiredError,
    RetryableAPIError,
    TransportError,
    UnauthorizedError,
)

__all__ = [
    "APIError",
    "APITimeoutError",
    "ConfigurationError",
    "EnvironmentResolutionError",
    "ForbiddenError",
    "NotFoundError",
    "PaymentRequiredError",
    "PrimeRunsError",
    "RetryableAPIError",
    "RunFinishedError",
    "TransportError",
    "UnauthorizedError",
    "is_transient",
]


class PrimeRunsError(Exception):
    """Base for errors raised by the SDK itself, before any request is made."""


class ConfigurationError(PrimeRunsError):
    """Missing API key, unreadable config file, unknown mode."""


class EnvironmentResolutionError(PrimeRunsError):
    """An environment named in ``init()`` could not be resolved to a hub ID —
    usually a typo or a permissions problem, not an outage."""


class RunFinishedError(PrimeRunsError):
    """A finished run was written to again: a producer bug."""


def is_transient(exc: BaseException) -> bool:
    """Whether a failure is about this moment (retry later) rather than this
    run (stop). Decides whether a sink is retired."""
    return isinstance(exc, (RetryableAPIError, TransportError))
