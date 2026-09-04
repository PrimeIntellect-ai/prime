"""Errors raised before any request derive from ``PrimeRunsError``; errors from
the platform are ``prime_traces``' own, so one set of ``except`` clauses covers
both SDKs. Under ``on_error="warn"`` none of them escape into a producer loop."""

from prime_traces.exceptions import (
    APIError,
    APITimeoutError,
    ForbiddenError,
    LineFormatConflictError,
    NotFoundError,
    PaymentRequiredError,
    RetryableAPIError,
    TraceTooLargeError,
    TransportError,
    UnauthorizedError,
    ValidationRejectedError,
)

__all__ = [
    "APIError",
    "APITimeoutError",
    "ConfigurationError",
    "EnvironmentResolutionError",
    "ForbiddenError",
    "LineFormatConflictError",
    "NotFoundError",
    "PaymentRequiredError",
    "PrimeRunsError",
    "RetryableAPIError",
    "RunFinishedError",
    "TraceTooLargeError",
    "TransportError",
    "UnauthorizedError",
    "ValidationRejectedError",
    "is_record_rejection",
    "is_transient",
]


class PrimeRunsError(Exception):
    """Base for errors raised by the SDK itself, before any request is made."""


class ConfigurationError(PrimeRunsError):
    """Missing API key, unreadable config file, unknown mode."""


class EnvironmentResolutionError(PrimeRunsError):
    """An environment named in ``init()`` could not be resolved to a hub id."""


class RunFinishedError(PrimeRunsError):
    """A finished run was written to again: a producer bug."""


def is_record_rejection(exc: BaseException) -> bool:
    """A failure specific to the submitted batch: a later batch can still succeed."""
    if isinstance(
        exc,
        (
            TypeError,
            ValueError,
            TraceTooLargeError,
            ValidationRejectedError,
            LineFormatConflictError,
        ),
    ):
        return True
    # The legacy samples API returns framework-generated 413/422 responses,
    # which the shared traces mapper leaves as plain APIError.
    return isinstance(exc, APIError) and exc.status_code in (413, 422)


def is_transient(exc: BaseException) -> bool:
    """A failure about this moment (retry later) rather than this run (stop)."""
    return isinstance(exc, (RetryableAPIError, TransportError))
