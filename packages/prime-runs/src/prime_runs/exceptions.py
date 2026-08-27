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
    """An environment named in ``init()`` could not be resolved to a hub ID —
    usually a typo or a permissions problem, not an outage."""


class RunFinishedError(PrimeRunsError):
    """A finished run was written to again: a producer bug."""


def is_record_rejection(exc: BaseException) -> bool:
    """Whether a failed write is specific to the submitted record batch.

    A later batch can still succeed after serialization, size, or validation
    failures, so these errors must not retire an otherwise healthy sink.
    """
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
    # which the shared traces mapper intentionally leaves as plain APIError.
    return isinstance(exc, APIError) and exc.status_code in (413, 422)


def is_transient(exc: BaseException) -> bool:
    """Whether a failure is about this moment (retry later) rather than this
    run (stop). Decides whether a sink is retired."""
    return isinstance(exc, (RetryableAPIError, TransportError))
