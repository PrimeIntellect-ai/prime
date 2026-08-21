"""Exceptions for the Prime Runs SDK. Nothing here escapes into a producer
loop by default (``on_error="warn"``); callers opting into ``on_error="raise"``
can branch on these types."""

from typing import Optional


class PrimeRunsError(Exception):
    """Base exception for the Prime Runs SDK."""


class ConfigurationError(PrimeRunsError):
    """Missing API key, unreadable config file, unknown mode. Raised before
    any request is made."""


class RunAPIError(PrimeRunsError):
    """An HTTP error response from a run backend."""

    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        code: Optional[str] = None,
    ):
        self.status_code = status_code
        self.code = code
        super().__init__(message)


class UnauthorizedError(RunAPIError):
    """401 — the credential was rejected. Stop rather than retry."""


class PaymentRequiredError(RunAPIError):
    """402 — payment required. Check billing status."""


class ForbiddenError(RunAPIError):
    """403 — authenticated, but not allowed: another owner's run, a team the
    key cannot act for, or a feature gated to an allowlist. Named to match
    ``prime_traces.ForbiddenError``."""


class NotFoundError(RunAPIError):
    """404 — the run, environment or evaluation does not exist for this owner."""


class RetryableAPIError(RunAPIError):
    """429/5xx — retry the same request after ``retry_after`` seconds."""

    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        code: Optional[str] = None,
        retry_after: Optional[float] = None,
    ):
        super().__init__(message, status_code=status_code, code=code)
        self.retry_after = retry_after


class TransportError(RunAPIError):
    """The request failed below HTTP — connection refused, TLS failure, timeout."""


def is_transient(exc: BaseException) -> bool:
    """Whether a failure is about this moment (retry later) rather than this
    run (stop). Decides whether a sink is retired. Covers the traces service's
    exception family too, since both reach the uploader through one path."""
    if isinstance(exc, (RetryableAPIError, TransportError)):
        return True
    try:
        from prime_traces.exceptions import RetryableAPIError as TracesRetryable
        from prime_traces.exceptions import TransportError as TracesTransport
    except ImportError:  # pragma: no cover - dependency is declared
        return False
    return isinstance(exc, (TracesRetryable, TracesTransport))


class EnvironmentResolutionError(PrimeRunsError):
    """An environment named in ``init()`` could not be resolved to a hub ID —
    usually a typo or a permissions problem, not an outage."""


class RunFinishedError(PrimeRunsError):
    """A finished run was written to again: a producer bug."""
