"""Exceptions for the Prime Runs SDK.

Producers run for hours; the default posture is that nothing here escapes into
a training loop (``on_error="warn"``). These types exist so that callers who
opt into ``on_error="raise"`` — tests, CI, hosted workers — can branch on what
actually failed instead of matching log strings.
"""

from typing import Optional


class PrimeRunsError(Exception):
    """Base exception for the Prime Runs SDK."""


class ConfigurationError(PrimeRunsError):
    """The SDK was asked to do something the local configuration cannot support.

    Missing API key, an unknown ``kind``, ``mode="online"`` with no way to reach
    the platform. Raised before any request is made.
    """


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
    """403 — authenticated, but not allowed to do this.

    Distinct from 401 because the credential is fine and re-authenticating will
    not help: the run belongs to another owner, the team header names a team the
    key cannot act for, or the feature is gated to an allowlist. Named to match
    ``prime_traces.ForbiddenError``, which the traces sink already branches on
    to retire itself when an account is outside the closed beta.
    """


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
    """Whether a failure is about this moment rather than this run.

    The distinction decides whether a sink is retired. A gated account or a bad
    credential will fail identically on every future batch, so the sink should
    stop. A gateway blip or a dropped connection will not, and retiring a sink
    for one of those means a single 502 empties the rest of the run's dashboard.

    Covers the traces service's exception family as well as this package's,
    since both reach the uploader through the same path.
    """
    if isinstance(exc, (RetryableAPIError, TransportError)):
        return True
    try:
        from prime_traces.exceptions import RetryableAPIError as TracesRetryable
        from prime_traces.exceptions import TransportError as TracesTransport
    except ImportError:  # pragma: no cover - dependency is declared
        return False
    return isinstance(exc, (TracesRetryable, TracesTransport))


class EnvironmentResolutionError(PrimeRunsError):
    """An environment named in ``init()`` could not be resolved to a hub ID.

    Distinct from a generic API error because it is usually a typo or a
    permissions problem on the environment, not an outage, and because an eval
    run cannot be created without at least one resolved environment.
    """


class RunFinishedError(PrimeRunsError):
    """A finished run was written to again.

    Terminal status is reported once. Logging after ``finish()`` is a producer
    bug — the data would land on a run the platform has already closed out.
    """
