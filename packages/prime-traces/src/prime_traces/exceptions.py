"""Exceptions for the Prime Traces SDK.

Two families:

- Local errors raised before any request is made (``TraceTooLargeError``).
- API errors mapped from HTTP responses. The service answers with a bounded
  set of error codes (see ``models.ErrorCode``); producers are expected to
  branch on the code, not the message.
"""

from typing import Optional


class PrimeTracesError(Exception):
    """Base exception for the Prime Traces SDK."""


class TraceTooLargeError(PrimeTracesError):
    """A single JSONL line exceeded the per-line limit.

    Raised locally during batching: an oversized line is rejected rather than
    split, because under bare-trace format the line is one trace and under
    episode format it is one complete episode.
    """

    def __init__(self, line_number: int, size: int, limit: int):
        self.line_number = line_number
        self.size = size
        self.limit = limit
        super().__init__(
            f"Line {line_number} is {size} bytes, exceeding the {limit} byte "
            "single-line limit. Oversized lines are rejected locally rather than split."
        )


class APIError(PrimeTracesError):
    """An HTTP error response from the Prime Traces service."""

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


class UnauthorizedError(APIError):
    """401 — the credential was rejected. Stop rather than retry."""


class PaymentRequiredError(APIError):
    """402 — payment required. Check billing status."""


class NotFoundError(APIError):
    """404 — the trace, episode or job does not exist for this owner."""


class ValidationRejectedError(APIError):
    """400 — the request was rejected; nothing was stored.

    ``code`` carries one of the bounded rejection codes (``models.ErrorCode``).
    Validation is deterministic, so resubmitting the same bytes yields the
    same verdict: correct the file and resubmit (corrected content hashes to
    a new batch ID).
    """


class LineFormatConflictError(APIError):
    """409 ``line_format_conflict`` — the same content-addressed bytes were
    previously submitted under a different ``X-Prime-Line-Format``."""


class RetryableAPIError(APIError):
    """429, 502, 503 or 504 — retry the exact same bytes after ``retry_after``
    seconds.

    On 429/503 ``code`` distinguishes what saturated: ``rate_limited``,
    ``writer_pool_saturated``, ``ingest_capacity_exceeded``,
    ``ingest_unavailable``, ``storage_unavailable``, or ``auth_unavailable``.
    502/504 come from gateways in front of the service, so they carry no
    service code.
    """

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


class TransportError(APIError):
    """The request failed at the transport level — connection refused, TLS
    failure, or a stream broken before the response completed — so there is
    no HTTP status to interpret.

    For content-addressed uploads these are always safe to retry, even the
    ambiguous ones where the request may have been processed: the same bytes
    resolve to the same idempotency key and the service replays the prior
    result instead of storing twice.
    """


class APITimeoutError(TransportError):
    """The request timed out at the transport level."""
