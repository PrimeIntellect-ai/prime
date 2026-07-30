"""Exceptions for the Prime Traces SDK.

Two families:

- Local errors raised before any request is made (``TraceTooLargeError``).
- API errors mapped from HTTP responses. The service persists a bounded set of
  rejection codes (see ``models.ErrorCode``); producers are expected to branch
  on the code, not the message.
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
    """400 — the request was durably rejected; nothing was stored.

    ``code`` carries one of the bounded rejection codes (``models.ErrorCode``).
    The same bytes will replay the same rejection: correct the file and
    resubmit (corrected content hashes to a new batch ID).
    """


class LineFormatConflictError(APIError):
    """409 ``line_format_conflict`` — the same content-addressed bytes were
    previously submitted under a different ``X-Prime-Line-Format``."""


class RetryableAPIError(APIError):
    """429 or 503 — retry the exact same bytes after ``retry_after`` seconds."""

    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        retry_after: Optional[float] = None,
    ):
        super().__init__(message, status_code=status_code)
        self.retry_after = retry_after


class APITimeoutError(APIError):
    """The request timed out at the transport level."""
