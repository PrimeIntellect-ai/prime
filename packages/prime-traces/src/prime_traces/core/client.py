"""HTTP client for the Prime Traces service."""

import gzip as gzip_module
import json as json_module
import random
import sys
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Dict, Iterator, Optional

import httpx

from ..exceptions import (
    AmbiguousDeleteError,
    APIError,
    APITimeoutError,
    ForbiddenError,
    LineFormatConflictError,
    NotFoundError,
    PaymentRequiredError,
    RetryableAPIError,
    TransportError,
    UnauthorizedError,
    ValidationRejectedError,
)
from ..models import ErrorCode, LineFormat
from .config import Config

IDEMPOTENT_RETRY_ATTEMPTS = 3

# Retryable statuses that leave it unknown whether the request reached the
# service.
_AMBIGUOUS_RETRY_STATUSES = frozenset({502, 504})

# Codes proving that a retryable response came from the service while it was
# declining the request
_SERVICE_REFUSAL_CODES = frozenset(
    {
        ErrorCode.RATE_LIMITED.value,
        ErrorCode.WRITER_POOL_SATURATED.value,
        ErrorCode.INGEST_CAPACITY_EXCEEDED.value,
        ErrorCode.INGEST_UNAVAILABLE.value,
        ErrorCode.STORAGE_UNAVAILABLE.value,
        ErrorCode.AUTH_UNAVAILABLE.value,
    }
)

# Transport failures that can happen after a request has started crossing the
# network.
_AMBIGUOUS_TRANSPORT_ERRORS = (
    httpx.CloseError,
    httpx.DecodingError,
    httpx.ReadError,
    httpx.ReadTimeout,
    httpx.RemoteProtocolError,
    httpx.WriteError,
    httpx.WriteTimeout,
)

_BACKOFF_BASE_SECONDS = 1.0
_BACKOFF_CAP_SECONDS = 30.0
_RETRY_AFTER_CAP_SECONDS = 60.0


def retry_delay(error: Exception, attempt: int) -> float:
    """Seconds to wait before retrying."""
    delay = getattr(error, "retry_after", None)
    if delay is not None:
        return min(delay, _RETRY_AFTER_CAP_SECONDS)
    return min(
        _BACKOFF_CAP_SECONDS,
        _BACKOFF_BASE_SECONDS * (2**attempt),
    ) * (0.5 + random.random())


def _default_user_agent() -> str:
    """Build default User-Agent string"""
    from prime_traces import __version__

    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    return f"prime-traces/{__version__} python/{python_version}"


def _normalize_base_url(url: str) -> str:
    """The same normalization ``Config`` applies to file/env URLs."""
    return url.rstrip("/").removesuffix("/api/v1")


def _parse_retry_after(response: httpx.Response) -> Optional[float]:
    value = response.headers.get("Retry-After")
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        target = parsedate_to_datetime(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if target.tzinfo is None:
        target = target.replace(tzinfo=timezone.utc)
    return max(0.0, (target - datetime.now(timezone.utc)).total_seconds())


def _extract_error(response: httpx.Response) -> tuple[Optional[str], str]:
    """Return (code, message) from an error body."""
    try:
        body = response.json()
    except ValueError:
        return None, response.text or f"HTTP {response.status_code}"
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict):
            return error.get("code"), error.get("message") or f"HTTP {response.status_code}"
        detail = body.get("detail")
        if detail:
            return None, str(detail)
    return None, response.text or f"HTTP {response.status_code}"


def raise_for_response(response: httpx.Response) -> None:
    """Map a non-2xx response to a typed exception."""
    if response.is_success:
        return
    status = response.status_code
    code, message = _extract_error(response)

    if status == 401:
        raise UnauthorizedError(
            "API key unauthorized. Check PRIME_API_KEY.", status_code=status, code=code
        )
    if status == 402:
        raise PaymentRequiredError(
            "Payment required. Check billing status.", status_code=status, code=code
        )
    if status == 403:
        # The service's message names the missing scope; pass it through
        # rather than substituting a canned line.
        raise ForbiddenError(message, status_code=status, code=code)
    if status == 404:
        raise NotFoundError(message, status_code=status, code=code)
    if status == 400:
        raise ValidationRejectedError(message, status_code=status, code=code)
    if status == 409:
        raise LineFormatConflictError(message, status_code=status, code=code)
    if status in (429, 502, 503, 504):
        raise RetryableAPIError(
            message, status_code=status, code=code, retry_after=_parse_retry_after(response)
        )
    raise APIError(f"HTTP {status}: {message}", status_code=status, code=code)


class TracesAPIClient:
    """Thin synchronous client for the Prime Traces REST API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        team_id: Optional[str] = None,
        user_agent: Optional[str] = None,
        timeout: Optional[httpx.Timeout] = None,
        transport: Optional[httpx.BaseTransport] = None,
    ):
        self.config = Config()
        self.api_key = api_key if api_key is not None else self.config.api_key
        self.base_url = _normalize_base_url(base_url or self.config.traces_url)
        self.team_id = team_id if team_id is not None else self.config.team_id

        # No default Content-Type here: uploads are multipart (httpx must own
        # the boundary header) and reads set JSON per-request.
        headers = {"User-Agent": user_agent or _default_user_agent()}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if self.team_id:
            headers["X-Prime-Team-ID"] = self.team_id

        self.client = httpx.Client(
            headers=headers,
            follow_redirects=True,
            # Generous read/write budget: one request body can be 256 MiB.
            timeout=timeout or httpx.Timeout(120.0, connect=10.0),
            transport=transport,
        )

    def _check_auth(self) -> None:
        if not self.api_key:
            raise APIError("No API key configured. Set PRIME_API_KEY environment variable.")

    def _url(self, endpoint: str) -> str:
        if not endpoint.startswith("/"):
            endpoint = f"/{endpoint}"
        return f"{self.base_url}/api/v1{endpoint}"

    def _wrap_transport_errors(self, exc: Exception) -> Exception:
        if isinstance(exc, httpx.TimeoutException):
            return APITimeoutError(f"Request timed out: {exc}")
        if isinstance(exc, httpx.RequestError):
            req = getattr(exc, "request", None)
            method = getattr(req, "method", "?")
            url = getattr(req, "url", "?")
            return TransportError(
                f"Request failed: {exc.__class__.__name__} at {method} {url}: {exc}"
            )
        return exc

    # -- write path ---------------------------------------------------------

    def upload_batch(
        self,
        body: bytes,
        idempotency_key: str,
        *,
        line_format: LineFormat = LineFormat.TRACE,
        schema_version: int = 1,
        context: Optional[Dict[str, str]] = None,
        compress: bool = True,
    ) -> Dict[str, Any]:
        """POST one content-addressed JSONL request.

        ``body`` is the exact uncompressed JSONL bytes; ``idempotency_key``
        must be ``sha256:<64 lowercase hex>`` over those exact bytes. Gzip is
        transport-only: the digest and all limits are defined over the
        uncompressed bytes, so compressing changes nothing but upload time.
        """
        self._check_auth()

        metadata: Dict[str, Any] = {"schema_version": schema_version}
        if context:
            metadata["context"] = context

        headers = {"Idempotency-Key": idempotency_key}
        if line_format is LineFormat.EPISODE:
            headers["X-Prime-Line-Format"] = LineFormat.EPISODE.value

        if compress:
            # mtime=0 keeps the compressed bytes reproducible across retries.
            traces_content = gzip_module.compress(body, mtime=0)
            traces_part = (
                "traces.jsonl",
                traces_content,
                "application/x-ndjson",
                {"Content-Encoding": "gzip"},
            )
        else:
            traces_part = ("traces.jsonl", body, "application/x-ndjson")  # type: ignore[assignment]

        files = {
            "metadata": (None, json_module.dumps(metadata).encode(), "application/json"),
            "traces": traces_part,
        }

        try:
            response = self.client.post(self._url("/traces"), headers=headers, files=files)
        except Exception as exc:
            raise self._wrap_transport_errors(exc) from exc

        raise_for_response(response)
        result = response.json()
        if not isinstance(result, dict):
            raise APIError("API response was not a dictionary")
        return result

    # -- read path ----------------------------------------------------------

    def _idempotent_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]],
    ) -> httpx.Response:
        """Send with bounded retries when replay is known to be safe.

        GET requests retry every transient failure. DELETE requests retry only
        failures known to happen before delivery, plus 429 and service-coded
        503 responses where the service declined the operation. A response-path
        transport failure or gateway 502/503/504 is ambiguous: the deletion may
        already have landed, and replaying it could delete a trace uploaded
        between attempts. Those failures therefore surface to the caller
        without a transparent retry.
        """
        is_delete = method.upper() == "DELETE"
        for attempt in range(IDEMPOTENT_RETRY_ATTEMPTS):
            last = attempt == IDEMPOTENT_RETRY_ATTEMPTS - 1
            try:
                response = self.client.request(method, self._url(endpoint), params=params)
            except Exception as exc:
                error = self._wrap_transport_errors(exc)
                if error is exc:
                    raise
                if is_delete and isinstance(exc, _AMBIGUOUS_TRANSPORT_ERRORS):
                    raise AmbiguousDeleteError(f"Delete outcome is unknown: {error}") from exc
                if last:
                    raise error from exc
                time.sleep(retry_delay(error, attempt))
                continue
            try:
                raise_for_response(response)
            except RetryableAPIError as exc:
                ambiguous_status = exc.status_code in _AMBIGUOUS_RETRY_STATUSES or (
                    exc.status_code == 503 and exc.code not in _SERVICE_REFUSAL_CODES
                )
                if is_delete and ambiguous_status:
                    raise AmbiguousDeleteError(
                        f"Delete outcome is unknown after HTTP {exc.status_code}: {exc}",
                        status_code=exc.status_code,
                        code=exc.code,
                    ) from exc
                if last:
                    raise
                time.sleep(retry_delay(exc, attempt))
                continue
            return response
        raise AssertionError("unreachable")  # pragma: no cover

    def get_json(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self._check_auth()
        response = self._idempotent_request("GET", endpoint, params)
        result = response.json()
        if not isinstance(result, dict):
            raise APIError("API response was not a dictionary")
        return result

    def delete(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> None:
        """Send a DELETE and discard the body — 202 carries none."""
        self._check_auth()
        self._idempotent_request("DELETE", endpoint, params)

    def stream_bytes(
        self, endpoint: str, params: Optional[Dict[str, Any]] = None
    ) -> Iterator[bytes]:
        """Stream a response body in chunks (a raw trace can be 64 MiB)."""
        self._check_auth()
        for attempt in range(IDEMPOTENT_RETRY_ATTEMPTS):
            last = attempt == IDEMPOTENT_RETRY_ATTEMPTS - 1
            yielded = False
            try:
                with self.client.stream("GET", self._url(endpoint), params=params) as response:
                    if not response.is_success:
                        response.read()
                        raise_for_response(response)
                    for chunk in response.iter_bytes():
                        yielded = True
                        yield chunk
                return
            except APIError as exc:
                # Already typed by raise_for_response.
                if yielded or last or not isinstance(exc, RetryableAPIError):
                    raise
                time.sleep(retry_delay(exc, attempt))
            except Exception as exc:
                error = self._wrap_transport_errors(exc)
                if error is exc:
                    raise
                if yielded or last:
                    raise error from exc
                time.sleep(retry_delay(error, attempt))

    def close(self) -> None:
        self.client.close()
