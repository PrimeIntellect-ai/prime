"""HTTP client for the Prime Traces service.

Speaks the wire contract and maps error responses to typed exceptions.

Retry policy is split by what makes a retry safe. Idempotent requests — GET,
and DELETE, which the service accepts for already-absent rows — are retried
here with a small fixed budget, matching the sibling SDKs (prime-sandboxes
retries idempotent methods on 502/503/504 and transport failures). Uploads
are POSTs whose retry safety comes from content addressing rather than the
method, so their loop lives in ``TracesClient``, where the attempt budget is
caller-configurable and Retry-After is honored against it.
"""

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
from ..models import LineFormat
from .config import Config

#: Attempts for idempotent requests, matching prime-sandboxes'
#: ``stop_after_attempt(3)``. Uploads have their own caller-configurable
#: budget — see ``TracesClient._send_with_retry``.
IDEMPOTENT_RETRY_ATTEMPTS = 3

_BACKOFF_BASE_SECONDS = 1.0
_BACKOFF_CAP_SECONDS = 30.0
# Retry-After is server-controlled input (and may come from a gateway's
# HTTP-date far in the future); honor it, but never let it park the client.
_RETRY_AFTER_CAP_SECONDS = 60.0


def retry_delay(error: Exception, attempt: int) -> float:
    """Seconds to wait before retrying ``attempt`` (0-based).

    A server-supplied Retry-After wins, capped; otherwise jittered exponential
    backoff. Shared by the read retries here and the upload retries in
    ``TracesClient`` so the two paths cannot drift.
    """
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
    """The same normalization ``Config`` applies to file/env URLs.

    ``_url()`` appends ``/api/v1`` itself, and URLs are commonly written with
    the suffix already on them; without stripping it here, an explicit
    ``base_url=`` would request ``/api/v1/api/v1/...`` while the identical
    value via config worked.
    """
    return url.rstrip("/").removesuffix("/api/v1")


def _parse_retry_after(response: httpx.Response) -> Optional[float]:
    """Retry-After in either RFC 9110 form: delta-seconds or HTTP-date.

    Gateways in front of the service emit the date form, so it converts to the
    remaining delay rather than being dropped — dropping it would substitute a
    shorter local backoff and retry before the server asked. Unparseable
    values return None and callers fall back to their own backoff.
    """
    value = response.headers.get("Retry-After")
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        target = parsedate_to_datetime(value)
    except ValueError:
        return None
    if target.tzinfo is None:
        # RFC 5322 allows -0000, which parses naive; it means UTC.
        target = target.replace(tzinfo=timezone.utc)
    return max(0.0, (target - datetime.now(timezone.utc)).total_seconds())


def _extract_error(response: httpx.Response) -> tuple[Optional[str], str]:
    """Return (code, message) from an error body.

    The service envelope is {"error": {"code", "message"}}; fall back to
    FastAPI's {"detail": ...} and then raw text so nothing is swallowed.
    """
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
        # 502/504 come from gateways in front of the service, not the service
        # itself. Content addressing makes retrying them safe even for uploads
        # whose first attempt may have been processed: the same bytes resolve
        # to the same idempotency key and replay the committed receipt.
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
        # For api_key and team_id, None means "resolve from config" and any
        # explicit value — including "" — is final. Injectors like the prime
        # CLI pass their own resolved values and rely on an unset field never
        # silently re-resolving against the SDK's static config, which may
        # belong to a different context.
        self.api_key = api_key if api_key is not None else self.config.api_key
        self.base_url = _normalize_base_url(base_url or self.config.traces_url)
        self.team_id = team_id if team_id is not None else self.config.team_id

        # No default Content-Type here: uploads are multipart (httpx must own
        # the boundary header) and reads set JSON per-request.
        headers = {"User-Agent": user_agent or _default_user_agent()}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if self.team_id:
            # Spelled exactly as the service declares it (auth/dependencies.py
            # TEAM_HEADER) — case-insensitive on the wire, greppable in code.
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
        self, method: str, endpoint: str, params: Optional[Dict[str, Any]]
    ) -> httpx.Response:
        """Send with bounded retries on transient failures.

        Only for idempotent methods: GET, and DELETE — deletion is idempotent
        at the API level (deleting already-absent rows is still accepted), so
        a replayed DELETE cannot do more than the first one did.
        """
        for attempt in range(IDEMPOTENT_RETRY_ATTEMPTS):
            last = attempt == IDEMPOTENT_RETRY_ATTEMPTS - 1
            try:
                response = self.client.request(method, self._url(endpoint), params=params)
            except Exception as exc:
                error = self._wrap_transport_errors(exc)
                if error is exc:
                    raise
                if last:
                    raise error from exc
                time.sleep(retry_delay(error, attempt))
                continue
            try:
                raise_for_response(response)
            except RetryableAPIError as exc:
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

    def delete_json(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self._check_auth()
        response = self._idempotent_request("DELETE", endpoint, params)
        if not response.content:
            return {}
        result = response.json()
        return result if isinstance(result, dict) else {}

    def stream_bytes(
        self, endpoint: str, params: Optional[Dict[str, Any]] = None
    ) -> Iterator[bytes]:
        """Stream a response body in chunks (a raw trace can be 64 MiB).

        Transient failures are retried only until the first body byte has been
        yielded: a mid-stream retry would silently restart the body under the
        consumer (``_stream_to_file`` would write the prefix twice), so once
        bytes have flowed a failure raises and the caller re-runs.
        """
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
