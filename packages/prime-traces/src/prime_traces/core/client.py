"""HTTP client for the Prime Traces service.

Speaks the wire contract and maps error responses to typed exceptions.
Retry policy deliberately lives in the caller (``TracesClient``), not here:
which responses are retryable is part of the upload contract, and the
uploader owns honoring Retry-After against its own attempt budget.
"""

import gzip as gzip_module
import json as json_module
import sys
from typing import Any, Dict, Iterator, Optional

import httpx

from ..exceptions import (
    APIError,
    APITimeoutError,
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


def _default_user_agent() -> str:
    """Build default User-Agent string"""
    from prime_traces import __version__

    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    return f"prime-traces/{__version__} python/{python_version}"


def _parse_retry_after(response: httpx.Response) -> Optional[float]:
    value = response.headers.get("Retry-After")
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        # HTTP-date form; callers fall back to their own backoff.
        return None


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
        self.base_url = (base_url or self.config.traces_url).rstrip("/")
        self.team_id = team_id if team_id is not None else self.config.team_id

        # No default Content-Type here: uploads are multipart (httpx must own
        # the boundary header) and reads set JSON per-request.
        headers = {"User-Agent": user_agent or _default_user_agent()}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if self.team_id:
            headers["X-Prime-Team-Id"] = self.team_id

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

    def get_json(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self._check_auth()
        try:
            response = self.client.get(self._url(endpoint), params=params)
        except Exception as exc:
            raise self._wrap_transport_errors(exc) from exc
        raise_for_response(response)
        result = response.json()
        if not isinstance(result, dict):
            raise APIError("API response was not a dictionary")
        return result

    def delete_json(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self._check_auth()
        try:
            response = self.client.delete(self._url(endpoint), params=params)
        except Exception as exc:
            raise self._wrap_transport_errors(exc) from exc
        raise_for_response(response)
        if not response.content:
            return {}
        result = response.json()
        return result if isinstance(result, dict) else {}

    def stream_bytes(
        self, endpoint: str, params: Optional[Dict[str, Any]] = None
    ) -> Iterator[bytes]:
        """Stream a response body in chunks (a raw trace can be 64 MiB)."""
        self._check_auth()
        try:
            with self.client.stream("GET", self._url(endpoint), params=params) as response:
                if not response.is_success:
                    response.read()
                    raise_for_response(response)
                yield from response.iter_bytes()
        except Exception as exc:
            raise self._wrap_transport_errors(exc) from exc

    def close(self) -> None:
        self.client.close()
