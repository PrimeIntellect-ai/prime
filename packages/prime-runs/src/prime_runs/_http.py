"""Shared HTTP client for the platform run APIs.

One client for all backends and the legacy samples sink, because they share a
host, a credential and a retry policy. Two things it does that a bare
``httpx.Client`` does not:

- maps status codes onto :mod:`prime_runs.exceptions` so callers branch on a
  type rather than on a message;
- retries 429/502/503/504 and transport failures with exponential backoff,
  honouring ``Retry-After`` when the server sends one.

Retries are safe here because every call it makes is either idempotent (PUT,
GET) or create-shaped and guarded upstream: run creation happens exactly once
per ``init()``, and sample POSTs that get retried after a lost response are the
known duplicate-append case the traces sink exists to replace.
"""

import json
import sys
import time
from typing import Any, Dict, Mapping, Optional, Union

import httpx

from .exceptions import (
    NotFoundError,
    PaymentRequiredError,
    RetryableAPIError,
    RunAPIError,
    TransportError,
    UnauthorizedError,
)

DEFAULT_TIMEOUT = httpx.Timeout(60.0, connect=10.0)
# Sample batches are megabytes and the platform fans them out to storage before
# answering, so uploads get their own, much longer budget.
UPLOAD_TIMEOUT = httpx.Timeout(300.0, connect=10.0)
RETRY_STATUS = frozenset({429, 502, 503, 504})
DEFAULT_MAX_ATTEMPTS = 5
MAX_BACKOFF_SECONDS = 16.0


def _user_agent() -> str:
    from . import __version__

    py = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    return f"prime-runs/{__version__} python/{py}"


def retry_delay(attempt: int, retry_after: Optional[float]) -> float:
    """Seconds to wait before ``attempt`` (1-based). Server wins if it spoke."""
    if retry_after is not None and retry_after >= 0:
        return min(retry_after, MAX_BACKOFF_SECONDS)
    return min(2.0 ** (attempt - 1), MAX_BACKOFF_SECONDS)


def _parse_retry_after(response: httpx.Response) -> Optional[float]:
    raw = response.headers.get("retry-after")
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        # HTTP-date form. Not worth parsing for a backoff hint — fall back to
        # the exponential schedule rather than guessing a clock skew.
        return None


def encode_json(value: Any) -> bytes:
    """Compact UTF-8 JSON, matching the encoding used to size batches.

    ``allow_nan=False`` matters: a NaN reward serialized as JavaScript's bare
    ``NaN`` is rejected by strict JSON parsers server-side, and the failure
    surfaces as an opaque 400 on a payload the producer cannot inspect.
    """
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode(
        "utf-8"
    )


class PlatformClient:
    """Minimal authenticated client for ``{base_url}/api/v1``."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        timeout: httpx.Timeout = DEFAULT_TIMEOUT,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        client: Optional[httpx.Client] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_prefix = f"{self.base_url}/api/v1"
        self.max_attempts = max(1, max_attempts)
        self._owns_client = client is None
        self._client = client or httpx.Client(
            headers={
                "Authorization": f"Bearer {api_key}",
                "User-Agent": _user_agent(),
            },
            follow_redirects=True,
            timeout=timeout,
        )

    def request(
        self,
        method: str,
        path: str,
        *,
        json_body: Optional[Mapping[str, Any]] = None,
        content: Optional[bytes] = None,
        params: Optional[Mapping[str, Any]] = None,
        timeout: Union[httpx.Timeout, float, None] = None,
        max_attempts: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Send one request, retrying transient failures. Returns the JSON body."""
        url = f"{self.api_prefix}{path}"
        body = content if content is not None else (encode_json(json_body) if json_body else None)
        headers = {"Content-Type": "application/json"} if body is not None else None
        attempts = max_attempts or self.max_attempts

        last_error: Optional[Exception] = None
        for attempt in range(1, attempts + 1):
            try:
                response = self._client.request(
                    method,
                    url,
                    content=body,
                    headers=headers,
                    params=dict(params) if params else None,
                    timeout=timeout,
                )
            except httpx.TimeoutException as exc:
                last_error = TransportError(f"{method} {path} timed out: {exc}")
            except httpx.RequestError as exc:
                last_error = TransportError(f"{method} {path} failed: {type(exc).__name__}: {exc}")
            else:
                if response.status_code in RETRY_STATUS:
                    last_error = RetryableAPIError(
                        _error_message(response),
                        status_code=response.status_code,
                        code=_error_code(response),
                        retry_after=_parse_retry_after(response),
                    )
                elif response.is_error:
                    raise _map_error(response)
                else:
                    return _decode(response)

            if attempt == attempts:
                break
            after = getattr(last_error, "retry_after", None)
            time.sleep(retry_delay(attempt, after))

        assert last_error is not None
        raise last_error

    def get(self, path: str, **kwargs: Any) -> Dict[str, Any]:
        return self.request("GET", path, **kwargs)

    def post(self, path: str, **kwargs: Any) -> Dict[str, Any]:
        return self.request("POST", path, **kwargs)

    def put(self, path: str, **kwargs: Any) -> Dict[str, Any]:
        return self.request("PUT", path, **kwargs)

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "PlatformClient":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()


def _decode(response: httpx.Response) -> Dict[str, Any]:
    if not response.content:
        return {}
    try:
        payload = response.json()
    except ValueError as exc:
        raise RunAPIError(
            f"{response.request.method} {response.request.url.path} returned non-JSON "
            f"({response.status_code}): {response.text[:200]!r}"
        ) from exc
    return payload if isinstance(payload, dict) else {"data": payload}


def _error_body(response: httpx.Response) -> Dict[str, Any]:
    try:
        payload = response.json()
    except ValueError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _error_code(response: httpx.Response) -> Optional[str]:
    body = _error_body(response)
    code = body.get("code") or body.get("error_code")
    return str(code) if code else None


def _error_message(response: httpx.Response) -> str:
    body = _error_body(response)
    detail = body.get("detail") or body.get("message") or body.get("error")
    if detail is None:
        detail = response.text[:200] or response.reason_phrase
    return (
        f"HTTP {response.status_code} from "
        f"{response.request.method} {response.request.url.path}: {detail}"
    )


def _map_error(response: httpx.Response) -> RunAPIError:
    message = _error_message(response)
    code = _error_code(response)
    status = response.status_code
    if status == 401:
        return UnauthorizedError(
            f"{message} — check PRIME_API_KEY or run `prime login`.",
            status_code=status,
            code=code,
        )
    if status == 402:
        return PaymentRequiredError(message, status_code=status, code=code)
    if status == 404:
        return NotFoundError(message, status_code=status, code=code)
    return RunAPIError(message, status_code=status, code=code)
