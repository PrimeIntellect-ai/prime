"""HTTP client for the platform run APIs.

Error mapping, backoff and the transport-failure classification come from
``prime_traces.core.client``; what is local is the retry *policy*, because it
is decided per call. A failure is *ambiguous* when the request may already have
been processed (a 502/504, a read timeout). Replaying one is fine for a GET or
PUT and not for ``POST /evaluations/``, which would create a second run.
Callers declare intent with ``idempotent=``; unambiguous failures (connect
errors, 429) are replayed for every method.
"""

import json
import sys
import time
from typing import Any, Dict, Mapping, Optional, Union

import httpx
from prime_traces.core.client import (
    AMBIGUOUS_TRANSPORT_ERRORS,
    raise_for_response,
    retry_delay,
)

from . import _fork
from .exceptions import APIError, APITimeoutError, RetryableAPIError, TransportError

DEFAULT_TIMEOUT = httpx.Timeout(60.0, connect=10.0)
# Sample batches are megabytes; uploads get a longer budget.
UPLOAD_TIMEOUT = httpx.Timeout(300.0, connect=10.0)
#: Refused before any work was done, so replaying cannot duplicate anything.
#: 503 is excluded: it may come from an intermediary after forwarding.
UNAMBIGUOUS_RETRY_STATUS = frozenset({429})
DEFAULT_MAX_ATTEMPTS = 5


def _user_agent() -> str:
    from . import __version__

    py = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    return f"prime-runs/{__version__} python/{py}"


def encode_json(value: Any) -> bytes:
    """Compact UTF-8 JSON. ``allow_nan=False``: strict parsers server-side
    reject bare ``NaN`` and the failure is an opaque 400."""
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
        # Strip a trailing /api/v1 — the same normalization prime_traces applies —
        # so an explicit base_url written with the suffix does not double it.
        self.base_url = base_url.rstrip("/").removesuffix("/api/v1")
        self.api_prefix = f"{self.base_url}/api/v1"
        self.max_attempts = max(1, max_attempts)
        self._owns_client = client is None
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "User-Agent": _user_agent(),
        }
        self._timeout = timeout
        self._client = client or self._new_client()
        if self._owns_client:
            # Only a pool we opened ourselves is ours to rebuild after a fork.
            _fork.register(self)

    def _new_client(self) -> httpx.Client:
        return httpx.Client(
            headers=dict(self._headers),
            follow_redirects=True,
            timeout=self._timeout,
        )

    def reset_after_fork(self) -> None:
        """Rebuild the pool in a forked child. The old one is dropped, not
        closed: closing could send ``close_notify`` on the parent's socket."""
        self._client = self._new_client()

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
        idempotent: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Send one request, retrying transient failures. Returns the JSON body.

        ``idempotent`` defaults to ``method != "POST"``; a POST that is safe to
        replay (get-or-create) passes ``idempotent=True`` explicitly.
        """
        url = f"{self.api_prefix}{path}"
        body = content if content is not None else (encode_json(json_body) if json_body else None)
        request_kwargs: Dict[str, Any] = {
            "content": body,
            "headers": {"Content-Type": "application/json"} if body is not None else None,
            "params": dict(params) if params else None,
        }
        # ``None`` would disable httpx timeouts, not restore the default.
        if timeout is not None:
            request_kwargs["timeout"] = timeout
        attempts = max_attempts or self.max_attempts
        replayable = idempotent if idempotent is not None else method.upper() != "POST"

        for attempt in range(attempts):
            error: APIError
            try:
                response = self._client.request(method, url, **request_kwargs)
            except httpx.TimeoutException as exc:
                error = APITimeoutError(f"{method} {path} timed out: {exc}")
                ambiguous = isinstance(exc, AMBIGUOUS_TRANSPORT_ERRORS)
            except httpx.RequestError as exc:
                error = TransportError(f"{method} {path} failed: {type(exc).__name__}: {exc}")
                ambiguous = isinstance(exc, AMBIGUOUS_TRANSPORT_ERRORS)
            else:
                try:
                    raise_for_response(response)
                except RetryableAPIError as exc:
                    error = exc
                    ambiguous = exc.status_code not in UNAMBIGUOUS_RETRY_STATUS
                else:
                    return _decode(response)

            last = attempt == attempts - 1
            if last or (ambiguous and not replayable):
                # Possibly processed already; a duplicate cannot be undone.
                raise error
            time.sleep(retry_delay(error, attempt))
        raise AssertionError("unreachable")  # pragma: no cover

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
        raise APIError(
            f"{response.request.method} {response.request.url.path} returned non-JSON "
            f"({response.status_code}): {response.text[:200]!r}"
        ) from exc
    return payload if isinstance(payload, dict) else {"data": payload}
