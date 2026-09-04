"""Authenticated client for the platform API with a per-call retry policy.

Requests are addressed under ``{base_url}/api/v1``. When ``base_url`` is the
platform's *internal* RFT root instead (``…/api/internal/rft``, what a hosted
run's launcher injects as ``$PRIME_API_BASE``), they go under
``{base_url}/api/internal`` and the token is repeated in ``x-api-key``, the
header that router authenticates a run's own token on. A trailing ``/rft`` is
accepted on either root; only attached runs are reachable through the internal
one (it registers nothing and has no status endpoint).

A failure is *ambiguous* when the server may already have processed the request
(a 502/504, a read timeout). Replaying one is safe for a GET or PUT and not for
``POST /evaluations/``, which would create a second run; callers declare intent
with ``idempotent=``. Connect errors and 429 are replayed for every method.
Error mapping and backoff come from ``prime_traces.core.client``.
"""

import json
import sys
import time
from typing import Any, Dict, Mapping, Optional, Tuple, Union

import httpx
from prime_traces.core.client import AMBIGUOUS_TRANSPORT_ERRORS, raise_for_response, retry_delay

from . import _fork
from .exceptions import APIError, APITimeoutError, RetryableAPIError, TransportError

DEFAULT_TIMEOUT = httpx.Timeout(60.0, connect=10.0)
#: Sample batches are megabytes.
UPLOAD_TIMEOUT = httpx.Timeout(300.0, connect=10.0)
#: Refused before any work was done. 503 is excluded: it may come from an
#: intermediary after forwarding.
UNAMBIGUOUS_RETRY_STATUS = frozenset({429})
DEFAULT_MAX_ATTEMPTS = 5
PUBLIC_API_PATH = "/api/v1"
#: The platform's internal RFT root, served to hosted runs; see the module docstring.
INTERNAL_API_PATH = "/api/internal"


def split_api_root(base_url: str) -> Tuple[str, str]:
    """``(origin, api_path)`` for a base URL that is a platform origin or one of
    its API roots, with or without the ``/rft`` the RFT client historically got."""
    root = base_url.rstrip("/").removesuffix("/rft")
    for api_path in (INTERNAL_API_PATH, PUBLIC_API_PATH):
        if root.endswith(api_path):
            return root[: -len(api_path)], api_path
    return root, PUBLIC_API_PATH


def _user_agent() -> str:
    from . import __version__

    py = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    return f"prime-runs/{__version__} python/{py}"


def encode_json(value: Any) -> bytes:
    """Compact UTF-8 JSON; bare ``NaN`` is refused here rather than as a 400."""
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode(
        "utf-8"
    )


class PlatformClient:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        timeout: httpx.Timeout = DEFAULT_TIMEOUT,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        client: Optional[httpx.Client] = None,
    ) -> None:
        self.base_url, api_path = split_api_root(base_url)
        self.api_prefix = f"{self.base_url}{api_path}"
        #: Addressing the internal RFT router rather than the public API.
        self.internal = api_path == INTERNAL_API_PATH
        self.max_attempts = max(1, max_attempts)
        self._owns_client = client is None
        self._headers = {"Authorization": f"Bearer {api_key}", "User-Agent": _user_agent()}
        if self.internal:
            # The internal router reads a hosted run's own token from this header.
            self._headers["x-api-key"] = api_key
        self._timeout = timeout
        self._client = client or self._new_client()
        if self._owns_client:  # only a pool we opened is ours to rebuild after a fork
            _fork.register(self)

    def _new_client(self) -> httpx.Client:
        return httpx.Client(
            headers=dict(self._headers), follow_redirects=True, timeout=self._timeout
        )

    def reset_after_fork(self) -> None:
        """Rebuild the pool; the old one is dropped, not closed."""
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
        """Send one request, retrying transient failures; returns the JSON body.
        ``idempotent`` defaults to ``method != "POST"``."""
        url = f"{self.api_prefix}{path}"
        body = content if content is not None else (encode_json(json_body) if json_body else None)
        # Sent per request as well as set on the pool, so an injected client is
        # authenticated the same way as one we opened.
        headers = dict(self._headers)
        if body is not None:
            headers["Content-Type"] = "application/json"
        request_kwargs: Dict[str, Any] = {
            "content": body,
            "headers": headers,
            "params": dict(params) if params else None,
        }
        if timeout is not None:  # None would disable httpx timeouts, not restore the default
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

            if attempt == attempts - 1 or (ambiguous and not replayable):
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
