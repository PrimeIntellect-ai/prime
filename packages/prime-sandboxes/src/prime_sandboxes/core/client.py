"""Lightweight HTTP client for SDK packages."""

import sys
from typing import Any, Dict, Optional

import httpx
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)
from tenacity.stop import stop_base
from tenacity.wait import wait_base

from .config import Config

# Requests that failed before the server could process them: safe to resend
# even for an operation that is not idempotent.
NOT_PROCESSED_EXCEPTIONS = (
    httpx.RemoteProtocolError,  # Server disconnected unexpectedly
    httpx.ConnectError,  # Connection refused/failed
    httpx.ConnectTimeout,  # Timed out before the request was sent
    httpx.PoolTimeout,  # No connection available in pool
)

# Statuses and codes that are definitive answers. Everything else a transport
# returns is a link or load fault and is retried (deny-list).
_RETRYABLE_4XX_STATUSES = frozenset({408, 425, 429})
_PERMANENT_5XX_STATUSES = frozenset({501, 505})
PERMANENT_CONNECT_CODES = frozenset(
    {
        Code.CANCELED,
        Code.INVALID_ARGUMENT,
        Code.NOT_FOUND,
        Code.ALREADY_EXISTS,
        Code.PERMISSION_DENIED,
        Code.FAILED_PRECONDITION,
        Code.OUT_OF_RANGE,
        Code.UNIMPLEMENTED,
        Code.UNAUTHENTICATED,
    }
)
# Local misconfiguration surfaced as transport errors; resending cannot fix it.
_PERMANENT_TRANSPORT_EXCEPTIONS = (httpx.UnsupportedProtocol, httpx.LocalProtocolError)


def _is_sandbox_not_found_response(response: httpx.Response) -> bool:
    """The gateway's 502 for a sandbox that no longer exists is an answer, not a fault."""
    if response.status_code != 502:
        return False
    try:
        body = response.json()
    except Exception:
        return False
    return isinstance(body, dict) and body.get("error") == "sandbox_not_found"


def is_retryable(exc: BaseException, *, idempotent: bool) -> bool:
    """The one retry classifier for SDK calls.

    A transport failure is retried unless it is a definitive answer. A call that
    is not idempotent retries only failures that prove the request was never
    processed (connection-phase errors and 429). Plain APIError wrappers are
    unwrapped to their transport cause; any other exception (SDK outcomes such
    as a command timeout or a terminated sandbox, or a local defect) is final.
    """
    while type(exc) is APIError and exc.__cause__ is not None:
        exc = exc.__cause__
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        if status == 429:
            return True
        if not idempotent:
            return False
        if status >= 500:
            return status not in _PERMANENT_5XX_STATUSES and not _is_sandbox_not_found_response(
                exc.response
            )
        return status in _RETRYABLE_4XX_STATUSES
    if isinstance(exc, ConnectError):
        return idempotent and exc.code not in PERMANENT_CONNECT_CODES
    if isinstance(exc, _PERMANENT_TRANSPORT_EXCEPTIONS):
        return False
    if isinstance(exc, httpx.TransportError):
        return idempotent or isinstance(exc, NOT_PROCESSED_EXCEPTIONS)
    return False


def retry_transient(
    *,
    idempotent: bool,
    stop: stop_base,
    wait: wait_base,
    rate_limit_delays: tuple[float, ...],
) -> Any:
    """Tenacity decorator over is_retryable; call sites choose only the budget."""
    return retry(
        retry=retry_if_exception(lambda exc: is_retryable(exc, idempotent=idempotent)),
        stop=stop,
        wait=_RateLimitAwareWait(wait, rate_limit_delays),
        reraise=True,
    )


IDEMPOTENT_HTTP_METHODS = frozenset({"GET", "HEAD", "PUT", "DELETE", "OPTIONS"})
_PLATFORM_RATE_LIMIT_DELAYS = (10.0, 30.0)


class _RateLimitAwareWait(wait_base):
    def __init__(self, default_wait: wait_base, rate_limit_delays: tuple[float, ...]):
        self._default_wait = default_wait
        self._rate_limit_delays = rate_limit_delays

    def __call__(self, retry_state: RetryCallState) -> float:
        exception = retry_state.outcome.exception() if retry_state.outcome else None
        if isinstance(exception, httpx.HTTPStatusError) and exception.response.status_code == 429:
            return self._rate_limit_delays[
                min(retry_state.attempt_number - 1, len(self._rate_limit_delays) - 1)
            ]
        return self._default_wait(retry_state)


# Three attempts permit only two sleeps, so 429s wait 10s then 30s (~40s total).
# Connection and 5xx errors keep the existing tight random-exponential budget.
def _platform_retry(idempotent: bool) -> Any:
    return retry_transient(
        idempotent=idempotent,
        stop=stop_after_attempt(3),
        wait=wait_random_exponential(multiplier=0.1, max=2),
        rate_limit_delays=_PLATFORM_RATE_LIMIT_DELAYS,
    )


def _default_user_agent() -> str:
    """Build default User-Agent string"""
    from prime_sandboxes import __version__

    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    return f"prime-sandboxes/{__version__} python/{python_version}"


class APIError(Exception):
    """Base API exception"""

    pass


class UnauthorizedError(APIError):
    """Raised when API returns 401 unauthorized"""

    pass


class PaymentRequiredError(APIError):
    """Raised when API returns 402 payment required"""

    pass


class APITimeoutError(APIError):
    """Raised when API request times out"""

    pass


class APIClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        require_auth: bool = True,
        user_agent: Optional[str] = None,
    ):
        self.config = Config()
        self.api_key = api_key or self.config.api_key
        self.require_auth = require_auth
        self.base_url = self.config.base_url

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        headers["User-Agent"] = user_agent if user_agent else _default_user_agent()

        self.client = httpx.Client(
            headers=headers,
            follow_redirects=True,
            timeout=httpx.Timeout(30.0, connect=10.0),
        )

    def _check_auth_required(self) -> None:
        if self.require_auth and not self.api_key:
            raise APIError(
                "No API key configured. Set PRIME_API_KEY environment variable.",
            )

    @_platform_retry(idempotent=True)
    def _idempotent_request_with_retry(
        self,
        method: str,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> httpx.Response:
        """Make idempotent HTTP request with retry on transient failures."""
        response = self.client.request(method, url, params=params, json=json, timeout=timeout)
        response.raise_for_status()
        return response

    @_platform_retry(idempotent=False)
    def _non_idempotent_request_with_retry(
        self,
        method: str,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> httpx.Response:
        """Make non-idempotent request with only pre-processing safe retries."""
        response = self.client.request(method, url, params=params, json=json, timeout=timeout)
        if response.status_code == 429:
            response.raise_for_status()
        return response

    @_platform_retry(idempotent=True)
    def _idempotent_post_request_with_retry(
        self,
        method: str,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> httpx.Response:
        """Make idempotent POST with retries for ambiguous transient failures."""
        response = self.client.request(method, url, params=params, json=json, timeout=timeout)
        response.raise_for_status()
        return response

    def request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        idempotent_post: bool = False,
    ) -> Dict[str, Any]:
        """Make a request to the API"""
        self._check_auth_required()

        if not endpoint.startswith("/"):
            endpoint = f"/api/v1/{endpoint}"
        else:
            endpoint = f"/api/v1{endpoint}"

        url = f"{self.base_url}{endpoint}"

        try:
            method_upper = method.upper()
            is_idempotent_post = method_upper == "POST" and idempotent_post
            if method_upper == "POST":
                request_fn = (
                    self._idempotent_post_request_with_retry
                    if is_idempotent_post
                    else self._non_idempotent_request_with_retry
                )
            elif method_upper in IDEMPOTENT_HTTP_METHODS:
                request_fn = self._idempotent_request_with_retry
            else:
                request_fn = self._non_idempotent_request_with_retry
            response = request_fn(method, url, params=params, json=json, timeout=timeout)
            if not is_idempotent_post:
                response.raise_for_status()
            if response.status_code == 204:
                return {}

            result = response.json()
            if not isinstance(result, dict):
                raise APIError("API response was not a dictionary")
            return result
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                raise UnauthorizedError("API key unauthorized. Check PRIME_API_KEY.") from e
            if e.response.status_code == 402:
                raise PaymentRequiredError("Payment required. Check billing status.") from e

            try:
                error_response = e.response.json()
                if isinstance(error_response, dict) and "detail" in error_response:
                    raise APIError(f"HTTP {e.response.status_code}: {error_response['detail']}")
            except (ValueError, KeyError):
                pass

            raise APIError(f"HTTP {e.response.status_code}: {e.response.text or str(e)}") from e
        except httpx.TimeoutException as e:
            raise APITimeoutError(f"Request timed out: {e}") from e
        except httpx.RequestError as e:
            req = getattr(e, "request", None)
            method = getattr(req, "method", "?")
            u = getattr(req, "url", "?")
            raise APIError(f"Request failed: {e.__class__.__name__} at {method} {u}: {e}") from e

    def get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self.request("GET", endpoint, params=params)

    def post(self, endpoint: str, json: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self.request("POST", endpoint, json=json)

    def patch(self, endpoint: str, json: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self.request("PATCH", endpoint, json=json)

    def delete(self, endpoint: str) -> Dict[str, Any]:
        return self.request("DELETE", endpoint)


class AsyncAPIClient:
    """Async version of APIClient"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        require_auth: bool = True,
        user_agent: Optional[str] = None,
    ):
        self.config = Config()
        self.api_key = api_key or self.config.api_key
        self.require_auth = require_auth
        self.base_url = self.config.base_url

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        headers["User-Agent"] = user_agent if user_agent else _default_user_agent()

        self.client = httpx.AsyncClient(
            headers=headers,
            follow_redirects=True,
            timeout=httpx.Timeout(30.0, connect=10.0),
        )

    def _check_auth_required(self) -> None:
        if self.require_auth and not self.api_key:
            raise APIError(
                "No API key configured. Set PRIME_API_KEY environment variable.",
            )

    @_platform_retry(idempotent=True)
    async def _idempotent_request_with_retry(
        self,
        method: str,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> httpx.Response:
        """Make async idempotent HTTP request with retry on transient failures."""
        response = await self.client.request(method, url, params=params, json=json, timeout=timeout)
        response.raise_for_status()
        return response

    @_platform_retry(idempotent=False)
    async def _non_idempotent_request_with_retry(
        self,
        method: str,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> httpx.Response:
        """Make async non-idempotent request with only pre-processing safe retries."""
        response = await self.client.request(method, url, params=params, json=json, timeout=timeout)
        if response.status_code == 429:
            response.raise_for_status()
        return response

    @_platform_retry(idempotent=True)
    async def _idempotent_post_request_with_retry(
        self,
        method: str,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> httpx.Response:
        """Make async idempotent POST with retries for ambiguous transient failures."""
        response = await self.client.request(method, url, params=params, json=json, timeout=timeout)
        response.raise_for_status()
        return response

    async def request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        idempotent_post: bool = False,
    ) -> Dict[str, Any]:
        """Make an async request to the API"""
        self._check_auth_required()

        if not endpoint.startswith("/"):
            endpoint = f"/api/v1/{endpoint}"
        else:
            endpoint = f"/api/v1{endpoint}"

        url = f"{self.base_url}{endpoint}"

        try:
            method_upper = method.upper()
            is_idempotent_post = method_upper == "POST" and idempotent_post
            if method_upper == "POST":
                request_fn = (
                    self._idempotent_post_request_with_retry
                    if is_idempotent_post
                    else self._non_idempotent_request_with_retry
                )
            elif method_upper in IDEMPOTENT_HTTP_METHODS:
                request_fn = self._idempotent_request_with_retry
            else:
                request_fn = self._non_idempotent_request_with_retry
            response = await request_fn(method, url, params=params, json=json, timeout=timeout)
            if not is_idempotent_post:
                response.raise_for_status()
            if response.status_code == 204:
                return {}

            result = response.json()
            if not isinstance(result, dict):
                raise APIError("API response was not a dictionary")
            return result
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                raise UnauthorizedError("API key unauthorized. Check PRIME_API_KEY.") from e
            if e.response.status_code == 402:
                raise PaymentRequiredError("Payment required. Check billing status.") from e

            try:
                error_response = e.response.json()
                if isinstance(error_response, dict) and "detail" in error_response:
                    raise APIError(f"HTTP {e.response.status_code}: {error_response['detail']}")
            except (ValueError, KeyError):
                pass

            raise APIError(f"HTTP {e.response.status_code}: {e.response.text or str(e)}") from e
        except httpx.TimeoutException as e:
            raise APITimeoutError(f"Request timed out: {e}") from e
        except httpx.RequestError as e:
            req = getattr(e, "request", None)
            method = getattr(req, "method", "?")
            u = getattr(req, "url", "?")
            raise APIError(f"Request failed: {e.__class__.__name__} at {method} {u}: {e}") from e

    async def get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return await self.request("GET", endpoint, params=params)

    async def post(self, endpoint: str, json: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return await self.request("POST", endpoint, json=json)

    async def patch(self, endpoint: str, json: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return await self.request("PATCH", endpoint, json=json)

    async def delete(self, endpoint: str) -> Dict[str, Any]:
        return await self.request("DELETE", endpoint)

    async def aclose(self) -> None:
        await self.client.aclose()

    async def __aenter__(self) -> "AsyncAPIClient":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.aclose()
