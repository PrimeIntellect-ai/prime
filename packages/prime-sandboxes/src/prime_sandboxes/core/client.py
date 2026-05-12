"""Lightweight HTTP client for SDK packages."""

import sys
from typing import Any, Dict, Optional

import httpx
from tenacity import (
    retry,
    retry_if_exception,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from .config import Config

# Retry configuration for transient connection errors. POST only retries errors
# that happen before the server can process the request; retrying ReadError on a
# non-idempotent POST can duplicate side effects after a committed response-path
# failure.
POST_RETRYABLE_EXCEPTIONS = (
    httpx.RemoteProtocolError,  # Server disconnected unexpectedly
    httpx.ConnectError,  # Connection refused/failed
    httpx.PoolTimeout,  # No connection available in pool
)

IDEMPOTENT_RETRYABLE_EXCEPTIONS = POST_RETRYABLE_EXCEPTIONS + (
    httpx.ReadError,  # Connection broken while reading response (e.g., TCP reset)
)

IDEMPOTENT_RETRYABLE_STATUSES = frozenset({502, 503, 504})
IDEMPOTENT_HTTP_METHODS = frozenset({"GET", "HEAD", "PUT", "DELETE", "OPTIONS"})


def _is_idempotent_request_retryable_error(exc: BaseException) -> bool:
    if isinstance(exc, IDEMPOTENT_RETRYABLE_EXCEPTIONS):
        return True
    return (
        isinstance(exc, httpx.HTTPStatusError)
        and exc.response.status_code in IDEMPOTENT_RETRYABLE_STATUSES
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

    @retry(
        retry=retry_if_exception(_is_idempotent_request_retryable_error),
        stop=stop_after_attempt(3),
        wait=wait_random_exponential(multiplier=0.1, max=2),
        reraise=True,
    )
    def _idempotent_request_with_retry(
        self,
        method: str,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
    ) -> httpx.Response:
        """Make idempotent HTTP request with retry on transient failures."""
        response = self.client.request(method, url, params=params, json=json, timeout=timeout)
        response.raise_for_status()
        return response

    @retry(
        retry=retry_if_exception_type(POST_RETRYABLE_EXCEPTIONS),
        stop=stop_after_attempt(3),
        wait=wait_random_exponential(multiplier=0.1, max=2),
        reraise=True,
    )
    def _non_idempotent_request_with_retry(
        self,
        method: str,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
    ) -> httpx.Response:
        """Make non-idempotent request with only pre-processing safe retries."""
        return self.client.request(method, url, params=params, json=json, timeout=timeout)

    @retry(
        retry=retry_if_exception(_is_idempotent_request_retryable_error),
        stop=stop_after_attempt(3),
        wait=wait_random_exponential(multiplier=0.1, max=2),
        reraise=True,
    )
    def _idempotent_post_request_with_retry(
        self,
        method: str,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
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
        timeout: Optional[int] = None,
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

    @retry(
        retry=retry_if_exception(_is_idempotent_request_retryable_error),
        stop=stop_after_attempt(3),
        wait=wait_random_exponential(multiplier=0.1, max=2),
        reraise=True,
    )
    async def _idempotent_request_with_retry(
        self,
        method: str,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
    ) -> httpx.Response:
        """Make async idempotent HTTP request with retry on transient failures."""
        response = await self.client.request(method, url, params=params, json=json, timeout=timeout)
        response.raise_for_status()
        return response

    @retry(
        retry=retry_if_exception_type(POST_RETRYABLE_EXCEPTIONS),
        stop=stop_after_attempt(3),
        wait=wait_random_exponential(multiplier=0.1, max=2),
        reraise=True,
    )
    async def _non_idempotent_request_with_retry(
        self,
        method: str,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
    ) -> httpx.Response:
        """Make async non-idempotent request with only pre-processing safe retries."""
        return await self.client.request(method, url, params=params, json=json, timeout=timeout)

    @retry(
        retry=retry_if_exception(_is_idempotent_request_retryable_error),
        stop=stop_after_attempt(3),
        wait=wait_random_exponential(multiplier=0.1, max=2),
        reraise=True,
    )
    async def _idempotent_post_request_with_retry(
        self,
        method: str,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
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
        timeout: Optional[int] = None,
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
