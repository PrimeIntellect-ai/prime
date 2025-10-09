from typing import Any, Dict, Optional

import httpx

from .config import Config


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


# Deprecated: Use APITimeoutError instead
TimeoutError = APITimeoutError


class APIClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        require_auth: bool = True,
    ):
        # Load config
        self.config = Config()

        # Use provided API key or fall back to config
        self.api_key = api_key or self.config.api_key
        if require_auth and not self.api_key:
            raise APIError(
                "No API key configured. Use command 'prime login' to configure your API key.",
            )

        # Setup client
        self.base_url = self.config.base_url
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        self.client = httpx.Client(
            headers=headers,
            follow_redirects=True,
            timeout=httpx.Timeout(30.0, connect=10.0),
        )

    def request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Make a request to the API"""
        # Ensure endpoint starts with /api/v1/
        if not endpoint.startswith("/"):
            endpoint = f"/api/v1/{endpoint}"
        else:
            endpoint = f"/api/v1{endpoint}"

        url = f"{self.base_url}{endpoint}"

        try:
            response = self.client.request(
                method, url, params=params, json=json, timeout=timeout
            )
            response.raise_for_status()

            result = response.json()
            if not isinstance(result, dict):
                raise APIError("API response was not a dictionary")
            return result
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                raise UnauthorizedError(
                    "API key unauthorized. "
                    "Please check that your API key has the correct permissions, "
                    "generate a new one at https://app.primeintellect.ai/dashboard/tokens, "
                    "or run 'prime login' to configure a new API key."
                )
            if e.response.status_code == 402:
                raise PaymentRequiredError(
                    "Payment required. Please check your billing status at "
                    "https://app.primeintellect.ai/dashboard/billing"
                )

            # For other HTTP errors, try to extract the error message from the response
            try:
                error_response = e.response.json()
                if isinstance(error_response, dict) and "detail" in error_response:
                    raise APIError(
                        f"HTTP {e.response.status_code}: {error_response['detail']}"
                    )
            except (ValueError, KeyError):
                pass

            raise APIError(
                f"HTTP {e.response.status_code}: {e.response.text or str(e)}"
            )
        except httpx.TimeoutException as e:
            raise APITimeoutError(f"Request timed out: {e}")
        except httpx.RequestError as e:
            raise APIError(f"Request failed: {e}")

    def get(
        self, endpoint: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Make a GET request to the API"""
        return self.request("GET", endpoint, params=params)

    def post(
        self, endpoint: str, json: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Make a POST request to the API"""
        return self.request("POST", endpoint, json=json)

    def patch(
        self, endpoint: str, json: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Make a PATCH request to the API"""
        return self.request("PATCH", endpoint, json=json)

    def delete(self, endpoint: str) -> Dict[str, Any]:
        """Make a DELETE request to the API"""
        return self.request("DELETE", endpoint)

    def __str__(self) -> str:
        """For debugging"""
        return f"APIClient(base_url={self.base_url})"


class AsyncAPIClient:
    """Async version of APIClient using httpx.AsyncClient"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        require_auth: bool = True,
    ):
        # Load config
        self.config = Config()

        # Use provided API key or fall back to config
        self.api_key = api_key or self.config.api_key
        if require_auth and not self.api_key:
            raise APIError(
                "No API key configured. Use command 'prime login' to configure your API key.",
            )

        # Setup client
        self.base_url = self.config.base_url
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        self.client = httpx.AsyncClient(
            headers=headers,
            follow_redirects=True,
            timeout=httpx.Timeout(30.0, connect=10.0),
        )

    async def request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Make an async request to the API"""
        # Ensure endpoint starts with /api/v1/
        if not endpoint.startswith("/"):
            endpoint = f"/api/v1/{endpoint}"
        else:
            endpoint = f"/api/v1{endpoint}"

        url = f"{self.base_url}{endpoint}"

        try:
            response = await self.client.request(
                method, url, params=params, json=json, timeout=timeout
            )
            response.raise_for_status()

            result = response.json()
            if not isinstance(result, dict):
                raise APIError("API response was not a dictionary")
            return result
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                raise UnauthorizedError(
                    "API key unauthorized. "
                    "Please check that your API key has the correct permissions, "
                    "generate a new one at https://app.primeintellect.ai/dashboard/tokens, "
                    "or run 'prime login' to configure a new API key."
                )
            if e.response.status_code == 402:
                raise PaymentRequiredError(
                    "Payment required. Please check your billing status at "
                    "https://app.primeintellect.ai/dashboard/billing"
                )

            # For other HTTP errors, try to extract the error message from the response
            try:
                error_response = e.response.json()
                if isinstance(error_response, dict) and "detail" in error_response:
                    raise APIError(
                        f"HTTP {e.response.status_code}: {error_response['detail']}"
                    )
            except (ValueError, KeyError):
                pass

            raise APIError(
                f"HTTP {e.response.status_code}: {e.response.text or str(e)}"
            )
        except httpx.TimeoutException as e:
            raise APITimeoutError(f"Request timed out: {e}")
        except httpx.RequestError as e:
            raise APIError(f"Request failed: {e}")

    async def get(
        self, endpoint: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Make an async GET request to the API"""
        return await self.request("GET", endpoint, params=params)

    async def post(
        self, endpoint: str, json: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Make an async POST request to the API"""
        return await self.request("POST", endpoint, json=json)

    async def patch(
        self, endpoint: str, json: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Make an async PATCH request to the API"""
        return await self.request("PATCH", endpoint, json=json)

    async def delete(self, endpoint: str) -> Dict[str, Any]:
        """Make an async DELETE request to the API"""
        return await self.request("DELETE", endpoint)

    async def aclose(self) -> None:
        """Close the async client"""
        await self.client.aclose()

    async def __aenter__(self) -> "AsyncAPIClient":
        """Async context manager entry"""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit"""
        await self.aclose()

    def __str__(self) -> str:
        """For debugging"""
        return f"AsyncAPIClient(base_url={self.base_url})"
