from typing import Any, Dict, Iterable, Optional

import httpx

from ..config import Config
from ..utils.debug import debug_log


class APIError(Exception):
    """Base API exception"""

    pass


class UnauthorizedError(APIError):
    """Raised when API returns 401 unauthorized"""

    pass


class PaymentRequiredError(APIError):
    """Raised when API returns 402 payment required"""

    pass


class TimeoutError(APIError):
    """Raised when API request times out"""

    pass


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

        self.client = httpx.Client(headers=headers, follow_redirects=True)

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
            response = self.client.request(method, url, params=params, json=json, timeout=timeout)
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
                    raise APIError(f"HTTP {e.response.status_code}: {error_response['detail']}")
            except (ValueError, KeyError):
                pass

            raise APIError(f"HTTP {e.response.status_code}: {e.response.text or str(e)}")
        except httpx.TimeoutException as e:
            raise TimeoutError(f"Request timed out: {e}")
        except httpx.RequestError as e:
            raise APIError(f"Request failed: {e}")

    def get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make a GET request to the API"""
        return self.request("GET", endpoint, params=params)

    def post(self, endpoint: str, json: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make a POST request to the API"""
        return self.request("POST", endpoint, json=json)

    def delete(self, endpoint: str) -> Dict[str, Any]:
        """Make a DELETE request to the API"""
        return self.request("DELETE", endpoint)

    def _build_url(self, endpoint: str) -> str:
        if not endpoint.startswith("/"):
            endpoint = f"/api/v1/{endpoint}"
        else:
            endpoint = f"/api/v1{endpoint}"
        return f"{self.base_url}{endpoint}"

    def stream_post(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Iterable[bytes]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        url = self._build_url(endpoint)
        req_headers = {"Accept": "application/json", "Content-Type": "application/octet-stream"}
        if headers:
            req_headers.update(headers)
        try:
            response = self.client.request(
                "POST",
                url,
                params=params,
                data=data,
                headers=req_headers,
                timeout=timeout,
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
            try:
                error_response = e.response.json()
                if isinstance(error_response, dict) and "detail" in error_response:
                    raise APIError(f"HTTP {e.response.status_code}: {error_response['detail']}")
            except (ValueError, KeyError):
                pass
            raise APIError(f"HTTP {e.response.status_code}: {e.response.text or str(e)}")
        except httpx.TimeoutException as e:
            raise TimeoutError(f"Request timed out: {e}")
        except httpx.RequestError as e:
            raise APIError(f"Request failed: {e}")

    def multipart_post(
        self,
        endpoint: str,
        files: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Send a multipart form POST request (for file uploads)"""
        url = self._build_url(endpoint)
        req_headers = {"Accept": "application/json"}

        # Debug logging
        import logging

        logger = logging.getLogger(__name__)
        logger.debug("🔍 Multipart POST Debug:")
        logger.debug(f"   URL: {url}")
        logger.debug(f"   Headers: {req_headers}")
        logger.debug(f"   Files: {files}")
        logger.debug(f"   Data: {data}")

        try:
            # For multipart requests, we need to let httpx set the Content-Type automatically
            # Create a temporary client without the default Content-Type header
            temp_headers = {"Accept": "application/json"}
            if self.api_key:
                temp_headers["Authorization"] = f"Bearer {self.api_key}"

            # Use a context manager for the temporary client
            with httpx.Client(headers=temp_headers, follow_redirects=True) as temp_client:
                response = temp_client.request(
                    "POST",
                    url,
                    files=files,
                    data=data,
                    timeout=timeout,
                )

                # Debug response
                logger.debug("📡 Response Debug:")
                logger.debug(f"   Status: {response.status_code}")
                logger.debug(f"   Headers: {dict(response.headers)}")
                logger.debug(f"   Content-Type: {response.headers.get('content-type', 'N/A')}")

                response.raise_for_status()
                result = response.json()
                if not isinstance(result, dict):
                    raise APIError("API response was not a dictionary")
                return result
        except httpx.HTTPStatusError as e:
            # Enhanced error logging
            logger.error(f"❌ HTTP Error {e.response.status_code}:")
            logger.error(f"   Response text: {e.response.text}")
            logger.error(f"   Response headers: {dict(e.response.headers)}")

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
            try:
                error_response = e.response.json()
                if isinstance(error_response, dict) and "detail" in error_response:
                    raise APIError(f"HTTP {e.response.status_code}: {error_response['detail']}")
            except (ValueError, KeyError):
                pass
            raise APIError(f"HTTP {e.response.status_code}: {e.response.text or str(e)}")
        except httpx.TimeoutException as e:
            raise TimeoutError(f"Request timed out: {e}")
        except httpx.RequestError as e:
            raise APIError(f"Request failed: {e}")

    def stream_get(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
    ) -> httpx.Response:
        url = self._build_url(endpoint)
        try:
            # Note: httpx uses stream() context manager for streaming responses
            # But we'll return the response directly for the caller to handle streaming
            response = self.client.request("GET", url, params=params, timeout=timeout)
            response.raise_for_status()
            return response
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
            try:
                error_response = e.response.json()
                if isinstance(error_response, dict) and "detail" in error_response:
                    raise APIError(f"HTTP {e.response.status_code}: {error_response['detail']}")
            except (ValueError, KeyError):
                pass
            raise APIError(f"HTTP {e.response.status_code}: {e.response.text or str(e)}")
        except httpx.TimeoutException as e:
            raise TimeoutError(f"Request timed out: {e}")
        except httpx.RequestError as e:
            raise APIError(f"Request failed: {e}")

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

        self.client = httpx.AsyncClient(headers=headers, follow_redirects=True)

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
                    "API key unauthorized. ",
                    "Please check that your API key has the correct permissions, "
                    "generate a new one at https://app.primeintellect.ai/dashboard/tokens, "
                    "or run 'prime login' to configure a new API key.",
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
                    raise APIError(f"HTTP {e.response.status_code}: {error_response['detail']}")
            except (ValueError, KeyError):
                pass

            raise APIError(f"HTTP {e.response.status_code}: {e.response.text or str(e)}")
        except httpx.TimeoutException as e:
            raise TimeoutError(f"Request timed out: {e}")
        except httpx.RequestError as e:
            raise APIError(f"Request failed: {e}")

    async def get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make an async GET request to the API"""
        return await self.request("GET", endpoint, params=params)

    async def post(self, endpoint: str, json: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make an async POST request to the API"""
        return await self.request("POST", endpoint, json=json)

    async def delete(self, endpoint: str) -> Dict[str, Any]:
        """Make an async DELETE request to the API"""
        return await self.request("DELETE", endpoint)

    async def multipart_post(
        self,
        endpoint: str,
        files: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Send an async multipart form POST request (for file uploads)"""
        url = self._build_url(endpoint)
        

        try:
            # For multipart requests, we need to let httpx set the Content-Type automatically
            temp_headers = {"Accept": "application/json"}
            if self.api_key:
                temp_headers["Authorization"] = f"Bearer {self.api_key}"

            # Create a temporary async client without the default Content-Type header
            async with httpx.AsyncClient(headers=temp_headers, follow_redirects=True) as temp_client:
                response = await temp_client.request(
                    "POST",
                    url,
                    files=files,
                    data=data,
                    timeout=timeout,
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
            try:
                error_response = e.response.json()
                if isinstance(error_response, dict) and "detail" in error_response:
                    raise APIError(f"HTTP {e.response.status_code}: {error_response['detail']}")
            except (ValueError, KeyError):
                pass
            raise APIError(f"HTTP {e.response.status_code}: {e.response.text or str(e)}")
        except httpx.TimeoutException as e:
            raise TimeoutError(f"Request timed out: {e}")
        except httpx.RequestError as e:
            raise APIError(f"Request failed: {e}")

    async def stream_get(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
    ) -> httpx.Response:
        """Make an async streaming GET request"""
        url = self._build_url(endpoint)
        try:
            response = await self.client.request("GET", url, params=params, timeout=timeout)
            response.raise_for_status()
            return response
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
            try:
                error_response = e.response.json()
                if isinstance(error_response, dict) and "detail" in error_response:
                    raise APIError(f"HTTP {e.response.status_code}: {error_response['detail']}")
            except (ValueError, KeyError):
                pass
            raise APIError(f"HTTP {e.response.status_code}: {e.response.text or str(e)}")
        except httpx.TimeoutException as e:
            raise TimeoutError(f"Request timed out: {e}")
        except httpx.RequestError as e:
            raise APIError(f"Request failed: {e}")

    def _build_url(self, endpoint: str) -> str:
        """Build the full URL for an endpoint"""
        if not endpoint.startswith("/"):
            endpoint = f"/api/v1/{endpoint}"
        else:
            endpoint = f"/api/v1{endpoint}"
        return f"{self.base_url}{endpoint}"

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
