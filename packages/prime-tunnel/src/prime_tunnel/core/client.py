import sys
from typing import Any, Dict, Optional

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from prime_tunnel.core.config import Config
from prime_tunnel.exceptions import TunnelAuthError, TunnelError, TunnelTimeoutError
from prime_tunnel.models import TunnelInfo

# Retry configuration for transient connection errors
RETRYABLE_EXCEPTIONS = (
    httpx.RemoteProtocolError,
    httpx.ConnectError,
    httpx.PoolTimeout,
)


def _default_user_agent() -> str:
    """Build default User-Agent string."""
    from prime_tunnel import __version__

    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    return f"prime-tunnel/{__version__} python/{python_version}"


class TunnelClient:
    """Client for interacting with Prime Tunnel API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        user_agent: Optional[str] = None,
        timeout: float = 30.0,
    ):
        """
        Initialize the tunnel client.

        Args:
            api_key: Optional API key (defaults to config)
            user_agent: Optional custom User-Agent string
            timeout: Request timeout in seconds
        """
        self.config = Config()
        self.api_key = api_key or self.config.api_key
        self.base_url = self.config.base_url
        self._timeout = timeout

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        headers["User-Agent"] = user_agent if user_agent else _default_user_agent()

        self._client: Optional[httpx.AsyncClient] = None
        self._headers = headers

    def _check_auth_required(self) -> None:
        """Check if API key is configured."""
        if not self.api_key:
            raise TunnelError("No API key configured. Set PRIME_API_KEY environment variable.")

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=self._timeout,
                headers=self._headers,
                follow_redirects=True,
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    @retry(
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.1, min=0.1, max=2),
        reraise=True,
    )
    async def _request_with_retry(
        self,
        method: str,
        url: str,
        json: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, str]] = None,
    ) -> httpx.Response:
        """Make async HTTP request with retry on transient connection errors."""
        client = await self._get_client()
        return await client.request(method, url, json=json, params=params)

    async def _handle_response(self, response: httpx.Response, operation: str) -> Dict[str, Any]:
        """Handle response and raise appropriate errors."""
        if response.status_code == 401:
            raise TunnelAuthError("API key unauthorized. Check PRIME_API_KEY.")
        elif response.status_code == 402:
            raise TunnelAuthError("Payment required. Check billing status.")
        elif response.status_code == 404:
            return {}  # Handle 404 specially in callers
        elif response.status_code >= 400:
            try:
                error_detail = response.json().get("detail", response.text)
            except Exception:
                error_detail = response.text
            raise TunnelError(f"Failed to {operation}: {error_detail}")

        if response.status_code == 204:
            return {}

        return response.json()

    async def create_tunnel(
        self,
        local_port: int,
        name: Optional[str] = None,
        team_id: Optional[str] = None,
    ) -> TunnelInfo:
        """
        Register a new tunnel with the backend.

        Args:
            local_port: Local port the tunnel will forward to
            name: Optional friendly name for the tunnel
            team_id: Optional team ID for team tunnels

        Returns:
            TunnelInfo with connection details

        Raises:
            TunnelAuthError: If authentication fails
            TunnelError: If registration fails
        """
        self._check_auth_required()

        if team_id is None:
            team_id = self.config.team_id

        url = f"{self.base_url}/api/v1/tunnel"
        payload: Dict[str, Any] = {"local_port": local_port}
        if name:
            payload["name"] = name
        if team_id:
            payload["teamId"] = team_id

        try:
            response = await self._request_with_retry("POST", url, json=payload)
        except httpx.TimeoutException as e:
            raise TunnelTimeoutError(f"Request timed out: {e}") from e
        except httpx.RequestError as e:
            raise TunnelError(f"Failed to connect to API: {e}") from e

        data = await self._handle_response(response, "create tunnel")
        if not data:
            raise TunnelError("Failed to create tunnel: unexpected empty response")

        return TunnelInfo(**data)

    async def get_tunnel(self, tunnel_id: str) -> Optional[TunnelInfo]:
        """
        Get tunnel status by ID.

        Args:
            tunnel_id: The tunnel identifier

        Returns:
            TunnelInfo if found, None otherwise
        """
        self._check_auth_required()

        url = f"{self.base_url}/api/v1/tunnel/{tunnel_id}"

        try:
            response = await self._request_with_retry("GET", url)
        except httpx.TimeoutException as e:
            raise TunnelTimeoutError(f"Request timed out: {e}") from e
        except httpx.RequestError as e:
            raise TunnelError(f"Failed to connect to API: {e}") from e

        if response.status_code == 404:
            return None

        data = await self._handle_response(response, "get tunnel")
        return TunnelInfo(
            tunnel_id=data["tunnel_id"],
            hostname=data["hostname"],
            url=data["url"],
            frp_token="",  # Token not returned on status check
            proxy_name=data["proxy_name"],
            server_host="",
            server_port=7000,
            expires_at=data["expires_at"],
            user_id=data.get("user_id"),
        )

    async def delete_tunnel(self, tunnel_id: str) -> bool:
        """
        Delete a tunnel.

        Args:
            tunnel_id: The tunnel identifier

        Returns:
            True if deleted successfully
        """
        self._check_auth_required()

        url = f"{self.base_url}/api/v1/tunnel/{tunnel_id}"

        try:
            response = await self._request_with_retry("DELETE", url)
        except httpx.TimeoutException as e:
            raise TunnelTimeoutError(f"Request timed out: {e}") from e
        except httpx.RequestError as e:
            raise TunnelError(f"Failed to connect to API: {e}") from e

        if response.status_code == 404:
            return False

        await self._handle_response(response, "delete tunnel")
        return True

    async def bulk_delete_tunnels(self, tunnel_ids: list[str]) -> dict:
        """Bulk delete multiple tunnels."""
        self._check_auth_required()

        url = f"{self.base_url}/api/v1/tunnel"
        payload = {"tunnel_ids": tunnel_ids}

        try:
            response = await self._request_with_retry("DELETE", url, json=payload)
        except httpx.TimeoutException as e:
            raise TunnelTimeoutError(f"Request timed out: {e}") from e
        except httpx.RequestError as e:
            raise TunnelError(f"Failed to connect to API: {e}") from e

        return await self._handle_response(response, "bulk delete tunnels")

    async def list_tunnels(self, team_id: Optional[str] = None) -> list[TunnelInfo]:
        """
        List all tunnels for the current user.

        Args:
            team_id: Optional team ID to include team tunnels

        Returns:
            List of TunnelInfo objects
        """
        self._check_auth_required()

        if team_id is None:
            team_id = self.config.team_id

        url = f"{self.base_url}/api/v1/tunnel"
        params = {"teamId": team_id} if team_id else None

        try:
            response = await self._request_with_retry("GET", url, params=params)
        except httpx.TimeoutException as e:
            raise TunnelTimeoutError(f"Request timed out: {e}") from e
        except httpx.RequestError as e:
            raise TunnelError(f"Failed to connect to API: {e}") from e

        data = await self._handle_response(response, "list tunnels")
        tunnels = []
        for t in data.get("tunnels", []):
            tunnels.append(
                TunnelInfo(
                    tunnel_id=t["tunnel_id"],
                    hostname=t["hostname"],
                    url=t["url"],
                    frp_token="",
                    proxy_name=t["proxy_name"],
                    server_host="",
                    server_port=7000,
                    expires_at=t["expires_at"],
                    user_id=t.get("user_id"),
                )
            )
        return tunnels

    async def __aenter__(self) -> "TunnelClient":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()
