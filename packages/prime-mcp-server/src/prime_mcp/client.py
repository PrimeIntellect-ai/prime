from typing import Any

from .core import AsyncAPIClient

_client = AsyncAPIClient()


async def make_prime_request(
    method: str,
    endpoint: str,
    params: dict[str, Any] | None = None,
    json_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Make a request to the PrimeIntellect API with proper error handling.

    Args:
        method: HTTP method (GET, POST, DELETE, PATCH)
        endpoint: API endpoint (e.g., "/pods", "availability/")
        params: Query parameters for GET requests
        json_data: JSON body for POST/PATCH requests

    Returns:
        API response as dictionary, or dict with "error" key on failure
    """
    try:
        if method == "GET":
            return await _client.get(endpoint, params=params)
        elif method == "POST":
            return await _client.post(endpoint, json=json_data)
        elif method == "DELETE":
            return await _client.delete(endpoint)
        elif method == "PATCH":
            return await _client.patch(endpoint, json=json_data)
        else:
            return {"error": f"Unsupported HTTP method: {method}"}
    except Exception as e:
        return {"error": str(e)}
