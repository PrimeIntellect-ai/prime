from typing import Any

from .core import AsyncAPIClient

_client = AsyncAPIClient()


async def make_prime_request(
    method: str,
    endpoint: str,
    params: dict[str, Any] | None = None,
    json_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Make a request to the PrimeIntellect API with proper error handling."""
    try:
        if method == "GET":
            return await _client.get(endpoint, params=params)
        elif method == "POST":
            return await _client.post(endpoint, json=json_data)
        elif method == "DELETE":
            return await _client.delete(endpoint, json=json_data)
        elif method == "PATCH":
            return await _client.patch(endpoint, json=json_data)
        else:
            return {"error": f"Unsupported HTTP method: {method}"}
    except Exception as e:
        return {"error": str(e)}
