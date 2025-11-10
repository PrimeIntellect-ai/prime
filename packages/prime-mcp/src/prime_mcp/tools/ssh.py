from typing import Any, Optional

from prime_mcp.client import make_prime_request


async def manage_ssh_keys(
    action: str = "list",
    key_name: Optional[str] = None,
    public_key: Optional[str] = None,
    key_id: Optional[str] = None,
    offset: int = 0,
    limit: int = 100,
) -> dict[str, Any]:
    """Manage SSH keys for pod access.

    Args:
        action: Action to perform ("list", "add", "delete", "set_primary")
        key_name: Name for the SSH key (required for "add")
        public_key: SSH public key content (required for "add")
        key_id: Key ID (required for "delete" and "set_primary")
        offset: Number of items to skip (for "list" action, default: 0)
        limit: Maximum number of items to return (for "list" action, default: 100)

    Returns:
        SSH key operation result
    """
    if action == "list":
        params = {"offset": offset, "limit": limit}
        response_data = await make_prime_request("GET", "ssh_keys/", params=params)

    elif action == "add":
        if not key_name or not public_key:
            return {"error": "key_name and public_key are required for adding SSH key"}

        request_body = {"name": key_name, "publicKey": public_key}
        response_data = await make_prime_request("POST", "ssh_keys/", json_data=request_body)

    elif action == "delete":
        if not key_id:
            return {"error": "key_id is required for deleting SSH key"}

        response_data = await make_prime_request("DELETE", f"ssh_keys/{key_id}")
        if response_data is not None and not response_data.get("error"):
            return {"success": True, "message": f"SSH key {key_id} deleted successfully"}

    elif action == "set_primary":
        if not key_id:
            return {"error": "key_id is required for setting primary SSH key"}

        request_body = {"isPrimary": True}
        response_data = await make_prime_request(
            "PATCH", f"ssh_keys/{key_id}", json_data=request_body
        )

    else:
        return {"error": f"Invalid action: {action}. Use 'list', 'add', 'delete', or 'set_primary'"}

    if not response_data:
        return {"error": f"Unable to {action} SSH keys"}

    return response_data
