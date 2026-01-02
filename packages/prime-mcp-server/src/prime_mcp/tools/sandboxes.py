from typing import Any, Optional

import httpx

from prime_mcp.client import make_prime_request


async def _get_sandbox_auth(sandbox_id: str) -> dict[str, Any]:
    """Get gateway auth credentials (gateway_url, token, user_ns, job_id)."""
    response = await make_prime_request("POST", f"sandbox/{sandbox_id}/auth")
    if not response or "error" in response:
        raise RuntimeError(f"Failed to get sandbox auth: {response}")
    return response


async def _gateway_request(
    method: str,
    gateway_url: str,
    user_ns: str,
    job_id: str,
    endpoint: str,
    token: str,
    json_data: Optional[dict[str, Any]] = None,
    timeout: int = 300,
) -> dict[str, Any]:
    """Make an authenticated request to the sandbox gateway."""
    url = f"{gateway_url.rstrip('/')}/{user_ns}/{job_id}/{endpoint}"
    headers = {"Authorization": f"Bearer {token}"}

    async with httpx.AsyncClient(timeout=timeout + 5) as client:
        if method == "POST":
            response = await client.post(url, json=json_data, headers=headers)
        elif method == "GET":
            response = await client.get(url, headers=headers)
        else:
            raise ValueError(f"Unsupported method: {method}")

        if response.status_code == 404:
            raise RuntimeError("Sandbox not found or not running")
        elif response.status_code == 408:
            raise RuntimeError("Command timed out")
        elif response.status_code == 503:
            raise RuntimeError("Sandbox service unavailable")

        response.raise_for_status()
        return response.json()


async def create_sandbox(
    name: str,
    docker_image: str = "python:3.11-slim",
    start_command: Optional[str] = "tail -f /dev/null",
    cpu_cores: int = 1,
    memory_gb: int = 2,
    disk_size_gb: int = 5,
    network_access: bool = True,
    timeout_minutes: int = 60,
    environment_vars: Optional[dict[str, str]] = None,
    secrets: Optional[dict[str, str]] = None,
    labels: Optional[list[str]] = None,
    team_id: Optional[str] = None,
    registry_credentials_id: Optional[str] = None,
) -> dict[str, Any]:
    """Create a new sandbox for isolated code execution."""
    if cpu_cores < 1 or cpu_cores > 16:
        return {"error": "cpu_cores must be between 1 and 16"}
    if memory_gb < 1 or memory_gb > 64:
        return {"error": "memory_gb must be between 1 and 64"}
    if disk_size_gb < 1 or disk_size_gb > 1000:
        return {"error": "disk_size_gb must be between 1 and 1000"}
    if timeout_minutes < 1 or timeout_minutes > 1440:
        return {"error": "timeout_minutes must be between 1 and 1440 (24 hours)"}

    request_body: dict[str, Any] = {
        "name": name,
        "docker_image": docker_image,
        "cpu_cores": cpu_cores,
        "memory_gb": memory_gb,
        "disk_size_gb": disk_size_gb,
        "gpu_count": 0,  # GPU support not yet available
        "network_access": network_access,
        "timeout_minutes": timeout_minutes,
    }

    if start_command:
        request_body["start_command"] = start_command
    if environment_vars:
        request_body["environment_vars"] = environment_vars
    if secrets:
        request_body["secrets"] = secrets
    if labels:
        request_body["labels"] = labels
    if team_id:
        request_body["team_id"] = team_id
    if registry_credentials_id:
        request_body["registry_credentials_id"] = registry_credentials_id

    response = await make_prime_request("POST", "sandbox", json_data=request_body)

    if not response:
        return {"error": "Unable to create sandbox"}

    return response


async def list_sandboxes(
    team_id: Optional[str] = None,
    status: Optional[str] = None,
    labels: Optional[list[str]] = None,
    page: int = 1,
    per_page: int = 50,
    exclude_terminated: bool = False,
) -> dict[str, Any]:
    """List all sandboxes in your account."""
    params: dict[str, Any] = {"page": max(1, page), "per_page": min(100, max(1, per_page))}

    if team_id:
        params["team_id"] = team_id
    if status:
        params["status"] = status
    if labels:
        params["labels"] = labels
    if exclude_terminated:
        params["is_active"] = True

    response = await make_prime_request("GET", "sandbox", params=params)

    if not response:
        return {"error": "Unable to list sandboxes"}

    return response


async def get_sandbox(sandbox_id: str) -> dict[str, Any]:
    """Get detailed information about a specific sandbox."""
    if not sandbox_id:
        return {"error": "sandbox_id is required"}

    response = await make_prime_request("GET", f"sandbox/{sandbox_id}")

    if not response:
        return {"error": f"Unable to get sandbox: {sandbox_id}"}

    return response


async def delete_sandbox(sandbox_id: str) -> dict[str, Any]:
    """Delete/terminate a sandbox."""
    if not sandbox_id:
        return {"error": "sandbox_id is required"}

    response = await make_prime_request("DELETE", f"sandbox/{sandbox_id}")

    if not response:
        return {"error": f"Unable to delete sandbox: {sandbox_id}"}

    return response


async def bulk_delete_sandboxes(
    sandbox_ids: Optional[list[str]] = None,
    labels: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Bulk delete multiple sandboxes by IDs or labels."""
    if not sandbox_ids and not labels:
        return {"error": "Must specify either sandbox_ids or labels"}
    if sandbox_ids and labels:
        return {"error": "Cannot specify both sandbox_ids and labels"}

    request_body: dict[str, Any] = {}
    if sandbox_ids:
        request_body["sandbox_ids"] = sandbox_ids
    if labels:
        request_body["labels"] = labels

    response = await make_prime_request("DELETE", "sandbox", json_data=request_body)

    if not response:
        return {"error": "Unable to bulk delete sandboxes"}

    return response


async def get_sandbox_logs(sandbox_id: str) -> dict[str, Any]:
    """Get logs from a sandbox."""
    if not sandbox_id:
        return {"error": "sandbox_id is required"}

    response = await make_prime_request("GET", f"sandbox/{sandbox_id}/logs")

    if not response:
        return {"error": f"Unable to get logs for sandbox: {sandbox_id}"}

    return response


async def execute_command(
    sandbox_id: str,
    command: str,
    working_dir: Optional[str] = None,
    env: Optional[dict[str, str]] = None,
    timeout: int = 300,
) -> dict[str, Any]:
    """Execute a command in a sandbox via the gateway."""
    if not sandbox_id:
        return {"error": "sandbox_id is required"}
    if not command:
        return {"error": "command is required"}
    if timeout < 1:
        return {"error": "timeout must be at least 1 second"}

    try:
        auth = await _get_sandbox_auth(sandbox_id)

        gateway_url = auth.get("gateway_url")
        token = auth.get("token")
        user_ns = auth.get("user_ns")
        job_id = auth.get("job_id")

        if not all([gateway_url, token, user_ns, job_id]):
            return {"error": "Invalid auth response from sandbox"}

        request_body: dict[str, Any] = {
            "command": command,
            "timeout": timeout,
            "sandbox_id": sandbox_id,
            "env": env or {},
        }

        if working_dir:
            request_body["working_dir"] = working_dir

        response = await _gateway_request(
            method="POST",
            gateway_url=gateway_url,
            user_ns=user_ns,
            job_id=job_id,
            endpoint="exec",
            token=token,
            json_data=request_body,
            timeout=timeout,
        )

        return response

    except httpx.TimeoutException:
        return {"error": f"Command timed out after {timeout} seconds"}
    except httpx.HTTPStatusError as e:
        return {"error": f"HTTP {e.response.status_code}: {e.response.text}"}
    except RuntimeError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Failed to execute command: {str(e)}"}


async def expose_port(
    sandbox_id: str,
    port: int,
    name: Optional[str] = None,
    protocol: str = "HTTP",
) -> dict[str, Any]:
    """Expose a port from a sandbox to the internet."""

    if not sandbox_id:
        return {"error": "sandbox_id is required"}
    if not port or port < 22 or port > 9000:
        return {"error": "port must be between 22 and 9000"}
    if port == 8080:
        return {"error": "port 8080 is reserved and cannot be exposed"}
    if protocol.upper() not in ("HTTP", "TCP", "UDP"):
        return {"error": "protocol must be HTTP, TCP, or UDP"}

    request_body: dict[str, Any] = {"port": port, "protocol": protocol.upper()}
    if name:
        request_body["name"] = name

    response = await make_prime_request(
        "POST", f"sandbox/{sandbox_id}/expose", json_data=request_body
    )

    if not response:
        return {"error": f"Unable to expose port {port} in sandbox: {sandbox_id}"}

    return response


async def unexpose_port(sandbox_id: str, exposure_id: str) -> dict[str, Any]:
    """Remove a port exposure from a sandbox."""
    if not sandbox_id:
        return {"error": "sandbox_id is required"}
    if not exposure_id:
        return {"error": "exposure_id is required"}

    response = await make_prime_request("DELETE", f"sandbox/{sandbox_id}/expose/{exposure_id}")

    if response is None:
        return {"error": f"Unable to unexpose port in sandbox: {sandbox_id}"}

    return response if response else {"success": True}


async def list_exposed_ports(sandbox_id: str) -> dict[str, Any]:
    """List all exposed ports for a sandbox."""
    if not sandbox_id:
        return {"error": "sandbox_id is required"}

    response = await make_prime_request("GET", f"sandbox/{sandbox_id}/expose")

    if not response:
        return {"error": f"Unable to list exposed ports for sandbox: {sandbox_id}"}

    return response


async def list_registry_credentials() -> dict[str, Any]:
    """List available registry credentials for private Docker images."""
    response = await make_prime_request("GET", "template/registry-credentials")

    if not response:
        return {"error": "Unable to list registry credentials"}

    return response


async def check_docker_image(
    image: str,
    registry_credentials_id: Optional[str] = None,
) -> dict[str, Any]:
    """Check if a Docker image is accessible."""
    if not image:
        return {"error": "image is required"}

    request_body: dict[str, Any] = {"image": image}
    if registry_credentials_id:
        request_body["registry_credentials_id"] = registry_credentials_id

    response = await make_prime_request(
        "POST", "template/check-docker-image", json_data=request_body
    )

    if not response:
        return {"error": f"Unable to check image: {image}"}

    return response
