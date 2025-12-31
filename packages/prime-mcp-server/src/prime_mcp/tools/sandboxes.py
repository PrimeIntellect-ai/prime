from typing import Any, Optional

from prime_mcp.client import make_prime_request


async def create_sandbox(
    name: str,
    docker_image: str = "python:3.11-slim",
    start_command: Optional[str] = "tail -f /dev/null",
    cpu_cores: int = 1,
    memory_gb: int = 2,
    disk_size_gb: int = 5,
    gpu_count: int = 0,
    network_access: bool = True,
    timeout_minutes: int = 60,
    environment_vars: Optional[dict[str, str]] = None,
    labels: Optional[list[str]] = None,
    team_id: Optional[str] = None,
    registry_credentials_id: Optional[str] = None,
) -> dict[str, Any]:
    """Create a new sandbox for isolated code execution.

    A sandbox is a containerized environment where you can safely execute code,
    run commands, and manage files in isolation.

    Args:
        name: Name for the sandbox (required)
        docker_image: Docker image to use (default: "python:3.11-slim")
        start_command: Command to run on startup (default: "tail -f /dev/null")
        cpu_cores: Number of CPU cores (default: 1, min: 1)
        memory_gb: Memory in GB (default: 2, min: 1)
        disk_size_gb: Disk size in GB (default: 5, min: 1)
        gpu_count: Number of GPUs (default: 0)
        network_access: Enable network access (default: True)
        timeout_minutes: Timeout before auto-termination (default: 60)
        environment_vars: Environment variables as key-value pairs
        labels: Labels for organizing and filtering sandboxes
        team_id: Team ID for organization accounts
        registry_credentials_id: ID of registry credentials for private images

    Returns:
        Created sandbox details including ID, status, and configuration
    """
    # Validate parameters
    if cpu_cores < 1:
        return {"error": "cpu_cores must be at least 1"}
    if memory_gb < 1:
        return {"error": "memory_gb must be at least 1"}
    if disk_size_gb < 1:
        return {"error": "disk_size_gb must be at least 1"}
    if gpu_count < 0:
        return {"error": "gpu_count cannot be negative"}
    if timeout_minutes < 1:
        return {"error": "timeout_minutes must be at least 1"}

    request_body: dict[str, Any] = {
        "name": name,
        "docker_image": docker_image,
        "cpu_cores": cpu_cores,
        "memory_gb": memory_gb,
        "disk_size_gb": disk_size_gb,
        "gpu_count": gpu_count,
        "network_access": network_access,
        "timeout_minutes": timeout_minutes,
    }

    if start_command:
        request_body["start_command"] = start_command
    if environment_vars:
        request_body["environment_vars"] = environment_vars
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
    """List all sandboxes in your account.

    Args:
        team_id: Filter by team ID
        status: Filter by status (PENDING, PROVISIONING, RUNNING, STOPPED, ERROR, TERMINATED)
        labels: Filter by labels
        page: Page number for pagination (default: 1)
        per_page: Results per page (default: 50, max: 100)
        exclude_terminated: Exclude terminated sandboxes (default: False)

    Returns:
        List of sandboxes with pagination info
    """
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
    """Get detailed information about a specific sandbox.

    Args:
        sandbox_id: Unique identifier of the sandbox

    Returns:
        Detailed sandbox information including status, configuration, and timestamps
    """
    if not sandbox_id:
        return {"error": "sandbox_id is required"}

    response = await make_prime_request("GET", f"sandbox/{sandbox_id}")

    if not response:
        return {"error": f"Unable to get sandbox: {sandbox_id}"}

    return response


async def delete_sandbox(sandbox_id: str) -> dict[str, Any]:
    """Delete/terminate a sandbox.

    Args:
        sandbox_id: Unique identifier of the sandbox to delete

    Returns:
        Deletion confirmation
    """
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
    """Bulk delete multiple sandboxes by IDs or labels.

    You must specify either sandbox_ids OR labels, but not both.

    Args:
        sandbox_ids: List of sandbox IDs to delete
        labels: Delete all sandboxes with these labels

    Returns:
        Results showing succeeded and failed deletions
    """
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
    """Get logs from a sandbox.

    Args:
        sandbox_id: Unique identifier of the sandbox

    Returns:
        Sandbox logs as text
    """
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
    """Execute a command in a sandbox.

    IMPORTANT: The sandbox must be in RUNNING status before executing commands.
    Use get_sandbox() to check status first.

    Args:
        sandbox_id: Unique identifier of the sandbox
        command: Command to execute (shell command)
        working_dir: Working directory for the command (optional)
        env: Additional environment variables (optional)
        timeout: Command timeout in seconds (default: 300)

    Returns:
        Command result with stdout, stderr, and exit_code
    """
    if not sandbox_id:
        return {"error": "sandbox_id is required"}
    if not command:
        return {"error": "command is required"}
    if timeout < 1:
        return {"error": "timeout must be at least 1 second"}

    request_body: dict[str, Any] = {
        "command": command,
        "timeout": timeout,
    }

    if working_dir:
        request_body["working_dir"] = working_dir
    if env:
        request_body["env"] = env

    # Note: Command execution goes through the gateway, not the main API
    # The MCP client needs to handle this specially - for now we route through backend
    response = await make_prime_request(
        "POST", f"sandbox/{sandbox_id}/exec", json_data=request_body
    )

    if not response:
        return {"error": f"Unable to execute command in sandbox: {sandbox_id}"}

    return response


async def expose_port(
    sandbox_id: str,
    port: int,
    name: Optional[str] = None,
) -> dict[str, Any]:
    """Expose an HTTP port from a sandbox to the internet.

    Creates a public URL that routes traffic to the specified port in the sandbox.
    Useful for web servers, APIs, Jupyter notebooks, etc.

    Args:
        sandbox_id: Unique identifier of the sandbox
        port: Port number to expose (e.g., 8080, 8888)
        name: Optional friendly name for the exposure

    Returns:
        Exposure details including the public URL
    """
    if not sandbox_id:
        return {"error": "sandbox_id is required"}
    if not port or port < 1 or port > 65535:
        return {"error": "port must be between 1 and 65535"}

    request_body: dict[str, Any] = {"port": port}
    if name:
        request_body["name"] = name

    response = await make_prime_request(
        "POST", f"sandbox/{sandbox_id}/expose", json_data=request_body
    )

    if not response:
        return {"error": f"Unable to expose port {port} in sandbox: {sandbox_id}"}

    return response


async def unexpose_port(sandbox_id: str, exposure_id: str) -> dict[str, Any]:
    """Remove a port exposure from a sandbox.

    Args:
        sandbox_id: Unique identifier of the sandbox
        exposure_id: ID of the exposure to remove

    Returns:
        Confirmation of removal
    """
    if not sandbox_id:
        return {"error": "sandbox_id is required"}
    if not exposure_id:
        return {"error": "exposure_id is required"}

    response = await make_prime_request("DELETE", f"sandbox/{sandbox_id}/expose/{exposure_id}")

    if response is None:
        return {"error": f"Unable to unexpose port in sandbox: {sandbox_id}"}

    return response if response else {"success": True}


async def list_exposed_ports(sandbox_id: str) -> dict[str, Any]:
    """List all exposed ports for a sandbox.

    Args:
        sandbox_id: Unique identifier of the sandbox

    Returns:
        List of exposed ports with their URLs
    """
    if not sandbox_id:
        return {"error": "sandbox_id is required"}

    response = await make_prime_request("GET", f"sandbox/{sandbox_id}/expose")

    if not response:
        return {"error": f"Unable to list exposed ports for sandbox: {sandbox_id}"}

    return response


async def list_registry_credentials() -> dict[str, Any]:
    """List available registry credentials for private Docker images.

    Returns:
        List of registry credentials (without secrets)
    """
    response = await make_prime_request("GET", "template/registry-credentials")

    if not response:
        return {"error": "Unable to list registry credentials"}

    return response


async def check_docker_image(
    image: str,
    registry_credentials_id: Optional[str] = None,
) -> dict[str, Any]:
    """Check if a Docker image is accessible.

    Args:
        image: Docker image name (e.g., "python:3.11-slim", "ghcr.io/org/image:tag")
        registry_credentials_id: Optional credentials ID for private registries

    Returns:
        Whether the image is accessible and any details
    """
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
