from typing import Any, Optional

from prime_sandboxes import (
    APIError,
    AsyncSandboxClient,
    AsyncTemplateClient,
    CommandTimeoutError,
    CreateSandboxRequest,
)

_sandbox_client: Optional[AsyncSandboxClient] = None
_template_client: Optional[AsyncTemplateClient] = None


def _get_sandbox_client() -> AsyncSandboxClient:
    """Get or create the sandbox client singleton."""
    global _sandbox_client
    if _sandbox_client is None:
        _sandbox_client = AsyncSandboxClient()
    return _sandbox_client


def _get_template_client() -> AsyncTemplateClient:
    """Get or create the template client singleton."""
    global _template_client
    if _template_client is None:
        _template_client = AsyncTemplateClient()
    return _template_client


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
    try:
        client = _get_sandbox_client()
        request = CreateSandboxRequest(
            name=name,
            docker_image=docker_image,
            start_command=start_command,
            cpu_cores=cpu_cores,
            memory_gb=memory_gb,
            disk_size_gb=disk_size_gb,
            gpu_count=0,  # GPU support not yet available
            network_access=network_access,
            timeout_minutes=timeout_minutes,
            environment_vars=environment_vars,
            secrets=secrets,
            labels=labels or [],
            team_id=team_id,
            registry_credentials_id=registry_credentials_id,
        )
        sandbox = await client.create(request)
        return sandbox.model_dump(by_alias=True)
    except APIError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Failed to create sandbox: {e}"}


async def list_sandboxes(
    team_id: Optional[str] = None,
    status: Optional[str] = None,
    labels: Optional[list[str]] = None,
    page: int = 1,
    per_page: int = 50,
    exclude_terminated: bool = False,
) -> dict[str, Any]:
    """List all sandboxes in your account."""
    try:
        client = _get_sandbox_client()
        response = await client.list(
            team_id=team_id,
            status=status,
            labels=labels,
            page=page,
            per_page=per_page,
            exclude_terminated=exclude_terminated,
        )
        return {
            "sandboxes": [s.model_dump(by_alias=True) for s in response.sandboxes],
            "total": response.total,
            "page": response.page,
            "per_page": response.per_page,
            "has_next": response.has_next,
        }
    except APIError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Failed to list sandboxes: {e}"}


async def get_sandbox(sandbox_id: str) -> dict[str, Any]:
    """Get detailed information about a specific sandbox."""
    if not sandbox_id:
        return {"error": "sandbox_id is required"}
    try:
        client = _get_sandbox_client()
        sandbox = await client.get(sandbox_id)
        return sandbox.model_dump(by_alias=True)
    except APIError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Failed to get sandbox: {e}"}


async def delete_sandbox(sandbox_id: str) -> dict[str, Any]:
    """Delete/terminate a sandbox."""
    if not sandbox_id:
        return {"error": "sandbox_id is required"}
    try:
        client = _get_sandbox_client()
        result = await client.delete(sandbox_id)
        return result
    except APIError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Failed to delete sandbox: {e}"}


async def bulk_delete_sandboxes(
    sandbox_ids: Optional[list[str]] = None,
    labels: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Bulk delete multiple sandboxes by IDs or labels."""
    if not sandbox_ids and not labels:
        return {"error": "Must specify either sandbox_ids or labels"}
    if sandbox_ids and labels:
        return {"error": "Cannot specify both sandbox_ids and labels"}
    try:
        client = _get_sandbox_client()
        response = await client.bulk_delete(sandbox_ids=sandbox_ids, labels=labels)
        return response.model_dump()
    except APIError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Failed to bulk delete sandboxes: {e}"}


async def get_sandbox_logs(sandbox_id: str) -> dict[str, Any]:
    """Get logs from a sandbox."""
    if not sandbox_id:
        return {"error": "sandbox_id is required"}
    try:
        client = _get_sandbox_client()
        logs = await client.get_logs(sandbox_id)
        return {"logs": logs}
    except APIError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Failed to get sandbox logs: {e}"}


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
        client = _get_sandbox_client()
        result = await client.execute_command(
            sandbox_id=sandbox_id,
            command=command,
            working_dir=working_dir,
            env=env,
            timeout=timeout,
        )
        return result.model_dump()
    except CommandTimeoutError:
        return {"error": f"Command timed out after {timeout} seconds"}
    except APIError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Failed to execute command: {e}"}


async def expose_port(
    sandbox_id: str,
    port: int,
    name: Optional[str] = None,
) -> dict[str, Any]:
    """Expose an HTTP port from a sandbox to the internet."""
    if not sandbox_id:
        return {"error": "sandbox_id is required"}
    if not port or port < 22 or port > 9000:
        return {"error": "port must be between 22 and 9000"}
    if port == 8080:
        return {"error": "port 8080 is reserved and cannot be exposed"}
    try:
        client = _get_sandbox_client()
        result = await client.expose(sandbox_id=sandbox_id, port=port, name=name)
        return result.model_dump()
    except APIError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Failed to expose port: {e}"}


async def unexpose_port(sandbox_id: str, exposure_id: str) -> dict[str, Any]:
    """Remove a port exposure from a sandbox."""
    if not sandbox_id:
        return {"error": "sandbox_id is required"}
    if not exposure_id:
        return {"error": "exposure_id is required"}
    try:
        client = _get_sandbox_client()
        await client.unexpose(sandbox_id=sandbox_id, exposure_id=exposure_id)
        return {"success": True}
    except APIError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Failed to unexpose port: {e}"}


async def list_exposed_ports(sandbox_id: str) -> dict[str, Any]:
    """List all exposed ports for a sandbox."""
    if not sandbox_id:
        return {"error": "sandbox_id is required"}
    try:
        client = _get_sandbox_client()
        response = await client.list_exposed_ports(sandbox_id)
        return {"exposures": [e.model_dump() for e in response.exposures]}
    except APIError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Failed to list exposed ports: {e}"}


async def list_registry_credentials() -> dict[str, Any]:
    """List available registry credentials for private Docker images."""
    try:
        client = _get_template_client()
        credentials = await client.list_registry_credentials()
        return {"credentials": [c.model_dump(by_alias=True) for c in credentials]}
    except APIError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Failed to list registry credentials: {e}"}


async def check_docker_image(
    image: str,
    registry_credentials_id: Optional[str] = None,
) -> dict[str, Any]:
    """Check if a Docker image is accessible."""
    if not image:
        return {"error": "image is required"}
    try:
        client = _get_template_client()
        result = await client.check_docker_image(
            image=image,
            registry_credentials_id=registry_credentials_id,
        )
        return result.model_dump()
    except APIError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Failed to check docker image: {e}"}
