from mcp.server.fastmcp import FastMCP

from prime_mcp.tools import availability, pods, ssh

mcp = FastMCP("primeintellect")


@mcp.tool()
async def check_gpu_availability(
    gpu_type: str | None = None,
    regions: str | None = None,
    socket: str | None = None,
    security: str | None = None,
) -> dict:
    """Check GPU availability across different providers.

    Args:
        gpu_type: GPU model (e.g., "A100_80GB", "H100_80GB", "RTX4090_24GB")
        regions: List of regions to filter (comma-separated string)
        socket: Socket for selected GPU model
            (options: "PCIe", "SXM2", "SXM3", "SXM4", "SXM5", "SXM6")
        security: Security type ("secure_cloud" or "community_cloud")

    Returns:
        Available GPU instances matching the criteria
    """
    return await availability.check_gpu_availability(gpu_type, regions, socket, security)


@mcp.tool()
async def check_cluster_availability(
    regions: list[str] | None = None,
    gpu_count: int | None = None,
    gpu_type: str | None = None,
    socket: str | None = None,
    security: str | None = None,
) -> dict:
    """Check cluster availability for multi-node deployments.

    Args:
        regions: List of regions to filter
        gpu_count: Desired number of GPUs
        gpu_type: GPU model (e.g., "H100_80GB", "A100_80GB", "RTX4090_24GB")
        socket: Socket for selected GPU model ("PCIe", "SXM2", "SXM3", "SXM4", "SXM5", "SXM6")
        security: Security type ("secure_cloud", "community_cloud")

    Returns:
        Available cluster configurations grouped by GPU type
    """
    return await availability.check_cluster_availability(
        regions, gpu_count, gpu_type, socket, security
    )


@mcp.tool()
async def create_pod(
    cloud_id: str,
    gpu_type: str,
    provider_type: str,
    name: str | None = None,
    gpu_count: int = 1,
    socket: str = "PCIe",
    disk_size: int | None = None,
    vcpus: int | None = None,
    memory: int | None = None,
    max_price: float | None = None,
    image: str = "ubuntu_22_cuda_12",
    custom_template_id: str | None = None,
    data_center_id: str | None = None,
    country: str | None = None,
    security: str | None = None,
    auto_restart: bool | None = None,
    jupyter_password: str | None = None,
    env_vars: dict[str, str] | None = None,
    team_id: str | None = None,
) -> dict:
    """Create a new GPU pod (compute instance).

    Args:
        cloud_id: Required cloud provider ID from availability check
        gpu_type: GPU model name
        provider_type: Provider type (e.g., "runpod", "fluidstack")
        name: Name for the pod (optional)
        gpu_count: Number of GPUs (default: 1)
        socket: GPU socket type (default: "PCIe")
        disk_size: Disk size in GB
        vcpus: Number of virtual CPUs
        memory: Memory in GB
        max_price: Maximum price per hour
        image: Environment image (default: "ubuntu_22_cuda_12")
        custom_template_id: Custom template ID
        data_center_id: Specific data center ID
        country: Country code
        security: Security level
        auto_restart: Auto-restart on failure
        jupyter_password: Jupyter password
        env_vars: Environment variables
        team_id: Team ID

    Returns:
        Created pod details
    """
    return await pods.create_pod(
        cloud_id=cloud_id,
        gpu_type=gpu_type,
        provider_type=provider_type,
        name=name,
        gpu_count=gpu_count,
        socket=socket,
        disk_size=disk_size,
        vcpus=vcpus,
        memory=memory,
        max_price=max_price,
        image=image,
        custom_template_id=custom_template_id,
        data_center_id=data_center_id,
        country=country,
        security=security,
        auto_restart=auto_restart,
        jupyter_password=jupyter_password,
        env_vars=env_vars,
        team_id=team_id,
    )


@mcp.tool()
async def list_pods(offset: int = 0, limit: int = 100) -> dict:
    """List all pods in your account.

    Args:
        offset: Number of pods to skip for pagination (default: 0)
        limit: Maximum number of pods to return (default: 100)

    Returns:
        Response containing list of pods
    """
    return await pods.list_pods(offset, limit)


@mcp.tool()
async def get_pods_history(
    limit: int = 100,
    offset: int = 0,
    sort_by: str = "terminatedAt",
    sort_order: str = "desc",
) -> dict:
    """Get pods history with sorting and pagination.

    Args:
        limit: Maximum number of entries (default: 100)
        offset: Number of entries to skip (default: 0)
        sort_by: Field to sort by (default: "terminatedAt")
        sort_order: Sort order (default: "desc")

    Returns:
        Historical pod data
    """
    return await pods.get_pods_history(limit, offset, sort_by, sort_order)


@mcp.tool()
async def get_pods_status(pod_ids: list[str] | None = None) -> dict:
    """Get pods status information.

    Args:
        pod_ids: List of specific pod IDs (optional)

    Returns:
        Pod status information
    """
    return await pods.get_pods_status(pod_ids)


@mcp.tool()
async def get_pod_details(pod_id: str) -> dict:
    """Get detailed information about a specific pod.

    Args:
        pod_id: Unique identifier of the pod

    Returns:
        Detailed pod information
    """
    return await pods.get_pod_details(pod_id)


@mcp.tool()
async def delete_pod(pod_id: str) -> dict:
    """Delete/terminate a pod.

    Args:
        pod_id: Unique identifier of the pod to delete

    Returns:
        Pod deletion response
    """
    return await pods.delete_pod(pod_id)


@mcp.tool()
async def manage_ssh_keys(
    action: str = "list",
    key_name: str | None = None,
    public_key: str | None = None,
    key_id: str | None = None,
    offset: int = 0,
    limit: int = 100,
) -> dict:
    """Manage SSH keys for pod access.

    Args:
        action: Action to perform ("list", "add", "delete", "set_primary")
        key_name: Name for the SSH key (required for "add")
        public_key: SSH public key content (required for "add")
        key_id: Key ID (required for "delete" and "set_primary")
        offset: Number of items to skip (for "list", default: 0)
        limit: Maximum items to return (for "list", default: 100)

    Returns:
        SSH key operation result
    """
    return await ssh.manage_ssh_keys(action, key_name, public_key, key_id, offset, limit)


if __name__ == "__main__":
    mcp.run(transport="stdio")
