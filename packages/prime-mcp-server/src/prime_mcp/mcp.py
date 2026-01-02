from mcp.server.fastmcp import FastMCP

from prime_mcp.tools import availability, pods, sandboxes, ssh

mcp = FastMCP("primeintellect")


@mcp.tool()
async def check_gpu_availability(
    gpu_type: str | None = None,
    regions: list[str] | None = None,
    socket: str | None = None,
    security: str | None = None,
    gpu_count: int | None = None,
) -> dict:
    """Check GPU availability across different providers.

    Args:
        gpu_type: GPU model (e.g., "A100_80GB", "H100_80GB", "RTX4090_24GB")
        regions: List of regions to filter (e.g., ["united_states", "eu_west"])
            Valid options: "africa", "asia_south", "asia_northeast", "australia", "canada",
            "eu_east", "eu_north", "eu_west", "middle_east", "south_america", "united_states"
        socket: Socket for selected GPU model
            (options: "PCIe", "SXM2", "SXM3", "SXM4", "SXM5", "SXM6")
        security: Security type ("secure_cloud" or "community_cloud")
        gpu_count: Number of GPUs to filter by

    Returns:
        Available GPU instances grouped by GPU type. IMPORTANT: Each instance has an 'isSpot' field:
        - Spot instances (isSpot=true): 50-90% CHEAPER but CAN BE TERMINATED AT ANY TIME
        - On-demand (isSpot=false/null): More expensive but GUARANTEED availability

        When presenting options to user, ALWAYS show both spot and on-demand prices and let them
        choose based on workload criticality. Also show available 'images' so user can select one.
    """
    return await availability.check_gpu_availability(gpu_type, regions, socket, security, gpu_count)


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
        regions: List of regions to filter (e.g., ["united_states", "eu_west"])
            Valid options: "africa", "asia_south", "asia_northeast", "australia", "canada",
            "eu_east", "eu_north", "eu_west", "middle_east", "south_america", "united_states"
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
    data_center_id: str,
    name: str | None = None,
    gpu_count: int = 1,
    socket: str = "PCIe",
    disk_size: int | None = None,
    vcpus: int | None = None,
    memory: int | None = None,
    max_price: float | None = None,
    image: str = "ubuntu_22_cuda_12",
    custom_template_id: str | None = None,
    country: str | None = None,
    security: str | None = None,
    auto_restart: bool | None = None,
    jupyter_password: str | None = None,
    env_vars: dict[str, str] | None = None,
    team_id: str | None = None,
) -> dict:
    """Create a new GPU pod (compute instance).

    IMPORTANT BEFORE CREATING:
    1. SSH KEYS: Ensure user has added SSH key via manage_ssh_keys() or they won't be able to
       access the pod!
    2. SPOT vs ON-DEMAND: Check the 'isSpot' field from availability results:
       - Spot (isSpot=true): 50-90% cheaper but can be TERMINATED AT ANY TIME
       - On-demand (isSpot=false/null): More expensive but GUARANTEED availability
       Ask user which type they prefer based on their workload criticality!
    3. IMAGE SELECTION: Ask user which image they need for their workload:
       - ubuntu_22_cuda_12 (base), cuda_12_4_pytorch_2_5 (PyTorch), prime_rl (RL), etc.
       Check the 'images' field in availability results for available options.

    Args:
        cloud_id: Required cloud provider ID from availability check
        gpu_type: GPU model name
        provider_type: Provider type (e.g., "runpod", "fluidstack", "hyperstack", "datacrunch")
        data_center_id: Required data center ID from availability check. Get this from the
            'dataCenter' field in availability results. Examples: "CANADA-1", "US-CA-2",
            "FIN-02", "ICE-01"
        name: Name for the pod (optional)
        gpu_count: Number of GPUs (default: 1)
        socket: GPU socket type (default: "PCIe")
        disk_size: Disk size in GB
        vcpus: Number of virtual CPUs
        memory: Memory in GB
        max_price: Maximum price per hour
        image: Environment image - ASK USER which image they need! Options include:
            ubuntu_22_cuda_12, cuda_12_4_pytorch_2_5, prime_rl, vllm_llama_*, etc.
        custom_template_id: Custom template ID
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
        data_center_id=data_center_id,
        name=name,
        gpu_count=gpu_count,
        socket=socket,
        disk_size=disk_size,
        vcpus=vcpus,
        memory=memory,
        max_price=max_price,
        image=image,
        custom_template_id=custom_template_id,
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
        sort_by: Field to sort by (default: "terminatedAt", options: "terminatedAt", "createdAt")
        sort_order: Sort order (default: "desc", options: "asc", "desc")

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

    CRITICAL: Users MUST have an SSH key added BEFORE creating pods, or they won't be able to
    access them! Always check if user has SSH keys with action="list" before creating pods.
    If no keys exist, guide user to add one with action="add".

    Args:
        action: Action to perform ("list", "add", "delete", "set_primary")
        key_name: Name for the SSH key (required for "add")
        public_key: SSH public key content (required for "add")
            Users can get their public key from ~/.ssh/id_rsa.pub or generate with:
            ssh-keygen -t rsa -b 4096
        key_id: Key ID (required for "delete" and "set_primary")
        offset: Number of items to skip (for "list", default: 0)
        limit: Maximum items to return (for "list", default: 100)

    Returns:
        SSH key operation result
    """
    return await ssh.manage_ssh_keys(action, key_name, public_key, key_id, offset, limit)


@mcp.tool()
async def create_sandbox(
    name: str,
    docker_image: str = "python:3.11-slim",
    start_command: str | None = "tail -f /dev/null",
    cpu_cores: int = 1,
    memory_gb: int = 2,
    disk_size_gb: int = 5,
    network_access: bool = True,
    timeout_minutes: int = 60,
    environment_vars: dict[str, str] | None = None,
    secrets: dict[str, str] | None = None,
    labels: list[str] | None = None,
    team_id: str | None = None,
    registry_credentials_id: str | None = None,
) -> dict:
    """Create a new sandbox for isolated code execution.

    A sandbox is a containerized environment where you can safely execute code,
    run commands, and manage files in isolation. Perfect for:
    - Running untrusted code safely
    - Testing and development
    - Data processing pipelines
    - CI/CD tasks

    WORKFLOW:
    1. Create sandbox with create_sandbox()
    2. Wait for status to become RUNNING (check with get_sandbox())
    3. Execute commands with execute_sandbox_command()
    4. Clean up with delete_sandbox()

    Args:
        name: Name for the sandbox (required)
        docker_image: Docker image to use (default: "python:3.11-slim")
            Popular options: python:3.11-slim, ubuntu:22.04, node:20-slim
        start_command: Command to run on startup (default: "tail -f /dev/null")
        cpu_cores: Number of CPU cores (1-16, default: 1)
        memory_gb: Memory in GB (1-64, default: 2)
        disk_size_gb: Disk size in GB (1-1000, default: 5)
        network_access: Enable network access (default: True)
        timeout_minutes: Auto-termination timeout (1-1440 minutes, default: 60)
        environment_vars: Environment variables as key-value pairs
        secrets: Sensitive environment variables (e.g., API keys) - stored securely
        labels: Labels for organizing and filtering sandboxes
        team_id: Team ID for organization accounts
        registry_credentials_id: ID for private Docker registry credentials

    Returns:
        Created sandbox details including ID, status, and configuration
    """
    return await sandboxes.create_sandbox(
        name=name,
        docker_image=docker_image,
        start_command=start_command,
        cpu_cores=cpu_cores,
        memory_gb=memory_gb,
        disk_size_gb=disk_size_gb,
        network_access=network_access,
        timeout_minutes=timeout_minutes,
        environment_vars=environment_vars,
        secrets=secrets,
        labels=labels,
        team_id=team_id,
        registry_credentials_id=registry_credentials_id,
    )


@mcp.tool()
async def list_sandboxes(
    team_id: str | None = None,
    status: str | None = None,
    labels: list[str] | None = None,
    page: int = 1,
    per_page: int = 50,
    exclude_terminated: bool = False,
) -> dict:
    """List all sandboxes in your account.

    Args:
        team_id: Filter by team ID
        status: Filter by status (PENDING, PROVISIONING, RUNNING, STOPPED, ERROR, TERMINATED)
        labels: Filter by labels (sandboxes must have ALL specified labels)
        page: Page number for pagination (default: 1)
        per_page: Results per page (default: 50, max: 100)
        exclude_terminated: Exclude terminated sandboxes (default: False)

    Returns:
        List of sandboxes with pagination info (sandboxes, total, page, per_page, has_next)
    """
    return await sandboxes.list_sandboxes(
        team_id=team_id,
        status=status,
        labels=labels,
        page=page,
        per_page=per_page,
        exclude_terminated=exclude_terminated,
    )


@mcp.tool()
async def get_sandbox(sandbox_id: str) -> dict:
    """Get detailed information about a specific sandbox.

    Use this to check sandbox status before executing commands.
    Sandbox must be in RUNNING status for command execution.

    Args:
        sandbox_id: Unique identifier of the sandbox

    Returns:
        Detailed sandbox information including:
        - id, name, status
        - docker_image, cpu_cores, memory_gb, disk_size_gb
        - created_at, started_at, terminated_at
        - labels, environment_vars
    """
    return await sandboxes.get_sandbox(sandbox_id)


@mcp.tool()
async def delete_sandbox(sandbox_id: str) -> dict:
    """Delete/terminate a sandbox.

    This will immediately terminate the sandbox and release resources.
    Any unsaved data will be lost.

    Args:
        sandbox_id: Unique identifier of the sandbox to delete

    Returns:
        Deletion confirmation
    """
    return await sandboxes.delete_sandbox(sandbox_id)


@mcp.tool()
async def bulk_delete_sandboxes(
    sandbox_ids: list[str] | None = None,
    labels: list[str] | None = None,
) -> dict:
    """Bulk delete multiple sandboxes by IDs or labels.

    Useful for cleanup operations. You must specify either sandbox_ids OR labels,
    but not both.

    Args:
        sandbox_ids: List of sandbox IDs to delete
        labels: Delete all sandboxes with ALL of these labels

    Returns:
        Results showing succeeded and failed deletions
    """
    return await sandboxes.bulk_delete_sandboxes(sandbox_ids=sandbox_ids, labels=labels)


@mcp.tool()
async def get_sandbox_logs(sandbox_id: str) -> dict:
    """Get logs from a sandbox.

    Returns container logs including stdout/stderr from the start command
    and any executed commands.

    Args:
        sandbox_id: Unique identifier of the sandbox

    Returns:
        Sandbox logs as text
    """
    return await sandboxes.get_sandbox_logs(sandbox_id)


@mcp.tool()
async def execute_sandbox_command(
    sandbox_id: str,
    command: str,
    working_dir: str | None = None,
    env: dict[str, str] | None = None,
    timeout: int = 300,
) -> dict:
    """Execute a command in a sandbox.

    IMPORTANT: The sandbox must be in RUNNING status before executing commands.
    Use get_sandbox() to check status first.

    Args:
        sandbox_id: Unique identifier of the sandbox
        command: Shell command to execute (e.g., "python script.py", "ls -la")
        working_dir: Working directory for the command (optional)
        env: Additional environment variables (optional)
        timeout: Command timeout in seconds (default: 300, max: 3600)

    Returns:
        Command result with:
        - stdout: Standard output
        - stderr: Standard error
        - exit_code: Exit code (0 = success)
    """
    return await sandboxes.execute_command(
        sandbox_id=sandbox_id,
        command=command,
        working_dir=working_dir,
        env=env,
        timeout=timeout,
    )


@mcp.tool()
async def expose_sandbox_port(
    sandbox_id: str,
    port: int,
    name: str | None = None,
    protocol: str = "HTTP",
) -> dict:
    """Expose a port from a sandbox to the internet.

    Creates a public URL that routes traffic to the specified port.
    Useful for web servers, APIs, Jupyter notebooks, Streamlit apps, etc.

    Args:
        sandbox_id: Unique identifier of the sandbox
        port: Port number to expose (22-9000, excluding 8080 which is reserved)
        name: Optional friendly name for the exposure
        protocol: Protocol type - HTTP (default), TCP, or UDP

    Returns:
        Exposure details including:
        - exposure_id: ID to use for unexpose_sandbox_port()
        - url: Public URL to access the service (for HTTP)
        - tls_socket: TLS socket address
        - external_port: External port (for TCP/UDP)
    """
    return await sandboxes.expose_port(
        sandbox_id=sandbox_id, port=port, name=name, protocol=protocol
    )


@mcp.tool()
async def unexpose_sandbox_port(sandbox_id: str, exposure_id: str) -> dict:
    """Remove a port exposure from a sandbox.

    Args:
        sandbox_id: Unique identifier of the sandbox
        exposure_id: ID of the exposure to remove (from expose_sandbox_port result)

    Returns:
        Confirmation of removal
    """
    return await sandboxes.unexpose_port(sandbox_id=sandbox_id, exposure_id=exposure_id)


@mcp.tool()
async def list_sandbox_exposed_ports(sandbox_id: str) -> dict:
    """List all exposed ports for a sandbox.

    Args:
        sandbox_id: Unique identifier of the sandbox

    Returns:
        List of exposed ports with their URLs and details
    """
    return await sandboxes.list_exposed_ports(sandbox_id)


@mcp.tool()
async def list_registry_credentials() -> dict:
    """List available registry credentials for private Docker images.

    Registry credentials allow you to pull images from private Docker registries
    like GitHub Container Registry, AWS ECR, Google Container Registry, etc.

    Returns:
        List of registry credentials (id, name, server - no secrets)
    """
    return await sandboxes.list_registry_credentials()


@mcp.tool()
async def check_docker_image(
    image: str,
    registry_credentials_id: str | None = None,
) -> dict:
    """Check if a Docker image is accessible before creating a sandbox.

    Validates that the image exists and can be pulled. Useful for:
    - Verifying public images exist
    - Testing private registry credentials

    Args:
        image: Docker image name (e.g., "python:3.11-slim", "ghcr.io/org/image:tag")
        registry_credentials_id: Optional credentials ID for private registries

    Returns:
        - accessible: Whether the image can be pulled
        - details: Additional information or error message
    """
    return await sandboxes.check_docker_image(
        image=image, registry_credentials_id=registry_credentials_id
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")
