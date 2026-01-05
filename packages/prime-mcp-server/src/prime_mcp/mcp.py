from mcp.server.fastmcp import FastMCP

from prime_mcp.tools import availability, pods, rft, ssh

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
async def list_rft_models() -> dict:
    """List all available RFT models for training.

    Returns models from healthy RFT clusters that are ready to accept training jobs.
    Check this before creating a run to see which models are available.

    Returns:
        List of available RFT models with their names
    """
    return await rft.list_rft_models()


@mcp.tool()
async def list_rft_runs(team_id: str | None = None) -> dict:
    """List RFT training runs for the authenticated user.

    If team_id is provided, returns runs for that team only (requires team membership).
    If team_id is None, returns user's personal runs AND all runs from teams they're in.

    Args:
        team_id: Optional team ID to filter runs by team

    Returns:
        List of RFT runs with status, configuration, and progress information
    """
    return await rft.list_rft_runs(team_id)


@mcp.tool()
async def get_rft_run(run_id: str) -> dict:
    """Get detailed information about a specific RFT training run.

    Args:
        run_id: Unique identifier of the RFT run

    Returns:
        Detailed run information including:
        - status: QUEUED, PENDING, RUNNING, COMPLETED, FAILED, STOPPED
        - configuration: model, environments, hyperparameters
        - progress: current step, started_at, completed_at
        - error_message: if run failed
    """
    return await rft.get_rft_run(run_id)


@mcp.tool()
async def create_rft_run(
    model_name: str,
    environments: list[dict],
    rollouts_per_example: int,
    seq_len: int,
    max_steps: int,
    name: str | None = None,
    eval_config: dict | None = None,
    wandb_entity: str | None = None,
    wandb_project: str | None = None,
    wandb_run_name: str | None = None,
    wandb_api_key: str | None = None,
    secrets: list[dict] | None = None,
    team_id: str | None = None,
) -> dict:
    """Create a new RFT (Reinforcement Fine-Tuning) training run.

    WORKFLOW:
    1. First check available models with list_rft_models()
    2. Configure your training environments
    3. Optionally set up W&B monitoring with your API key
    4. Create the run - it will be queued and start automatically

    Args:
        model_name: Model to fine-tune (e.g., "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B").
            Use list_rft_models() to see available models.
        environments: Training environments list. Each environment dict should have:
            - id (required): Environment ID like "reverse-text" or hub slug "primeintellect/vf-math"
            - name (optional): Display name for this environment
            - args (optional): Dict of environment-specific arguments
        rollouts_per_example: Number of rollouts per training example.
            MUST be one of: 1, 2, 4, 8, 16, 32, 64, 128 (must divide batch size 128 evenly)
        seq_len: Sequence length for training (context window size)
        max_steps: Maximum number of training steps
        name: Optional run name (auto-generated if not provided)
        eval_config: Optional evaluation configuration dict with:
            - environments: List of eval environments (same format as training)
            - interval: Evaluate every N steps (default: 100)
            - num_examples: Examples per environment (-1 for all)
            - rollouts_per_example: Rollouts per eval example (default: 1)
            - eval_base_model: Whether to eval base model first (default: True)
        wandb_entity: W&B entity (username or team name) for metrics logging
        wandb_project: W&B project name - REQUIRED if you want monitoring
        wandb_run_name: W&B run name (optional)
        wandb_api_key: Your W&B API key - REQUIRED for W&B monitoring
        secrets: Additional secrets as list of {"key": "NAME", "value": "secret"} dicts
        team_id: Team ID to create run for (requires team membership)

    Returns:
        Created RFT run details including run ID and initial status (QUEUED)

    Example:
        create_rft_run(
            model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            environments=[{"id": "reverse-text"}],
            rollouts_per_example=8,
            seq_len=2048,
            max_steps=1000,
            wandb_project="my-rft-project",
            wandb_api_key="your-wandb-key"
        )
    """
    return await rft.create_rft_run(
        model_name=model_name,
        environments=environments,
        rollouts_per_example=rollouts_per_example,
        seq_len=seq_len,
        max_steps=max_steps,
        name=name,
        eval_config=eval_config,
        wandb_entity=wandb_entity,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,
        wandb_api_key=wandb_api_key,
        secrets=secrets,
        team_id=team_id,
    )


@mcp.tool()
async def stop_rft_run(run_id: str) -> dict:
    """Stop/abort a running RFT training run.

    Can only stop runs that are in QUEUED, PENDING, or RUNNING status.
    The run will save any adapter weights produced so far before stopping.

    Args:
        run_id: Unique identifier of the RFT run to stop

    Returns:
        Updated run details with STOPPED status
    """
    return await rft.stop_rft_run(run_id)


@mcp.tool()
async def delete_rft_run(run_id: str) -> dict:
    """Delete an RFT training run.

    This will:
    - Cleanup Kubernetes resources if still running
    - Delete the run record from the database
    - Note: Adapter weights from completed runs are preserved separately

    Args:
        run_id: Unique identifier of the RFT run to delete

    Returns:
        Deletion confirmation with run_id and success status
    """
    return await rft.delete_rft_run(run_id)


@mcp.tool()
async def get_rft_run_logs(run_id: str, tail_lines: int = 1000) -> dict:
    """Get training logs for an RFT run.

    Fetches logs from the orchestrator pod running the training job.
    Useful for debugging failed runs or monitoring progress.

    Args:
        run_id: Unique identifier of the RFT run
        tail_lines: Number of lines to return from the end of logs (default: 1000)

    Returns:
        Log content as a string
    """
    return await rft.get_rft_run_logs(run_id, tail_lines)


@mcp.tool()
async def list_rft_adapters(team_id: str | None = None) -> dict:
    """List trained adapters (LoRA weights) from completed RFT runs.

    Adapters are the output of successful RFT training - they contain the
    fine-tuned LoRA weights that can be used for inference.

    Args:
        team_id: Optional team ID to filter adapters by team

    Returns:
        List of adapters with:
        - id: Adapter ID for use with inference
        - display_name: Optional friendly name
        - base_model: The base model these weights are for
        - rft_run_id: The training run that produced this adapter
        - status: PENDING, READY, or FAILED
    """
    return await rft.list_rft_adapters(team_id)


@mcp.tool()
async def get_rft_adapter(adapter_id: str) -> dict:
    """Get detailed information about a specific adapter.

    Args:
        adapter_id: Unique identifier of the adapter

    Returns:
        Adapter details including base model, status, and source run
    """
    return await rft.get_rft_adapter(adapter_id)


@mcp.tool()
async def delete_rft_adapter(adapter_id: str) -> dict:
    """Delete an adapter.

    Removes the adapter record from the database.
    Note: This deletes the database record but storage files may be retained.

    Args:
        adapter_id: Unique identifier of the adapter to delete

    Returns:
        Deletion confirmation with adapter_id and success status
    """
    return await rft.delete_rft_adapter(adapter_id)


if __name__ == "__main__":
    mcp.run(transport="stdio")
