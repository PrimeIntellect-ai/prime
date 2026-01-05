from typing import Any

from prime_mcp.client import make_prime_request


async def list_rft_models() -> dict[str, Any]:
    """List all available RFT models.

    Returns models from healthy RFT clusters (heartbeat within last 1 minute).

    Returns:
        List of available RFT models with their names
    """
    response_data = await make_prime_request("GET", "rft/models")

    if not response_data:
        return {"error": "Unable to fetch RFT models"}

    return response_data


async def list_rft_runs(team_id: str | None = None) -> dict[str, Any]:
    """List RFT runs for the authenticated user.

    If team_id is provided, returns runs for that team only (requires team membership).
    If team_id is None, returns user's personal runs AND all runs from teams they're in.

    Args:
        team_id: Optional team ID to filter runs

    Returns:
        List of RFT runs with their details
    """
    params = {}
    if team_id:
        params["team_id"] = team_id

    response_data = await make_prime_request("GET", "rft/runs", params=params if params else None)

    if not response_data:
        return {"error": "Unable to fetch RFT runs"}

    return response_data


async def get_rft_run(run_id: str) -> dict[str, Any]:
    """Get details of a specific RFT run.

    Args:
        run_id: Unique identifier of the RFT run

    Returns:
        Detailed RFT run information including status, configuration, and progress
    """
    response_data = await make_prime_request("GET", f"rft/runs/{run_id}")

    if not response_data:
        return {"error": f"Unable to fetch RFT run: {run_id}"}

    return response_data


async def create_rft_run(
    model_name: str,
    environments: list[dict[str, Any]],
    rollouts_per_example: int,
    seq_len: int,
    max_steps: int,
    name: str | None = None,
    eval_config: dict[str, Any] | None = None,
    wandb_entity: str | None = None,
    wandb_project: str | None = None,
    wandb_run_name: str | None = None,
    wandb_api_key: str | None = None,
    secrets: list[dict[str, str]] | None = None,
    team_id: str | None = None,
) -> dict[str, Any]:
    """Create a new RFT training run.

    IMPORTANT PREREQUISITES:
    1. Check available models with list_rft_models() first
    2. Ensure you have a W&B API key if you want monitoring

    Args:
        model_name: Model name/path (e.g., "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
        environments: List of training environments. Each environment should have:
            - id: Environment ID (e.g., "reverse-text" or "primeintellect/vf-math")
            - name: Optional display name
            - args: Optional environment-specific arguments dict
        rollouts_per_example: Number of rollouts per example.
            Must divide 128: valid values 1,2,4,8,16,32,64,128
        seq_len: Sequence length for training
        max_steps: Maximum training steps
        name: Optional run name (auto-generated if not provided)
        eval_config: Optional evaluation configuration with:
            - environments: List of eval environments (same format as training)
            - interval: Evaluate every N steps (default: 100)
            - num_examples: Number of examples per environment (-1 for all)
            - rollouts_per_example: Rollouts per example (default: 1)
            - eval_base_model: Whether to eval base model before training (default: True)
        wandb_entity: W&B entity (username or team) for monitoring
        wandb_project: W&B project name for monitoring
        wandb_run_name: W&B run name
        wandb_api_key: W&B API key for authentication (passed as secret)
        secrets: Additional secrets as list of {key, value} dicts
        team_id: Optional team ID to create run for team

    Returns:
        Created RFT run details including run ID and status
    """
    # Validate rollouts_per_example
    valid_rollouts = [1, 2, 4, 8, 16, 32, 64, 128]
    if rollouts_per_example not in valid_rollouts:
        return {"error": f"rollouts_per_example must be one of {valid_rollouts}"}

    # Build request body
    request_body: dict[str, Any] = {
        "model": {"name": model_name},
        "environments": environments,
        "rollouts_per_example": rollouts_per_example,
        "seq_len": seq_len,
        "max_steps": max_steps,
    }

    if name:
        request_body["name"] = name

    # Build secrets list (copy to avoid mutating caller's list)
    all_secrets = list(secrets) if secrets else []
    if wandb_api_key:
        all_secrets.append({"key": "WANDB_API_KEY", "value": wandb_api_key})

    if all_secrets:
        request_body["secrets"] = all_secrets

    # Add monitoring config if W&B settings provided
    if wandb_project:
        request_body["monitoring"] = {
            "wandb": {
                "project": wandb_project,
            }
        }
        if wandb_entity:
            request_body["monitoring"]["wandb"]["entity"] = wandb_entity
        if wandb_run_name:
            request_body["monitoring"]["wandb"]["name"] = wandb_run_name

    # Add eval config if provided
    if eval_config:
        request_body["eval"] = eval_config

    if team_id:
        request_body["team_id"] = team_id

    response_data = await make_prime_request("POST", "rft/runs", json_data=request_body)

    if not response_data:
        return {"error": "Unable to create RFT run"}

    return response_data


async def stop_rft_run(run_id: str) -> dict[str, Any]:
    """Stop/abort a running RFT training run.

    Can only stop runs in QUEUED, PENDING, or RUNNING status.

    Args:
        run_id: Unique identifier of the RFT run to stop

    Returns:
        Updated RFT run details with STOPPED status
    """
    response_data = await make_prime_request("PUT", f"rft/runs/{run_id}/stop")

    if not response_data:
        return {"error": f"Unable to stop RFT run: {run_id}"}

    return response_data


async def delete_rft_run(run_id: str) -> dict[str, Any]:
    """Delete an RFT training run.

    This will cleanup Kubernetes resources and delete the run from the database.

    Args:
        run_id: Unique identifier of the RFT run to delete

    Returns:
        Deletion confirmation with run_id and success status
    """
    response_data = await make_prime_request("DELETE", f"rft/runs/{run_id}")

    if not response_data:
        return {"error": f"Unable to delete RFT run: {run_id}"}

    return response_data


async def get_rft_run_logs(run_id: str, tail_lines: int = 1000) -> dict[str, Any]:
    """Get orchestrator logs for an RFT run.

    Args:
        run_id: Unique identifier of the RFT run
        tail_lines: Number of lines to tail from the end of logs (default: 1000)

    Returns:
        Pod logs as a string
    """
    params = {"tail_lines": tail_lines}

    response_data = await make_prime_request("GET", f"rft/runs/{run_id}/logs", params=params)

    if not response_data:
        return {"error": f"Unable to fetch logs for RFT run: {run_id}"}

    return response_data


async def list_rft_adapters(team_id: str | None = None) -> dict[str, Any]:
    """List adapters for the authenticated user.

    Adapters are LoRA weights produced by completed RFT training runs.

    Args:
        team_id: Optional team ID to filter adapters

    Returns:
        List of adapters with their details (ID, base model, status, etc.)
    """
    params = {}
    if team_id:
        params["team_id"] = team_id

    response_data = await make_prime_request(
        "GET", "rft/adapters", params=params if params else None
    )

    if not response_data:
        return {"error": "Unable to fetch RFT adapters"}

    return response_data


async def get_rft_adapter(adapter_id: str) -> dict[str, Any]:
    """Get a specific adapter by ID.

    Args:
        adapter_id: Unique identifier of the adapter

    Returns:
        Adapter details including ID, base model, status, and associated run
    """
    response_data = await make_prime_request("GET", f"rft/adapters/{adapter_id}")

    if not response_data:
        return {"error": f"Unable to fetch adapter: {adapter_id}"}

    return response_data


async def delete_rft_adapter(adapter_id: str) -> dict[str, Any]:
    """Delete an adapter by ID.

    Note: This only deletes the database record. Storage files are not automatically cleaned up.

    Args:
        adapter_id: Unique identifier of the adapter to delete

    Returns:
        Deletion confirmation with adapter_id and success status
    """
    response_data = await make_prime_request("DELETE", f"rft/adapters/{adapter_id}")

    if not response_data:
        return {"error": f"Unable to delete adapter: {adapter_id}"}

    return response_data
