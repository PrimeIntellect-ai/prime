from typing import Any

from prime_mcp.client import make_prime_request


async def list_rl_models() -> dict[str, Any]:
    """List available RL models from healthy clusters."""
    response_data = await make_prime_request("GET", "rft/models")
    if not response_data:
        return {"error": "Unable to fetch RL models"}
    return response_data


async def list_rl_runs(team_id: str | None = None) -> dict[str, Any]:
    """List RL runs. If team_id is None, returns personal + team runs."""
    params = {"team_id": team_id} if team_id else {}
    response_data = await make_prime_request("GET", "rft/runs", params=params if params else None)
    if not response_data:
        return {"error": "Unable to fetch RL runs"}
    return response_data


async def get_rl_run(run_id: str) -> dict[str, Any]:
    """Get details of a specific RL run."""
    response_data = await make_prime_request("GET", f"rft/runs/{run_id}")
    if not response_data:
        return {"error": f"Unable to fetch RL run: {run_id}"}
    return response_data


async def create_rl_run(
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
    """Create a new RL training run."""
    valid_rollouts = [1, 2, 4, 8, 16, 32, 64, 128]
    if rollouts_per_example not in valid_rollouts:
        return {"error": f"rollouts_per_example must be one of {valid_rollouts}"}

    request_body: dict[str, Any] = {
        "model": {"name": model_name},
        "environments": environments,
        "rollouts_per_example": rollouts_per_example,
        "seq_len": seq_len,
        "max_steps": max_steps,
    }

    if name:
        request_body["name"] = name

    all_secrets = list(secrets) if secrets else []
    if wandb_api_key:
        all_secrets.append({"key": "WANDB_API_KEY", "value": wandb_api_key})
    if all_secrets:
        request_body["secrets"] = all_secrets

    if wandb_project:
        request_body["monitoring"] = {"wandb": {"project": wandb_project}}
        if wandb_entity:
            request_body["monitoring"]["wandb"]["entity"] = wandb_entity
        if wandb_run_name:
            request_body["monitoring"]["wandb"]["name"] = wandb_run_name

    if eval_config:
        request_body["eval"] = eval_config
    if team_id:
        request_body["team_id"] = team_id

    response_data = await make_prime_request("POST", "rft/runs", json_data=request_body)
    if not response_data:
        return {"error": "Unable to create RL run"}
    return response_data


async def stop_rl_run(run_id: str) -> dict[str, Any]:
    """Stop an RL training run."""
    response_data = await make_prime_request("PUT", f"rft/runs/{run_id}/stop")
    if not response_data:
        return {"error": f"Unable to stop RL run: {run_id}"}
    return response_data


async def delete_rl_run(run_id: str) -> dict[str, Any]:
    """Delete an RL training run."""
    response_data = await make_prime_request("DELETE", f"rft/runs/{run_id}")
    if response_data is not None and not response_data.get("error"):
        return {"success": True, "run_id": run_id}
    return response_data or {"error": f"Unable to delete RL run: {run_id}"}


async def get_rl_run_logs(run_id: str, tail_lines: int = 1000) -> dict[str, Any]:
    """Get logs for an RL run."""
    response_data = await make_prime_request(
        "GET", f"rft/runs/{run_id}/logs", params={"tail_lines": tail_lines}
    )
    if not response_data:
        return {"error": f"Unable to fetch logs for RL run: {run_id}"}
    return response_data


async def list_rl_adapters(team_id: str | None = None) -> dict[str, Any]:
    """List LoRA adapters from completed RL runs."""
    params = {"team_id": team_id} if team_id else {}
    response_data = await make_prime_request(
        "GET", "rft/adapters", params=params if params else None
    )
    if not response_data:
        return {"error": "Unable to fetch RL adapters"}
    return response_data


async def get_rl_adapter(adapter_id: str) -> dict[str, Any]:
    """Get a specific adapter by ID."""
    response_data = await make_prime_request("GET", f"rft/adapters/{adapter_id}")
    if not response_data:
        return {"error": f"Unable to fetch adapter: {adapter_id}"}
    return response_data


async def delete_rl_adapter(adapter_id: str) -> dict[str, Any]:
    """Delete an adapter by ID."""
    response_data = await make_prime_request("DELETE", f"rft/adapters/{adapter_id}")
    if response_data is not None and not response_data.get("error"):
        return {"success": True, "adapter_id": adapter_id}
    return response_data or {"error": f"Unable to delete adapter: {adapter_id}"}
