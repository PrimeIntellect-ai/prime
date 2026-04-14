from typing import Any

from prime_mcp.client import make_prime_request


async def list_rl_runs(team_id: str | None = None) -> dict[str, Any]:
    """List hosted RL training runs for the authenticated user."""
    params = {"team_id": team_id} if team_id else None
    response_data = await make_prime_request("GET", "rft/runs", params=params)

    if not response_data:
        return {"error": "Unable to fetch RL runs"}

    return response_data


async def get_rl_run_progress(run_id: str) -> dict[str, Any]:
    """Get rollout/distribution progress metadata for a hosted RL training run."""
    response_data = await make_prime_request("GET", f"rft/runs/{run_id}/progress")

    if not response_data:
        return {"error": f"Unable to fetch progress for RL run: {run_id}"}

    return response_data


async def get_rl_run_rollouts(
    run_id: str,
    step: int,
    page: int = 1,
    limit: int = 100,
) -> dict[str, Any]:
    """Get rollout samples for a hosted RL training run step."""
    if step < 0:
        return {"error": "step must be greater than or equal to 0"}
    if page < 1:
        return {"error": "page must be greater than 0"}
    if limit < 1:
        return {"error": "limit must be greater than 0"}

    response_data = await make_prime_request(
        "GET",
        f"rft/runs/{run_id}/samples",
        params={"step": step, "page": page, "limit": limit},
    )

    if not response_data:
        return {"error": f"Unable to fetch rollouts for RL run: {run_id}"}

    return response_data
