"""Hosted RL (Reinforcement Learning) API client."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from prime_cli.core import APIClient, APIError


class RLModel(BaseModel):
    """Model available for RL training."""

    name: str = Field(..., description="Model name")

    model_config = ConfigDict(populate_by_name=True)


class RLRun(BaseModel):
    """RL Training Run."""

    id: str = Field(..., description="Run ID")
    name: Optional[str] = Field(None, description="Run name")
    user_id: str = Field(..., alias="userId")
    team_id: Optional[str] = Field(None, alias="teamId")
    cluster_id: str = Field(..., alias="rftClusterId")
    status: str = Field(..., description="Run status")

    # Training configuration
    rollouts_per_example: int = Field(..., alias="rolloutsPerExample")
    seq_len: int = Field(..., alias="seqLen")
    max_steps: int = Field(..., alias="maxSteps")
    save_steps: Optional[int] = Field(None, alias="saveSteps")
    base_model: str = Field(..., alias="baseModel")
    environments: List[Dict[str, Any]] = Field(default_factory=list)
    run_config: Optional[Dict[str, Any]] = Field(None, alias="runConfig")
    eval_config: Optional[Dict[str, Any]] = Field(None, alias="evalConfig")

    # Monitoring
    wandb_entity: Optional[str] = Field(None, alias="wandbEntity")
    wandb_project: Optional[str] = Field(None, alias="wandbProject")
    wandb_run_name: Optional[str] = Field(None, alias="wandbRunName")

    # Timestamps
    started_at: Optional[datetime] = Field(None, alias="startedAt")
    completed_at: Optional[datetime] = Field(None, alias="completedAt")
    error_message: Optional[str] = Field(None, alias="errorMessage")
    created_at: datetime = Field(..., alias="createdAt")
    updated_at: datetime = Field(..., alias="updatedAt")

    model_config = ConfigDict(populate_by_name=True)


class RLClient:
    """Client for hosted RL API."""

    def __init__(self, client: APIClient) -> None:
        self.client = client

    def list_models(self) -> List[RLModel]:
        """List available models for RL training."""
        try:
            response = self.client.get("/rft/models")
            models_data = response.get("models", [])
            return [RLModel.model_validate(model) for model in models_data]
        except Exception as e:
            if hasattr(e, "response") and hasattr(e.response, "text"):
                raise APIError(f"Failed to list RL models: {e.response.text}")
            raise APIError(f"Failed to list RL models: {str(e)}")

    def list_runs(self, team_id: Optional[str] = None) -> List[RLRun]:
        """List RL training runs for the authenticated user."""
        try:
            params = {}
            if team_id:
                params["team_id"] = team_id
            response = self.client.get("/rft/runs", params=params if params else None)
            runs_data = response.get("runs", [])
            return [RLRun.model_validate(run) for run in runs_data]
        except Exception as e:
            if hasattr(e, "response") and hasattr(e.response, "text"):
                raise APIError(f"Failed to list RL runs: {e.response.text}")
            raise APIError(f"Failed to list RL runs: {str(e)}")

    def create_run(
        self,
        model_name: str,
        environments: List[Dict[str, Any]],
        rollouts_per_example: int = 8,
        seq_len: int = 4096,
        max_steps: int = 100,
        save_steps: Optional[int] = None,
        name: Optional[str] = None,
        wandb_entity: Optional[str] = None,
        wandb_project: Optional[str] = None,
        wandb_run_name: Optional[str] = None,
        wandb_api_key: Optional[str] = None,
        team_id: Optional[str] = None,
        run_config: Optional[Dict[str, Any]] = None,
        eval_config: Optional[Dict[str, Any]] = None,
    ) -> RLRun:
        """Create a new RL training run."""
        try:
            secrets: List[Dict[str, str]] = []

            # Add W&B API key as a secret if provided
            if wandb_api_key:
                secrets.append({"key": "WANDB_API_KEY", "value": wandb_api_key})

            payload: Dict[str, Any] = {
                "model": {"name": model_name},
                "environments": environments,
                "rollouts_per_example": rollouts_per_example,
                "seq_len": seq_len,
                "max_steps": max_steps,
                "secrets": secrets,
            }

            if save_steps is not None:
                payload["save_steps"] = save_steps

            if name:
                payload["name"] = name

            # Add monitoring config if W&B is specified
            if wandb_entity or wandb_project:
                payload["monitoring"] = {
                    "wandb": {
                        "entity": wandb_entity,
                        "project": wandb_project,
                        "name": wandb_run_name,
                    }
                }

            if team_id:
                payload["team_id"] = team_id

            if run_config:
                payload["run_config"] = run_config

            if eval_config:
                payload["eval"] = eval_config

            response = self.client.post("/rft/runs", json=payload)
            return RLRun.model_validate(response.get("run"))
        except Exception as e:
            if hasattr(e, "response") and hasattr(e.response, "text"):
                raise APIError(f"Failed to create RL run: {e.response.text}")
            raise APIError(f"Failed to create RL run: {str(e)}")

    def stop_run(self, run_id: str) -> RLRun:
        """Stop a running RL training run."""
        try:
            response = self.client.request("PUT", f"/rft/runs/{run_id}/stop")
            return RLRun.model_validate(response.get("run"))
        except Exception as e:
            if hasattr(e, "response") and hasattr(e.response, "text"):
                raise APIError(f"Failed to stop RL run: {e.response.text}")
            raise APIError(f"Failed to stop RL run: {str(e)}")

    def delete_run(self, run_id: str) -> None:
        """Delete an RL run."""
        try:
            self.client.delete(f"/rft/runs/{run_id}")
        except Exception as e:
            if hasattr(e, "response") and hasattr(e.response, "text"):
                raise APIError(f"Failed to delete RL run: {e.response.text}")
            raise APIError(f"Failed to delete RL run: {str(e)}")

    def get_logs(self, run_id: str, tail_lines: int = 1000) -> str:
        """Get logs for an RL run."""
        try:
            response = self.client.get(
                f"/rft/runs/{run_id}/logs", params={"tail_lines": tail_lines}
            )
            return response.get("logs", "")
        except Exception as e:
            if hasattr(e, "response") and hasattr(e.response, "text"):
                raise APIError(f"Failed to get RL run logs: {e.response.text}")
            raise APIError(f"Failed to get RL run logs: {str(e)}")
