"""RFT (Reinforcement Fine-Tuning) API client."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from prime_cli.core import APIClient, APIError


class RFTModel(BaseModel):
    """Model available for RFT training."""

    name: str = Field(..., description="Model name")

    model_config = ConfigDict(populate_by_name=True)


class RFTRun(BaseModel):
    """RFT Training Run."""

    id: str = Field(..., description="Run ID")
    user_id: str = Field(..., alias="userId")
    team_id: Optional[str] = Field(None, alias="teamId")
    cluster_id: str = Field(..., alias="rftClusterId")
    status: str = Field(..., description="Run status")

    # Training configuration
    rollouts_per_example: int = Field(..., alias="rolloutsPerExample")
    seq_len: int = Field(..., alias="seqLen")
    max_steps: int = Field(..., alias="maxSteps")
    model_name: str = Field(..., alias="modelName")
    environments: List[Dict[str, Any]] = Field(default_factory=list)
    run_config: Optional[Dict[str, Any]] = Field(None, alias="runConfig")

    # Monitoring
    wandb_project: Optional[str] = Field(None, alias="wandbProject")
    wandb_run_name: Optional[str] = Field(None, alias="wandbRunName")

    # Timestamps
    started_at: Optional[datetime] = Field(None, alias="startedAt")
    completed_at: Optional[datetime] = Field(None, alias="completedAt")
    error_message: Optional[str] = Field(None, alias="errorMessage")
    created_at: datetime = Field(..., alias="createdAt")
    updated_at: datetime = Field(..., alias="updatedAt")

    model_config = ConfigDict(populate_by_name=True)


class RFTClient:
    """Client for RFT (Reinforcement Fine-Tuning) API."""

    def __init__(self, client: APIClient) -> None:
        self.client = client

    def list_models(self) -> List[RFTModel]:
        """List available models for RFT training."""
        try:
            response = self.client.get("/rft/models")
            models_data = response.get("models", [])
            return [RFTModel.model_validate(model) for model in models_data]
        except Exception as e:
            if hasattr(e, "response") and hasattr(e.response, "text"):
                raise APIError(f"Failed to list RFT models: {e.response.text}")
            raise APIError(f"Failed to list RFT models: {str(e)}")

    def list_runs(self, team_id: Optional[str] = None) -> List[RFTRun]:
        """List RFT training runs for the authenticated user."""
        try:
            params = {}
            if team_id:
                params["team_id"] = team_id
            response = self.client.get("/rft/runs", params=params if params else None)
            runs_data = response.get("runs", [])
            return [RFTRun.model_validate(run) for run in runs_data]
        except Exception as e:
            if hasattr(e, "response") and hasattr(e.response, "text"):
                raise APIError(f"Failed to list RFT runs: {e.response.text}")
            raise APIError(f"Failed to list RFT runs: {str(e)}")

    def create_run(
        self,
        model_name: str,
        environments: List[Dict[str, Any]],
        rollouts_per_example: int = 8,
        seq_len: int = 4096,
        max_steps: int = 100,
        wandb_project: Optional[str] = None,
        wandb_run_name: Optional[str] = None,
        wandb_api_key: Optional[str] = None,
        team_id: Optional[str] = None,
        run_config: Optional[Dict[str, Any]] = None,
    ) -> RFTRun:
        """Create a new RFT training run."""
        try:
            payload: Dict[str, Any] = {
                "model": {"name": model_name},
                "environments": environments,
                "rollouts_per_example": rollouts_per_example,
                "seq_len": seq_len,
                "max_steps": max_steps,
                "secrets": [],
            }

            # Add monitoring config if W&B is specified
            if wandb_project:
                payload["monitoring"] = {
                    "wandb": {
                        "project": wandb_project,
                        "name": wandb_run_name,
                    }
                }

            # Add W&B API key as a secret if provided
            if wandb_api_key:
                payload["secrets"].append({"key": "WANDB_API_KEY", "value": wandb_api_key})

            if team_id:
                payload["team_id"] = team_id

            if run_config:
                payload["run_config"] = run_config

            response = self.client.post("/rft/runs", json=payload)
            return RFTRun.model_validate(response.get("run"))
        except Exception as e:
            if hasattr(e, "response") and hasattr(e.response, "text"):
                raise APIError(f"Failed to create RFT run: {e.response.text}")
            raise APIError(f"Failed to create RFT run: {str(e)}")

    def stop_run(self, run_id: str) -> RFTRun:
        """Stop a running RFT training run."""
        try:
            response = self.client.request("PUT", f"/rft/runs/{run_id}/stop")
            return RFTRun.model_validate(response.get("run"))
        except Exception as e:
            if hasattr(e, "response") and hasattr(e.response, "text"):
                raise APIError(f"Failed to stop RFT run: {e.response.text}")
            raise APIError(f"Failed to stop RFT run: {str(e)}")

    def delete_run(self, run_id: str) -> bool:
        """Delete an RFT run."""
        try:
            response = self.client.delete(f"/rft/runs/{run_id}")
            return response.get("success", False)
        except Exception as e:
            if hasattr(e, "response") and hasattr(e.response, "text"):
                raise APIError(f"Failed to delete RFT run: {e.response.text}")
            raise APIError(f"Failed to delete RFT run: {str(e)}")
