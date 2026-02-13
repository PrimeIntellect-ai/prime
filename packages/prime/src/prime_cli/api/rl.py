"""Hosted RL (Reinforcement Learning) API client."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from prime_cli.core import APIClient, APIError, ValidationError


class RLModel(BaseModel):
    """Model available for RL training."""

    name: str = Field(..., description="Model name")
    at_capacity: bool = Field(False, alias="atCapacity")

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
    max_tokens: Optional[int] = Field(None, alias="maxTokens")
    batch_size: int = Field(..., alias="batchSize")
    base_model: str = Field(..., alias="baseModel")
    environments: List[Dict[str, Any]] = Field(default_factory=list)
    run_config: Optional[Dict[str, Any]] = Field(None, alias="runConfig")
    eval_config: Optional[Dict[str, Any]] = Field(None, alias="evalConfig")
    val_config: Optional[Dict[str, Any]] = Field(None, alias="valConfig")
    buffer_config: Optional[Dict[str, Any]] = Field(None, alias="bufferConfig")
    learning_rate: Optional[float] = Field(None, alias="learningRate")
    lora_alpha: Optional[int] = Field(None, alias="loraAlpha")
    oversampling_factor: Optional[float] = Field(None, alias="oversamplingFactor")
    max_async_level: Optional[int] = Field(None, alias="maxAsyncLevel")

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


class RLCheckpoint(BaseModel):
    """Checkpoint from an RL training run."""

    id: str = Field(..., description="Checkpoint ID")
    rft_run_id: str = Field(..., alias="rftRunId")
    step: int = Field(..., description="Training step number")
    storage_url: str = Field(..., alias="storageUrl")
    status: str = Field(..., description="Checkpoint status")
    size_bytes: Optional[int] = Field(None, alias="sizeBytes")
    created_at: datetime = Field(..., alias="createdAt")
    uploaded_at: Optional[datetime] = Field(None, alias="uploadedAt")

    model_config = ConfigDict(populate_by_name=True)


class RLClient:
    """Client for hosted RL API."""

    def __init__(self, client: APIClient) -> None:
        self.client = client

    def list_models(self, team_id: Optional[str] = None) -> List[RLModel]:
        """List available models for RL training."""
        try:
            params = {}
            if team_id:
                params["team_id"] = team_id
            response = self.client.get("/rft/models", params=params if params else None)
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
        max_steps: int = 100,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        batch_size: int = 128,
        name: Optional[str] = None,
        wandb_entity: Optional[str] = None,
        wandb_project: Optional[str] = None,
        wandb_run_name: Optional[str] = None,
        secrets: Optional[Dict[str, str]] = None,
        team_id: Optional[str] = None,
        eval_config: Optional[Dict[str, Any]] = None,
        val_config: Optional[Dict[str, Any]] = None,
        buffer_config: Optional[Dict[str, Any]] = None,
        learning_rate: Optional[float] = None,
        lora_alpha: Optional[int] = None,
        oversampling_factor: Optional[float] = None,
        max_async_level: Optional[int] = None,
        checkpoints_config: Optional[Dict[str, Any]] = None,
        adapters_config: Optional[Dict[str, Any]] = None,
        checkpoint_id: Optional[str] = None,
        cluster_name: Optional[str] = None,
    ) -> RLRun:
        """Create a new RL training run."""
        try:
            secrets_list: List[Dict[str, str]] = []
            if secrets:
                for key, value in secrets.items():
                    secrets_list.append({"key": key, "value": value})

            payload: Dict[str, Any] = {
                "model": {"name": model_name},
                "environments": environments,
                "rollouts_per_example": rollouts_per_example,
                "max_steps": max_steps,
                "batch_size": batch_size,
                "secrets": secrets_list,
            }

            if name:
                payload["name"] = name

            # Add monitoring config if any W&B field is set (backend validates completeness)
            if wandb_entity or wandb_project or wandb_run_name:
                wandb_config: Dict[str, Any] = {}
                if wandb_entity:
                    wandb_config["entity"] = wandb_entity
                if wandb_project:
                    wandb_config["project"] = wandb_project
                if wandb_run_name:
                    wandb_config["name"] = wandb_run_name
                payload["monitoring"] = {"wandb": wandb_config}

            if team_id:
                payload["team_id"] = team_id

            if max_tokens:
                payload["max_tokens"] = max_tokens

            if temperature is not None:
                payload["temperature"] = temperature

            if eval_config:
                payload["eval"] = eval_config

            if val_config:
                payload["val"] = val_config

            if buffer_config:
                payload["buffer"] = buffer_config

            if learning_rate is not None:
                payload["learning_rate"] = learning_rate

            if lora_alpha is not None:
                payload["lora_alpha"] = lora_alpha

            if oversampling_factor is not None:
                payload["oversampling_factor"] = oversampling_factor

            if max_async_level is not None:
                payload["max_async_level"] = max_async_level

            if checkpoints_config:
                payload["checkpoints"] = checkpoints_config

            if adapters_config:
                payload["adapters"] = adapters_config

            if checkpoint_id:
                payload["checkpoint_id"] = checkpoint_id

            if cluster_name:
                payload["cluster_name"] = cluster_name

            response = self.client.post("/rft/runs", json=payload)
            return RLRun.model_validate(response.get("run"))
        except ValidationError:
            raise  # Let ValidationError pass through for proper CLI handling
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

    def restart_run(self, run_id: str) -> RLRun:
        """Restart a running RL training run from its latest checkpoint.

        Only RUNNING runs can be restarted (checkpoints still on PVC).
        For STOPPED/FAILED/COMPLETED runs, checkpoints have been cleaned up.
        """
        try:
            response = self.client.request("PUT", f"/rft/runs/{run_id}/restart")
            return RLRun.model_validate(response.get("run"))
        except Exception as e:
            if hasattr(e, "response") and hasattr(e.response, "text"):
                raise APIError(f"Failed to restart RL run: {e.response.text}")
            raise APIError(f"Failed to restart RL run: {str(e)}")

    def list_checkpoints(
        self, run_id: str, status_filter: Optional[str] = None
    ) -> List[RLCheckpoint]:
        """List checkpoints for an RL run."""
        try:
            params: Dict[str, str] = {}
            if status_filter:
                params["status_filter"] = status_filter
            response = self.client.get(
                f"/rft/runs/{run_id}/checkpoints",
                params=params if params else None,
            )
            checkpoints_data = response.get("checkpoints", [])
            return [RLCheckpoint.model_validate(cp) for cp in checkpoints_data]
        except Exception as e:
            if hasattr(e, "response") and hasattr(e.response, "text"):
                raise APIError(f"Failed to list checkpoints: {e.response.text}")
            raise APIError(f"Failed to list checkpoints: {str(e)}")

    def get_run(self, run_id: str) -> RLRun:
        """Get details of a specific RL run."""
        try:
            response = self.client.get(f"/rft/runs/{run_id}")
            return RLRun.model_validate(response.get("run"))
        except Exception as e:
            if hasattr(e, "response") and hasattr(e.response, "text"):
                raise APIError(f"Failed to get RL run: {e.response.text}")
            raise APIError(f"Failed to get RL run: {str(e)}")

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

    def get_metrics(
        self,
        run_id: str,
        min_step: Optional[int] = None,
        max_step: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Get metrics for an RL run."""
        try:
            params: Dict[str, Any] = {}
            if min_step is not None:
                params["min_step"] = min_step
            if max_step is not None:
                params["max_step"] = max_step
            if limit is not None:
                params["limit"] = limit

            response = self.client.get(
                f"/rft/runs/{run_id}/metrics", params=params if params else None
            )
            return response.get("metrics", [])
        except Exception as e:
            if hasattr(e, "response") and hasattr(e.response, "text"):
                raise APIError(f"Failed to get RL run metrics: {e.response.text}")
            raise APIError(f"Failed to get RL run metrics: {str(e)}")

    def get_rollouts(
        self,
        run_id: str,
        step: int,
        page: int = 1,
        limit: int = 100,
    ) -> Dict[str, Any]:
        """Get rollout samples for an RL run."""
        try:
            params: Dict[str, Any] = {
                "page": page,
                "limit": limit,
                "step": step,
            }

            response = self.client.get(f"/rft/runs/{run_id}/samples", params=params)
            return {
                "run_id": response.get("run_id", run_id),
                "samples": response.get("samples", []),
                "total": response.get("total", 0),
                "page": response.get("page", page),
                "limit": response.get("limit", limit),
                "total_pages": response.get("total_pages", 0),
            }
        except Exception as e:
            if hasattr(e, "response") and hasattr(e.response, "text"):
                raise APIError(f"Failed to get RL run rollouts: {e.response.text}")
            raise APIError(f"Failed to get RL run rollouts: {str(e)}")

    def get_progress(self, run_id: str) -> Dict[str, Any]:
        """Get progress information for an RL run."""
        try:
            response = self.client.get(f"/rft/runs/{run_id}/progress")
            return {
                "latest_step": response.get("latestStep"),
                "steps_with_samples": response.get("stepsWithSamples", []),
                "steps_with_distributions": response.get("stepsWithDistributions", []),
                "last_updated_at": response.get("lastUpdatedAt"),
            }
        except Exception as e:
            if hasattr(e, "response") and hasattr(e.response, "text"):
                raise APIError(f"Failed to get RL run progress: {e.response.text}")
            raise APIError(f"Failed to get RL run progress: {str(e)}")

    def get_distributions(
        self,
        run_id: str,
        distribution_type: Optional[str] = None,
        step: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Get reward/advantage distribution histogram for an RL run."""
        try:
            params: Dict[str, Any] = {}
            if distribution_type is not None:
                params["type"] = distribution_type
            if step is not None:
                params["step"] = step

            response = self.client.get(f"/rft/runs/{run_id}/distributions", params=params)
            return {
                "bins": response.get("bins", []),
                "step": response.get("step"),
            }
        except Exception as e:
            if hasattr(e, "response") and hasattr(e.response, "text"):
                raise APIError(f"Failed to get RL run distributions: {e.response.text}")
            raise APIError(f"Failed to get RL run distributions: {str(e)}")

    def get_environment_status(self, owner: str, name: str) -> Dict[str, Any]:
        """Get status for an environment including latest version and action info."""
        try:
            response = self.client.get(f"/environmentshub/{owner}/{name}/status")
            return response.get("data") or {}
        except Exception as e:
            if hasattr(e, "response") and hasattr(e.response, "text"):
                raise APIError(f"Failed to get status for {owner}/{name}: {e.response.text}")
            raise APIError(f"Failed to get status for {owner}/{name}: {str(e)}")
