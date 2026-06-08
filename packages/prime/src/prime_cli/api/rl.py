"""Hosted Training API client."""

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from prime_cli.core import APIClient, APIError, ValidationError


class RLModel(BaseModel):
    """Model available for Hosted Training."""

    name: str = Field(..., description="Model name")
    at_capacity: bool = Field(False, alias="atCapacity")
    training_price_per_mtok: Optional[float] = Field(None, alias="trainingPricePerMtok")
    inference_input_price_per_mtok: Optional[float] = Field(
        None, alias="inferenceInputPricePerMtok"
    )
    inference_output_price_per_mtok: Optional[float] = Field(
        None, alias="inferenceOutputPricePerMtok"
    )
    list_training_price_per_mtok: Optional[float] = Field(None, alias="listTrainingPricePerMtok")
    list_inference_input_price_per_mtok: Optional[float] = Field(
        None, alias="listInferenceInputPricePerMtok"
    )
    list_inference_output_price_per_mtok: Optional[float] = Field(
        None, alias="listInferenceOutputPricePerMtok"
    )
    effective_training_price_per_mtok: Optional[float] = Field(
        None, alias="effectiveTrainingPricePerMtok"
    )
    effective_inference_input_price_per_mtok: Optional[float] = Field(
        None, alias="effectiveInferenceInputPricePerMtok"
    )
    effective_inference_output_price_per_mtok: Optional[float] = Field(
        None, alias="effectiveInferenceOutputPricePerMtok"
    )
    promo_label: Optional[str] = Field(None, alias="promoLabel")

    model_config = ConfigDict(populate_by_name=True)

    def resolve_prices(self, category: str) -> tuple[Optional[float], Optional[float]]:
        """Return ``(list_price, effective_price)`` for ``category``.

        Categories: ``training``, ``inference_input``, ``inference_output``.

        Post-swap backends populate ``list_*`` with the un-discounted price and
        the legacy ``*_price_per_mtok`` with the effective (charged) price.
        Pre-swap backends omit ``list_*`` and keep ``legacy = list``,
        ``effective = charged``.
        """
        list_attr = f"list_{category}_price_per_mtok"
        legacy_attr = f"{category}_price_per_mtok"
        effective_attr = f"effective_{category}_price_per_mtok"
        list_value = getattr(self, list_attr)
        if list_value is not None:
            return list_value, getattr(self, legacy_attr)
        return getattr(self, legacy_attr), getattr(self, effective_attr)


class RLRun(BaseModel):
    """Hosted Training run."""

    id: str = Field(..., description="Run ID")
    name: Optional[str] = Field(None, description="Run name")
    user_id: str = Field(..., alias="userId")
    team_id: Optional[str] = Field(None, alias="teamId")
    project_id: Optional[str] = Field(None, alias="projectId")
    cluster_id: Optional[str] = Field(None, alias="rftClusterId")
    status: str = Field(..., description="Run status")

    # Training configuration
    rollouts_per_example: int = Field(..., alias="rolloutsPerExample")
    seq_len: int = Field(..., alias="seqLen")
    max_steps: int = Field(..., alias="maxSteps")
    max_tokens: Optional[int] = Field(None, alias="maxTokens")
    batch_size: int = Field(..., alias="batchSize")
    loss: Optional[str] = "rl"
    teacher: Optional[Dict[str, Any]] = Field(
        None,
        validation_alias=AliasChoices("teacher", "teacherConfig"),
        serialization_alias="teacher",
    )
    base_model: str = Field(..., alias="baseModel")
    environments: List[Dict[str, Any]] = Field(default_factory=list)
    run_config: Optional[Dict[str, Any]] = Field(None, alias="runConfig")
    eval_config: Optional[Dict[str, Any]] = Field(None, alias="evalConfig")
    val_config: Optional[Dict[str, Any]] = Field(None, alias="valConfig")
    buffer_config: Optional[Dict[str, Any]] = Field(None, alias="bufferConfig")
    learning_rate: Optional[float] = Field(None, alias="learningRate")
    lora_alpha: Optional[int] = Field(None, alias="loraAlpha")
    max_inflight_rollouts: Optional[int] = Field(None, alias="maxInflightRollouts")
    oversampling_factor: Optional[float] = Field(None, alias="oversamplingFactor")
    max_async_level: Optional[int] = Field(None, alias="maxAsyncLevel")

    # Monitoring
    wandb_entity: Optional[str] = Field(None, alias="wandbEntity")
    wandb_project: Optional[str] = Field(None, alias="wandbProject")
    wandb_run_name: Optional[str] = Field(None, alias="wandbRunName")

    # Queue info
    runs_ahead: Optional[int] = Field(None, alias="runsAhead")
    queue_reason: Optional[str] = Field(None, alias="queueReason")
    notice: Optional[str] = Field(None, alias="notice")

    # Timestamps
    started_at: Optional[datetime] = Field(None, alias="startedAt")
    completed_at: Optional[datetime] = Field(None, alias="completedAt")
    error_message: Optional[str] = Field(None, alias="errorMessage")
    created_at: datetime = Field(..., alias="createdAt")
    updated_at: datetime = Field(..., alias="updatedAt")

    # Automated failure classification (only set for terminal FAILED runs).
    # Stored as a dict to stay forward-compatible if the API adds new fields.
    failure_analysis: Optional[Dict[str, Any]] = Field(None, alias="failureAnalysis")

    model_config = ConfigDict(populate_by_name=True)


class RLCheckpoint(BaseModel):
    """Checkpoint from a Hosted Training run."""

    id: str = Field(..., description="Checkpoint ID")
    rft_run_id: str = Field(..., alias="rftRunId")
    step: int = Field(..., description="Training step number")
    storage_url: str = Field(..., alias="storageUrl")
    status: str = Field(..., description="Checkpoint status")
    size_bytes: Optional[int] = Field(None, alias="sizeBytes")
    created_at: datetime = Field(..., alias="createdAt")
    uploaded_at: Optional[datetime] = Field(None, alias="uploadedAt")

    model_config = ConfigDict(populate_by_name=True)


class EnvServerInfo(BaseModel):
    """Env-server pod info for a Hosted Training run."""

    env_name: Optional[str] = Field(None, description="Environment name")
    env_index: Optional[int] = Field(None, description="Environment server index")
    pod_name: str = Field(..., description="Kubernetes pod name")
    status: str = Field(..., description="Pod status")

    model_config = ConfigDict(populate_by_name=True)


class RLClient:
    """Client for the Hosted Training API."""

    def __init__(self, client: APIClient) -> None:
        self.client = client

    def list_models(self, team_id: Optional[str] = None) -> List[RLModel]:
        """List available models for Hosted Training."""
        try:
            params = {}
            if team_id:
                params["team_id"] = team_id
            response = self.client.get("/rft/models", params=params if params else None)
            models_data = response.get("models", [])
            return [RLModel.model_validate(model) for model in models_data]
        except Exception as e:
            if hasattr(e, "response") and hasattr(e.response, "text"):
                raise APIError(f"Failed to list Hosted Training models: {e.response.text}")
            raise APIError(f"Failed to list Hosted Training models: {str(e)}")

    def list_runs(self, team_id: Optional[str] = None) -> List[RLRun]:
        """List Hosted Training runs for the authenticated user."""
        try:
            params = {}
            if team_id:
                params["team_id"] = team_id
            response = self.client.get("/rft/runs", params=params if params else None)
            runs_data = response.get("runs", [])
            return [RLRun.model_validate(run) for run in runs_data]
        except Exception as e:
            if hasattr(e, "response") and hasattr(e.response, "text"):
                raise APIError(f"Failed to list Hosted Training runs: {e.response.text}")
            raise APIError(f"Failed to list Hosted Training runs: {str(e)}")

    def create_run(
        self,
        model_name: str,
        environments: List[Dict[str, Any]],
        rollouts_per_example: int = 8,
        max_steps: int = 100,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        temp_scheduler: Optional[Dict[str, Any]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        batch_size: int = 128,
        name: Optional[str] = None,
        wandb_entity: Optional[str] = None,
        wandb_project: Optional[str] = None,
        wandb_run_name: Optional[str] = None,
        secrets: Optional[Dict[str, str]] = None,
        team_id: Optional[str] = None,
        project_id: Optional[str] = None,
        eval_config: Optional[Dict[str, Any]] = None,
        val_config: Optional[Dict[str, Any]] = None,
        buffer_config: Optional[Dict[str, Any]] = None,
        learning_rate: Optional[float] = None,
        lora_alpha: Optional[int] = None,
        max_inflight_rollouts: Optional[int] = None,
        oversampling_factor: Optional[float] = None,
        checkpoints_config: Optional[Dict[str, Any]] = None,
        adapters_config: Optional[Dict[str, Any]] = None,
        checkpoint_id: Optional[str] = None,
        cluster_name: Optional[str] = None,
        infrastructure_config: Optional[Dict[str, Any]] = None,
        tailscale_config: Optional[Dict[str, Any]] = None,
        enable_thinking: Optional[bool] = None,
        reasoning_effort: Optional[Literal["low", "medium", "high"]] = None,
        run_config: Optional[Dict[str, Any]] = None,
        loss: str = "rl",
        teacher: Optional[Dict[str, Any]] = None,
    ) -> RLRun:
        """Create a new Hosted Training run."""
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

            if loss != "rl":
                payload["loss"] = loss

            if teacher:
                payload["teacher"] = teacher

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

            if project_id:
                payload["project_id"] = project_id

            if max_tokens:
                payload["max_tokens"] = max_tokens

            if temperature is not None:
                payload["temperature"] = temperature

            if temp_scheduler is not None:
                payload["temp_scheduler"] = temp_scheduler

            if extra_body is not None:
                payload["extra_body"] = extra_body

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

            if max_inflight_rollouts is not None:
                payload["max_inflight_rollouts"] = max_inflight_rollouts

            if oversampling_factor is not None:
                payload["oversampling_factor"] = oversampling_factor

            if checkpoints_config:
                payload["checkpoints"] = checkpoints_config

            if adapters_config:
                payload["adapters"] = adapters_config

            if checkpoint_id:
                payload["checkpoint_id"] = checkpoint_id

            if cluster_name:
                payload["cluster_name"] = cluster_name

            if infrastructure_config:
                if "compute_size" in infrastructure_config:
                    payload["compute_size"] = infrastructure_config["compute_size"]

            if tailscale_config:
                payload["tailscale"] = tailscale_config

            if enable_thinking is not None:
                payload["enable_thinking"] = enable_thinking

            if reasoning_effort is not None:
                payload["reasoning_effort"] = reasoning_effort

            if run_config:
                payload["run_config"] = run_config

            response = self.client.post("/rft/runs", json=payload)
            return RLRun.model_validate(response.get("run"))
        except ValidationError:
            raise  # Let ValidationError pass through for proper CLI handling
        except Exception as e:
            if hasattr(e, "response") and hasattr(e.response, "text"):
                raise APIError(f"Failed to create Hosted Training run: {e.response.text}")
            raise APIError(f"Failed to create Hosted Training run: {str(e)}")

    def preview_run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Preview or validate an RL run payload without creating a run."""

        try:
            return self.client.post("/rft/runs/preview", json=payload)
        except Exception as e:
            if hasattr(e, "response") and hasattr(e.response, "text"):
                raise APIError(f"Failed to preview RL run: {e.response.text}")
            raise APIError(f"Failed to preview RL run: {str(e)}")

    def stop_run(self, run_id: str) -> RLRun:
        """Stop a running Hosted Training run."""
        try:
            response = self.client.request("PUT", f"/rft/runs/{run_id}/stop")
            return RLRun.model_validate(response.get("run"))
        except Exception as e:
            if hasattr(e, "response") and hasattr(e.response, "text"):
                raise APIError(f"Failed to stop Hosted Training run: {e.response.text}")
            raise APIError(f"Failed to stop Hosted Training run: {str(e)}")

    def delete_run(self, run_id: str) -> None:
        """Delete a Hosted Training run."""
        try:
            self.client.delete(f"/rft/runs/{run_id}")
        except Exception as e:
            if hasattr(e, "response") and hasattr(e.response, "text"):
                raise APIError(f"Failed to delete Hosted Training run: {e.response.text}")
            raise APIError(f"Failed to delete Hosted Training run: {str(e)}")

    def restart_run(self, run_id: str) -> RLRun:
        """Restart a running Hosted Training run from its latest checkpoint.

        Only RUNNING runs can be restarted (checkpoints still on PVC).
        For STOPPED/FAILED/COMPLETED runs, checkpoints have been cleaned up.
        """
        try:
            response = self.client.request("PUT", f"/rft/runs/{run_id}/restart")
            return RLRun.model_validate(response.get("run"))
        except Exception as e:
            if hasattr(e, "response") and hasattr(e.response, "text"):
                raise APIError(f"Failed to restart Hosted Training run: {e.response.text}")
            raise APIError(f"Failed to restart Hosted Training run: {str(e)}")

    def update_run_project(
        self,
        run_id: str,
        project_id: Optional[str],
        *,
        operation: str = "set",
        move_adapters: bool = True,
    ) -> tuple[RLRun, int]:
        """Update Hosted Training run project memberships."""
        try:
            response = self.client.request(
                "PATCH",
                f"/rft/runs/{run_id}/project",
                json={
                    "projectId": project_id,
                    "operation": operation,
                    "moveAdapters": move_adapters,
                },
            )
            run = RLRun.model_validate(response.get("run"))
            adapters_updated = int(
                response.get("adaptersUpdated", response.get("adapters_updated", 0))
            )
            return run, adapters_updated
        except Exception as e:
            if hasattr(e, "response") and hasattr(e.response, "text"):
                raise APIError(f"Failed to update Hosted Training run project: {e.response.text}")
            raise APIError(f"Failed to update Hosted Training run project: {str(e)}")

    def list_checkpoints(
        self, run_id: str, status_filter: Optional[str] = None
    ) -> List[RLCheckpoint]:
        """List checkpoints for a Hosted Training run."""
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
        """Get details of a specific Hosted Training run."""
        try:
            response = self.client.get(f"/rft/runs/{run_id}")
            return RLRun.model_validate(response.get("run"))
        except Exception as e:
            if hasattr(e, "response") and hasattr(e.response, "text"):
                raise APIError(f"Failed to get Hosted Training run: {e.response.text}")
            raise APIError(f"Failed to get Hosted Training run: {str(e)}")

    def get_logs(
        self,
        run_id: str,
        tail_lines: int = 1000,
        *,
        search: Optional[str] = None,
        regex: bool = False,
        level: Optional[str] = None,
        since_seconds: Optional[int] = None,
    ) -> str:
        """Get orchestrator logs for a Hosted Training run.

        Optional filters narrow the result via the platform's log search
        backend:
          - search: substring (or regex if regex=True) line filter
          - level:  one of ERROR/WARNING/SUCCESS/INFO/DEBUG
          - since_seconds: how far back to look (60–86400)
        """
        params: Dict[str, object] = {"tail_lines": tail_lines}
        if search:
            params["search"] = search
        if regex:
            params["regex"] = True
        if level:
            params["level"] = level
        if since_seconds is not None:
            params["since_seconds"] = since_seconds
        try:
            response = self.client.get(f"/rft/runs/{run_id}/logs", params=params)
            return response.get("logs", "")
        except Exception as e:
            if hasattr(e, "response") and hasattr(e.response, "text"):
                raise APIError(f"Failed to get Hosted Training run logs: {e.response.text}")
            raise APIError(f"Failed to get Hosted Training run logs: {str(e)}")

    def list_env_servers(self, run_id: str) -> List[EnvServerInfo]:
        """List env-server pods for a Hosted Training run."""
        try:
            response = self.client.get(f"/rft/runs/{run_id}/env-servers")
            return [EnvServerInfo.model_validate(p) for p in response.get("env_servers", [])]
        except Exception as e:
            if hasattr(e, "response") and hasattr(e.response, "text"):
                raise APIError(f"Failed to list env servers: {e.response.text}")
            raise APIError(f"Failed to list env servers: {str(e)}")

    def get_env_server_logs(
        self,
        run_id: str,
        env_name: str,
        env_index: int = 0,
        tail_lines: int = 1000,
        *,
        search: Optional[str] = None,
        regex: bool = False,
        level: Optional[str] = None,
        since_seconds: Optional[int] = None,
    ) -> str:
        """Get logs for a specific env-server pod of a Hosted Training run."""
        params: Dict[str, object] = {
            "env_name": env_name,
            "env_index": env_index,
            "tail_lines": tail_lines,
        }
        if search:
            params["search"] = search
        if regex:
            params["regex"] = True
        if level:
            params["level"] = level
        if since_seconds is not None:
            params["since_seconds"] = since_seconds
        try:
            response = self.client.get(
                f"/rft/runs/{run_id}/env-server-logs",
                params=params,
            )
            return response.get("logs", "")
        except Exception as e:
            if hasattr(e, "response") and hasattr(e.response, "text"):
                raise APIError(f"Failed to get env server logs: {e.response.text}")
            raise APIError(f"Failed to get env server logs: {str(e)}")

    def get_metrics(
        self,
        run_id: str,
        min_step: Optional[int] = None,
        max_step: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Get metrics for a Hosted Training run."""
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
                raise APIError(f"Failed to get Hosted Training run metrics: {e.response.text}")
            raise APIError(f"Failed to get Hosted Training run metrics: {str(e)}")

    def get_rollouts(
        self,
        run_id: str,
        step: int,
        page: int = 1,
        limit: int = 100,
    ) -> Dict[str, Any]:
        """Get rollout samples for a Hosted Training run."""
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
                raise APIError(f"Failed to get Hosted Training run rollouts: {e.response.text}")
            raise APIError(f"Failed to get Hosted Training run rollouts: {str(e)}")

    def get_progress(self, run_id: str) -> Dict[str, Any]:
        """Get progress information for a Hosted Training run."""
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
                raise APIError(f"Failed to get Hosted Training run progress: {e.response.text}")
            raise APIError(f"Failed to get Hosted Training run progress: {str(e)}")

    def get_distributions(
        self,
        run_id: str,
        distribution_type: Optional[str] = None,
        step: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Get reward/advantage distribution histogram for a Hosted Training run."""
        try:
            params: Dict[str, Any] = {}
            if distribution_type is not None:
                params["type"] = distribution_type
            if step is not None:
                params["step"] = step

            response = self.client.get(f"/rft/runs/{run_id}/distributions", params=params)
            chart_data = response.get("chartData") or response.get("chart_data")
            bins = response.get("bins")
            if bins is None and isinstance(chart_data, dict):
                bins = chart_data.get("histogramData")
            return {
                "bins": bins or [],
                "step": response.get("step"),
            }
        except Exception as e:
            if hasattr(e, "response") and hasattr(e.response, "text"):
                raise APIError(
                    f"Failed to get Hosted Training run distributions: {e.response.text}"
                )
            raise APIError(f"Failed to get Hosted Training run distributions: {str(e)}")

    def get_environment_status(self, owner: str, name: str) -> Dict[str, Any]:
        """Get status for an environment including latest version and action info."""
        try:
            response = self.client.get(f"/environmentshub/{owner}/{name}/status")
            return response.get("data") or {}
        except Exception as e:
            if hasattr(e, "response") and hasattr(e.response, "text"):
                raise APIError(f"Failed to get status for {owner}/{name}: {e.response.text}")
            raise APIError(f"Failed to get status for {owner}/{name}: {str(e)}")
