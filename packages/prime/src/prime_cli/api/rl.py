"""Hosted Training API client."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from prime_cli.core import APIClient, APIError


class RLRun(BaseModel):
    """Hosted Training run."""

    id: str = Field(..., description="Run ID")
    name: Optional[str] = Field(None, description="Run name")
    user_id: str = Field(..., alias="userId")
    team_id: Optional[str] = Field(None, alias="teamId")
    cluster_id: Optional[str] = Field(None, alias="rftClusterId")
    # DEDICATED_FULL_FT: the PrimeCluster hosting the run's helm release.
    # The backend writes rftClusterId NULL for dedicated runs, so this is
    # the only cluster field they carry (shared runs use cluster_id).
    prime_cluster_id: Optional[str] = Field(None, alias="primeClusterId")
    status: str = Field(..., description="Run status")
    # Discriminator: SHARED_RFT_HOSTED (LoRA) | DEDICATED_FULL_FT (own
    # helm release on a PrimeCluster) | EXTERNAL (CLI-side prime-rl).
    # Optional for backward-compat with older API versions.
    kind: Optional[str] = Field(None, description="Run kind discriminator")

    # Training configuration
    rollouts_per_example: Optional[int] = Field(None, alias="rolloutsPerExample")
    seq_len: Optional[int] = Field(None, alias="seqLen")
    max_steps: Optional[int] = Field(None, alias="maxSteps")
    max_tokens: Optional[int] = Field(None, alias="maxTokens")
    batch_size: Optional[int] = Field(None, alias="batchSize")
    loss: Optional[str] = "rl"
    teacher: Optional[Dict[str, Any]] = Field(
        None,
        validation_alias=AliasChoices("teacher", "teacherConfig"),
        serialization_alias="teacher",
    )
    base_model: Optional[str] = Field(None, alias="baseModel")
    environments: List[Dict[str, Any]] = Field(default_factory=list)
    run_config: Optional[Dict[str, Any]] = Field(None, alias="runConfig")
    eval_config: Optional[Dict[str, Any]] = Field(None, alias="evalConfig")
    val_config: Optional[Dict[str, Any]] = Field(None, alias="valConfig")
    # DEPRECATED: buffer removed in prime-rl orch v2; populated only on historical runs
    buffer_config: Optional[Dict[str, Any]] = Field(None, alias="bufferConfig")
    learning_rate: Optional[float] = Field(None, alias="learningRate")
    lora_alpha: Optional[int] = Field(None, alias="loraAlpha")
    max_inflight_rollouts: Optional[int] = Field(None, alias="maxInflightRollouts")
    oversampling_factor: Optional[float] = Field(None, alias="oversamplingFactor")
    max_async_level: Optional[int] = Field(None, alias="maxAsyncLevel")
    # User-facing name of the named volume a DEDICATED_FULL_FT run
    # mounted via --volume (outputs under runs/<runId>/ on it). Only
    # set on the run-detail read; None for runs without a volume.
    volume_name: Optional[str] = Field(None, alias="volumeName")
    # Resolved GPU footprint (trainer + inference). NULL on legacy rows
    # that pre-date the column.
    gpu_count: Optional[int] = Field(None, alias="gpuCount")

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

    # Full-FT source overlay: the prime-rl commit the pods run on top of the
    # image when the run was dispatched with `--ref` / `--pr`. None for runs
    # that use the image's baked source.
    source_commit: Optional[str] = Field(None, alias="sourceCommit")

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


class RLRunPage(BaseModel):
    """One page of GET /rft/runs, as returned by the backend.

    The pagination metadata fields are Optional: a backend that hasn't
    deployed pagination support yet silently ignores the page/limit/mine
    query params (FastAPI drops undeclared query params rather than
    erroring) and returns the old `{"runs": [...]}` shape - the caller
    must detect that (via `total is None`) and fall back to client-side
    filtering/pagination over the full list it got back.
    """

    runs: List[RLRun] = Field(default_factory=list)
    total: Optional[int] = Field(None, description="Total number of matching runs")
    page: Optional[int] = Field(None, description="Current page number")
    limit: Optional[int] = Field(None, description="Runs per page")
    total_pages: Optional[int] = Field(None, alias="total_pages")

    model_config = ConfigDict(populate_by_name=True)


class RLClient:
    """Client for the Hosted Training API."""

    def __init__(self, client: APIClient) -> None:
        self.client = client

    def list_runs(
        self,
        team_id: Optional[str] = None,
        *,
        mine: bool = False,
        page: int = 1,
        limit: int = 20,
    ) -> RLRunPage:
        """List Hosted Training runs for the authenticated user, one page
        at a time - pagination and the `mine` filter are applied
        server-side rather than fetched-in-full-then-sliced locally.
        """
        try:
            params: Dict[str, Any] = {"page": page, "limit": limit}
            if team_id:
                params["team_id"] = team_id
            if mine:
                params["mine"] = "true"
            response = self.client.get("/rft/runs", params=params)
            return RLRunPage.model_validate(response)
        except Exception as e:
            if hasattr(e, "response") and hasattr(e.response, "text"):
                raise APIError(f"Failed to list Hosted Training runs: {e.response.text}")
            raise APIError(f"Failed to list Hosted Training runs: {str(e)}")

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

    def get_run(self, run_id: str, timeout: Optional[int] = None) -> RLRun:
        """Get details of a specific Hosted Training run.

        `timeout` bounds this specific request (e.g. so a caller polling
        against a wall-clock deadline can't have a single stalled request
        blow past it) - the client-wide default applies when omitted.
        """
        try:
            # Only forward `timeout` when a caller actually set it, so
            # existing callers/mocks that don't expect the kwarg at all
            # see the exact same call shape as before.
            kwargs = {"timeout": timeout} if timeout is not None else {}
            response = self.client.get(f"/rft/runs/{run_id}", **kwargs)
            return RLRun.model_validate(response.get("run"))
        except APIError:
            # Preserve typed subclasses (NotFoundError, UnauthorizedError, …)
            # so callers like `prime train delete` can branch on the run-
            # missing case rather than string-matching a wrapped message.
            raise
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
        component: Optional[str] = None,
        pod_index: Optional[int] = None,
        env_name: Optional[str] = None,
    ) -> str:
        """Get logs for one component of a Hosted Training run.

        Defaults to the orchestrator pod. Dedicated full-FT runs additionally
        expose `trainer`, `inference`, and `env-server` components.
        `pod_index` narrows to a specific replica for multi-node
        trainer/inference; `env_name` picks among per-env env-server
        StatefulSets when `component='env-server'`.

        Optional filters narrow the result via the platform's log search
        backend:
          - search: substring (or regex if regex=True) line filter
          - level:  one of ERROR/WARNING/SUCCESS/INFO/DEBUG
          - since_seconds: how far back to look (60-86400)
        """
        params: Dict[str, Any] = {"tail_lines": tail_lines}
        if search:
            params["search"] = search
        if regex:
            params["regex"] = True
        if level:
            params["level"] = level
        if since_seconds is not None:
            params["since_seconds"] = since_seconds
        if component:
            params["component"] = component
        if pod_index is not None:
            params["pod_index"] = pod_index
        if env_name:
            params["env_name"] = env_name
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
