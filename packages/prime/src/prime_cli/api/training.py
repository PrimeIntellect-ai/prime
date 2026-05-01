"""Hosted full-FT training API client (POST/DELETE /v1/training/runs).

Sibling to api/rl.py — that's the LoRA-shared path. This client speaks to
the dedicated full-parameter prime-rl endpoint where each run gets its
own helm release on a registered PrimeCluster. Auth is the standard API
token; admin role is gated server-side.
"""

from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field

from prime_cli.core import APIClient, APIError


class HostedTrainingRunResponse(BaseModel):
    """Response from POST /v1/training/runs."""

    run_id: str = Field(..., alias="runId")
    job_id: str = Field(..., alias="jobId")
    token_value: str = Field(..., alias="tokenValue")

    model_config = ConfigDict(populate_by_name=True)


class HostedTrainingClient:
    """Client for the hosted full-FT training endpoint."""

    def __init__(self, client: APIClient) -> None:
        self.client = client

    def create_run(self, payload: Dict[str, Any]) -> HostedTrainingRunResponse:
        """POST /v1/training/runs. Backend mints a per-run API token and
        kicks off the helm install asynchronously; returns immediately with
        identifiers."""
        try:
            response = self.client.post("/training/runs", json=payload)
            return HostedTrainingRunResponse.model_validate(response)
        except Exception as e:
            if hasattr(e, "response") and hasattr(e.response, "text"):
                raise APIError(f"Failed to create hosted training run: {e.response.text}")
            raise APIError(f"Failed to create hosted training run: {str(e)}")

    def delete_run(self, run_id: str) -> Dict[str, Any]:
        """DELETE /v1/training/runs/{run_id}. Idempotent: helm uninstall +
        namespace delete + RFTRun soft-delete. Returns the wire payload
        (typically {runId, deleted})."""
        try:
            response = self.client.request("DELETE", f"/training/runs/{run_id}")
            return response if isinstance(response, dict) else {"runId": run_id}
        except Exception as e:
            if hasattr(e, "response") and hasattr(e.response, "text"):
                raise APIError(f"Failed to delete hosted training run: {e.response.text}")
            raise APIError(f"Failed to delete hosted training run: {str(e)}")


def build_payload_from_toml(
    cfg: Dict[str, Any],
    *,
    prime_cluster_id: Optional[str] = None,
    name: Optional[str] = None,
    wandb_api_key: Optional[str] = None,
    hf_token: Optional[str] = None,
) -> Dict[str, Any]:
    """Map a prime-rl-style TOML dict onto the /v1/training/runs payload.

    Mirrors the shape used in `prime-rl/examples/*/rl.toml` so the same
    file can be passed to both `uv run rl @ rl.toml` and `prime train
    rl.toml`. Unknown fields are ignored — only the explicit whitelist
    below is forwarded; the chart owns the rest.

    When `prime_cluster_id` is None the backend auto-picks the first
    uncordoned PrimeCluster — matches the common single-cluster setup.
    """
    payload: Dict[str, Any] = {}
    if prime_cluster_id:
        payload["primeClusterId"] = prime_cluster_id
    if name:
        payload["name"] = name

    def _dig(d: Any, *path: str) -> Any:
        cur = d
        for key in path:
            if not isinstance(cur, dict) or key not in cur:
                return None
            cur = cur[key]
        return cur

    mapping = {
        "model": ("model", "name"),
        "trainerGpus": ("deployment", "num_train_gpus"),
        "inferenceGpus": ("deployment", "num_infer_gpus"),
        "imageTag": ("image", "tag"),
        "seqLen": ("seq_len",),
        "maxSteps": ("max_steps",),
        "learningRate": ("trainer", "optim", "lr"),
        "batchSize": ("orchestrator", "batch_size"),
        "rolloutsPerExample": ("orchestrator", "rollouts_per_example"),
        "maxCompletionTokens": (
            "orchestrator",
            "train",
            "sampling",
            "max_completion_tokens",
        ),
        "wandbEntity": ("wandb", "entity"),
        "wandbProject": ("wandb", "project"),
        "wandbRunName": ("wandb", "name"),
    }
    for api_key, path in mapping.items():
        v = _dig(cfg, *path)
        if v is not None:
            payload[api_key] = v

    train_envs = _dig(cfg, "orchestrator", "train", "env")
    if isinstance(train_envs, list) and train_envs:
        env_id = train_envs[0].get("id") if isinstance(train_envs[0], dict) else None
        if env_id:
            payload["envId"] = env_id

    if wandb_api_key:
        payload["wandbApiKey"] = wandb_api_key
    if hf_token:
        payload["hfToken"] = hf_token
    return payload
