"""Hosted full-FT training API client (POST/DELETE /v1/training/runs).

Sibling to api/rl.py — that's the LoRA-shared path. This client speaks to
the dedicated full-parameter prime-rl endpoint where each run gets its
own helm release on a registered PrimeCluster. Auth is the standard API
token; admin role is gated server-side.
"""

from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field

from prime_cli.core import APIClient


class HostedTrainingRunResponse(BaseModel):
    """Response from POST /v1/training/runs."""

    run_id: str = Field(..., alias="runId")
    token_value: str = Field(..., alias="tokenValue")

    model_config = ConfigDict(populate_by_name=True)


class HostedTrainingClient:
    """Client for the hosted full-FT training endpoint."""

    def __init__(self, client: APIClient) -> None:
        self.client = client

    def create_run(self, payload: Dict[str, Any]) -> HostedTrainingRunResponse:
        """POST /v1/training/runs. Backend mints a per-run API token and
        kicks off the helm install asynchronously; returns immediately with
        identifiers.

        Lets typed APIError subclasses (NotFoundError, UnauthorizedError,
        …) propagate so callers can branch by exception class instead of
        string-matching the message.
        """
        response = self.client.post("/training/runs", json=payload)
        return HostedTrainingRunResponse.model_validate(response)

    def delete_run(self, run_id: str) -> Dict[str, Any]:
        """DELETE /v1/training/runs/{run_id}. Idempotent: helm uninstall +
        namespace delete + RFTRun soft-delete. Returns the wire payload
        (typically {runId, deleted}). Re-raises typed APIError subclasses
        (NotFoundError on 404, etc.) so callers can branch by exception
        class — `prime train delete` uses NotFoundError as the 'not a
        hosted run, try LoRA' fallback signal.
        """
        response = self.client.request("DELETE", f"/training/runs/{run_id}")
        return response if isinstance(response, dict) else {"runId": run_id}


def build_payload_from_toml(
    cfg: Dict[str, Any],
    *,
    name: Optional[str] = None,
    team_id: Optional[str] = None,
    image_tag: Optional[str] = None,
    wandb_api_key: Optional[str] = None,
    hf_token: Optional[str] = None,
    gpu_type: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the /v1/training/runs payload from a prime-rl-style TOML dict.

    Ships the *whole* TOML as `config` so the backend can split it
    per-component (trainer / orchestrator / inference) and bake each
    into the corresponding pod's startup command. Anything outside the
    handful of platform-authoritative overlays (chart-side scrape ports,
    monitor URL, secret name) flows through unchanged - same e2e
    behaviour as `uv run rl @ rl.toml`.

    What stays out of `config`:
      - secrets (wandb / hf): materialised into a per-run k8s Secret,
      - run name: lives on the platform's RFTRun row, not the TOML,
      - team_id: links the RFTRun to a team for billing/access scoping,
      - image_tag: chart-level (which prime-rl image to pull),
      - gpu_type: narrows the backend picker to clusters with matching
        PrimeCluster.gpuType (e.g. "H200_141GB"); omit for the default
        auto-pick with no type preference.

    Cluster targeting is backend-side (auto-pick first uncordoned).
    """
    payload: Dict[str, Any] = {"config": cfg}
    if name:
        payload["name"] = name
    if team_id:
        payload["teamId"] = team_id
    if image_tag:
        payload["imageTag"] = image_tag
    if wandb_api_key:
        payload["wandbApiKey"] = wandb_api_key
    if hf_token:
        payload["hfToken"] = hf_token
    if gpu_type:
        payload["gpuType"] = gpu_type
    return payload
