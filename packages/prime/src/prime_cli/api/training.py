"""Hosted full-FT training API client (POST/DELETE /v1/training/runs).

Sibling to api/rl.py — that's the LoRA-shared path. This client speaks to
the dedicated full-parameter prime-rl endpoint where each run gets its
own helm release on a registered PrimeCluster. Auth is the standard API
token; admin role is gated server-side.
"""

import os
from typing import Any, Dict, List, Optional

import httpx
from pydantic import BaseModel, ConfigDict, Field
from pydantic import ValidationError as PydanticValidationError

from prime_cli.core import APIClient, APIError, NotFoundError, default_user_agent

# The only repo the hosted source overlay runs code from. The platform and
# validator enforce the same restriction server-side; the CLI just resolves
# `--pr N` against it so a fork PR fails here with a clear message instead
# of a 422 from the backend.
PRIME_RL_GITHUB_REPO = "PrimeIntellect-ai/prime-rl"
_GITHUB_API = "https://api.github.com"


class HostedTrainingRunResponse(BaseModel):
    """Response from POST /v1/training/runs."""

    run_id: str = Field(..., alias="runId")
    token_value: str = Field(..., alias="tokenValue")

    model_config = ConfigDict(populate_by_name=True)


class AvailableGpuTypesResponse(BaseModel):
    """Response from GET /v1/training/available-gpu-types."""

    gpu_types: List[str] = Field(default_factory=list, alias="gpuTypes")

    model_config = ConfigDict(populate_by_name=True)


class FFTModelClusterInfo(BaseModel):
    """One cluster the caller can dispatch an FFT run to that already has
    the parent model cached.

    Intentionally narrower than the backend `FFTModelClusterInfo`
    schema: `cache_synced_at` is dropped because cache freshness is an
    implementation detail the user shouldn't reason about — the
    dispatch picker already gates on PRESENT-cache clusters, so any
    entry surfaced here is warm.
    """

    cluster_id: str = Field(..., alias="clusterId")
    cluster_name: str = Field(..., alias="clusterName")
    gpu_type: str | None = Field(None, alias="gpuType")

    model_config = ConfigDict(populate_by_name=True)


class AvailableFFTModel(BaseModel):
    """A model that is cached and ready for FFT dispatch on at least one
    eligible PrimeCluster."""

    name: str = Field(..., description="Model name")
    clusters: list[FFTModelClusterInfo] = Field(default_factory=list)

    model_config = ConfigDict(populate_by_name=True)


class AvailableFFTModelsResponse(BaseModel):
    """Response from GET /v1/training/available-fft-models."""

    models: list[AvailableFFTModel] = Field(default_factory=list)

    model_config = ConfigDict(populate_by_name=True)


class Volume(BaseModel):
    """A named volume (GET/POST /v1/training/volumes)."""

    name: str
    size: Optional[str] = None
    status: str
    cluster_id: str = Field(..., alias="clusterId")
    # Cluster name; absent from older backends.
    cluster: Optional[str] = None
    pvc_name: str = Field(..., alias="pvcName")
    created_by: Optional[str] = Field(None, alias="createdBy")
    created_at: Optional[str] = Field(None, alias="createdAt")

    model_config = ConfigDict(populate_by_name=True)


class VolumeSessionGateway(BaseModel):
    host: str
    port: int
    cert_sha256: str = Field(alias="certSha256")

    model_config = ConfigDict(populate_by_name=True)


class VolumeSession(BaseModel):
    id: str
    volume_name: str = Field(alias="volumeName")
    status: str
    read_only: bool = Field(alias="readOnly")
    # Same per-session endpoint carries shell, sftp/scp and rsync.
    ssh_connection: Optional[str] = Field(None, alias="sshConnection")
    # Public gateway that reaches the session from outside the tailnet.
    gateway: Optional[VolumeSessionGateway] = None
    # The session pod's sshd host public key, for scoped known_hosts pinning.
    host_public_key: Optional[str] = Field(None, alias="hostPublicKey")
    # Why the session failed (FAILED/TOMBSTONED only).
    error_message: Optional[str] = Field(None, alias="errorMessage")

    model_config = ConfigDict(populate_by_name=True)


class VolumeTransferProgress(BaseModel):
    """The latest rclone stats line of a running volume transfer.

    Values are rclone's own human strings ("412.345 GiB", "182.400 MiB/s",
    "1h48m0s"); only present once the transfer is running.
    """

    transferred: Optional[str] = None
    total: Optional[str] = None
    percentage: Optional[int] = None
    rate: Optional[str] = None
    eta: Optional[str] = None
    errors: Optional[int] = None


class VolumeTransfer(BaseModel):
    """An S3 import/export of a named volume (…/volumes/{name}/transfers).

    Credentials are never part of this model: the API accepts them in the
    create request body only, and no response or log ever carries them.
    """

    id: str
    volume_name: str = Field(..., alias="volumeName")
    direction: str
    path: str
    url: str
    status: str
    progress: Optional[VolumeTransferProgress] = None
    created_at: Optional[str] = Field(None, alias="createdAt")
    started_at: Optional[str] = Field(None, alias="startedAt")
    completed_at: Optional[str] = Field(None, alias="completedAt")
    error_message: Optional[str] = Field(None, alias="errorMessage")

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

    def create_volume(
        self,
        name: str,
        size: str,
        team_id: Optional[str] = None,
        cluster: Optional[str] = None,
    ) -> Volume:
        payload: Dict[str, Any] = {"name": name, "size": size}
        if team_id:
            payload["teamId"] = team_id
        if cluster:
            payload["cluster"] = cluster
        return Volume.model_validate(self.client.post("/training/volumes", json=payload))

    def list_volumes(self, team_id: Optional[str] = None) -> List[Volume]:
        params = {"teamId": team_id} if team_id else None
        response = self.client.get("/training/volumes", params=params)
        return [Volume.model_validate(v) for v in response.get("volumes", [])]

    def resize_volume(self, name: str, size: str, team_id: Optional[str] = None) -> Volume:
        payload: Dict[str, Any] = {"size": size}
        if team_id:
            payload["teamId"] = team_id
        return Volume.model_validate(self.client.patch(f"/training/volumes/{name}", json=payload))

    def delete_volume(self, name: str, team_id: Optional[str] = None) -> None:
        params = {"teamId": team_id} if team_id else None
        self.client.delete(f"/training/volumes/{name}", params=params)

    def create_volume_session(
        self,
        name: str,
        *,
        read_only: bool = True,
        allow_writable: bool = False,
        team_id: Optional[str] = None,
    ) -> VolumeSession:
        payload: Dict[str, Any] = {"readOnly": read_only}
        if allow_writable:
            payload["allowWritable"] = True
        if team_id:
            payload["teamId"] = team_id
        return VolumeSession.model_validate(
            self.client.post(f"/training/volumes/{name}/sessions", json=payload)
        )

    def get_volume_session(
        self, name: str, session_id: str, *, team_id: Optional[str] = None
    ) -> VolumeSession:
        params = {"teamId": team_id} if team_id else None
        return VolumeSession.model_validate(
            self.client.get(f"/training/volumes/{name}/sessions/{session_id}", params=params)
        )

    def stop_volume_session(
        self, name: str, session_id: str, *, team_id: Optional[str] = None
    ) -> None:
        params = {"teamId": team_id} if team_id else None
        self.client.delete(f"/training/volumes/{name}/sessions/{session_id}", params=params)

    def create_volume_transfer(
        self,
        name: str,
        *,
        direction: str,
        path: str,
        url: str,
        credentials: Dict[str, Any],
        region: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        team_id: Optional[str] = None,
    ) -> VolumeTransfer:
        """POST …/volumes/{name}/transfers. Queues an rclone job on the
        volume's cluster and returns at once with a transfer id.

        `credentials` is the `{accessKeyId, secretAccessKey, sessionToken}`
        mapping built from the caller's AWS environment; it travels in this
        request body only and is never persisted or returned.
        """
        payload: Dict[str, Any] = {
            "direction": direction,
            "path": path,
            "url": url,
            "credentials": credentials,
        }
        if region:
            payload["region"] = region
        if endpoint_url:
            payload["endpointUrl"] = endpoint_url
        if team_id:
            payload["teamId"] = team_id
        return VolumeTransfer.model_validate(
            self.client.post(f"/training/volumes/{name}/transfers", json=payload)
        )

    def list_volume_transfers(
        self, name: str, *, team_id: Optional[str] = None
    ) -> List[VolumeTransfer]:
        """GET …/volumes/{name}/transfers. The caller's transfers on this
        volume, newest first, finished ones included."""
        params = {"teamId": team_id} if team_id else None
        response = self.client.get(f"/training/volumes/{name}/transfers", params=params)
        return [VolumeTransfer.model_validate(t) for t in response.get("transfers", [])]

    def get_volume_transfer(
        self, name: str, transfer_id: str, *, team_id: Optional[str] = None
    ) -> VolumeTransfer:
        """GET …/volumes/{name}/transfers/{id}."""
        params = {"teamId": team_id} if team_id else None
        return VolumeTransfer.model_validate(
            self.client.get(f"/training/volumes/{name}/transfers/{transfer_id}", params=params)
        )

    def cancel_volume_transfer(
        self, name: str, transfer_id: str, *, team_id: Optional[str] = None
    ) -> None:
        """DELETE …/volumes/{name}/transfers/{id}. Tears down the job and its
        Secret; the row stays as `cancelled`. Idempotent."""
        params = {"teamId": team_id} if team_id else None
        self.client.delete(f"/training/volumes/{name}/transfers/{transfer_id}", params=params)

    def list_available_gpu_types(self, team_id: Optional[str] = None) -> AvailableGpuTypesResponse:
        """GET /v1/training/available-gpu-types. Distinct GPU types the
        caller could dispatch a dedicated FFT run on. Same principal
        collapse as the dispatch picker: `team_id` wins when set,
        otherwise the caller's personal allocations.
        """
        params: Dict[str, Any] = {}
        if team_id:
            params["team_id"] = team_id
        response = self.client.get("/training/available-gpu-types", params=params)
        return AvailableGpuTypesResponse.model_validate(response)

    def list_available_fft_models(self, team_id: str | None = None) -> list[AvailableFFTModel]:
        """GET /v1/training/available-fft-models. Models that are already
        cached on at least one PrimeCluster the caller can dispatch a
        full-FT run to.

        404 is swallowed to an empty list so the CLI still renders on
        older backends that haven't shipped the endpoint yet. Every
        other error (auth failure, forbidden, server errors) propagates
        — the caller decides whether to surface or hide it based on
        whether the LoRA section already ran.

        A schema-drifted response (pydantic ValidationError) is
        re-raised as APIError so the command layer's existing
        `except APIError` fallback catches it — otherwise a
        non-conforming backend payload would kill the LoRA table too.
        """
        params: dict[str, Any] = {}
        if team_id:
            params["team_id"] = team_id
        try:
            response = self.client.get("/training/available-fft-models", params=params)
        except NotFoundError:
            return []
        try:
            return AvailableFFTModelsResponse.model_validate(response).models
        except PydanticValidationError as exc:
            raise APIError(f"Failed to parse available FFT models response: {exc}") from exc


def _github_error_body(resp: httpx.Response) -> Dict[str, str]:
    """The string fields of a GitHub error body (`message`,
    `documentation_url`), or {} when the body isn't a JSON object."""
    try:
        body = resp.json()
    except ValueError:
        return {}
    if not isinstance(body, dict):
        return {}
    return {k: v for k, v in body.items() if isinstance(k, str) and isinstance(v, str)}


def _github_message(resp: httpx.Response) -> str:
    """`: <message>` from a GitHub error body, or "" when there is none."""
    message = _github_error_body(resp).get("message")
    return f": {message}" if message else ""


def _github_rate_limited(resp: httpx.Response) -> bool:
    """Whether a 403 / 429 from GitHub is a quota, not a refusal.

    A primary limit sets x-ratelimit-remaining: 0 and a secondary limit
    usually sets retry-after, but GitHub documents both headers as optional
    on a secondary limit; the one guaranteed signal there is the error
    message ("You have exceeded a secondary rate limit", or the older "abuse
    detection mechanism"), so fall back to the body. Any other 403 (SSO
    enforcement, a bad token) is a refusal and must not be reported as
    "retry later".
    """
    if resp.status_code == 429:
        return True
    if resp.headers.get("x-ratelimit-remaining") == "0" or "retry-after" in resp.headers:
        return True
    body = _github_error_body(resp)
    text = f"{body.get('message', '')} {body.get('documentation_url', '')}".lower()
    return "rate limit" in text or "rate-limit" in text or "abuse detection" in text


def resolve_pull_request_head(pr_number: int) -> str:
    """Head commit sha of a prime-rl pull request, for `prime train --pr N`.

    Public GitHub API (prime-rl is public); a GITHUB_TOKEN / GH_TOKEN in the
    environment is sent so a shared egress IP (CI, office NAT) isn't stuck
    on the anonymous 60/hour quota. Only PRs whose head branch lives in the
    canonical repo are accepted:
    the hosted pods can only fetch refs from PrimeIntellect-ai/prime-rl,
    so a fork PR would fail at dispatch anyway. Raises APIError with a
    user-facing message on any failure so the command layer can print
    and exit without string-matching.
    """
    url = f"{_GITHUB_API}/repos/{PRIME_RL_GITHUB_REPO}/pulls/{pr_number}"
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": default_user_agent(),
    }
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        resp = httpx.get(url, headers=headers, timeout=15.0)
    except httpx.HTTPError as exc:
        raise APIError(f"Could not reach GitHub to resolve PR #{pr_number}: {exc}") from exc
    if resp.status_code == 404:
        raise APIError(f"PR #{pr_number} not found in {PRIME_RL_GITHUB_REPO}.")
    if resp.status_code in (403, 429):
        if _github_rate_limited(resp):
            hint = "" if token else " (set GITHUB_TOKEN to lift the anonymous limit)"
            raise APIError(
                f"GitHub API rate limit hit while resolving PR #{pr_number}; retry "
                f"later{hint}, or pass --ref <branch-or-sha> directly."
            )
        raise APIError(
            f"GitHub refused the request for PR #{pr_number} (HTTP 403"
            f"{_github_message(resp)}); pass --ref <branch-or-sha> directly."
        )
    if resp.status_code != 200:
        raise APIError(f"GitHub returned HTTP {resp.status_code} while resolving PR #{pr_number}.")
    try:
        pull = resp.json()
        head = pull["head"]
        head_repo = (head.get("repo") or {}).get("full_name")
        sha = head["sha"]
        merged = bool(pull.get("merged"))
        merge_commit = pull.get("merge_commit_sha")
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        raise APIError(f"Unexpected GitHub response while resolving PR #{pr_number}.") from exc
    if merged:
        # A merged PR's head sha is usually gone from the repo's branches
        # (deleted branch, squash merge), so the platform would refuse it as
        # a commit it can't attribute; the merged code lives on main.
        via = f" (its merge commit is {merge_commit})" if isinstance(merge_commit, str) else ""
        raise APIError(f"PR #{pr_number} is already merged; run its code with --ref main{via}.")
    if not isinstance(head_repo, str) or head_repo.lower() != PRIME_RL_GITHUB_REPO.lower():
        raise APIError(
            f"PR #{pr_number} comes from a fork ({head_repo or 'unknown'}); fork PRs are "
            f"not supported. Push the branch to {PRIME_RL_GITHUB_REPO} and use --ref."
        )
    if not isinstance(sha, str) or not sha:
        raise APIError(f"PR #{pr_number} has no head commit sha in the GitHub response.")
    return sha


def build_payload_from_toml(
    cfg: Dict[str, Any],
    *,
    name: Optional[str] = None,
    team_id: Optional[str] = None,
    image_tag: Optional[str] = None,
    wandb_api_key: Optional[str] = None,
    hf_token: Optional[str] = None,
    gpu_type: Optional[str] = None,
    volume: Optional[str] = None,
    mode: Optional[str] = None,
    source_ref: Optional[str] = None,
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
      - volume: a named volume the run writes its outputs to, under
        runs/<runId>/, instead of a per-run PVC. For SFT runs the volume
        also supplies the dataset: it is mounted read-only at /volume,
        so config data.name must point at a path on the volume (e.g.
        "/volume/datasets/<name>" or the relative "datasets/<name>" —
        the platform resolves it).
      - mode: the prime-rl schema family the config belongs to ("rl" or
        "sft"). Omitted by default so existing RL payloads stay
        byte-compatible; set to "sft" for SFT configs so the backend +
        validator dispatch to SFTConfig instead of the RL schema.
      - source_ref: a prime-rl git ref (branch / tag / sha) the pods
        overlay onto the image at startup, for testing unmerged code
        without an image build. The platform pins the resolved commit.

    Cluster targeting is backend-side (auto-pick first uncordoned).
    """
    payload: Dict[str, Any] = {"config": cfg}
    if mode:
        payload["mode"] = mode
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
    if volume:
        payload["volume"] = volume
    if source_ref:
        payload["sourceRef"] = source_ref
    return payload
