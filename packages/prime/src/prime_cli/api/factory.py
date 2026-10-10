"""Model Factory fleet status API client."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, PrivateAttr, ValidationError

from prime_cli.core import APIClient, APIError


class FactoryPool(BaseModel):
    """GPU allocation summary for one workload pool on a cluster.

    ``reserved`` is GPUs claimed by the pool, ``in_use`` GPUs actually
    allocated to leaf workloads, ``idle_inside`` the difference when the
    evidence is complete. ``None`` means the value could not be observed.
    """

    type: str
    reserved_gpus: Optional[int] = None
    in_use_gpus: Optional[int] = None
    idle_inside_gpus: Optional[int] = None
    unknown_gpus: Optional[int] = None


class FactorySource(BaseModel):
    """Coarse freshness/coverage entry for one data source.

    Only whitelisted public fields: fixed source ``kind``, coarse
    ``status`` and the last ``observed_at`` timestamp.
    """

    kind: str
    status: str
    observed_at: Optional[datetime] = None


class FactoryCluster(BaseModel):
    """One dedicated cluster with its pool allocation summary."""

    display_name: str
    gpu_type: Optional[str] = None
    total_gpus: Optional[int] = None
    status: Optional[str] = None
    unassigned_gpus: Optional[int] = None
    unknown_gpus: Optional[int] = None
    # Both contract fields are required: a 200 response without them is a
    # malformed payload, not an empty allocation summary.
    pools: List[FactoryPool]
    sources: List[FactorySource]


class FactoryStatus(BaseModel):
    """Response envelope for ``GET /api/v1/factory/status``."""

    schema_version: int
    as_of: Optional[datetime] = None
    # Required: a 200 response without `clusters` is a malformed payload,
    # not an empty fleet — keep those distinguishable.
    clusters: List[FactoryCluster]

    # The raw API response is retained so ``--json`` can echo the exact
    # server payload instead of a re-serialization of the parsed model.
    _raw_response: Dict[str, Any] = PrivateAttr(default_factory=dict)

    @property
    def raw_response(self) -> Dict[str, Any]:
        return self._raw_response


class FactoryWorkloadOwner(BaseModel):
    """Coarse owner attribution for one workload.

    ``kind`` is one of ``prime`` / ``slurm`` / ``unknown``; ``display_name``
    is the reviewed public label (Prime user name/slug, native username) or
    ``None``. Emails, user ids and internal identities never appear here.
    """

    kind: str
    display_name: Optional[str] = None


class FactoryWorkload(BaseModel):
    """One leaf workload (training run, inference job, or Slurm job).

    ``id`` is a stable opaque public id. ``requested_gpus``/``allocated_gpus``
    are GPU counts; ``None`` means the value could not be observed, never 0.
    ``reason`` is a coarse blocking/waiting label supplied by the backend; it
    is rendered only when present and never invented client-side.
    """

    id: str
    type: str
    cluster_display_name: Optional[str] = None
    name: Optional[str] = None
    state: str
    native_state: Optional[str] = None
    owner: FactoryWorkloadOwner
    requested_gpus: Optional[int] = None
    allocated_gpus: Optional[int] = None
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    reason: Optional[str] = None
    source: FactorySource


class FactoryWorkloads(BaseModel):
    """Response envelope for ``GET /api/v1/factory/workloads``."""

    schema_version: int
    as_of: Optional[datetime] = None
    # Required: a 200 response without `workloads` is a malformed payload,
    # not an empty fleet — keep those distinguishable.
    workloads: List[FactoryWorkload]
    sources: List[FactorySource]

    # The raw API response is retained so ``--json`` can echo the exact
    # server payload instead of a re-serialization of the parsed model.
    _raw_response: Dict[str, Any] = PrivateAttr(default_factory=dict)

    @property
    def raw_response(self) -> Dict[str, Any]:
        return self._raw_response


SUPPORTED_SCHEMA_VERSION = 1


def _format_validation_error(exc: "ValidationError", label: str = "status") -> str:
    """Summarize pydantic validation errors without bracketed metadata.

    Pydantic messages contain ``[type=..., input_value=...]`` segments; those
    brackets would crash Rich markup rendering when the CLI prints the error.
    """
    details = "; ".join(
        f"{'.'.join(str(loc) for loc in err['loc'])}: {err['msg']}" for err in exc.errors()
    )
    return f"Unexpected factory {label} response shape: {details}"


class FactoryClient:
    """Client for the Model Factory fleet status API."""

    def __init__(self, client: APIClient) -> None:
        self.client = client

    def get_status(self, team_id: str) -> FactoryStatus:
        """Fetch the fleet status for a team.

        The backend validates team membership of the caller; the CLI only
        resolves which team context to ask about.
        """
        response = self.client.get("/factory/status", params={"team_id": team_id})
        try:
            status = FactoryStatus.model_validate(response)
        except ValidationError as exc:
            # Wrap shape drift as APIError so the command's except-APIError
            # branch surfaces a clean CLI error instead of a traceback.
            raise APIError(_format_validation_error(exc)) from exc
        if status.schema_version != SUPPORTED_SCHEMA_VERSION:
            # A newer schema still validating against the v1 model would be
            # silently misinterpreted; fail loudly instead.
            raise APIError(
                f"Unsupported factory status schema version: {status.schema_version} "
                f"(expected {SUPPORTED_SCHEMA_VERSION})"
            )
        status._raw_response = response
        return status

    def get_workloads(
        self,
        team_id: str,
        type: Optional[str] = None,
        state: Optional[str] = None,
    ) -> FactoryWorkloads:
        """Fetch the leaf workloads for a team.

        ``type`` (training/inference/slurm) and ``state`` (running/queued)
        are server-side filters; other narrowing happens client-side. The
        backend validates team membership of the caller.
        """
        params: Dict[str, Any] = {"team_id": team_id}
        if type is not None:
            params["type"] = type
        if state is not None:
            params["state"] = state
        response = self.client.get("/factory/workloads", params=params)
        try:
            workloads = FactoryWorkloads.model_validate(response)
        except ValidationError as exc:
            # Wrap shape drift as APIError so the command's except-APIError
            # branch surfaces a clean CLI error instead of a traceback.
            raise APIError(_format_validation_error(exc, label="workloads")) from exc
        if workloads.schema_version != SUPPORTED_SCHEMA_VERSION:
            # A newer schema still validating against the v1 model would be
            # silently misinterpreted; fail loudly instead.
            raise APIError(
                f"Unsupported factory workloads schema version: {workloads.schema_version} "
                f"(expected {SUPPORTED_SCHEMA_VERSION})"
            )
        workloads._raw_response = response
        return workloads
