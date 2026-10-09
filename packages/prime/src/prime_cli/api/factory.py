"""Model Factory fleet status API client."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, PrivateAttr, ValidationError

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
    pools: List[FactoryPool] = Field(default_factory=list)
    sources: List[FactorySource] = Field(default_factory=list)


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
            raise APIError(f"Unexpected factory status response shape: {exc}") from exc
        status._raw_response = response
        return status
