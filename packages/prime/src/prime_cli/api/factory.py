"""Model Factory fleet status API client."""

from datetime import datetime, timezone
from typing import Annotated, Any, Dict, List, Optional

from pydantic import AfterValidator, AliasChoices, BaseModel, Field, PrivateAttr, ValidationError

from prime_cli.core import APIClient, APIError


def _aware_utc(value: Optional[datetime]) -> Optional[datetime]:
    """Normalize a parsed timestamp to offset-aware UTC.

    Pydantic accepts both offset-aware strings (``...Z``) and naive ones;
    mixing them in later comparisons (oldest-source, ages) raises TypeError
    or compares across wrong bases. One timezone, always UTC.
    """
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


AwareUTCDatetime = Annotated[Optional[datetime], AfterValidator(_aware_utc)]


class FactoryPool(BaseModel):
    """GPU allocation summary for one workload group on a cluster.

    ``reserved`` is GPUs claimed by the group, ``in_use`` GPUs actually
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
    observed_at: AwareUTCDatetime = None


class FactoryCluster(BaseModel):
    """One dedicated cluster with its per-workload-group allocation summary."""

    display_name: str
    gpu_type: Optional[str] = None
    total_gpus: Optional[int] = None
    status: Optional[str] = None
    unassigned_gpus: Optional[int] = None
    unknown_gpus: Optional[int] = None
    # Both contract fields are required: a 200 response without them is a
    # malformed payload, not an empty allocation summary. The backend renamed
    # the envelope key from `pools` to `allocations` (user vocabulary); accept
    # `allocations` as the primary shape and tolerate the legacy `pools` key
    # (passthrough stays exact either way).
    pools: List[FactoryPool] = Field(validation_alias=AliasChoices("allocations", "pools"))
    sources: List[FactorySource]


class FactoryStatus(BaseModel):
    """Response envelope for ``GET /api/v1/factory/status``."""

    schema_version: int
    as_of: AwareUTCDatetime = None
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
    created_at: AwareUTCDatetime = None
    started_at: AwareUTCDatetime = None
    # Terminal rows carry a nullable end timestamp.
    ended_at: AwareUTCDatetime = None
    reason: Optional[str] = None
    source: FactorySource


class FactoryWorkloads(BaseModel):
    """Response envelope for ``GET /api/v1/factory/workloads``."""

    schema_version: int
    as_of: AwareUTCDatetime = None
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


class FactoryNode(BaseModel):
    """One node in a dedicated factory cluster.

    ``state`` is the coarse public state (ready/cordoned/offline/unknown);
    ``assigned_to`` is the workload group the node is claimed by
    (training/inference/slurm) or ``None`` when unassigned. ``None`` GPU
    counts mean unobserved, never zero.
    """

    name: str
    state: Optional[str] = None
    gpu_type: Optional[str] = None
    gpus_total: Optional[int] = None
    gpus_used: Optional[int] = None
    assigned_to: Optional[str] = None


class FactoryNodesCluster(BaseModel):
    """One cluster with its node inventory.

    Freshness lives on the envelope: `FactoryNodes.sources` carries one
    capacity entry per cluster, in the same order as `clusters` — clusters
    themselves carry no `sources` field (frozen nodes contract).
    """

    display_name: str
    status: Optional[str] = None
    # Required: a 200 response without `nodes` is a malformed payload.
    nodes: List[FactoryNode]


class FactoryNodes(BaseModel):
    """Response envelope for ``GET /api/v1/factory/nodes``.

    ``sources`` carries one capacity entry per cluster, in the same order
    as ``clusters``, so a degraded snapshot stays visible (frozen nodes
    contract; clusters themselves carry no ``sources`` field).
    """

    schema_version: int
    as_of: AwareUTCDatetime = None
    # Required: a 200 response without `clusters` is a malformed payload,
    # not an empty fleet — keep those distinguishable.
    clusters: List[FactoryNodesCluster]
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

    def _get_json(
        self,
        endpoint: str,
        params: Dict[str, Any],
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """GET one factory endpoint and normalize malformed success bodies.

        An HTTP 200 with a non-JSON body (e.g. an HTML proxy error page)
        escapes the transport client as a raw decoding error, before any
        validation runs. Normalize it to a coarse APIError so the commands'
        except-APIError branch surfaces a clean error instead of a
        traceback — without changing APIClient's semantics for other
        commands.
        """
        try:
            if timeout is None:
                # Omit the argument entirely: passing timeout=None to httpx
                # explicitly DISABLES the client's configured default
                # timeout, letting primary commands hang indefinitely.
                return self.client.get(endpoint, params=params)
            return self.client.get(endpoint, params=params, timeout=timeout)
        except ValueError as exc:
            raise APIError("Factory API returned a malformed response body.") from exc

    def get_status(self, team_id: str) -> FactoryStatus:
        """Fetch the fleet status for a team.

        The backend validates team membership of the caller; the CLI only
        resolves which team context to ask about.
        """
        response = self._get_json("/factory/status", {"team_id": team_id})
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
        workload_type: Optional[str] = None,
        state: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> FactoryWorkloads:
        """Fetch the leaf workloads for a team.

        ``workload_type`` (training/inference/slurm), ``state``
        (running/queued/completed/failed/all), ``since`` (ISO 8601) and
        ``limit`` are server-side filters; other narrowing happens
        client-side. The backend validates team membership of the caller.
        The query parameter is named ``workload_type`` exactly as the
        FastAPI route declares it — sending ``type`` would be ignored.
        """
        params: Dict[str, Any] = {"team_id": team_id}
        if workload_type is not None:
            params["workload_type"] = workload_type
        if state is not None:
            params["state"] = state
        if since is not None:
            normalized_since = _aware_utc(since)
            if normalized_since is not None:
                params["since"] = normalized_since.isoformat()
        if limit is not None:
            params["limit"] = limit
        response = self._get_json("/factory/workloads", params)
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

    def get_nodes(self, team_id: str, timeout: Optional[float] = None) -> FactoryNodes:
        """Fetch the node inventory for a team.

        The backend validates team membership of the caller; the CLI only
        resolves which team context to ask about. ``timeout`` caps the
        request for best-effort callers (the compact status glance).
        """
        response = self._get_json("/factory/nodes", {"team_id": team_id}, timeout=timeout)
        try:
            nodes = FactoryNodes.model_validate(response)
        except ValidationError as exc:
            # Wrap shape drift as APIError so the command's except-APIError
            # branch surfaces a clean CLI error instead of a traceback.
            raise APIError(_format_validation_error(exc, label="nodes")) from exc
        if nodes.schema_version != SUPPORTED_SCHEMA_VERSION:
            # A newer schema still validating against the v1 model would be
            # silently misinterpreted; fail loudly instead.
            raise APIError(
                f"Unsupported factory nodes schema version: {nodes.schema_version} "
                f"(expected {SUPPORTED_SCHEMA_VERSION})"
            )
        if len(nodes.sources) != len(nodes.clusters):
            # The nodes contract pairs sources with clusters positionally,
            # exactly one capacity entry per cluster. Any other length
            # mispairs every entry after the first omission — fail closed
            # instead of attempting index-based pairing.
            raise APIError(
                "Unexpected factory nodes response shape: sources must pair 1:1 with clusters"
            )
        if any(source.kind != "capacity" for source in nodes.sources):
            # A non-capacity entry at a paired index is equally out of
            # contract: the pair would render an unexplained em-dash NODES
            # cell while the badge claims fresh.
            raise APIError(
                "Unexpected factory nodes response shape: sources must be capacity entries"
            )
        nodes._raw_response = response
        return nodes
