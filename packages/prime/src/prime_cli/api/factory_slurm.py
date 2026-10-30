"""Slurm deployment roster API client (factory `slurm` subcommand)."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from prime_cli.core import APIClient, APIError


class FactorySlurmCluster(BaseModel):
    """One Slurm deployment row, per the /slurm-clusters list response.

    ``id`` is the Job id — the addressable cluster id for every route.
    """

    id: str
    prime_cluster_id: str = Field(alias="primeClusterId")
    display_name: str = Field(alias="displayName")
    status: str
    gpu_type: Optional[str] = Field(None, alias="gpuType")
    gpu_count: int = Field(alias="gpuCount")
    created_at: str = Field(alias="createdAt")
    started_at: Optional[str] = Field(None, alias="startedAt")

    model_config = ConfigDict(populate_by_name=True)

    # The raw row is retained so --json can echo the exact server payload.
    _raw: Dict[str, Any] = {}

    def model_post_init(self, __context: Any) -> None:
        # pydantic strips unknown keys; keep the exact raw dict for --json.
        self._raw = self.model_dump(by_alias=True, exclude_none=True)


class FactorySlurmMember(BaseModel):
    """One member roster row (public SSH keys and POSIX uid are deliberately
    exposed by the API for troubleshooting — not credentials)."""

    username: str
    uid: int
    ssh_authorized_keys: List[str] = Field(alias="sshAuthorizedKeys")
    sudo: bool
    status: str
    linked_user_id: Optional[str] = Field(None, alias="linkedUserId")
    linked_user_name: Optional[str] = Field(None, alias="linkedUserName")
    linked_user_email: Optional[str] = Field(None, alias="linkedUserEmail")

    model_config = ConfigDict(populate_by_name=True)


class FactorySlurmClient:
    """Client for the team-scoped /slurm-clusters roster surface."""

    def __init__(self, client: APIClient) -> None:
        self.client = client

    def _get_json(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """GET one endpoint; normalize malformed success bodies to APIError."""
        try:
            return self.client.get(endpoint, params=params)
        except ValueError as exc:
            raise APIError("Slurm cluster API returned a malformed response body.") from exc

    def list_clusters(self, team_id: str) -> Dict[str, Any]:
        """The team's Slurm deployments (raw envelope for exact --json)."""
        return self._get_json(f"/slurm-clusters/{team_id}")

    def list_members(self, team_id: str, cluster_id: str) -> Dict[str, Any]:
        return self._get_json(f"/slurm-clusters/{team_id}/{cluster_id}/members")

    def add_member(
        self,
        team_id: str,
        cluster_id: str,
        username: str,
        ssh_authorized_keys: List[str],
        linked_user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {"username": username, "sshAuthorizedKeys": ssh_authorized_keys}
        if linked_user_id:
            body["linkedUserId"] = linked_user_id
        try:
            return self.client.post(f"/slurm-clusters/{team_id}/{cluster_id}/members", json=body)
        except ValueError as exc:
            # An HTTP 2xx with a non-JSON body: the same coarse
            # malformed-response normalization the GETs apply.
            raise APIError("Slurm cluster API returned a malformed response body.") from exc

    def remove_member(self, team_id: str, cluster_id: str, username: str) -> None:
        self.client.delete(f"/slurm-clusters/{team_id}/{cluster_id}/members/{username}")

    def resize_cluster(
        self, team_id: str, cluster_id: str, num_nodes: int, timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """Resize a Slurm cluster to the specified node count.

        Platform API: POST /api/v1/slurm-clusters/{team_id}/{job_id}/resize
        Contract: github.com/PrimeIntellect-ai/platform PR #6917

        Args:
            team_id: The team ID that owns the cluster
            cluster_id: The cluster/job identifier (id field from list_clusters)
            num_nodes: The desired total node count (absolute target, not delta)
                       Must be >= 1 and <= max(64, current_size)
            timeout: Optional request timeout in seconds

        Returns:
            API response dict with:
            - numNodes: int (new total node count)
            - added: list[str] (names of nodes added)
            - removed: list[str] (names of nodes removed)

        Raises:
            APIError: If the resize request fails
            - 400: invalid numNodes or insufficient resources
            - 403: API key lacks the factory:write scope, not team admin,
                   or self-service disabled
            - 404: cluster not found or wrong team
            - 409: not running / concurrent operation / insufficient idle nodes
            - 502: labels changed but verification failed
            - 503: Slurm state unavailable (retryable)
        """
        endpoint = f"/slurm-clusters/{team_id}/{cluster_id}/resize"
        try:
            return self.client.post(endpoint, json={"numNodes": num_nodes}, timeout=timeout)
        except ValueError as exc:
            raise APIError("Slurm cluster resize API returned a malformed response body.") from exc


__all__ = ["FactorySlurmClient", "FactorySlurmCluster", "FactorySlurmMember"]
