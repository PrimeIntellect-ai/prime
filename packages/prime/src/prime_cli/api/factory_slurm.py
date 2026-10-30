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

    def create_cluster(
        self,
        team_id: str,
        prime_cluster_id: str,
        slurm_cluster_name: str,
        num_nodes: int,
        members: List[Dict[str, Any]],
        gpus_per_node: Optional[int] = None,
        home_size: Optional[str] = None,
        tailscale_auth_key: Optional[str] = None,
        containers_enabled: bool = False,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Create a new Slurm cluster deployment.

        Platform API: POST /api/v1/slurm-clusters/{team_id}/deploy
        Contract: backend/app/api/v1/slurm_clusters/router.py (develop branch)

        Args:
            team_id: The team ID that will own the cluster
            prime_cluster_id: Target Model Factory or dedicated cluster id
            slurm_cluster_name: Display name for the Slurm deployment (2-32 chars)
            num_nodes: Number of nodes to deploy (required)
            members: List of at least 1 member dict with:
                - username: str (matching ^[a-z][a-z0-9-]{0,30}[a-z0-9]$, length 2-32)
                - sshAuthorizedKeys: list[str] (at least 1)
                - sudo: bool (default false)
                - linkedUserId: Optional[str]
            gpus_per_node: Optional number of GPUs per node
            home_size: Optional home directory size
            tailscale_auth_key: Optional Tailscale auth key
            containers_enabled: Enable containers (default false)
            timeout: Optional request timeout in seconds

        Returns:
            API response dict with:
            - id: str (cluster/job identifier)
            - primeClusterId: str
            - status: str (starts PENDING)
            - createdAt: str

        Raises:
            APIError: If the create request fails
            - 403: API key lacks slurm_clusters:write scope or not team admin
            - 404: prime cluster not found
            - 409: conflict (e.g., name already exists, not enough nodes, port conflict)
            - 422: validation error (invalid fields or missing requirements)
        """
        endpoint = f"/slurm-clusters/{team_id}/deploy"
        body: Dict[str, Any] = {
            "primeClusterId": prime_cluster_id,
            "slurmClusterName": slurm_cluster_name,
            "numNodes": num_nodes,
            "members": members,
            "containersEnabled": containers_enabled,
        }
        if gpus_per_node is not None:
            body["gpusPerNode"] = gpus_per_node
        if home_size is not None:
            body["homeSize"] = home_size
        if tailscale_auth_key is not None:
            body["tailscaleAuthKey"] = tailscale_auth_key
        try:
            return self.client.post(endpoint, json=body, timeout=timeout)
        except ValueError as exc:
            raise APIError("Slurm cluster create API returned a malformed response body.") from exc

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
