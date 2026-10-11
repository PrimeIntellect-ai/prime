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
        return self.client.post(f"/slurm-clusters/{team_id}/{cluster_id}/members", json=body)

    def remove_member(self, team_id: str, cluster_id: str, username: str) -> None:
        self.client.delete(f"/slurm-clusters/{team_id}/{cluster_id}/members/{username}")


__all__ = ["FactorySlurmClient", "FactorySlurmCluster", "FactorySlurmMember"]
