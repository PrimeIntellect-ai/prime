from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field

from prime_cli.core import APIClient, APIError


class SlurmClusterSummary(BaseModel):
    id: str
    prime_cluster_id: str = Field(alias="primeClusterId")
    display_name: str = Field(alias="displayName")
    status: str
    gpu_type: Optional[str] = Field(None, alias="gpuType")
    gpu_count: int = Field(alias="gpuCount")
    created_at: str = Field(alias="createdAt")
    started_at: Optional[str] = Field(None, alias="startedAt")

    model_config = ConfigDict(populate_by_name=True)


class SlurmClusterMember(BaseModel):
    username: str
    uid: int
    ssh_authorized_keys: List[str] = Field(alias="sshAuthorizedKeys")
    sudo: bool
    status: str
    linked_user_id: Optional[str] = Field(None, alias="linkedUserId")
    linked_user_name: Optional[str] = Field(None, alias="linkedUserName")
    linked_user_email: Optional[str] = Field(None, alias="linkedUserEmail")

    model_config = ConfigDict(populate_by_name=True)


class SlurmClustersClient:
    def __init__(self, client: APIClient) -> None:
        self.client = client

    def list(self, team_id: str) -> List[SlurmClusterSummary]:
        response = self.client.get(f"/slurm-clusters/{team_id}")
        return [SlurmClusterSummary.model_validate(c) for c in response.get("data", [])]

    def list_members(self, team_id: str, cluster_id: str) -> List[SlurmClusterMember]:
        response = self.client.get(f"/slurm-clusters/{team_id}/{cluster_id}/members")
        return [SlurmClusterMember.model_validate(m) for m in response.get("data", [])]

    def add_member(
        self,
        team_id: str,
        cluster_id: str,
        username: str,
        ssh_authorized_keys: List[str],
        linked_user_id: Optional[str] = None,
    ) -> SlurmClusterMember:
        body = {"username": username, "sshAuthorizedKeys": ssh_authorized_keys}
        if linked_user_id:
            body["linkedUserId"] = linked_user_id
        response = self.client.post(f"/slurm-clusters/{team_id}/{cluster_id}/members", json=body)
        return SlurmClusterMember.model_validate(response)

    def remove_member(self, team_id: str, cluster_id: str, username: str) -> None:
        self.client.delete(f"/slurm-clusters/{team_id}/{cluster_id}/members/{username}")


__all__ = [
    "SlurmClustersClient",
    "SlurmClusterSummary",
    "SlurmClusterMember",
    "APIError",
]
