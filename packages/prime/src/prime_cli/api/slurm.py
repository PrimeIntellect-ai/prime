from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field

from prime_cli.core import APIClient, APIError


class CordonedNode(BaseModel):
    name: str
    reason: str
    automated: bool


class NodeHealth(BaseModel):
    total_nodes: int = Field(alias="totalNodes")
    healthy_nodes: int = Field(alias="healthyNodes")
    cordoned_nodes: List[CordonedNode] = Field(alias="cordonedNodes")

    model_config = ConfigDict(populate_by_name=True)


class GpuNode(BaseModel):
    name: str
    ready: bool
    is_cordoned: bool = Field(alias="isCordoned")
    cordon_reason: Optional[str] = Field(None, alias="cordonReason")
    allocatable_gpus: int = Field(alias="allocatableGpus")
    used_gpus: int = Field(alias="usedGpus")
    free_gpus: int = Field(alias="freeGpus")
    slurm_states: Optional[List[str]] = Field(None, alias="slurmStates")

    model_config = ConfigDict(populate_by_name=True)


class SlurmClusterSummary(BaseModel):
    id: str
    prime_cluster_id: str = Field(alias="primeClusterId")
    display_name: str = Field(alias="displayName")
    status: str
    gpu_type: Optional[str] = Field(None, alias="gpuType")
    gpu_count: int = Field(alias="gpuCount")
    total_gpus: Optional[int] = Field(None, alias="totalGpus")
    free_gpus: Optional[int] = Field(None, alias="freeGpus")
    total_nodes: int = Field(alias="totalNodes")
    healthy_nodes: int = Field(alias="healthyNodes")
    cordoned_node_count: int = Field(alias="cordonedNodeCount")
    created_at: str = Field(alias="createdAt")
    started_at: Optional[str] = Field(None, alias="startedAt")

    model_config = ConfigDict(populate_by_name=True)


class SlurmClusterDetail(BaseModel):
    id: str
    prime_cluster_id: str = Field(alias="primeClusterId")
    display_name: str = Field(alias="displayName")
    status: str
    gpu_type: Optional[str] = Field(None, alias="gpuType")
    gpu_count: int = Field(alias="gpuCount")
    connectable: bool
    ssh_host: Optional[str] = Field(None, alias="sshHost")
    ssh_port: Optional[int] = Field(None, alias="sshPort")
    node_health: NodeHealth = Field(alias="nodeHealth")
    nodes: List[GpuNode]
    created_at: str = Field(alias="createdAt")

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


class ThroughputPoint(BaseModel):
    day: str
    completed: int = 0
    failed: int = 0
    cancelled: int = 0


class QueueWaitPoint(BaseModel):
    day: str
    median_seconds: float


class GpuHoursRow(BaseModel):
    user: str
    gpu_hours: float


class OutcomeRow(BaseModel):
    outcome: str
    count: int


class SlurmAccounting(BaseModel):
    # Backend response is snake_case (not aliased) — see
    # platform backend/app/packages/jobs/schemas.py TeamSlurmAccountingResponse.
    available: bool
    days: int = 0
    total_jobs: int = 0
    throughput: List[ThroughputPoint] = Field(default_factory=list)
    queue_wait: List[QueueWaitPoint] = Field(default_factory=list)
    gpu_hours_by_user: List[GpuHoursRow] = Field(default_factory=list)
    outcomes: List[OutcomeRow] = Field(default_factory=list)


class UtilizationPoint(BaseModel):
    timestamp: float
    value: float


class SlurmClustersClient:
    def __init__(self, client: APIClient) -> None:
        self.client = client

    def list(self, team_id: str) -> List[SlurmClusterSummary]:
        response = self.client.get(f"/slurm-clusters/{team_id}")
        return [SlurmClusterSummary.model_validate(c) for c in response.get("data", [])]

    def get(self, team_id: str, cluster_id: str) -> SlurmClusterDetail:
        response = self.client.get(f"/slurm-clusters/{team_id}/{cluster_id}")
        return SlurmClusterDetail.model_validate(response)

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

    def set_sudo(
        self, team_id: str, cluster_id: str, username: str, enabled: bool
    ) -> SlurmClusterMember:
        response = self.client.patch(
            f"/slurm-clusters/{team_id}/{cluster_id}/members/{username}/sudo",
            json={"enabled": enabled},
        )
        return SlurmClusterMember.model_validate(response)

    def rename(self, team_id: str, cluster_id: str, display_name: str) -> Optional[str]:
        response = self.client.patch(
            f"/slurm-clusters/{team_id}/{cluster_id}", json={"displayName": display_name}
        )
        return response.get("displayName")

    def delete(self, team_id: str, cluster_id: str, force: bool = False) -> None:
        params = {"force": "true"} if force else None
        self.client.delete(f"/slurm-clusters/{team_id}/{cluster_id}", params=params)

    def accounting(self, team_id: str, cluster_id: str, days: int = 30) -> SlurmAccounting:
        response = self.client.get(
            f"/slurm-clusters/{team_id}/{cluster_id}/accounting", params={"days": days}
        )
        return SlurmAccounting.model_validate(response)

    def utilization(
        self, team_id: str, cluster_id: str, range_seconds: int = 21_600
    ) -> List[UtilizationPoint]:
        response = self.client.get(
            f"/slurm-clusters/{team_id}/{cluster_id}/utilization",
            params={"range_seconds": range_seconds},
        )
        return [UtilizationPoint.model_validate(p) for p in response.get("gpuUtil", [])]


__all__ = [
    "SlurmClustersClient",
    "SlurmClusterSummary",
    "SlurmClusterDetail",
    "SlurmClusterMember",
    "NodeHealth",
    "GpuNode",
    "CordonedNode",
    "SlurmAccounting",
    "UtilizationPoint",
    "APIError",
]
