from typing import List, Optional, Union
from pydantic import BaseModel, Field
from prime_cli.api.client import APIError


class PortMapping(BaseModel):
    internal: str
    external: str
    protocol: str
    used_by: Optional[str] = Field(None, alias="usedBy")
    description: Optional[str]

    class Config:
        populate_by_name = True


class PodStatus(BaseModel):
    pod_id: str = Field(..., alias="podId")
    provider_type: str = Field(..., alias="providerType")
    status: str
    ssh_connection: Optional[Union[str, List[str]]] = Field(None, alias="sshConnection")
    cost_per_hr: Optional[float] = Field(None, alias="priceHr")
    prime_port_mapping: Optional[List[PortMapping]] = Field(
        None, alias="primePortMapping"
    )
    ip: Optional[Union[str, List[str]]]
    installation_failure: Optional[str] = Field(None, alias="installationFailure")
    installation_progress: Optional[int] = Field(None, alias="installationProgress")
    installation_status: Optional[str] = Field(None, alias="installationStatus")
    team_id: Optional[str] = Field(None, alias="teamId")

    class Config:
        populate_by_name = True


class Pod(BaseModel):
    id: str
    name: Optional[str]
    gpu_type: str = Field(..., alias="gpuName")
    gpu_count: int = Field(..., alias="gpuCount")
    status: str
    created_at: str = Field(..., alias="createdAt")
    provider_type: str = Field(..., alias="providerType")
    installation_status: Optional[str] = Field(None, alias="installationStatus")
    team_id: Optional[str] = Field(None, alias="teamId")

    class Config:
        populate_by_name = True


class PodList(BaseModel):
    total_count: int = Field(..., alias="total_count")
    offset: int
    limit: int
    data: List[Pod]

    class Config:
        populate_by_name = True


class PodConfig(BaseModel):
    name: Optional[str]
    cloud_id: str = Field(..., alias="cloudId")
    gpu_type: str = Field(..., alias="gpuType")
    socket: str
    gpu_count: int = Field(..., alias="gpuCount")
    disk_size: Optional[int] = Field(None, alias="diskSize")
    vcpus: Optional[int]
    memory: Optional[int]
    image: Optional[str]
    custom_template_id: Optional[str] = Field(None, alias="customTemplateId")
    data_center_id: Optional[str] = Field(None, alias="dataCenterId")
    country: Optional[str]
    security: Optional[str]
    provider: dict
    team: Optional[dict]

    class Config:
        populate_by_name = True


class PodsClient:
    def __init__(self, client):
        self.client = client

    def list(self, offset: int = 0, limit: int = 100) -> PodList:
        """List all pods"""
        try:
            params = {"offset": offset, "limit": limit}
            response = self.client.get("/pods", params=params)
            return PodList(**response)
        except Exception as e:
            if hasattr(e, "response") and hasattr(e.response, "text"):
                raise APIError(f"Failed to list pods: {e.response.text}")
            raise APIError(f"Failed to list pods: {str(e)}")

    def get_status(self, pod_ids: List[str]) -> List[PodStatus]:
        """Get status for specified pods"""
        try:
            params = {"pod_ids": pod_ids}
            response = self.client.get("/pods/status", params=params)
            return [PodStatus(**status) for status in response.get("data", [])]
        except Exception as e:
            if hasattr(e, "response") and hasattr(e.response, "text"):
                raise APIError(f"Failed to get pod status: {e.response.text}")
            raise APIError(f"Failed to get pod status: {str(e)}")

    def get(self, pod_id: str) -> Pod:
        """Get details of a specific pod"""
        try:
            response = self.client.get(f"/pods/{pod_id}")
            return Pod(**response)
        except Exception as e:
            if hasattr(e, "response") and hasattr(e.response, "text"):
                raise APIError(f"Failed to get pod details: {e.response.text}")
            raise APIError(f"Failed to get pod details: {str(e)}")

    def create(self, pod_config: dict) -> Pod:
        """Create a new pod"""
        try:
            response = self.client.request("POST", "/pods", json=pod_config)
            return Pod(**response)
        except Exception as e:
            if hasattr(e, "response") and hasattr(e.response, "text"):
                raise APIError(f"Failed to create pod: {e.response.text}")
            raise APIError(f"Failed to create pod: {str(e)}")

    def delete(self, pod_id: str) -> None:
        """Delete a pod"""
        try:
            self.client.delete(f"/pods/{pod_id}")
        except Exception as e:
            if hasattr(e, "response") and hasattr(e.response, "text"):
                raise APIError(f"Failed to delete pod: {e.response.text}")
            raise APIError(f"Failed to delete pod: {str(e)}")
