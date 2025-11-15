import json
from typing import Any, List, Optional, Union

from prime_core import APIClient, APIError
from pydantic import BaseModel, ConfigDict, Field, field_validator


def clean_connection_fields(v: Any) -> Any:
    """Shared validator to clean connection fields (ssh_connection, ip).

    Removes empty list wrappers around None values.
    """
    if v == [None]:
        return None
    if isinstance(v, list) and len(v) == 1 and v[0] is None:
        return None
    return v


class PortMapping(BaseModel):
    internal: str
    external: str
    protocol: str
    used_by: Optional[str] = Field(None, alias="usedBy")
    description: Optional[str]

    model_config = ConfigDict(populate_by_name=True)


class PodStatus(BaseModel):
    pod_id: str = Field(..., alias="podId")
    provider_type: str = Field(..., alias="providerType")
    status: str
    ssh_connection: Optional[Union[str, List[str]]] = Field(None, alias="sshConnection")
    cost_per_hr: Optional[float] = Field(None, alias="priceHr")
    prime_port_mapping: Optional[List[PortMapping]] = Field(None, alias="primePortMapping")
    ip: Optional[Union[str, List[str]]]
    installation_failure: Optional[str] = Field(None, alias="installationFailure")
    installation_progress: Optional[int] = Field(None, alias="installationProgress")

    @field_validator("ssh_connection", "ip", mode="before")
    @classmethod
    def validate_connection_fields(cls, v: Any) -> Any:
        return clean_connection_fields(v)

    model_config = ConfigDict(populate_by_name=True)


class AttachedResource(BaseModel):
    id: Union[str, int, None] = Field(None, alias="id")
    type: Optional[str] = Field(None, alias="type")
    status: Optional[str] = None
    created_at: Optional[str] = Field(None, alias="createdAt")
    size: Optional[int]
    mount_path: Optional[str] = Field(None, alias="mountPath")
    resource_path: Optional[str] = Field(None, alias="resourcePath")
    is_detachable: Optional[bool] = Field(None, alias="isDetachable")
    resource_type: Optional[str] = Field(None, alias="resourceType")

    model_config = ConfigDict(
        populate_by_name=True, str_strip_whitespace=True, validate_assignment=True
    )


class Pod(BaseModel):
    id: str
    name: Optional[str]
    gpu_type: str = Field(..., alias="gpuName")
    gpu_count: int = Field(..., alias="gpuCount")
    status: str
    created_at: str = Field(..., alias="createdAt")
    provider_type: str = Field(..., alias="providerType")
    installation_status: Optional[str] = Field(None, alias="installationStatus")
    installation_failure: Optional[str] = Field(None, alias="installationFailure")
    installation_progress: Optional[int] = Field(None, alias="installationProgress")
    team_id: Optional[str] = Field(None, alias="teamId")
    resources: Optional[dict]
    attached_resources: Optional[List[AttachedResource]] = Field(None, alias="attachedResources")
    prime_port_mapping: Optional[List[PortMapping]] = Field(None, alias="primePortMapping")
    ssh_connection: Optional[Union[str, List[str]]] = Field(None, alias="sshConnection")
    ip: Optional[Union[str, List[str]]]
    price_hr: Optional[float] = Field(None, alias="priceHr")
    environment_type: Optional[str] = Field(None, alias="environmentType")
    socket: Optional[str]
    type: Optional[str]
    user_id: Optional[str] = Field(None, alias="userId")
    wallet_id: Optional[str] = Field(None, alias="walletId")
    updated_at: Optional[str] = Field(None, alias="updatedAt")
    jupyter_password: Optional[str] = Field(None, alias="jupyterPassword")
    stopped_price_hr: Optional[float] = Field(None, alias="stoppedPriceHr")
    provisioning_price_hr: Optional[float] = Field(None, alias="provisioningPriceHr")
    base_price_hr: Optional[float] = Field(None, alias="basePriceHr")
    base_currency: Optional[str] = Field(None, alias="baseCurrency")
    custom_template_id: Optional[str] = Field(None, alias="customTemplateId")
    is_spot: Optional[bool] = Field(None, alias="isSpot")
    auto_restart: Optional[bool] = Field(None, alias="autoRestart")

    @field_validator("ssh_connection", "ip", mode="before")
    @classmethod
    def validate_connection_fields(cls, v: Any) -> Any:
        return clean_connection_fields(v)

    model_config = ConfigDict(populate_by_name=True)


class PodList(BaseModel):
    total_count: int = Field(..., alias="total_count")
    offset: int
    limit: int
    data: List[Pod]

    model_config = ConfigDict(populate_by_name=True)


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

    model_config = ConfigDict(populate_by_name=True)


class HistoryObj(BaseModel):
    id: str
    name: str
    provider_type: str = Field(..., alias="providerType")
    provisioned_by: Optional[str] = Field(None, alias="provisionedBy")
    type: str
    created_at: str = Field(..., alias="createdAt")
    terminated_at: Optional[str] = Field(None, alias="terminatedAt")
    gpu_name: str = Field(..., alias="gpuName")
    count: int = Field(..., alias="gpuCount")
    socket: Optional[str] = None
    price_hr: float = Field(..., alias="priceHr")
    user_id: str = Field(..., alias="userId")
    team_id: Optional[str] = Field(None, alias="teamId")
    total_billed_price: float = Field(..., alias="totalBilledPrice")

    model_config = ConfigDict(populate_by_name=True)


class HistoryList(BaseModel):
    total_count: int = Field(..., alias="total_count")
    offset: int
    limit: int
    data: List[HistoryObj]

    model_config = ConfigDict(populate_by_name=True)


class PodsClient:
    def __init__(self, client: APIClient) -> None:
        self.client = client

    def list(self, offset: int = 0, limit: int = 100) -> PodList:
        """List all pods"""
        try:
            params = {"offset": offset, "limit": limit}
            response = self.client.get("/pods", params=params)
            return PodList.model_validate(response)
        except Exception as e:
            if hasattr(e, "response") and hasattr(e.response, "text"):
                raise APIError(f"Failed to list pods: {e.response.text}")
            raise APIError(f"Failed to list pods: {str(e)}")

    def get_status(self, pod_ids: List[str]) -> List[PodStatus]:
        """Get status for specified pods"""
        try:
            params = {"pod_ids": pod_ids}
            response = self.client.get("/pods/status", params=params)
            return [PodStatus.model_validate(status) for status in response.get("data", [])]
        except Exception as e:
            if hasattr(e, "response") and hasattr(e.response, "text"):
                raise APIError(f"Failed to get pod status: {e.response.text}")
            raise APIError(f"Failed to get pod status: {str(e)}")

    def get(self, pod_id: str) -> Pod:
        """Get details of a specific pod"""
        try:
            response = self.client.get(f"/pods/{pod_id}")
            return Pod.model_validate(response)
        except Exception as e:
            if hasattr(e, "response") and hasattr(e.response, "text"):
                raise APIError(f"Failed to get pod details: {e.response.text}")
            raise APIError(f"Failed to get pod details: {str(e)}")

    def create(self, pod_config: dict) -> Pod:
        """Create a new pod"""
        # Auto-populate team_id from config if not specified
        if not pod_config.get("team") and self.client.config.team_id:
            pod_config["team"] = {"teamId": self.client.config.team_id}

        try:
            response = self.client.request("POST", "/pods", json=pod_config)
            return Pod.model_validate(response)
        except Exception as e:
            if hasattr(e, "response") and hasattr(e.response, "text"):
                error_text = str(e.response.text)
                error_json = json.loads(error_text)
                if "detail" in error_json:
                    error_text = error_json["detail"]
                raise APIError(f"Failed to create pod: {error_text}")
            raise APIError(f"Failed to create pod: {str(e)}")

    def delete(self, pod_id: str) -> None:
        """Delete a pod"""
        try:
            self.client.delete(f"/pods/{pod_id}")
        except Exception as e:
            if hasattr(e, "response") and hasattr(e.response, "text"):
                raise APIError(f"Failed to delete pod: {e.response.text}")
            raise APIError(f"Failed to delete pod: {str(e)}")

    def history(
        self,
        offset: int = 0,
        limit: int = 100,
    ) -> HistoryList:
        """Get pods history"""
        try:
            params = {"offset": offset, "limit": limit}
            response = self.client.get("/pods/history", params=params)
            return HistoryList.model_validate(response)
        except Exception as e:
            if hasattr(e, "response") and hasattr(e.response, "text"):
                raise APIError(f"Failed to get pods history: {e.response.text}")
            raise APIError(f"Failed to get pods history: {str(e)}")
