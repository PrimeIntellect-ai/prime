import json
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field

from prime_cli.core import APIClient, APIError


class DiskInfo(BaseModel):
    country: Optional[str]
    data_center_id: Optional[str] = Field(None, alias="dataCenterId")
    cloud_id: Optional[str] = Field(None, alias="cloudId")
    is_multinode: Optional[bool] = Field(None, alias="isMultinode")

    model_config = ConfigDict(populate_by_name=True, extra="allow")


class Disk(BaseModel):
    id: str
    name: str
    created_at: str = Field(..., alias="createdAt")
    updated_at: str = Field(..., alias="updatedAt")
    terminated_at: Optional[str] = Field(None, alias="terminatedAt")
    status: str
    provider_type: str = Field(..., alias="providerType")
    size: int
    info: Optional[dict]
    price_hr: Optional[float] = Field(None, alias="priceHr")
    stopped_price_hr: Optional[float] = Field(None, alias="stoppedPriceHr")
    provisioning_price_hr: Optional[float] = Field(None, alias="provisioningPriceHr")
    user_id: Optional[str] = Field(None, alias="userId")
    team_id: Optional[str] = Field(None, alias="teamId")
    wallet_id: Optional[str] = Field(None, alias="walletId")
    pods: List[str] = Field(default_factory=list)
    clusters: List[str] = Field(default_factory=list)

    model_config = ConfigDict(populate_by_name=True)


class DiskList(BaseModel):
    total_count: int = Field(..., alias="total_count")
    offset: int
    limit: int
    data: List[Disk]

    model_config = ConfigDict(populate_by_name=True)


class DiskCreateRequest(BaseModel):
    size: int = Field(..., description="Size of the disk in GB", gt=0)
    name: Optional[str] = Field(None, description="Optional name for the disk")
    country: Optional[str] = None
    cloud_id: Optional[str] = Field(None, alias="cloudId")
    data_center_id: Optional[str] = Field(None, alias="dataCenterId")

    model_config = ConfigDict(populate_by_name=True)


class DiskUpdateRequest(BaseModel):
    name: str = Field(..., description="New name for the disk")

    model_config = ConfigDict(populate_by_name=True)


class DiskDeleteResponse(BaseModel):
    status: str

    model_config = ConfigDict(populate_by_name=True)


class DisksClient:
    def __init__(self, client: APIClient) -> None:
        self.client = client

    def list(self, offset: int = 0, limit: int = 100) -> DiskList:
        """List all disks"""
        try:
            params = {"offset": offset, "limit": limit}
            response = self.client.get("/disks", params=params)
            return DiskList.model_validate(response)
        except Exception as e:
            if hasattr(e, "response") and hasattr(e.response, "text"):
                raise APIError(f"Failed to list disks: {e.response.text}")
            raise APIError(f"Failed to list disks: {str(e)}")

    def get(self, disk_id: str) -> Disk:
        """Get details of a specific disk"""
        try:
            response = self.client.get(f"/disks/{disk_id}")
            return Disk.model_validate(response)
        except Exception as e:
            if hasattr(e, "response") and hasattr(e.response, "text"):
                raise APIError(f"Failed to get disk details: {e.response.text}")
            raise APIError(f"Failed to get disk details: {str(e)}")

    def create(self, disk_config: dict) -> Disk:
        """Create a new disk"""
        # Auto-populate team_id from config if not specified
        if not disk_config.get("team") and self.client.config.team_id:
            disk_config["team"] = {"teamId": self.client.config.team_id}

        try:
            response = self.client.request("POST", "/disks", json=disk_config)
            return Disk.model_validate(response)
        except Exception as e:
            if hasattr(e, "response") and hasattr(e.response, "text"):
                error_text = str(e.response.text)
                try:
                    error_json = json.loads(error_text)
                    if "detail" in error_json:
                        error_text = error_json["detail"]
                except (json.JSONDecodeError, TypeError):
                    pass
                raise APIError(f"Failed to create disk: {error_text}")
            raise APIError(f"Failed to create disk: {str(e)}")

    def update(self, disk_id: str, name: str) -> dict:
        """Update a disk's name"""
        try:
            update_data = {"name": name}
            response = self.client.request("PATCH", f"/disks/{disk_id}", json=update_data)
            return response
        except Exception as e:
            if hasattr(e, "response") and hasattr(e.response, "text"):
                error_text = str(e.response.text)
                try:
                    error_json = json.loads(error_text)
                    if "detail" in error_json:
                        error_text = error_json["detail"]
                except (json.JSONDecodeError, TypeError):
                    pass
                raise APIError(f"Failed to update disk: {error_text}")
            raise APIError(f"Failed to update disk: {str(e)}")

    def delete(self, disk_id: str) -> DiskDeleteResponse:
        """Delete a disk"""
        try:
            response = self.client.delete(f"/disks/{disk_id}")
            return DiskDeleteResponse.model_validate(response)
        except Exception as e:
            if hasattr(e, "response") and hasattr(e.response, "text"):
                error_text = str(e.response.text)
                try:
                    error_json = json.loads(error_text)
                    if "detail" in error_json:
                        error_text = error_json["detail"]
                except (json.JSONDecodeError, TypeError):
                    pass
                raise APIError(f"Failed to delete disk: {error_text}")
            raise APIError(f"Failed to delete disk: {str(e)}")
