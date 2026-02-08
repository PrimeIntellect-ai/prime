"""Adapter deployments API client."""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field

from prime_cli.core import APIClient, APIError


class Adapter(BaseModel):
    """LoRA adapter from an RL training run."""

    id: str = Field(..., description="Adapter ID")
    display_name: Optional[str] = Field(None, alias="displayName")
    user_id: str = Field(..., alias="userId")
    team_id: Optional[str] = Field(None, alias="teamId")
    rft_run_id: str = Field(..., alias="rftRunId")
    base_model: str = Field(..., alias="baseModel")
    status: str = Field(..., description="Adapter status: PENDING, UPLOADING, READY, FAILED")
    deployment_status: str = Field(
        default="NOT_DEPLOYED",
        alias="deploymentStatus",
        description="NOT_DEPLOYED, DEPLOYING, DEPLOYED, UNLOADING, DEPLOY_FAILED, UNLOAD_FAILED",
    )
    deployed_at: Optional[datetime] = Field(None, alias="deployedAt")
    deployment_error: Optional[str] = Field(None, alias="deploymentError")
    created_at: datetime = Field(..., alias="createdAt")
    updated_at: datetime = Field(..., alias="updatedAt")

    model_config = ConfigDict(populate_by_name=True)


class DeploymentsClient:
    """Client for adapter deployments API."""

    def __init__(self, client: APIClient) -> None:
        self.client = client

    def list_adapters(self, team_id: Optional[str] = None) -> List[Adapter]:
        """List adapters and their deployment status."""
        try:
            params = {}
            if team_id:
                params["team_id"] = team_id
            response = self.client.get("/rft/adapters", params=params if params else None)
            adapters_data = response.get("adapters", [])
            return [Adapter.model_validate(adapter) for adapter in adapters_data]
        except Exception as e:
            if hasattr(e, "response") and hasattr(e.response, "text"):
                raise APIError(f"Failed to list adapters: {e.response.text}")
            raise APIError(f"Failed to list adapters: {str(e)}")

    def get_adapter(self, adapter_id: str) -> Adapter:
        """Get details of a specific adapter."""
        try:
            response = self.client.get(f"/rft/adapters/{adapter_id}")
            return Adapter.model_validate(response.get("adapter"))
        except Exception as e:
            if hasattr(e, "response") and hasattr(e.response, "text"):
                raise APIError(f"Failed to get adapter: {e.response.text}")
            raise APIError(f"Failed to get adapter: {str(e)}")

    def deploy_adapter(self, adapter_id: str) -> Adapter:
        """Deploy an adapter for inference."""
        try:
            response = self.client.post(f"/rft/adapters/{adapter_id}/deploy")
            return Adapter.model_validate(response.get("adapter"))
        except Exception as e:
            if hasattr(e, "response") and hasattr(e.response, "text"):
                raise APIError(f"Failed to deploy adapter: {e.response.text}")
            raise APIError(f"Failed to deploy adapter: {str(e)}")

    def unload_adapter(self, adapter_id: str) -> Adapter:
        """Unload an adapter from inference."""
        try:
            response = self.client.post(f"/rft/adapters/{adapter_id}/unload")
            return Adapter.model_validate(response.get("adapter"))
        except Exception as e:
            if hasattr(e, "response") and hasattr(e.response, "text"):
                raise APIError(f"Failed to unload adapter: {e.response.text}")
            raise APIError(f"Failed to unload adapter: {str(e)}")
