from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field

from prime_cli.core import APIClient, APIError


class SSHKey(BaseModel):
    id: str
    user_id: str = Field(..., alias="userId")
    name: str
    public_key: Optional[str] = Field(None, alias="publicKey")
    is_primary: bool = Field(..., alias="isPrimary")
    is_user_key: bool = Field(..., alias="isUserKey")

    model_config = ConfigDict(populate_by_name=True)


class SSHKeyList(BaseModel):
    total_count: int = Field(..., alias="totalCount")
    offset: int
    limit: int
    data: List[SSHKey]

    model_config = ConfigDict(populate_by_name=True)


class SSHKeyCreateRequest(BaseModel):
    name: str = Field(..., description="Name for the SSH key")
    public_key: str = Field(..., alias="publicKey", description="The public key content")

    model_config = ConfigDict(populate_by_name=True)


class SSHKeysClient:
    def __init__(self, client: APIClient) -> None:
        self.client = client

    def list(self, offset: int = 0, limit: int = 100) -> SSHKeyList:
        """List all SSH keys for the authenticated user"""
        try:
            params = {"offset": offset, "limit": limit}
            response = self.client.get("/ssh_keys", params=params)
            return SSHKeyList.model_validate(response)
        except Exception as e:
            if hasattr(e, "response") and hasattr(e.response, "text"):
                raise APIError(f"Failed to list SSH keys: {e.response.text}")
            raise APIError(f"Failed to list SSH keys: {str(e)}")

    def create(self, name: str, public_key: str) -> SSHKey:
        """Create a new SSH key"""
        try:
            data = {"name": name, "publicKey": public_key}
            response = self.client.request("POST", "/ssh_keys", json=data)
            return SSHKey.model_validate(response)
        except Exception as e:
            if hasattr(e, "response") and hasattr(e.response, "text"):
                raise APIError(f"Failed to create SSH key: {e.response.text}")
            raise APIError(f"Failed to create SSH key: {str(e)}")

    def delete(self, key_id: str) -> None:
        """Delete an SSH key"""
        try:
            self.client.delete(f"/ssh_keys/{key_id}")
        except Exception as e:
            if hasattr(e, "response") and hasattr(e.response, "text"):
                raise APIError(f"Failed to delete SSH key: {e.response.text}")
            raise APIError(f"Failed to delete SSH key: {str(e)}")

    def set_primary(self, key_id: str) -> dict:
        """Set an SSH key as primary"""
        try:
            data = {"isPrimary": True}
            response = self.client.request("PATCH", f"/ssh_keys/{key_id}", json=data)
            return response
        except Exception as e:
            if hasattr(e, "response") and hasattr(e.response, "text"):
                raise APIError(f"Failed to set primary SSH key: {e.response.text}")
            raise APIError(f"Failed to set primary SSH key: {str(e)}")
