from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field, SecretStr


class TunnelInfo(BaseModel):
    """Information about a registered tunnel."""

    tunnel_id: str = Field(..., description="Unique tunnel identifier")
    name: Optional[str] = Field(None, description="Friendly name")
    hostname: str = Field(..., description="Tunnel hostname")
    url: str = Field(..., description="Full HTTPS URL")
    frp_token: SecretStr = Field(..., description="Authentication token for frpc")
    binding_secret: SecretStr = Field(default=SecretStr(""), description="Per-tunnel secret for frpc metadata")
    server_host: str = Field(..., description="frps server hostname")
    server_port: int = Field(7000, description="frps server port")
    local_port: Optional[int] = Field(None, description="Local port forwarded by the tunnel")
    labels: List[str] = Field(default_factory=list, description="Tunnel labels")
    http_user: Optional[str] = Field(
        None, description="HTTP basic auth username, if auth is enabled"
    )
    http_password: Optional[SecretStr] = Field(
        None,
        description=(
            "Auto-generated HTTP basic auth password. Only present in the "
            "create response; never retrievable afterwards."
        ),
    )
    expires_at: datetime = Field(..., description="Token expiration time")
    # Optional because create_tunnel response doesn't include user_id
    user_id: Optional[str] = Field(None, description="Owner user ID")
    team_id: Optional[str] = Field(None, description="Team ID")
    status: Optional[str] = Field(None, description="Current tunnel status")
    created_at: Optional[datetime] = Field(None, description="Creation time")
    connected_at: Optional[datetime] = Field(None, description="Connection time")
    terminated_at: Optional[datetime] = Field(None, description="Termination time")

    class Config:
        from_attributes = True


class TunnelListPage(BaseModel):
    """A page of tunnel results with pagination metadata."""

    tunnels: List[TunnelInfo] = Field(default_factory=list, description="Tunnels on this page")
    total: int = Field(0, description="Total number of tunnels matching the query")
    page: int = Field(1, description="Current page number")
    per_page: int = Field(50, description="Items per page")
    has_next: bool = Field(False, description="Whether more pages are available")
