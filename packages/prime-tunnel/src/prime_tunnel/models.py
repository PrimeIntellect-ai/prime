from datetime import datetime

from pydantic import BaseModel, Field


class TunnelInfo(BaseModel):
    """Information about a registered tunnel."""

    tunnel_id: str = Field(..., description="Unique tunnel identifier")
    hostname: str = Field(..., description="Tunnel hostname")
    url: str = Field(..., description="Full HTTPS URL")
    frp_token: str = Field(..., description="Authentication token for frpc")
    binding_secret: str = Field("", description="Per-tunnel secret for frpc metadata")
    server_host: str = Field(..., description="frps server hostname")
    server_port: int = Field(7000, description="frps server port")
    expires_at: datetime = Field(..., description="Token expiration time")

    class Config:
        from_attributes = True
