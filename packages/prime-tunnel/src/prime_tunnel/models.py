from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class TunnelStatus(str, Enum):
    """Tunnel connection status."""

    PENDING = "pending"
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    EXPIRED = "expired"


class TunnelConfig(BaseModel):
    """Configuration for a tunnel."""

    local_port: int = Field(8765, description="Local port to tunnel")
    local_addr: str = Field("127.0.0.1", description="Local address to tunnel")
    name: Optional[str] = Field(None, description="Friendly name for the tunnel")


class TunnelInfo(BaseModel):
    """Information about a registered tunnel."""

    tunnel_id: str = Field(..., description="Unique tunnel identifier")
    subdomain: str = Field(..., description="Tunnel subdomain")
    url: str = Field(..., description="Full HTTPS URL")
    frp_token: str = Field(..., description="Authentication token for frpc")
    server_addr: str = Field(..., description="frps server address")
    server_port: int = Field(7000, description="frps server port")
    expires_at: datetime = Field(..., description="Token expiration time")

    class Config:
        from_attributes = True


class TunnelRegistrationRequest(BaseModel):
    """Request to register a new tunnel."""

    name: Optional[str] = None
    local_port: int = 8765


class TunnelRegistrationResponse(BaseModel):
    """Response from tunnel registration."""

    tunnel_id: str
    subdomain: str
    url: str
    frp_token: str
    server_addr: str
    server_port: int
    expires_at: datetime

    class Config:
        from_attributes = True
