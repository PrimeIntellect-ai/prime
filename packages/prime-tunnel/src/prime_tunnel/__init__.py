"""Prime Tunnel SDK - Expose local services via secure tunnels."""

__version__ = "0.1.0"

from prime_tunnel.core import Config, TunnelClient
from prime_tunnel.exceptions import (
    TunnelAuthError,
    TunnelConnectionError,
    TunnelError,
    TunnelTimeoutError,
)
from prime_tunnel.models import TunnelConfig, TunnelInfo
from prime_tunnel.tunnel import Tunnel

__all__ = [
    "__version__",
    # Core
    "Config",
    "TunnelClient",
    # Main interface
    "Tunnel",
    # Models
    "TunnelConfig",
    "TunnelInfo",
    # Exceptions
    "TunnelError",
    "TunnelAuthError",
    "TunnelConnectionError",
    "TunnelTimeoutError",
]
