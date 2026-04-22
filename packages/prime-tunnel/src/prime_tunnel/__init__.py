"""Prime Tunnel SDK - Expose local services via secure tunnels."""

__version__ = "0.1.6"

from prime_tunnel.core import Config, TunnelClient
from prime_tunnel.exceptions import (
    BinaryDownloadError,
    TunnelAuthError,
    TunnelConnectionError,
    TunnelError,
    TunnelLimitReachedError,
    TunnelTimeoutError,
)
from prime_tunnel.models import TunnelInfo
from prime_tunnel.tunnel import Tunnel

__all__ = [
    "__version__",
    # Core
    "Config",
    "TunnelClient",
    # Main interface
    "Tunnel",
    # Models
    "TunnelInfo",
    # Exceptions
    "BinaryDownloadError",
    "TunnelAuthError",
    "TunnelError",
    "TunnelLimitReachedError",
    "TunnelConnectionError",
    "TunnelTimeoutError",
]
