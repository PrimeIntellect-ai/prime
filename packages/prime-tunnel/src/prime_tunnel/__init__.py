"""Prime Tunnel SDK - Expose local services via secure tunnels."""

__version__ = "0.1.10"

from prime_tunnel.core import Config, TunnelClient
from prime_tunnel.exceptions import (
    BinaryDownloadError,
    TunnelAuthError,
    TunnelConnectionError,
    TunnelError,
    TunnelLimitReachedError,
    TunnelTimeoutError,
)
from prime_tunnel.models import TunnelInfo, TunnelListPage

__all__ = [
    "__version__",
    # Core
    "Config",
    "TunnelClient",
    # Main interface
    "Tunnel",
    # Models
    "TunnelInfo",
    "TunnelListPage",
    # Exceptions
    "BinaryDownloadError",
    "TunnelAuthError",
    "TunnelError",
    "TunnelLimitReachedError",
    "TunnelConnectionError",
    "TunnelTimeoutError",
]


def __getattr__(name: str):
    if name == "Tunnel":
        from prime_tunnel.tunnel import Tunnel

        return Tunnel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
