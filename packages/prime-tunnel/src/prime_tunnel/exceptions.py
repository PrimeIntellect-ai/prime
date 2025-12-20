class TunnelError(Exception):
    """Base exception for tunnel errors."""

    pass


class TunnelAuthError(TunnelError):
    """Authentication failed when registering tunnel."""

    pass


class TunnelConnectionError(TunnelError):
    """Failed to establish tunnel connection."""

    pass


class TunnelTimeoutError(TunnelError):
    """Tunnel operation timed out."""

    pass


class BinaryDownloadError(TunnelError):
    """Failed to download frpc binary."""

    pass
