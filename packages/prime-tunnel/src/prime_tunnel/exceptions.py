class TunnelError(Exception):
    """Base exception for tunnel errors."""

    pass


class TunnelConnectionError(TunnelError):
    """Tunnel connection failure with optional tunnel ID for diagnostics."""

    def __init__(
        self,
        message: str | None = None,
        *,  # keyword-only below for backwards compat
        tunnel_id: str | None = None,
    ):
        self.tunnel_id = tunnel_id

        if message is not None:
            msg = message
        elif tunnel_id:
            msg = f"Tunnel {tunnel_id} is not running"
        else:
            msg = "Tunnel is not running"
        super().__init__(msg)


class TunnelAuthError(TunnelError):
    """Authentication failed when registering tunnel."""

    pass


class TunnelTimeoutError(TunnelError):
    """Tunnel operation timed out."""

    pass


class TunnelLimitReachedError(TunnelError):
    """Tunnel quota exceeded."""

    pass


class BinaryDownloadError(TunnelError):
    """Failed to download frpc binary."""

    pass
