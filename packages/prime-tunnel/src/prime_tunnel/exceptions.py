class TunnelError(Exception):
    """Base exception for tunnel errors."""

    pass


class TunnelNotRunningError(TunnelError):
    """Structured error for tunnel failures with diagnostic fields."""

    def __init__(
        self,
        tunnel_id: str | None = None,
        error_type: str | None = None,
        message: str | None = None,
    ):
        self.tunnel_id = tunnel_id
        self.error_type = error_type

        if message:
            msg = message
        elif error_type:
            msg = f"Tunnel {tunnel_id} failed ({error_type})"
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
