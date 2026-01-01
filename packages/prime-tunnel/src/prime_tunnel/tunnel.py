import asyncio
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional

from prime_tunnel.binary import get_frpc_path
from prime_tunnel.core.client import TunnelClient
from prime_tunnel.exceptions import TunnelConnectionError, TunnelError, TunnelTimeoutError
from prime_tunnel.models import TunnelInfo


class Tunnel:
    """Tunnel interface for exposing local services."""

    def __init__(
        self,
        local_port: int,
        local_addr: str = "127.0.0.1",
        name: Optional[str] = None,
        connection_timeout: float = 30.0,
    ):
        """
        Initialize a tunnel.

        Args:
            local_port: Local port to tunnel
            local_addr: Local address to tunnel (default: 127.0.0.1)
            name: Optional friendly name for the tunnel
            connection_timeout: Timeout for establishing connection (seconds)
        """
        self.local_port = local_port
        self.local_addr = local_addr
        self.name = name
        self.connection_timeout = connection_timeout

        self._client = TunnelClient()
        self._process: Optional[subprocess.Popen] = None
        self._tunnel_info: Optional[TunnelInfo] = None
        self._config_file: Optional[Path] = None
        self._started = False

    @property
    def tunnel_id(self) -> Optional[str]:
        """Get the tunnel ID."""
        return self._tunnel_info.tunnel_id if self._tunnel_info else None

    @property
    def url(self) -> Optional[str]:
        """Get the tunnel URL."""
        return self._tunnel_info.url if self._tunnel_info else None

    @property
    def subdomain(self) -> Optional[str]:
        """Get the tunnel subdomain."""
        return self._tunnel_info.subdomain if self._tunnel_info else None

    @property
    def is_running(self) -> bool:
        """Check if the tunnel is running."""
        if self._process is None:
            return False
        return self._process.poll() is None

    async def start(self) -> str:
        """
        Start the tunnel.

        Returns:
            The tunnel URL

        Raises:
            TunnelError: If tunnel registration fails
            TunnelConnectionError: If frpc fails to connect
            TunnelTimeoutError: If connection times out
        """
        if self._started:
            raise TunnelError("Tunnel is already started")

        # 1. Get frpc binary
        frpc_path = get_frpc_path()

        # 2. Register tunnel with backend
        try:
            self._tunnel_info = await self._client.create_tunnel(
                local_port=self.local_port,
                name=self.name,
            )
        except Exception as e:
            raise TunnelError(f"Failed to register tunnel: {e}") from e

        # 3. Generate frpc config
        try:
            self._config_file = self._write_frpc_config()
        except Exception as e:
            await self._cleanup()
            raise TunnelError(f"Failed to write frpc config: {e}") from e

        # 4. Start frpc process
        try:
            self._process = subprocess.Popen(
                [str(frpc_path), "-c", str(self._config_file)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except Exception as e:
            await self._cleanup()
            raise TunnelConnectionError(f"Failed to start frpc: {e}") from e

        # 5. Wait for connection
        try:
            await self._wait_for_connection()
        except Exception:
            await self._cleanup()
            raise

        self._started = True

        return self.url

    async def stop(self) -> None:
        """Stop the tunnel and cleanup resources."""
        if not self._started:
            return

        await self._cleanup()
        self._started = False

    async def _cleanup(self) -> None:
        """Clean up tunnel resources."""
        # Stop frpc process
        if self._process is not None:
            try:
                self._process.terminate()
                try:
                    self._process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait(timeout=2)
            except Exception:
                pass
            finally:
                self._process = None

        # Delete tunnel registration
        if self._tunnel_info is not None:
            try:
                await self._client.delete_tunnel(self._tunnel_info.tunnel_id)
            except Exception:
                pass
            finally:
                self._tunnel_info = None

        # Clean up config file
        if self._config_file is not None:
            try:
                if self._config_file.exists():
                    self._config_file.unlink()
            except Exception:
                pass
            finally:
                self._config_file = None

        # Close HTTP client
        try:
            await self._client.close()
        except Exception:
            pass

    def _write_frpc_config(self) -> Path:
        """Generate and write frpc configuration file."""
        if self._tunnel_info is None:
            raise TunnelError("Tunnel not registered")

        server_host = self._tunnel_info.server_host
        server_port = self._tunnel_info.server_port

        # Generate config content
        config = f"""# Prime Tunnel frpc configuration
# Tunnel ID: {self._tunnel_info.tunnel_id}

serverAddr = "{server_host}"
serverPort = {server_port}

# Authentication
user = "{self._tunnel_info.tunnel_id}"
auth.method = "token"
auth.token = "{self._tunnel_info.frp_token}"

# Transport settings
transport.tcpMux = true
transport.tcpMuxKeepaliveInterval = 30
transport.poolCount = 5

# Logging
log.to = "console"
log.level = "info"

# HTTP proxy configuration
[[proxies]]
name = "http"
type = "http"
localIP = "{self.local_addr}"
localPort = {self.local_port}
subdomain = "{self._tunnel_info.tunnel_id}"
"""

        # Write to temp file
        config_dir = Path(tempfile.gettempdir()) / "prime-tunnel"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_file = config_dir / f"{self._tunnel_info.tunnel_id}.toml"
        config_file.write_text(config)

        return config_file

    async def _wait_for_connection(self) -> None:
        """Wait for frpc to establish connection."""
        start_time = time.time()

        while time.time() - start_time < self.connection_timeout:
            if self._process is None:
                raise TunnelConnectionError("frpc process not running")

            return_code = self._process.poll()
            if return_code is not None:
                stderr = ""
                if self._process.stderr:
                    stderr = self._process.stderr.read()
                raise TunnelConnectionError(f"frpc exited with code {return_code}: {stderr}")

            if self._process.stderr:
                if os.name == "posix":
                    import fcntl

                    fd = self._process.stderr.fileno()
                    fl = fcntl.fcntl(fd, fcntl.F_GETFL)
                    fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)

                    try:
                        line = self._process.stderr.readline()
                        if line:
                            if "start proxy success" in line.lower():
                                return
                            if "login failed" in line.lower():
                                raise TunnelConnectionError(f"frpc login failed: {line.strip()}")
                    except (BlockingIOError, IOError):
                        pass
                    finally:
                        fcntl.fcntl(fd, fcntl.F_SETFL, fl)

            await asyncio.sleep(0.1)

        raise TunnelTimeoutError(f"Tunnel connection timed out after {self.connection_timeout}s")

    async def __aenter__(self) -> "Tunnel":
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.stop()
