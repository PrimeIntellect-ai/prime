import asyncio
import fcntl
import os
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional

import httpx

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
        log_level: str = "info",
        team_id: Optional[str] = None,
    ):
        """
        Initialize a tunnel.

        Args:
            local_port: Local port to tunnel
            local_addr: Local address to tunnel (default: 127.0.0.1)
            name: Optional friendly name for the tunnel
            team_id: Optional team ID for team tunnels
            connection_timeout: Timeout for establishing connection (seconds)
            log_level: frpc log level (trace, debug, info, warn, error)
        """
        self.local_port = local_port
        self.local_addr = local_addr
        self.name = name
        self.team_id = team_id
        self.connection_timeout = connection_timeout
        self.log_level = log_level

        self._client = TunnelClient()
        self._process: Optional[subprocess.Popen] = None
        self._tunnel_info: Optional[TunnelInfo] = None
        self._config_file: Optional[Path] = None
        self._started = False
        self._output_lines: list[str] = []

    @property
    def tunnel_id(self) -> Optional[str]:
        """Get the tunnel ID."""
        return self._tunnel_info.tunnel_id if self._tunnel_info else None

    @property
    def url(self) -> Optional[str]:
        """Get the tunnel URL."""
        return self._tunnel_info.url if self._tunnel_info else None

    @property
    def hostname(self) -> Optional[str]:
        """Get the tunnel hostname."""
        return self._tunnel_info.hostname if self._tunnel_info else None

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
        frpc_path = await asyncio.to_thread(get_frpc_path)

        # 2. Register tunnel with backend
        try:
            self._tunnel_info = await self._client.create_tunnel(
                local_port=self.local_port,
                name=self.name,
                team_id=self.team_id,
            )
        except BaseException as e:
            await self._cleanup()
            if isinstance(e, asyncio.CancelledError):
                raise
            raise TunnelError(f"Failed to register tunnel: {e}") from e

        # 3. Generate frpc config
        try:
            self._config_file = self._write_frpc_config()
        except BaseException as e:
            await self._cleanup()
            if isinstance(e, asyncio.CancelledError):
                raise
            raise TunnelError(f"Failed to write frpc config: {e}") from e

        # 4. Start frpc process
        try:
            self._process = subprocess.Popen(
                [str(frpc_path), "-c", str(self._config_file)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except BaseException as e:
            await self._cleanup()
            if isinstance(e, asyncio.CancelledError):
                raise
            raise TunnelConnectionError(f"Failed to start frpc: {e}") from e

        # 5. Wait for connection
        try:
            await self._wait_for_connection()
        except BaseException:
            await self._cleanup()
            raise

        # 6. Start background thread to drain pipes (prevents buffer exhaustion)
        try:
            self._start_pipe_drain()
        except BaseException as e:
            await self._cleanup()
            if isinstance(e, asyncio.CancelledError):
                raise
            raise TunnelConnectionError(f"Failed to start pipe drain: {e}") from e

        self._started = True

        return self.url

    async def stop(self) -> None:
        """Stop the tunnel and cleanup resources."""
        if not self._started:
            return

        await self._cleanup()
        self._started = False

    def sync_stop(self) -> None:
        """Stop the tunnel synchronously. Safe for signal handlers and atexit."""
        if not self._started:
            return

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

        if self._tunnel_info is not None:
            try:
                httpx.delete(
                    f"{self._client.base_url}/api/v1/tunnel/{self._tunnel_info.tunnel_id}",
                    headers=self._client._headers,
                    timeout=5.0,
                )
            except Exception:
                pass
            finally:
                self._tunnel_info = None

        if self._config_file is not None:
            try:
                if self._config_file.exists():
                    self._config_file.unlink()
            except Exception:
                pass
            finally:
                self._config_file = None

        self._started = False

    async def _cleanup(self) -> None:
        """Clean up tunnel resources."""
        # Stop frpc process (this will cause drain threads to exit via EOF)
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

    def _start_pipe_drain(self) -> None:
        """Start background threads to drain subprocess pipes.

        This prevents the pipe buffer from filling up and blocking frpc
        when it produces output (logs, reconnection attempts, etc.).
        """
        if self._process is None:
            return

        def drain_pipe(pipe):
            """Read and discard output from a pipe until EOF."""
            if pipe is None:
                return
            try:
                for _ in pipe:
                    pass  # Discard all output
            except (OSError, ValueError):
                pass  # Pipe closed

        # Use separate threads for stdout/stderr to avoid blocking on one
        stdout_thread = threading.Thread(
            target=drain_pipe, args=(self._process.stdout,), daemon=True
        )
        stderr_thread = threading.Thread(
            target=drain_pipe, args=(self._process.stderr,), daemon=True
        )
        stdout_thread.start()
        stderr_thread.start()

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

# Per-tunnel binding secret
metadatas.binding_secret = "{self._tunnel_info.binding_secret}"

# Transport settings
transport.tcpMux = true
transport.tcpMuxKeepaliveInterval = 30
transport.poolCount = 5

# Logging - always use console so we can detect connection via stdout
log.to = "console"
log.level = "{self.log_level}"

# HTTP proxy configuration
[[proxies]]
name = "{self._tunnel_info.tunnel_id}"
type = "http"
localIP = "{self.local_addr}"
localPort = {self.local_port}
subdomain = "{self._tunnel_info.tunnel_id}"
"""

        # Write to temp file
        config_dir = Path(tempfile.gettempdir()) / "prime-tunnel"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_file = config_dir / f"{self._tunnel_info.tunnel_id}.toml"

        # Create file with 0600 permissions
        fd = os.open(str(config_file), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.write(fd, config.encode())
        finally:
            os.close(fd)

        return config_file

    async def _wait_for_connection(self) -> None:
        """Wait for frpc to establish connection."""
        start_time = time.time()
        self._output_lines = []

        while time.time() - start_time < self.connection_timeout:
            if self._process is None:
                raise TunnelConnectionError("frpc process not running")

            return_code = self._process.poll()
            if return_code is not None:
                remaining_output = []
                if self._process.stdout:
                    remaining_output.extend(self._process.stdout.readlines())
                if self._process.stderr:
                    remaining_output.extend(self._process.stderr.readlines())
                self._output_lines.extend(line.strip() for line in remaining_output if line.strip())

                # Build detailed error message
                output_text = (
                    "\n".join(self._output_lines) if self._output_lines else "(no output captured)"
                )
                raise TunnelConnectionError(
                    f"frpc exited with code {return_code}\n"
                    f"--- frpc output ---\n{output_text}\n-------------------"
                )

            if os.name == "posix":
                # Set both pipes to non-blocking mode to drain them without deadlock
                pipes_to_drain = []
                original_flags = {}

                for pipe in (self._process.stdout, self._process.stderr):
                    if pipe:
                        fd = pipe.fileno()
                        fl = fcntl.fcntl(fd, fcntl.F_GETFL)
                        original_flags[fd] = fl
                        fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)
                        pipes_to_drain.append(pipe)

                try:
                    # Drain both stdout and stderr to prevent buffer exhaustion
                    for pipe in pipes_to_drain:
                        try:
                            while True:
                                line = pipe.readline()
                                if not line:
                                    break
                                line = line.strip()
                                if line:
                                    self._output_lines.append(line)
                                    # Check for success/failure indicators
                                    if "start proxy success" in line.lower():
                                        return
                                    if "login failed" in line.lower():
                                        raise TunnelConnectionError(f"frpc login failed: {line}")
                                    if "authorization failed" in line.lower():
                                        raise TunnelConnectionError(
                                            f"frpc authorization failed: {line}"
                                        )
                        except (BlockingIOError, IOError):
                            pass  # No more data available on this pipe
                finally:
                    # Restore original flags
                    for fd, fl in original_flags.items():
                        try:
                            fcntl.fcntl(fd, fcntl.F_SETFL, fl)
                        except (OSError, ValueError):
                            pass  # Pipe may have closed

            await asyncio.sleep(0.1)

        # Timeout - include any captured output
        output_text = (
            "\n".join(self._output_lines) if self._output_lines else "(no output captured)"
        )
        raise TunnelTimeoutError(
            f"Tunnel connection timed out after {self.connection_timeout}s\n"
            f"--- frpc output ---\n{output_text}\n-------------------"
        )

    async def __aenter__(self) -> "Tunnel":
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.stop()
