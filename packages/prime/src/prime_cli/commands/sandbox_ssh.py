"""Ephemeral SSH transport for VM sandboxes.

Shared by ``prime sandbox ssh`` and the sandbox-volume import flow:
generate an Ed25519 key, authorize it through the sandbox API, and expose the
connection parameters an external ssh/rsync command needs.
"""

import os
import shlex
import shutil
import sys
import tempfile
from contextlib import contextmanager
from typing import Iterator, NamedTuple, Optional

import typer
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from prime_sandboxes import SandboxClient

from ..utils import get_console

console = get_console()


class SandboxSshTransport(NamedTuple):
    """Connection parameters for one authorized ephemeral SSH session.

    The SSH username is the session id: the gateway routes by it.
    """

    user: str
    host: str
    port: int
    key_path: str
    proxy_command: str


@contextmanager
def ssh_transport(sandbox_client: SandboxClient, sandbox_id: str) -> Iterator[SandboxSshTransport]:
    """Yield an authorized SSH transport for a running VM sandbox.

    The key is written 0600 into a fresh temp directory. The session is
    revoked and the key removed on exit, both best effort.
    """
    if not shutil.which("ssh"):
        console.print("[red]Error:[/red] OpenSSH is required.")
        raise typer.Exit(1)

    sandbox = sandbox_client.get(sandbox_id)
    if sandbox.status != "RUNNING":
        console.print(f"[red]Error:[/red] Sandbox is not running (status: {sandbox.status})")
        raise typer.Exit(1)

    temp_dir = tempfile.mkdtemp(prefix="prime-ssh-")
    session_id: Optional[str] = None
    try:
        key_path = os.path.join(temp_dir, "id_ed25519")
        private_key = Ed25519PrivateKey.generate()
        key_fd = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(key_fd, "wb") as key_file:
            key_file.write(
                private_key.private_bytes(
                    serialization.Encoding.PEM,
                    serialization.PrivateFormat.OpenSSH,
                    serialization.NoEncryption(),
                )
            )
        public_key = (
            private_key.public_key()
            .public_bytes(serialization.Encoding.OpenSSH, serialization.PublicFormat.OpenSSH)
            .decode()
        )

        session = sandbox_client.create_ssh_session(sandbox_id, public_key)
        session_id = session.session_id
        yield SandboxSshTransport(
            user=session.session_id,
            host=session.host,
            port=session.port,
            key_path=key_path,
            proxy_command=shlex.join(
                [
                    sys.executable,
                    "-m",
                    "prime_cli.ssh_proxy",
                    session.host,
                    str(session.port),
                    session.session_id,
                ]
            ),
        )
    finally:
        if session_id:
            try:
                sandbox_client.close_ssh_session(sandbox_id, session_id)
            except Exception:
                pass
        shutil.rmtree(temp_dir, ignore_errors=True)
