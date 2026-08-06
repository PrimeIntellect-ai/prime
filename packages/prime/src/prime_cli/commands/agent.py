"""Prime Agent launcher."""

import hashlib
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import typer

from .. import __version__
from ..utils import get_console

PRIME_AGENT_COMMAND = "prime-agent"
PRIME_AGENT_INSTALLER_URL = "https://app.primeintellect.ai/prime-agent/install.sh"
# Update only after reviewing a new installer published at the URL above.
PRIME_AGENT_INSTALLER_SHA256 = (
    "38d14a1be73b325652c7ce8342e3bf19335721837192855a7907732caf8e6d04"
)
MAX_INSTALLER_BYTES = 1024 * 1024

console = get_console()


def _standalone_agent_path() -> Path:
    data_home = os.environ.get("XDG_DATA_HOME")
    base_dir = Path(data_home) if data_home else Path.home() / ".local" / "share"
    return base_dir / "prime-agent-node" / "current" / "bin" / PRIME_AGENT_COMMAND


def _find_agent() -> str | None:
    executable = shutil.which(PRIME_AGENT_COMMAND)
    if executable is not None:
        return executable

    standalone_path = _standalone_agent_path()
    if standalone_path.is_file():
        return str(standalone_path)

    return None


def _download_installer() -> bytes:
    request = Request(
        PRIME_AGENT_INSTALLER_URL,
        headers={"User-Agent": f"prime-cli/{__version__}"},
    )
    try:
        with urlopen(request, timeout=30) as response:
            installer = response.read(MAX_INSTALLER_BYTES + 1)
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        console.print(f"[red]Failed to download the Prime Agent installer:[/red] {exc}")
        raise typer.Exit(1) from exc

    if len(installer) > MAX_INSTALLER_BYTES:
        console.print("[red]Failed to download Prime Agent:[/red] installer is unexpectedly large.")
        raise typer.Exit(1)

    actual_sha256 = hashlib.sha256(installer).hexdigest()
    if actual_sha256 != PRIME_AGENT_INSTALLER_SHA256:
        console.print("[red]Failed to download Prime Agent:[/red] installer checksum mismatch.")
        raise typer.Exit(1)

    if not installer.startswith(b"#!/bin/sh"):
        console.print("[red]Failed to download Prime Agent:[/red] invalid installer response.")
        raise typer.Exit(1)

    return installer


def _install_agent() -> None:
    shell = shutil.which("sh")
    if shell is None:
        console.print("[red]Prime Agent is not installed and `sh` is unavailable.[/red]")
        console.print(f"[dim]Install it from {PRIME_AGENT_INSTALLER_URL} and try again.[/dim]")
        raise typer.Exit(1)

    console.print("[dim]Prime Agent is not installed. Downloading the installer...[/dim]")
    installer = _download_installer()
    installer_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            prefix="prime-agent-install-",
            suffix=".sh",
            delete=False,
        ) as file:
            installer_path = Path(file.name)
            file.write(installer)

        result = subprocess.run([shell, str(installer_path)])
    except OSError as exc:
        console.print(f"[red]Failed to run the Prime Agent installer:[/red] {exc}")
        raise typer.Exit(1) from exc
    finally:
        if installer_path is not None:
            installer_path.unlink(missing_ok=True)

    if result.returncode != 0:
        raise typer.Exit(result.returncode)


def _run_agent(executable: str, args: list[str]) -> None:
    try:
        os.execv(executable, [executable, *args])
    except OSError as exc:
        console.print(f"[red]Failed to launch Prime Agent:[/red] {exc}")
        raise typer.Exit(1) from exc


def agent_command(ctx: typer.Context) -> None:
    """Launch Prime Agent, installing it if needed."""
    executable = _find_agent()
    if executable is None:
        _install_agent()
        executable = _find_agent()

    if executable is None:
        console.print("[red]Prime Agent was installed, but its command could not be found.[/red]")
        console.print("[dim]Restart your shell and run `prime agent` again.[/dim]")
        raise typer.Exit(1)

    _run_agent(executable, list(ctx.args))
