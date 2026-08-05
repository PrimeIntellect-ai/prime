"""Prime Agent launcher."""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import typer

from ..utils import get_console

PRIME_AGENT_COMMAND = "prime-agent"
PRIME_AGENT_INSTALLER_URL = "https://pub-728493de92a943e2a9b2d17b4719f318.r2.dev/install.sh"
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
    try:
        with urlopen(PRIME_AGENT_INSTALLER_URL, timeout=30) as response:
            installer = response.read(MAX_INSTALLER_BYTES + 1)
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        console.print(f"[red]Failed to download the Prime Agent installer:[/red] {exc}")
        raise typer.Exit(1) from exc

    if len(installer) > MAX_INSTALLER_BYTES:
        console.print("[red]Failed to download Prime Agent:[/red] installer is unexpectedly large.")
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
            file.write(installer)
            installer_path = Path(file.name)

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
        result = subprocess.run([executable, *args])
    except OSError as exc:
        console.print(f"[red]Failed to launch Prime Agent:[/red] {exc}")
        raise typer.Exit(1) from exc

    if result.returncode != 0:
        raise typer.Exit(result.returncode)


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
