import shutil
import subprocess
import sys

from prime_cli import __version__
from prime_cli.utils import get_console
from prime_cli.utils.version_check import get_latest_pypi_version

from .upgrade_configs import UpgradeConfig

console = get_console()


def _detect_install_method() -> str:
    """Detect how prime was installed.

    Returns one of: 'uv_tool', 'pipx', 'pip', or None if unknown.
    """
    # Check if running from uv tool
    # uv tool installs typically live in ~/.local/share/uv/tools/
    exe_path = sys.executable
    if "uv/tools" in exe_path or "uv\\tools" in exe_path:
        return "uv_tool"

    # Check if running from pipx
    # pipx installs typically live in ~/.local/pipx/venvs/
    if "pipx/venvs" in exe_path or "pipx\\venvs" in exe_path:
        return "pipx"

    # Default to pip
    return "pip"


def _run_upgrade(method: str) -> bool:
    """Run the upgrade command for the detected install method.

    Returns True if upgrade succeeded.
    """
    commands: dict[str, list[list[str]]] = {
        "uv_tool": [["uv", "tool", "upgrade", "prime"]],
        "pipx": [["pipx", "upgrade", "prime"]],
        "pip": [
            ["uv", "pip", "install", "--upgrade", "prime"],
            ["pip", "install", "--upgrade", "prime"],
        ],
    }

    cmd_list = commands.get(method, commands["pip"])

    for cmd in cmd_list:
        # Check if the command exists
        if shutil.which(cmd[0]) is None:
            continue

        console.print(f"[dim]Running: {' '.join(cmd)}[/dim]")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode == 0:
                return True
            else:
                console.print(f"[yellow]Command failed: {result.stderr.strip()}[/yellow]")
        except subprocess.TimeoutExpired:
            console.print("[red]Upgrade command timed out[/red]")
        except Exception as e:
            console.print(f"[red]Error running upgrade: {e}[/red]")

    return False


def upgrade(config: UpgradeConfig) -> None:
    """Upgrade the Prime CLI to the latest version.

    Automatically detects how prime was installed (uv tool, pipx, or pip)
    and runs the appropriate upgrade command.
    """
    check = config.check
    force = config.force

    latest_version = get_latest_pypi_version()

    if latest_version is None:
        console.print("[red]Could not fetch latest version from PyPI[/red]")
        raise SystemExit(1)

    console.print(f"[cyan]Installed version:[/cyan] {__version__}")
    console.print(f"[cyan]Latest version:[/cyan]    {latest_version}")

    from packaging import version

    installed = version.parse(__version__)
    latest = version.parse(latest_version)

    if installed >= latest and not force:
        console.print("\n[green]✓ You are already on the latest version![/green]")
        raise SystemExit(0)

    if installed < latest:
        console.print(f"\n[yellow]A newer version is available: {latest_version}[/yellow]")

    if check:
        if installed < latest:
            console.print("\n[dim]Run 'prime upgrade' to upgrade[/dim]")
        raise SystemExit(0)

    # Perform upgrade
    method = _detect_install_method()
    console.print(f"\n[dim]Detected install method: {method}[/dim]")

    if _run_upgrade(method):
        console.print(f"\n[green]✓ Successfully upgraded to {latest_version}![/green]")
    else:
        console.print("\n[red]Upgrade failed. You can try manually:[/red]")
        console.print("  [dim]uv tool upgrade prime[/dim]")
        console.print("  [dim]pipx upgrade prime[/dim]")
        console.print("  [dim]pip install --upgrade prime[/dim]")
        raise SystemExit(1)
