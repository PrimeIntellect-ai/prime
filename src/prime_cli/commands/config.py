import typer
from rich.console import Console
from rich.table import Table

from ..config import Config

app = typer.Typer(help="Configure the CLI")
console = Console()


@app.command()
def view() -> None:
    """View current configuration"""
    config = Config()
    settings = config.view()

    table = Table(title="Prime CLI Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    # Show API key (partially hidden)
    api_key = settings["api_key"]
    if api_key:
        masked_key = f"{api_key[:6]}...{api_key[-4:]}" if len(api_key) > 10 else "***"
    else:
        masked_key = "Not set"
    table.add_row("API Key", masked_key)

    # Show Team ID
    team_id = settings["team_id"]
    table.add_row("Team ID", team_id or "Personal Account")

    # Show base URL
    table.add_row("Base URL", settings["base_url"])

    # Show SSH key path
    table.add_row("SSH Key Path", settings["ssh_key_path"])

    console.print(table)


@app.command()
def set_api_key(
    api_key: str = typer.Option(
        ...,
        prompt="Enter your API key",
        help="Your Prime Intellect API key",
        hide_input=True,
    ),
) -> None:
    """Set your API key"""
    config = Config()
    config.set_api_key(api_key)
    masked_key = f"{api_key[:6]}***{api_key[-4:]}" if len(api_key) > 10 else "***"
    console.print(f"[green]API key {masked_key} configured successfully![/green]")
    console.print("[blue]You can verify your API key with 'prime config view'[/blue]")


@app.command()
def set_team_id(
    team_id: str = typer.Option(
        ..., prompt="Enter your team ID", help="Your Prime Intellect team ID"
    ),
) -> None:
    """Set your team ID"""
    config = Config()
    config.set_team_id(team_id)
    console.print("[green]Team ID configured successfully![/green]")


@app.command()
def remove_team_id() -> None:
    """Remove team ID to use personal account"""
    config = Config()
    config.set_team_id("")
    console.print("[green]Team ID removed. Using personal account.[/green]")


@app.command()
def set_base_url(
    url: str = typer.Option(
        ...,
        prompt="Enter the API base URL",
        help="Base URL for the Prime Intellect API",
    ),
) -> None:
    """Set the API base URL"""
    config = Config()
    config.set_base_url(url)
    console.print("[green]Base URL configured successfully![/green]")


@app.command()
def set_ssh_key_path(
    path: str = typer.Option(
        ...,
        prompt="Enter the SSH private key path",
        help="Path to your SSH private key file",
    ),
) -> None:
    """Set the SSH private key path"""
    config = Config()
    config.set_ssh_key_path(path)
    console.print("[green]SSH key path configured successfully![/green]")


@app.command()
def reset() -> None:
    """Reset configuration to defaults"""
    if typer.confirm("Are you sure you want to reset all settings?"):
        config = Config()
        config.set_api_key("")
        config.set_team_id("")
        config.set_base_url(Config.DEFAULT_BASE_URL)
        config.set_ssh_key_path(Config.DEFAULT_SSH_KEY_PATH)
        console.print("[green]Configuration reset to defaults![/green]")
