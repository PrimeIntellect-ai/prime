from typing import Optional

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

    # Show current environment
    table.add_row("Current Environment", settings["current_environment"])

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

    # Show frontend URL
    table.add_row("Frontend URL", settings["frontend_url"])

    # Show SSH key path
    table.add_row("SSH Key Path", settings["ssh_key_path"])

    console.print(table)


@app.command()
def set_api_key(
    api_key: Optional[str] = typer.Argument(
        None,
        help="Your Prime Intellect API key. If not provided, you'll be prompted securely.",
    ),
) -> None:
    """Set your API key (prompts securely if not provided)"""
    if not api_key:
        # Interactive mode with secure prompt
        api_key = typer.prompt(
            "Enter your Prime Intellect API key",
            hide_input=True,
            confirmation_prompt=True,
        )
        if not api_key:
            console.print("[red]API key is required[/red]")
            return

    config = Config()
    config.set_api_key(api_key)
    masked_key = f"{api_key[:6]}***{api_key[-4:]}" if len(api_key) > 10 else "***"
    console.print(f"[green]API key {masked_key} configured successfully![/green]")
    console.print("[blue]You can verify your API key with 'prime config view'[/blue]")
    console.print(
        "\n[yellow]Tip: Get your API key at https://app.primeintellect.ai/dashboard/tokens[/yellow]"
    )


@app.command()
def set_team_id(
    team_id: str = typer.Argument(..., help="Your Prime Intellect team ID"),
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
    url: Optional[str] = typer.Argument(
        None,
        help="Base URL for the Prime Intellect API. If not provided, you'll be prompted.",
    ),
) -> None:
    """Set the API base URL (prompts if not provided)"""
    if not url:
        config = Config()
        url = typer.prompt(
            "Enter the base URL for the Prime Intellect API",
            default=config.base_url,
        )
        if not url:
            console.print("[red]Base URL is required[/red]")
            return

    config = Config()
    config.set_base_url(url)
    console.print(f"[green]Base URL set to: {url}[/green]")


@app.command()
def set_frontend_url(
    url: Optional[str] = typer.Argument(
        None,
        help="Frontend URL for the Prime Intellect web app. If not provided, you'll be prompted.",
    ),
) -> None:
    """Set the frontend URL (prompts if not provided)"""
    if not url:
        config = Config()
        url = typer.prompt(
            "Enter the frontend URL for the Prime Intellect web app",
            default=config.frontend_url,
        )
        if not url:
            console.print("[red]Frontend URL is required[/red]")
            return

    config = Config()
    config.set_frontend_url(url)
    console.print(f"[green]Frontend URL set to: {url}[/green]")


# Helper functions (not commands)
def _set_environment(
    env: str,
) -> None:
    """Set URLs for a specific environment"""
    config = Config()

    # Try to load the environment (handles both built-in and custom)
    try:
        if config.load_environment(env):
            console.print(f"[green]Switched to environment '{env}'![/green]")
        else:
            console.print(f"[red]Unknown environment: {env}[/red]")
            console.print("[yellow]Available environments:[/yellow]")
            for env_name in config.list_environments():
                console.print(f"  - {env_name}")
            raise typer.Exit(1)
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    console.print("[blue]Run 'prime config view' to see the current configuration[/blue]")


def _save_environment(
    name: str,
) -> None:
    """Save current configuration as a named environment (including API key)"""
    try:
        config = Config()
        config.save_environment(name)
        console.print(f"[green]Saved current configuration as environment '{name}'![/green]")
        console.print("[yellow]Note: This includes your API key and team ID[/yellow]")
        console.print(f"[blue]Use 'prime config use {name}' to load it later[/blue]")
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


def _list_environments() -> None:
    """List all available environments"""
    config = Config()
    environments = config.list_environments()

    table = Table(title="Available Environments")
    table.add_column("Environment", style="cyan")
    table.add_column("Type", style="green")

    for env in environments:
        env_type = "Built-in" if env == "production" else "Custom"
        table.add_row(env, env_type)

    console.print(table)


@app.command()
def set_ssh_key_path(
    path: str = typer.Argument(
        ...,
        help="Path to your SSH private key file",
    ),
) -> None:
    """Set the SSH private key path"""
    config = Config()
    config.set_ssh_key_path(path)
    console.print("[green]SSH key path configured successfully![/green]")


@app.command()
def reset(
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
) -> None:
    """Reset configuration to defaults"""
    if yes or typer.confirm("Are you sure you want to reset all settings?"):
        config = Config()
        config.set_api_key("")
        config.set_team_id("")
        config.set_base_url(Config.DEFAULT_BASE_URL)
        config.set_frontend_url(Config.DEFAULT_FRONTEND_URL)
        config.set_ssh_key_path(Config.DEFAULT_SSH_KEY_PATH)
        config.set_current_environment("production")
        console.print("[green]Configuration reset to defaults![/green]")


# Environment commands
@app.command(name="use")
def use_environment(
    env: str = typer.Argument(
        ..., help="Environment name: 'production' or a custom saved environment"
    ),
) -> None:
    """Switch to a different environment"""
    _set_environment(env)


@app.command(name="save")
def save_env(name: str = typer.Argument(..., help="Name for the environment")) -> None:
    """Save current config as environment (including API key)"""
    _save_environment(name)


@app.command(name="envs")
def list_envs() -> None:
    """List available environments"""
    _list_environments()
