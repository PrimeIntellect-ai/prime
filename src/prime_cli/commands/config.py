import os
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from ..api.client import APIClient, APIError
from ..config import Config

app = typer.Typer(help="Configure the CLI", no_args_is_help=True)
console = Console()


@app.command()
def view() -> None:
    """View current configuration"""
    config = Config()
    settings = config.view()

    table = Table(title="Prime CLI Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    def _env_set(*names: str) -> bool:
        return any((val := os.getenv(n)) and val.strip() for n in names)

    # Show current environment
    table.add_row("Current Environment", settings["current_environment"])

    api_key = settings["api_key"]
    if api_key:
        masked_key = f"{api_key[:6]}...{api_key[-4:]}" if len(api_key) > 10 else "***"
        if _env_set("PRIME_API_KEY"):
            masked_key += " (from env var)"
    else:
        masked_key = "Not set"
    table.add_row("API Key", masked_key)

    # Show Team ID
    team_id = settings["team_id"]
    team_label = team_id or "Personal Account"
    if team_id and _env_set("PRIME_TEAM_ID"):
        team_label += " (from env var)"
    table.add_row("Team ID", team_label)

    # Show User ID
    user_id = settings.get("user_id")
    user_label = user_id or "Not set"
    if user_id and _env_set("PRIME_USER_ID"):
        user_label += " (from env var)"
    table.add_row("User ID", user_label)

    # Show base URL
    base_label = settings["base_url"]
    if _env_set("PRIME_API_BASE_URL", "PRIME_BASE_URL"):
        base_label += " (from env var)"
    table.add_row("Base URL", base_label)

    # Show frontend URL
    front_label = settings["frontend_url"]
    if _env_set("PRIME_FRONTEND_URL"):
        front_label += " (from env var)"
    table.add_row("Frontend URL", front_label)

    # Show inference URL
    inf_label = settings["inference_url"]
    if _env_set("PRIME_INFERENCE_URL"):
        inf_label += " (from env var)"
    table.add_row("Inference URL", inf_label)

    # Show SSH key path
    ssh_label = settings["ssh_key_path"]
    if _env_set("PRIME_SSH_KEY_PATH"):
        ssh_label += " (from env var)"
    table.add_row("SSH Key Path", ssh_label)

    console.print(table)


@app.command()
def set_api_key(
    api_key: Optional[str] = typer.Argument(
        None,
        help="Your Prime Intellect API key. If not provided, you'll be prompted securely.",
    ),
) -> None:
    """Set your API key (prompts securely if not provided)"""
    if api_key is None:
        # Interactive mode with secure prompt
        api_key = typer.prompt(
            "Enter your Prime Intellect API key (or press Enter to clear)",
            hide_input=True,
            confirmation_prompt=False,
            default="",
        )

    config = Config()
    config.set_api_key(api_key)

    if api_key:
        masked_key = f"{api_key[:6]}***{api_key[-4:]}" if len(api_key) > 10 else "***"

        # Try to fetch user id like in login flow
        try:
            client = APIClient(api_key=api_key)
            whoami_resp = client.get("/user/whoami")
            data = whoami_resp.get("data") if isinstance(whoami_resp, dict) else None
            if isinstance(data, dict):
                user_id = data.get("id")
                if user_id:
                    config.set_user_id(user_id)
                    config.update_current_environment_file()
        except (APIError, Exception):
            pass

        console.print(f"[green]API key {masked_key} configured successfully![/green]")
        console.print("[blue]You can verify your API key with 'prime config view'[/blue]")
        console.print(
            "\n[yellow]Tip: Get your API key at https://app.primeintellect.ai/dashboard/tokens[/yellow]"
        )
    else:
        console.print("[green]API key cleared successfully![/green]")


@app.command()
def set_team_id(
    team_id: Optional[str] = typer.Argument(
        None,
        help="Your Prime Intellect team ID. Leave empty for personal account.",
    ),
) -> None:
    """Set your team ID. Empty team ID means personal account."""
    if team_id is None:
        # Interactive mode with prompt
        team_id = typer.prompt(
            "Enter your Prime Intellect team ID (leave empty for personal account)",
            default="",
        )

    config = Config()
    config.set_team_id(team_id)
    if team_id:
        console.print(f"[green]Team ID '{team_id}' configured successfully![/green]")
    else:
        console.print("[green]Team ID cleared. Using personal account.[/green]")


@app.command()
def remove_team_id() -> None:
    """Remove team ID to use personal account"""
    config = Config()
    config.set_team_id(None)
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


@app.command()
def set_inference_url(
    url: Optional[str] = typer.Argument(
        None,
        help="Inference URL for Prime Inference API. If not provided, you'll be prompted.",
    ),
) -> None:
    """Set the inference URL (prompts if not provided)"""
    if not url:
        config = Config()
        url = typer.prompt(
            "Enter the inference URL for Prime Inference API",
            default=config.inference_url,
        )
        if not url:
            console.print("[red]Inference URL is required[/red]")
            return

    config = Config()
    config.set_inference_url(url)
    console.print(f"[green]Inference URL set to: {url}[/green]")


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


@app.command(no_args_is_help=True)
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
        config.set_team_id(None)
        config.set_base_url(Config.DEFAULT_BASE_URL)
        config.set_frontend_url(Config.DEFAULT_FRONTEND_URL)
        config.set_ssh_key_path(Config.DEFAULT_SSH_KEY_PATH)
        config.set_current_environment("production")
        console.print("[green]Configuration reset to defaults![/green]")


# Environment commands
@app.command(name="use", no_args_is_help=True)
def use_environment(
    env: str = typer.Argument(
        ..., help="Environment name: 'production' or a custom saved environment"
    ),
) -> None:
    """Switch to a different environment"""
    _set_environment(env)


@app.command(name="save", no_args_is_help=True)
def save_env(name: str = typer.Argument(..., help="Name for the environment")) -> None:
    """Save current config as environment (including API key)"""
    _save_environment(name)


@app.command(name="envs")
def list_envs() -> None:
    """List available environments"""
    _list_environments()
