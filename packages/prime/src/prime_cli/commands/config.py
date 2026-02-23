import os
import re
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from prime_cli.core import Config

from ..client import APIClient, APIError
from .teams import fetch_teams

app = typer.Typer(help="Configure the CLI", no_args_is_help=True)
console = Console()

# Team ID validation pattern: CUID (v1)
TEAM_ID_PATTERN = re.compile(r"^c[a-z0-9]{24}$")


def validate_team_id(team_id: str) -> bool:
    """Validate team ID format.

    Args:
        team_id: The team ID to validate

    Returns:
        True if valid, False otherwise
    """
    if not team_id:  # Empty string is valid (means personal account)
        return True
    return bool(TEAM_ID_PATTERN.match(team_id))


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

    # Show Team
    team_id = settings["team_id"]
    team_from_env = _env_set("PRIME_TEAM_ID")
    if team_id:
        if team_from_env:
            team_label = f"{team_id} (from env var)"
        else:
            team_name = settings.get("team_name")
            team_label = f"{team_name} ({team_id})" if team_name else team_id
    else:
        team_label = "Personal Account"
    table.add_row("Team", team_label)

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

    # Show share resources with team
    share_label = str(settings.get("share_resources_with_team", False))
    table.add_row("Share Resources With Team", share_label)

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
        help="Your Prime Intellect team ID. Leave empty for interactive selection.",
    ),
) -> None:
    """Set your team ID. Shows interactive team selection if no ID is provided."""
    config = Config()

    if team_id is None:
        # Interactive mode: fetch teams and let user pick
        try:
            client = APIClient()
            all_teams: list[dict] = []
            offset = 0
            limit = 100
            while True:
                response = client.get("/user/teams", params={"offset": offset, "limit": limit})
                batch = response.get("data", []) if isinstance(response, dict) else []
                all_teams.extend(batch)
                total = (
                    response.get("total_count", len(all_teams))
                    if isinstance(response, dict)
                    else len(all_teams)
                )
                if len(all_teams) >= total or not batch:
                    break
                offset += limit

            if not all_teams:
                console.print(
                    "[yellow]You are not a member of any team. Using personal account.[/yellow]"
                )
                config.set_team(None)
                return

            console.print("\n[bold]Select a team:[/bold]\n")
            console.print("  [cyan]0[/cyan] - Personal Account (no team)")
            for idx, t in enumerate(all_teams, start=1):
                name = t.get("name", "Unknown")
                slug = t.get("slug", "")
                role = t.get("role", "")
                slug_part = f" ({slug})" if slug else ""
                console.print(f"  [cyan]{idx}[/cyan] - {name}{slug_part} [dim]{role}[/dim]")

            console.print()
            choice = typer.prompt(
                "Enter number",
                type=int,
                default=0,
            )

            if choice == 0:
                config.set_team(None)
                console.print("[green]Team ID cleared. Using personal account.[/green]")
                return

            if choice < 0 or choice > len(all_teams):
                console.print("[red]Error: Invalid selection.[/red]")
                raise typer.Exit(1)

            selected = all_teams[choice - 1]
            team_id = selected.get("teamId", "")
            team_name = selected.get("name")
            team_role = selected.get("role")
            config.set_team(team_id, team_name=team_name, team_role=team_role)
            console.print(f"[green]Team '{team_name}' ({team_id}) configured successfully![/green]")
            return

        except APIError as e:
            console.print(f"[red]Error fetching teams:[/red] {str(e)}")
            console.print(
                "[yellow]Falling back to manual input. You can enter a team ID directly.[/yellow]\n"
            )
            team_id = typer.prompt(
                "Enter your Prime Intellect team ID (leave empty for personal account)",
                default="",
            )
        except Exception as e:
            console.print(f"[red]Unexpected error:[/red] {str(e)}")
            console.print(
                "[yellow]Falling back to manual input. You can enter a team ID directly.[/yellow]\n"
            )
            team_id = typer.prompt(
                "Enter your Prime Intellect team ID (leave empty for personal account)",
                default="",
            )

    # Validate team ID format
    if not validate_team_id(team_id):
        console.print(
            "[red]Error: Invalid team ID format. "
            "Team ID must be a CUID v1 (start with 'c' followed by 24 lowercase "
            "alphanumeric characters).[/red]"
        )
        raise typer.Exit(code=1)

    team_name = None
    if team_id:
        try:
            client = APIClient()
            teams = fetch_teams(client)
            for team in teams:
                if team.get("teamId") == team_id:
                    team_name = team.get("name")
                    break
        except (APIError, Exception):
            pass

    config.set_team(team_id, team_name=team_name)
    if team_id:
        if team_name:
            console.print(f"[green]Team '{team_name}' ({team_id}) configured successfully![/green]")
        else:
            console.print(f"[green]Team ID '{team_id}' configured successfully![/green]")
    else:
        console.print("[green]Team ID cleared. Using personal account.[/green]")


@app.command()
def remove_team_id() -> None:
    """Remove team ID to use personal account"""
    config = Config()
    config.set_team(None)
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
def set_share_resources_with_team(
    enabled: str = typer.Argument(
        ...,
        help="Enable or disable auto-sharing with team: true or false",
    ),
) -> None:
    """Set whether to automatically share new resources with all team members"""
    value = enabled.lower()
    if value not in ("true", "false"):
        console.print("[red]Error: Value must be 'true' or 'false'[/red]")
        raise typer.Exit(1)

    config = Config()
    config.set_share_resources_with_team(value == "true")
    console.print(f"[green]Share resources with team set to: {value}[/green]")


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
        config.set_team(None)
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
