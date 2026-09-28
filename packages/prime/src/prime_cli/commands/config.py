import os
import re
from pathlib import Path
from typing import Optional

import typer
from rich.markup import escape
from rich.table import Table
from rich.text import Text

from prime_cli.core import Config
from prime_cli.core.config import (
    find_local_context_file,
    read_local_context,
    write_local_context,
)

from ..client import APIClient, APIError
from ..utils import PlainTyper, get_console, require_persistent_context
from ..utils.context import (
    apply_team,
    describe_local_context,
    local_context_target,
    require_loadable_config,
)
from .teams import fetch_teams

app = PlainTyper(help="Configure the CLI", no_args_is_help=True)
console = get_console()


@app.callback()
def _config_callback(ctx: typer.Context) -> None:
    # `unpin` must still work when the directory context it removes is broken.
    if ctx.invoked_subcommand != "unpin":
        require_loadable_config()


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

    local_file = config.local_context_file
    pinned_context = config.local_context.get("context")

    # Show current environment
    env_label = settings["current_environment"]
    if config.context_override:
        env_label += " (from --context)"
    elif pinned_context:
        env_label += " (directory context)"
    table.add_row("Current Environment", Text(env_label))
    table.add_row("Directory Context", Text(str(local_file) if local_file else "None"))

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
    if config.team_pinned and not team_from_env:
        team_label += " (directory context)"
    table.add_row("Team", Text(team_label))

    # Show User
    user_id = settings.get("user_id")
    if user_id:
        if _env_set("PRIME_USER_ID"):
            user_label = f"{user_id} (from env var)"
        else:
            user_name = settings.get("user_name")
            user_label = f"{user_name} ({user_id})" if user_name else user_id
    else:
        user_label = "Not set"
    table.add_row("User", Text(user_label))

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

    # Show traces URL (effective value: falls back to the traces service default)
    traces_label = settings["traces_url"]
    if _env_set("PRIME_TRACES_URL"):
        traces_label += " (from env var)"
    table.add_row("Traces URL", Text(traces_label))

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
    require_persistent_context()

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
                    config.set_user_id(user_id, user_name=data.get("name"))
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
    team_id: str = typer.Argument(
        ...,
        help="Your Prime Intellect team ID.",
    ),
) -> None:
    """Set your team ID."""
    require_persistent_context()
    config = Config()

    # Validate team ID format
    if not validate_team_id(team_id):
        console.print(
            "[red]Error: Invalid team ID format. "
            "Team ID must be a CUID v1 (start with 'c' followed by 24 lowercase "
            "alphanumeric characters).[/red]"
        )
        raise typer.Exit(code=1)

    team_name = None
    team_role = None
    if team_id:
        try:
            client = APIClient()
            teams = fetch_teams(client)
            for team in teams:
                if team.get("teamId") == team_id:
                    team_name = team.get("name")
                    team_role = team.get("role")
                    break
        except (APIError, Exception):
            pass

    where = apply_team(config, config.local_context_file, team_id, team_name, team_role)
    if team_id:
        if team_name:
            console.print(
                f"[green]Team '{team_name}' ({team_id}) configured successfully{where}![/green]"
            )
        else:
            console.print(f"[green]Team ID '{team_id}' configured successfully{where}![/green]")
    else:
        console.print(f"[green]Team ID cleared. Using personal account{where}.[/green]")


@app.command()
def remove_team_id() -> None:
    """Remove team ID to use personal account"""
    require_persistent_context()
    config = Config()
    where = apply_team(config, config.local_context_file, None)
    console.print(f"[green]Team ID removed. Using personal account{where}.[/green]")


@app.command()
def set_base_url(
    url: Optional[str] = typer.Argument(
        None,
        help="Base URL for the Prime Intellect API. If not provided, you'll be prompted.",
    ),
) -> None:
    """Set the API base URL (prompts if not provided)"""
    require_persistent_context()

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
    require_persistent_context()

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
    require_persistent_context()

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


@app.command()
def set_traces_url(
    url: Optional[str] = typer.Argument(
        None,
        help=(
            "URL of the Prime Traces service. Pass '' or - to clear the override "
            "and follow the base URL. If not provided, you'll be prompted."
        ),
    ),
) -> None:
    """Set the Prime Traces service URL (prompts if not provided)"""
    require_persistent_context()

    if url is None:
        config = Config()
        url = typer.prompt(
            "Enter the URL of the Prime Traces service ('-' follows the base URL)",
            default=config._configured_traces_url() or "",
        )

    if url == "-":
        url = ""

    config = Config()
    try:
        config.set_traces_url_for_active_environment(url)
    except ValueError as e:
        console.print(f"[red]Error: {escape(str(e))}[/red]")
        raise typer.Exit(1)
    if url:
        console.print(f"[green]Traces URL set to: {escape(url)}[/green]")
    else:
        console.print("[green]Traces URL override cleared; following the base URL[/green]")


# Helper functions (not commands)
def _set_environment(env: str, local: bool = False, global_: bool = False) -> None:
    """Set URLs for a specific environment"""
    require_persistent_context()
    config = Config()
    target = local_context_target(config, local, global_)
    if target is not None:
        _pin_environment(config, env, target)
        return
    if config.local_context_file is not None:
        # --global from inside a directory context: write the global config.
        config = Config(use_context=False)

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


def _pin_environment(config: Config, env: str, target: Path) -> None:
    """Select a saved environment for a directory via its context file."""
    try:
        known = {name.casefold(): name for name in config.list_environments()}
        config._sanitize_environment_name(env)
    except ValueError as e:
        console.print(f"[red]Error: {escape(str(e))}[/red]")
        raise typer.Exit(1)
    if env.casefold() not in known:
        console.print(f"[red]Unknown environment: {escape(env)}[/red]")
        console.print("[yellow]Available environments:[/yellow]")
        for env_name in known.values():
            console.print(f"  - {escape(env_name)}")
        raise typer.Exit(1)

    data = read_local_context(target) if target.is_file() else {}
    data["context"] = known[env.casefold()]
    # A pinned team belongs to the previous environment's account.
    for key in ("team_id", "team_name", "team_role"):
        data.pop(key, None)
    write_local_context(target, data)
    console.print(
        f"[green]Using environment '{escape(known[env.casefold()])}' "
        f"{describe_local_context(target)}.[/green]"
    )
    console.print(
        "[dim]Commands and SDKs run in this directory now use it; "
        "'prime config unpin' removes it.[/dim]"
    )


def _save_environment(
    name: str,
) -> None:
    """Save current configuration as a named environment (including API key)"""
    require_persistent_context()
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


def _delete_environment(
    name: str,
) -> None:
    """Delete a named saved environment."""
    require_persistent_context()
    try:
        config = Config()
        config.delete_environment(name)
        console.print(f"[green]Deleted environment '{name}'![/green]")
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command(no_args_is_help=True)
def set_share_resources_with_team(
    enabled: str = typer.Argument(
        ...,
        help="Enable or disable auto-sharing with team: true or false",
    ),
) -> None:
    """Set whether to automatically share new resources with all team members"""
    require_persistent_context()
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
    require_persistent_context()
    config = Config()
    config.set_ssh_key_path(path)
    console.print("[green]SSH key path configured successfully![/green]")


@app.command()
def reset(
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
) -> None:
    """Reset configuration to defaults"""
    require_persistent_context()
    if yes or typer.confirm("Are you sure you want to reset all settings?"):
        config = Config(use_context=False)
        config.set_api_key("")
        config.set_team(None)
        config.set_user_id(None)
        config.set_base_url(Config.DEFAULT_BASE_URL)
        config.set_frontend_url(Config.DEFAULT_FRONTEND_URL)
        config.set_inference_url(Config.DEFAULT_INFERENCE_URL)
        config.set_traces_url("")
        config.set_ssh_key_path(Config.DEFAULT_SSH_KEY_PATH)
        config.set_current_environment("production")
        console.print("[green]Configuration reset to defaults![/green]")
        local_file = find_local_context_file()
        if local_file is not None:
            console.print(
                f"[yellow]Note:[/yellow] {escape(str(local_file))} still selects this "
                "directory's team or context; 'prime config unpin' removes it."
            )


# Environment commands
@app.command(name="use", no_args_is_help=True)
def use_environment(
    env: str = typer.Argument(
        ..., help="Environment name: 'production' or a custom saved environment"
    ),
    local: bool = typer.Option(
        False,
        "--local",
        help="Use it only in the current directory (writes ./.prime/context.json)",
    ),
    global_: bool = typer.Option(
        False,
        "--global",
        help="Change the global environment even inside a directory with its own context",
    ),
) -> None:
    """Switch to a different environment.

    Inside a directory with a .prime/context.json (see --local), this changes
    that directory's environment; elsewhere it changes the global one.
    """
    _set_environment(env, local=local, global_=global_)


@app.command(name="unpin")
def unpin() -> None:
    """Remove the directory context (.prime/context.json) in effect here"""
    path = find_local_context_file()
    if path is None:
        console.print("[yellow]No directory context applies here.[/yellow]")
        return
    write_local_context(path, {})
    console.print(f"[green]Removed directory context {escape(str(path))}.[/green]")


@app.command(name="save", no_args_is_help=True)
def save_env(name: str = typer.Argument(..., help="Name for the environment")) -> None:
    """Save current config as environment (including API key)"""
    _save_environment(name)


@app.command(name="delete", no_args_is_help=True)
def delete_env(name: str = typer.Argument(..., help="Name of the saved environment")) -> None:
    """Delete a saved environment"""
    _delete_environment(name)


@app.command(name="envs")
def list_envs() -> None:
    """List available environments"""
    _list_environments()
