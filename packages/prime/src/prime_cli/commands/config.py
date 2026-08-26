import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import typer
from rich.markup import escape
from rich.table import Table
from rich.text import Text

from prime_cli.core import Config
from prime_cli.core.config import (
    CONFIG_DIR_NAME,
    CONFIG_FILE_NAME,
    trust_local_config,
    untrust_local_config,
)

from ..client import APIClient, APIError
from ..utils import PlainTyper, get_console
from .teams import fetch_teams

app = PlainTyper(help="Configure the CLI", no_args_is_help=True)
console = get_console()

LOCAL_OPTION_HELP = (
    "Use a project-local config at ./.prime/config.json instead of ~/.prime/config.json. "
    "Commands run from this directory (or any directory below it) will use it."
)

TRUST_PATH_HELP = (
    "Project directory (or its .prime/ or config.json). Defaults to the current directory."
)

CONFIG_SOURCE_LABELS = {
    "env": "from PRIME_CONFIG_DIR",
    "local": "project-local",
    "global": "global",
    "explicit": "project-local",
}


def describe_config_file(config: Config) -> str:
    """The active config file path plus how it was chosen, for display."""
    label = CONFIG_SOURCE_LABELS.get(config.config_source, config.config_source)
    return f"{config.config_file} ({label})"


def _git_repo_root(path: Path) -> Optional[Path]:
    """The nearest enclosing git repository root (`.git` dir or worktree file), or None."""
    for directory in (path, *path.parents):
        if (directory / ".git").exists():
            return directory
    return None


def _is_git_ignored(repo_root: Path, path: Path) -> Optional[bool]:
    """Ask git whether `path` is ignored; None when git can't tell us."""
    git = shutil.which("git")
    if git is None:
        return None
    try:
        result = subprocess.run(
            [git, "-C", str(repo_root), "check-ignore", "-q", "--", str(path)],
            capture_output=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode == 0:
        return True
    if result.returncode == 1:
        return False
    return None


def _gitignore_mentions_prime(repo_root: Path, config_dir: Path) -> bool:
    """Fallback without git: scan .gitignore files between the repo root and the config."""
    for directory in (config_dir.parent, *config_dir.parent.parents):
        gitignore = directory / ".gitignore"
        try:
            patterns = gitignore.read_text().splitlines() if gitignore.is_file() else []
        except OSError:
            patterns = []
        if any(line.strip().strip("/") in (".prime", ".prime/") for line in patterns):
            return True
        if directory == repo_root:
            break
    return False


def print_local_config_notice(config: Config) -> None:
    """Tell the user where a project-local config landed and how to keep it out of git.

    The config may sit in a subdirectory of a repository, so look for the
    enclosing repo rather than only the config's parent, and let git decide
    whether the file is actually ignored.
    """
    console.print(f"[blue]Using project-local config: {config.config_file}[/blue]")
    config_dir = config.config_dir.resolve()
    repo_root = _git_repo_root(config_dir.parent)
    if repo_root is None:
        return
    ignored = _is_git_ignored(repo_root, config_dir / CONFIG_FILE_NAME)
    if ignored is None:
        ignored = _gitignore_mentions_prime(repo_root, config_dir)
    if ignored:
        return
    console.print(
        "[yellow]Warning: .prime/ is not ignored by the git repository at "
        f"{repo_root} — add it to .gitignore so your API key is never committed.[/yellow]"
    )


def new_local_config() -> Config:
    """A project-local Config for the working directory that is not written yet.

    Nothing touches disk until the first `set_*` call, so an aborted login
    does not leave an empty file behind that would shadow the global config.
    A brand-new local config inherits the service URLs of whatever config is
    currently active, so it keeps targeting the same deployment.
    """
    config = Config.local(create=False)
    if not config.config_file.exists():
        config.adopt_urls(Config())
    return config


def _resolve_local_config_file(path: Optional[Path]) -> Path:
    """Accept a project dir, its .prime dir, or the config.json itself."""
    target = (path or Path.cwd()).expanduser()
    if target.is_file():
        return target
    if target.name == CONFIG_DIR_NAME:
        return target / CONFIG_FILE_NAME
    return target / CONFIG_DIR_NAME / CONFIG_FILE_NAME


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

    table.add_row("Config File", describe_config_file(config))
    for ignored in config.untrusted_local_configs:
        table.add_row("Ignored Config", f"{ignored} (untrusted; see 'prime config trust')")

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

    # Show traces URL (effective value: falls back to the base URL)
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
    local: bool = typer.Option(False, "--local", help=LOCAL_OPTION_HELP),
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

    config = new_local_config() if local else Config()
    config.set_api_key(api_key)
    if local:
        print_local_config_notice(config)

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
    team_id: str = typer.Argument(
        ...,
        help="Your Prime Intellect team ID.",
    ),
) -> None:
    """Set your team ID."""
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

    config.set_team(team_id, team_name=team_name, team_role=team_role)
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


def _delete_environment(
    name: str,
) -> None:
    """Delete a named saved environment."""
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
        config.set_inference_url(Config.DEFAULT_INFERENCE_URL)
        config.set_traces_url("")
        config.set_ssh_key_path(Config.DEFAULT_SSH_KEY_PATH)
        config.set_current_environment("production")
        console.print("[green]Configuration reset to defaults![/green]")


@app.command()
def trust(
    path: Optional[Path] = typer.Argument(None, help=TRUST_PATH_HELP),
) -> None:
    """Allow a project-local .prime/config.json to be used by commands run below it.

    Discovered project configs are ignored until trusted, because one committed
    to a repository you clone could redirect your API key elsewhere. Trust is
    tied to the file's content: if it changes, run this again.
    """
    config_file = _resolve_local_config_file(path)
    if not config_file.is_file():
        console.print(f"[red]No config file at {config_file}[/red]")
        raise typer.Exit(1)
    try:
        resolved = trust_local_config(config_file)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    console.print(f"[green]Trusted project config {resolved}[/green]")


@app.command()
def untrust(
    path: Optional[Path] = typer.Argument(None, help=TRUST_PATH_HELP),
) -> None:
    """Stop using a project-local .prime/config.json."""
    config_file = _resolve_local_config_file(path)
    if untrust_local_config(config_file):
        console.print(f"[green]Untrusted project config {config_file.resolve()}[/green]")
    else:
        console.print(f"[yellow]{config_file.resolve()} was not trusted[/yellow]")


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


@app.command(name="delete", no_args_is_help=True)
def delete_env(name: str = typer.Argument(..., help="Name of the saved environment")) -> None:
    """Delete a saved environment"""
    _delete_environment(name)


@app.command(name="envs")
def list_envs() -> None:
    """List available environments"""
    _list_environments()
