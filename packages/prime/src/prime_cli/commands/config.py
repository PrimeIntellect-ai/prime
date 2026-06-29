from __future__ import annotations

import os
import re

from pydantic import AliasChoices, Field
from pydantic_config import BaseConfig
from rich.table import Table

from prime_cli.core import Config as PrimeConfig

from ..client import APIClient, APIError
from ..utils import get_console
from ..utils.prompt import confirm, prompt
from .teams import fetch_teams

console = get_console()

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


def view(config: ConfigViewConfig) -> None:
    """View current configuration"""
    settings = PrimeConfig().view()

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


def set_api_key(config: ConfigSetApiKeyConfig) -> None:
    """Set your API key (prompts securely if not provided)"""
    api_key = config.api_key

    if api_key is None:
        # Interactive mode with secure prompt
        api_key = prompt(
            "Enter your Prime Intellect API key (or press Enter to clear)",
            hide_input=True,
            default="",
        )

    prime_config = PrimeConfig()
    prime_config.set_api_key(api_key)

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
                    prime_config.set_user_id(user_id)
                    prime_config.update_current_environment_file()
        except (APIError, Exception):
            pass

        console.print(f"[green]API key {masked_key} configured successfully![/green]")
        console.print("[blue]You can verify your API key with 'prime config view'[/blue]")
        console.print(
            "\n[yellow]Tip: Get your API key at https://app.primeintellect.ai/dashboard/tokens[/yellow]"
        )
    else:
        console.print("[green]API key cleared successfully![/green]")


def set_team_id(config: ConfigSetTeamIdConfig) -> None:
    """Set your team ID."""
    team_id = config.team_id

    prime_config = PrimeConfig()

    # Validate team ID format
    if not validate_team_id(team_id):
        console.print(
            "[red]Error: Invalid team ID format. "
            "Team ID must be a CUID v1 (start with 'c' followed by 24 lowercase "
            "alphanumeric characters).[/red]"
        )
        raise SystemExit(1)

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

    prime_config.set_team(team_id, team_name=team_name, team_role=team_role)
    if team_id:
        if team_name:
            console.print(f"[green]Team '{team_name}' ({team_id}) configured successfully![/green]")
        else:
            console.print(f"[green]Team ID '{team_id}' configured successfully![/green]")
    else:
        console.print("[green]Team ID cleared. Using personal account.[/green]")


def remove_team_id(config: ConfigRemoveTeamIdConfig) -> None:
    """Remove team ID to use personal account"""
    PrimeConfig().set_team(None)
    console.print("[green]Team ID removed. Using personal account.[/green]")


def set_base_url(config: ConfigSetBaseUrlConfig) -> None:
    """Set the API base URL (prompts if not provided)"""
    url = config.url

    if not url:
        prime_config = PrimeConfig()
        url = prompt(
            "Enter the base URL for the Prime Intellect API",
            default=prime_config.base_url,
        )
        if not url:
            console.print("[red]Base URL is required[/red]")
            return

    PrimeConfig().set_base_url(url)
    console.print(f"[green]Base URL set to: {url}[/green]")


def set_frontend_url(config: ConfigSetFrontendUrlConfig) -> None:
    """Set the frontend URL (prompts if not provided)"""
    url = config.url

    if not url:
        prime_config = PrimeConfig()
        url = prompt(
            "Enter the frontend URL for the Prime Intellect web app",
            default=prime_config.frontend_url,
        )
        if not url:
            console.print("[red]Frontend URL is required[/red]")
            return

    PrimeConfig().set_frontend_url(url)
    console.print(f"[green]Frontend URL set to: {url}[/green]")


def set_inference_url(config: ConfigSetInferenceUrlConfig) -> None:
    """Set the inference URL (prompts if not provided)"""
    url = config.url

    if not url:
        prime_config = PrimeConfig()
        url = prompt(
            "Enter the inference URL for Prime Inference API",
            default=prime_config.inference_url,
        )
        if not url:
            console.print("[red]Inference URL is required[/red]")
            return

    PrimeConfig().set_inference_url(url)
    console.print(f"[green]Inference URL set to: {url}[/green]")


# Helper functions (not commands)
def _set_environment(
    env: str,
) -> None:
    """Set URLs for a specific environment"""
    config = PrimeConfig()

    # Try to load the environment (handles both built-in and custom)
    try:
        if config.load_environment(env):
            console.print(f"[green]Switched to environment '{env}'![/green]")
        else:
            console.print(f"[red]Unknown environment: {env}[/red]")
            console.print("[yellow]Available environments:[/yellow]")
            for env_name in config.list_environments():
                console.print(f"  - {env_name}")
            raise SystemExit(1)
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise SystemExit(1)

    console.print("[blue]Run 'prime config view' to see the current configuration[/blue]")


def _save_environment(
    name: str,
) -> None:
    """Save current configuration as a named environment (including API key)"""
    try:
        config = PrimeConfig()
        config.save_environment(name)
        console.print(f"[green]Saved current configuration as environment '{name}'![/green]")
        console.print("[yellow]Note: This includes your API key and team ID[/yellow]")
        console.print(f"[blue]Use 'prime config use {name}' to load it later[/blue]")
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise SystemExit(1)


def _list_environments() -> None:
    """List all available environments"""
    config = PrimeConfig()
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
        config = PrimeConfig()
        config.delete_environment(name)
        console.print(f"[green]Deleted environment '{name}'![/green]")
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise SystemExit(1)


def set_share_resources_with_team(config: ConfigSetShareResourcesWithTeamConfig) -> None:
    """Set whether to automatically share new resources with all team members"""
    enabled = config.enabled

    value = enabled.lower()
    if value not in ("true", "false"):
        console.print("[red]Error: Value must be 'true' or 'false'[/red]")
        raise SystemExit(1)

    PrimeConfig().set_share_resources_with_team(value == "true")
    console.print(f"[green]Share resources with team set to: {value}[/green]")


def set_ssh_key_path(config: ConfigSetSshKeyPathConfig) -> None:
    """Set the SSH private key path"""
    path = config.path

    PrimeConfig().set_ssh_key_path(path)
    console.print("[green]SSH key path configured successfully![/green]")


def reset(config: ConfigResetConfig) -> None:
    """Reset configuration to defaults"""
    yes = config.yes

    if yes or confirm("Are you sure you want to reset all settings?"):
        prime_config = PrimeConfig()
        prime_config.set_api_key("")
        prime_config.set_team(None)
        prime_config.set_base_url(PrimeConfig.DEFAULT_BASE_URL)
        prime_config.set_frontend_url(PrimeConfig.DEFAULT_FRONTEND_URL)
        prime_config.set_ssh_key_path(PrimeConfig.DEFAULT_SSH_KEY_PATH)
        prime_config.set_current_environment("production")
        console.print("[green]Configuration reset to defaults![/green]")


# Environment commands
def use_environment(config: ConfigUseConfig) -> None:
    """Switch to a different environment"""
    env = config.env

    _set_environment(env)


def save_env(config: ConfigSaveConfig) -> None:
    """Save current config as environment (including API key)"""
    name = config.name

    _save_environment(name)


def delete_env(config: ConfigDeleteConfig) -> None:
    """Delete a saved environment"""
    name = config.name

    _delete_environment(name)


def list_envs(config: ConfigEnvsConfig) -> None:
    """List available environments"""
    _list_environments()


# --- inlined config schemas (previously in config_configs) ---
class ConfigDeleteConfig(BaseConfig):
    """Delete a saved environment"""

    name: str = Field(..., description="Name of the saved environment")


class ConfigEnvsConfig(BaseConfig):
    """List available environments"""

    pass


class ConfigRemoveTeamIdConfig(BaseConfig):
    """Remove team ID to use personal account"""

    pass


class ConfigResetConfig(BaseConfig):
    """Reset configuration to defaults"""

    yes: bool = Field(
        False, validation_alias=AliasChoices("yes", "y"), description="Skip confirmation prompt"
    )


class ConfigSaveConfig(BaseConfig):
    """Save current config as environment (including API key)"""

    name: str = Field(..., description="Name for the environment")


class ConfigSetApiKeyConfig(BaseConfig):
    """Set your API key (prompts securely if not provided)"""

    api_key: str | None = Field(
        None,
        description="Your Prime Intellect API key. If not provided, you'll be prompted securely.",
    )


class ConfigSetBaseUrlConfig(BaseConfig):
    """Set the API base URL (prompts if not provided)"""

    url: str | None = Field(
        None,
        description="Base URL for the Prime Intellect API. If not provided, you'll be prompted.",
    )


class ConfigSetFrontendUrlConfig(BaseConfig):
    """Set the frontend URL (prompts if not provided)"""

    url: str | None = Field(
        None,
        description="Prime Intellect web app URL. Prompts when omitted.",
    )


class ConfigSetInferenceUrlConfig(BaseConfig):
    """Set the inference URL (prompts if not provided)"""

    url: str | None = Field(
        None,
        description="Inference URL for Prime Inference API. If not provided, you'll be prompted.",
    )


class ConfigSetShareResourcesWithTeamConfig(BaseConfig):
    """Set whether to automatically share new resources with all team members"""

    enabled: str = Field(..., description="Enable or disable auto-sharing with team: true or false")


class ConfigSetSshKeyPathConfig(BaseConfig):
    """Set the SSH private key path"""

    path: str = Field(..., description="Path to your SSH private key file")


class ConfigSetTeamIdConfig(BaseConfig):
    """Set your team ID."""

    team_id: str = Field(..., description="Your Prime Intellect team ID.")


class ConfigUseConfig(BaseConfig):
    """Switch to a different environment"""

    env: str = Field(
        ..., description="Environment name: 'production' or a custom saved environment"
    )


class ConfigViewConfig(BaseConfig):
    """View current configuration"""

    pass
