import json
import os

import typer

from prime_cli.core import Config

from ..utils import PlainTyper, confirm, get_console

app = PlainTyper(help="Log out of Prime Intellect", no_args_is_help=False)
console = get_console()


_ENV_OVERRIDES = ("PRIME_API_KEY", "PRIME_TEAM_ID", "PRIME_USER_ID")


def _clear_env_file(config: Config) -> None:
    """Rewrite the current environment's saved file from raw cleared config values.

    Config.update_current_environment_file() reads env-precedence properties, so
    PRIME_* shell vars would otherwise leak back onto disk right after logout.
    """
    if config.current_environment == "production":
        return
    try:
        sanitized = config._sanitize_environment_name(config.current_environment)
    except ValueError:
        return
    env_file = config.environments_dir / f"{sanitized}.json"
    if not env_file.exists():
        return
    raw = config.config
    env_file.write_text(
        json.dumps(
            {
                "api_key": raw.get("api_key", ""),
                "team_id": raw.get("team_id"),
                "team_name": raw.get("team_name"),
                "team_role": raw.get("team_role"),
                "user_id": raw.get("user_id"),
                "base_url": raw.get("base_url", Config.DEFAULT_BASE_URL),
                "frontend_url": raw.get("frontend_url", Config.DEFAULT_FRONTEND_URL),
                "inference_url": raw.get("inference_url", Config.DEFAULT_INFERENCE_URL),
            },
            indent=2,
        )
    )


@app.callback(invoke_without_command=True)
def logout(
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
) -> None:
    """Clear the stored API key, team selection, and user id."""
    config = Config()

    raw = config.config
    if not raw.get("api_key") and not raw.get("user_id") and not raw.get("team_id"):
        console.print("[yellow]Not logged in.[/yellow]")
        set_overrides = [name for name in _ENV_OVERRIDES if os.getenv(name)]
        if set_overrides:
            console.print(
                f"[dim]{', '.join(set_overrides)} set in your environment; "
                "unset to fully log out.[/dim]"
            )
        raise typer.Exit(0)

    env_name = config.current_environment
    if not yes and not confirm(
        f"Log out of '{env_name}' (clears API key, team, and user id)?",
        default=True,
    ):
        raise typer.Exit(0)

    config.set_api_key("")
    config.set_team(None)
    config.set_user_id(None)
    _clear_env_file(config)

    console.print("[green]Logged out.[/green]")

    set_overrides = [name for name in _ENV_OVERRIDES if os.getenv(name)]
    if set_overrides:
        console.print(
            f"[yellow]Note:[/yellow] {', '.join(set_overrides)} set in your environment "
            "and will override the cleared config. Unset to fully log out."
        )
