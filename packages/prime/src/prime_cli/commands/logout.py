import os

import typer

from prime_cli.core import Config

from ..utils import PlainTyper, get_console, require_persistent_context

app = PlainTyper(help="Log out of Prime Intellect", no_args_is_help=False)
console = get_console()


_ENV_OVERRIDES = ("PRIME_API_KEY", "PRIME_TEAM_ID", "PRIME_USER_ID")


@app.callback(invoke_without_command=True)
def logout(
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
) -> None:
    """Clear the stored API key, team selection, and user id."""
    require_persistent_context()
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
    if not yes and not typer.confirm(
        f"Log out of '{env_name}' (clears API key, team, and user id)?",
        default=True,
    ):
        raise typer.Exit(0)

    config.set_api_key("")
    config.set_team(None)
    config.set_user_id(None)
    # Mirrors stored values only, so PRIME_* shell vars never leak back onto disk.
    config.update_current_environment_file()

    console.print("[green]Logged out.[/green]")

    set_overrides = [name for name in _ENV_OVERRIDES if os.getenv(name)]
    if set_overrides:
        console.print(
            f"[yellow]Note:[/yellow] {', '.join(set_overrides)} set in your environment "
            "and will override the cleared config. Unset to fully log out."
        )
