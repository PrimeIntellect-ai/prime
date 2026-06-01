import os

import typer

from prime_cli.core import Config

from ..utils import PlainTyper, get_console

app = PlainTyper(help="Log out of Prime Intellect", no_args_is_help=False)
console = get_console()


@app.callback(invoke_without_command=True)
def logout(
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
) -> None:
    """Clear the stored API key, team selection, and user id."""
    config = Config()

    if not config.config.get("api_key") and not config.config.get("user_id"):
        console.print("[yellow]Not logged in.[/yellow]")
        if os.getenv("PRIME_API_KEY"):
            console.print(
                "[dim]PRIME_API_KEY is set in your environment; unset it to fully log out.[/dim]"
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
    config.update_current_environment_file()

    console.print("[green]Logged out.[/green]")

    if os.getenv("PRIME_API_KEY"):
        console.print(
            "[yellow]Note:[/yellow] PRIME_API_KEY is set in your environment "
            "and will still authenticate requests. Unset it to fully log out."
        )
