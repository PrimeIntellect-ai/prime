from typing import Optional

import typer
from click.exceptions import Abort
from rich.console import Console

from prime_cli.core import Config

from ..client import APIClient, APIError
from .teams import fetch_teams

app = typer.Typer(
    help="Switch between your personal account and team contexts",
    no_args_is_help=False,
)
console = Console()


def _select_team_by_target(teams: list[dict], target: str) -> Optional[dict]:
    normalized_target = target.strip().lower()
    for team in teams:
        team_slug = str(team.get("slug", "")).strip().lower()
        if team_slug and team_slug == normalized_target:
            return team

    for team in teams:
        team_id = str(team.get("teamId", "")).strip().lower()
        if team_id and team_id == normalized_target:
            return team

    return None


def _switch_to_personal(config: Config) -> None:
    config.set_team(None)
    config.update_current_environment_file()
    console.print("[green]Switched to personal account.[/green]")


def _switch_to_team(config: Config, team: dict) -> None:
    team_id = team.get("teamId")
    team_name = team.get("name", "Unknown")
    team_role = team.get("role", "member")

    if not team_id:
        console.print("[red]Error:[/red] Selected team is missing a team ID.")
        raise typer.Exit(1)

    config.set_team(team_id, team_name=team_name, team_role=team_role)
    config.update_current_environment_file()
    console.print(f"[green]Switched to team '{team_name}'.[/green]")


def _print_available_slugs(teams: list[dict]) -> None:
    slugs = [str(team.get("slug", "")).strip() for team in teams if team.get("slug")]
    if slugs:
        console.print(f"[dim]Available teams: {', '.join(sorted(slugs))}[/dim]")


@app.callback(invoke_without_command=True)
def switch(
    target: Optional[str] = typer.Argument(None, help="'personal', a team slug, or a team ID"),
) -> None:
    """Switch the active account context."""
    config = Config()

    if config.team_id_from_env:
        console.print(
            "[red]Error:[/red] PRIME_TEAM_ID is set in your environment. "
            "Clear it before using [bold]prime switch[/bold]."
        )
        raise typer.Exit(1)

    try:
        client = APIClient()
        teams = fetch_teams(client)
    except APIError as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {str(e)}")
        raise typer.Exit(1)

    if target is not None:
        normalized_target = target.strip().lower()
        if normalized_target == "personal":
            _switch_to_personal(config)
            return

        selected_team = _select_team_by_target(teams, normalized_target)
        if selected_team is None:
            console.print(f"[red]Team '{target}' not found.[/red]")
            _print_available_slugs(teams)
            raise typer.Exit(1)

        _switch_to_team(config, selected_team)
        return

    console.print("\n[bold]Switch account:[/bold]\n")
    current_team_id = config.team_id

    personal_label = "Personal"
    if current_team_id is None:
        personal_label += " [green](current)[/green]"
    console.print(f"  [cyan](1)[/cyan] {personal_label}")

    for idx, team in enumerate(teams, start=2):
        name = team.get("name", "Unknown")
        slug = str(team.get("slug") or "").strip()
        role = str(team.get("role", "member")).lower()
        current_badge = " [green](current)[/green]" if team.get("teamId") == current_team_id else ""
        details = f"slug: {slug}, role: {role}" if slug else f"role: {role}"
        console.print(f"  [cyan]({idx})[/cyan] {name} [dim]({details})[/dim]{current_badge}")

    while True:
        try:
            selection = typer.prompt("Select", type=int, default=1)
            if selection == 1:
                _switch_to_personal(config)
                return
            if 2 <= selection <= len(teams) + 1:
                _switch_to_team(config, teams[selection - 2])
                return
            console.print(f"[red]Invalid selection. Enter 1-{len(teams) + 1}.[/red]")
        except Abort:
            raise typer.Exit(1)
