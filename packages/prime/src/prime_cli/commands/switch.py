from typing import Optional

import questionary
import typer

from prime_cli.core import Config

from ..client import APIClient, APIError
from ..utils import PlainTyper, get_console
from .teams import fetch_teams

app = PlainTyper(
    help="Switch between your personal account and team contexts",
    no_args_is_help=False,
)
console = get_console()

PERSONAL_TARGET = "personal"


def _account_choices(teams: list[dict], current_team_id: Optional[str]) -> list[questionary.Choice]:
    personal_label = "Personal (current)" if current_team_id is None else "Personal"
    choices = [questionary.Choice(personal_label, value=PERSONAL_TARGET)]
    for team in teams:
        name = team.get("name", "Unknown")
        slug = str(team.get("slug") or "").strip()
        role = str(team.get("role", "member")).lower()
        current = " (current)" if team.get("teamId") == current_team_id else ""
        details = f"slug: {slug}, role: {role}" if slug else f"role: {role}"
        choices.append(questionary.Choice(f"{name} ({details}){current}", value=team))
    return choices


def _select_team_by_target(teams: list[dict], target: str) -> Optional[dict]:
    normalized_target = target.strip().lower()

    for team in teams:
        slug = team.get("slug")
        if slug and slug.strip().lower() == normalized_target:
            return team

    for team in teams:
        team_id = team.get("teamId")
        if team_id and team_id.strip().lower() == normalized_target:
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
    target: Optional[str] = typer.Argument(
        None, help=f"'{PERSONAL_TARGET}', a team slug, or a team ID"
    ),
) -> None:
    """Switch the active account context."""
    config = Config()

    if config.team_id_from_env:
        console.print(
            "[red]Error:[/red] PRIME_TEAM_ID is set in your environment. "
            "Clear it before using [bold]prime switch[/bold]."
        )
        raise typer.Exit(1)

    if target is not None:
        normalized_target = target.strip().lower()
        if normalized_target == PERSONAL_TARGET:
            _switch_to_personal(config)
            return

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
        selected_team = _select_team_by_target(teams, normalized_target)
        if selected_team is None:
            console.print(f"[red]Team '{target}' not found.[/red]")
            _print_available_slugs(teams)
            raise typer.Exit(1)

        _switch_to_team(config, selected_team)
        return

    current_team_id = config.team_id
    choices = _account_choices(teams, current_team_id)

    selected = questionary.select("Switch account", choices=choices).ask()
    if selected is None:
        raise typer.Exit(1)
    if selected == "personal":
        _switch_to_personal(config)
        return
    _switch_to_team(config, selected)
