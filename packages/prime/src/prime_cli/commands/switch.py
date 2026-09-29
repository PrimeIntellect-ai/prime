from pathlib import Path
from typing import Optional

import typer
from click.exceptions import Abort
from rich.markup import escape

from prime_cli.core import Config

from ..client import APIClient, APIError
from ..utils import PlainTyper, get_console, require_persistent_context
from ..utils.context import apply_team, local_context_target
from .teams import fetch_teams

app = PlainTyper(
    help="Switch between your personal account and team contexts",
    no_args_is_help=False,
    # Accept `prime switch edison --local` although the command is a group callback.
    context_settings={"allow_interspersed_args": True},
)
console = get_console()

PERSONAL_TARGET = "personal"


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


def _switch_to_personal(config: Config, target: Optional[Path]) -> None:
    where = apply_team(config, target, None)
    console.print(f"[green]Switched to personal account{where}.[/green]")


def _switch_to_team(config: Config, team: dict, target: Optional[Path]) -> None:
    team_id = team.get("teamId")
    team_name = team.get("name", "Unknown")
    team_role = team.get("role", "member")

    if not team_id:
        console.print("[red]Error:[/red] Selected team is missing a team ID.")
        raise typer.Exit(1)

    where = apply_team(config, target, team_id, team_name, team_role)
    console.print(f"[green]Switched to team '{escape(team_name)}'{where}.[/green]")


def _print_available_slugs(teams: list[dict]) -> None:
    slugs = [str(team.get("slug", "")).strip() for team in teams if team.get("slug")]
    if slugs:
        console.print(f"[dim]Available teams: {', '.join(sorted(slugs))}[/dim]")


@app.callback(invoke_without_command=True)
def switch(
    target: Optional[str] = typer.Argument(
        None, help=f"'{PERSONAL_TARGET}', a team slug, or a team ID"
    ),
    local: bool = typer.Option(
        False,
        "--local",
        help="Pin the team to this repository (writes .prime/context.json at the git root)",
    ),
    global_: bool = typer.Option(
        False,
        "--global",
        help="Change the global team even inside a directory with its own context",
    ),
) -> None:
    """Switch the active account context.

    Inside a directory with a .prime/context.json (see --local), this changes
    that directory's team; elsewhere it changes the global team.
    """
    require_persistent_context()
    config = Config()
    destination = local_context_target(config, local, global_)

    if config.team_id_from_env:
        console.print(
            "[red]Error:[/red] PRIME_TEAM_ID is set in your environment. "
            "Clear it before using [bold]prime switch[/bold]."
        )
        raise typer.Exit(1)

    if target is not None:
        normalized_target = target.strip().lower()
        if normalized_target == PERSONAL_TARGET:
            _switch_to_personal(config, destination)
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

        _switch_to_team(config, selected_team, destination)
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
                _switch_to_personal(config, destination)
                return
            if 2 <= selection <= len(teams) + 1:
                _switch_to_team(config, teams[selection - 2], destination)
                return
            console.print(f"[red]Invalid selection. Enter 1-{len(teams) + 1}.[/red]")
        except Abort:
            raise typer.Exit(1)
