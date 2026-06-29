from typing import Optional

from prime_cli.command_configs import SwitchConfig
from prime_cli.core import Config as PrimeConfig

from ..client import APIClient, APIError
from ..utils import get_console
from ..utils.prompt import prompt
from .teams import fetch_teams

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


def _switch_to_personal(config: PrimeConfig) -> None:
    config.set_team(None)
    config.update_current_environment_file()
    console.print("[green]Switched to personal account.[/green]")


def _switch_to_team(config: PrimeConfig, team: dict) -> None:
    team_id = team.get("teamId")
    team_name = team.get("name", "Unknown")
    team_role = team.get("role", "member")

    if not team_id:
        console.print("[red]Error:[/red] Selected team is missing a team ID.")
        raise SystemExit(1)

    config.set_team(team_id, team_name=team_name, team_role=team_role)
    config.update_current_environment_file()
    console.print(f"[green]Switched to team '{team_name}'.[/green]")


def _print_available_slugs(teams: list[dict]) -> None:
    slugs = [str(team.get("slug", "")).strip() for team in teams if team.get("slug")]
    if slugs:
        console.print(f"[dim]Available teams: {', '.join(sorted(slugs))}[/dim]")


def switch(config: SwitchConfig) -> None:
    """Switch the active account context."""
    target = config.target

    prime_config = PrimeConfig()

    if prime_config.team_id_from_env:
        console.print(
            "[red]Error:[/red] PRIME_TEAM_ID is set in your environment. "
            "Clear it before using [bold]prime switch[/bold]."
        )
        raise SystemExit(1)

    if target is not None:
        normalized_target = target.strip().lower()
        if normalized_target == PERSONAL_TARGET:
            _switch_to_personal(prime_config)
            return

    try:
        client = APIClient()
        teams = fetch_teams(client)
    except APIError as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        raise SystemExit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {str(e)}")
        raise SystemExit(1)

    if target is not None:
        selected_team = _select_team_by_target(teams, normalized_target)
        if selected_team is None:
            console.print(f"[red]Team '{target}' not found.[/red]")
            _print_available_slugs(teams)
            raise SystemExit(1)

        _switch_to_team(prime_config, selected_team)
        return

    console.print("\n[bold]Switch account:[/bold]\n")
    current_team_id = prime_config.team_id

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
            selection = prompt("Select", type=int, default=1)
            if selection == 1:
                _switch_to_personal(prime_config)
                return
            if 2 <= selection <= len(teams) + 1:
                _switch_to_team(prime_config, teams[selection - 2])
                return
            console.print(f"[red]Invalid selection. Enter 1-{len(teams) + 1}.[/red]")
        except (EOFError, KeyboardInterrupt):
            raise SystemExit(1)
