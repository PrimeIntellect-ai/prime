from typing import List, Optional

import typer
from rich.markup import escape
from rich.table import Table
from rich.text import Text

from prime_cli.core import APIClient, APIError, Config

from ..api.slurm import SlurmClusterMember, SlurmClustersClient
from ..utils import (
    PlainTyper,
    confirm_or_skip,
    get_console,
    json_output_help,
    output_data_as_json,
    status_color,
    validate_output_format,
)
from ..utils.display import SLURM_CLUSTER_STATUS_COLORS

app = PlainTyper(help="Manage Slurm cluster member access", no_args_is_help=True)
console = get_console()

LIST_CLUSTERS_JSON_HELP = json_output_help(
    ".data[] = {id, prime_cluster_id, display_name, status, gpu_type, gpu_count, "
    "created_at, started_at?}",
)

MEMBERS_JSON_HELP = json_output_help(
    ".data[] = {username, uid, ssh_authorized_keys[], sudo, status, "
    "linked_user_id?, linked_user_name?, linked_user_email?}",
)


def _resolve_team_id(team_id: Optional[str]) -> str:
    resolved = team_id or Config().team_id
    if not resolved:
        console.print(
            "[red]Error: No team selected. "
            "Use --team-id or set a team with 'prime config set-team-id'[/red]"
        )
        raise typer.Exit(1)
    return resolved


def _slurm_client() -> SlurmClustersClient:
    return SlurmClustersClient(APIClient())


_TEAM_ID_OPTION = typer.Option(
    None, "--team-id", help="Team ID (uses config team_id if not specified)"
)


@app.command(name="list", epilog=LIST_CLUSTERS_JSON_HELP)
def list_clusters(
    team_id: Optional[str] = _TEAM_ID_OPTION,
    output: str = typer.Option("table", "--output", "-o", help="Output format: table or json"),
) -> None:
    """List your team's Slurm clusters."""
    validate_output_format(output, console)
    resolved_team_id = _resolve_team_id(team_id)

    try:
        clusters = _slurm_client().list(resolved_team_id)

        if output == "json":
            output_data_as_json({"data": [c.model_dump() for c in clusters]}, console)
            return

        table = Table(title=f"Slurm Clusters (Total: {len(clusters)})", show_lines=True)
        table.add_column("ID", style="cyan", no_wrap=True)
        table.add_column("Name", style="blue")
        table.add_column("Status", style="yellow")
        table.add_column("GPU Type", style="magenta")
        table.add_column("GPU Count", style="green")

        for c in clusters:
            table.add_row(
                c.id,
                Text(c.display_name),
                Text(c.status, style=status_color(c.status, SLURM_CLUSTER_STATUS_COLORS)),
                c.gpu_type or "N/A",
                str(c.gpu_count),
            )
        console.print(table)
        console.print("\n[blue]Use 'prime slurm members <cluster-id>' to see who has access[/blue]")

    except APIError as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {str(e)}")
        raise typer.Exit(1)


@app.command(name="members", no_args_is_help=True, epilog=MEMBERS_JSON_HELP)
def list_members(
    cluster_id: str,
    team_id: Optional[str] = _TEAM_ID_OPTION,
    output: str = typer.Option("table", "--output", "-o", help="Output format: table or json"),
) -> None:
    """List members with SSH access to a Slurm cluster."""
    validate_output_format(output, console)
    resolved_team_id = _resolve_team_id(team_id)

    try:
        members = _slurm_client().list_members(resolved_team_id, cluster_id)

        if output == "json":
            output_data_as_json({"data": [m.model_dump() for m in members]}, console)
            return

        table = Table(title=f"Members (Total: {len(members)})", show_lines=True)
        table.add_column("Username", style="cyan")
        table.add_column("UID", style="magenta")
        table.add_column("SSH Keys", style="green")
        table.add_column("Sudo", style="white")
        table.add_column("Status", style="yellow")
        table.add_column("Linked User", style="blue")
        for m in members:
            table.add_row(
                m.username,
                str(m.uid),
                Text("\n").join(Text(_truncate_ssh_key(k)) for k in m.ssh_authorized_keys)
                or Text("N/A"),
                "yes" if m.sudo else "no",
                m.status,
                Text(m.linked_user_email or m.linked_user_name or "N/A"),
            )
        console.print(table)

    except APIError as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {str(e)}")
        raise typer.Exit(1)


def _truncate_ssh_key(key: str, max_len: int = 60) -> str:
    """Shorten a full authorized_keys line for table display. Full value
    is always available via --output json."""
    return key if len(key) <= max_len else f"{key[:max_len]}..."


def _print_member(m: SlurmClusterMember) -> None:
    console.print(f"Username: {m.username}")
    console.print(f"UID: {m.uid}")
    console.print(f"Sudo: {'yes' if m.sudo else 'no'}")
    console.print(f"Status: {m.status}")
    for key in m.ssh_authorized_keys:
        console.print(f"SSH Key: {escape(key)}")


@app.command(name="add-member", no_args_is_help=True)
def add_member(
    cluster_id: str,
    username: str,
    ssh_key: List[str] = typer.Option(
        ..., "--ssh-key", help="Authorized SSH public key line. Repeatable."
    ),
    link_user: Optional[str] = typer.Option(
        None, "--link-user", help="Prime user ID to link this member to"
    ),
    team_id: Optional[str] = _TEAM_ID_OPTION,
) -> None:
    """Add a member with SSH access to a Slurm cluster. Requires team admin."""
    resolved_team_id = _resolve_team_id(team_id)

    try:
        with console.status("[bold blue]Adding member...", spinner="dots"):
            member = _slurm_client().add_member(
                resolved_team_id, cluster_id, username, ssh_key, link_user
            )
        console.print(
            f"[green]Successfully added {member.username} to cluster {cluster_id}[/green]"
        )
        _print_member(member)

    except APIError as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {str(e)}")
        raise typer.Exit(1)


@app.command(name="remove-member", no_args_is_help=True)
def remove_member(
    cluster_id: str,
    username: str,
    team_id: Optional[str] = _TEAM_ID_OPTION,
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
) -> None:
    """Remove a member's SSH access from a Slurm cluster. Requires team admin."""
    resolved_team_id = _resolve_team_id(team_id)

    if not confirm_or_skip(f"Remove {username}'s access to cluster {cluster_id}?", yes):
        console.print("Cancelled")
        raise typer.Exit(0)

    try:
        with console.status("[bold blue]Removing member...", spinner="dots"):
            _slurm_client().remove_member(resolved_team_id, cluster_id, username)
        console.print(f"[green]Successfully removed {username} from cluster {cluster_id}[/green]")

    except APIError as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {str(e)}")
        raise typer.Exit(1)
