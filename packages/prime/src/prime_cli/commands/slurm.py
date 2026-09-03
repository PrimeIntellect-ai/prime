import os
import subprocess
from datetime import datetime, timezone
from typing import List, Optional

import typer
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

app = PlainTyper(help="Manage Slurm clusters", no_args_is_help=True)
console = get_console()

LIST_CLUSTERS_JSON_HELP = json_output_help(
    ".data[] = {id, prime_cluster_id, display_name, status, gpu_type, gpu_count, "
    "total_gpus, free_gpus, total_nodes, healthy_nodes, cordoned_node_count, "
    "created_at, started_at?}",
)

CLUSTER_DETAIL_JSON_HELP = json_output_help(
    ". = {id, prime_cluster_id, display_name, status, gpu_type, gpu_count, "
    "connectable, ssh_host?, ssh_port?, node_health, nodes[]}",
)

MEMBERS_JSON_HELP = json_output_help(
    ".data[] = {username, sudo, status, linked_user_id?, linked_user_name?, linked_user_email?}",
)

ACCOUNTING_JSON_HELP = json_output_help(
    ". = {available, days, total_jobs, throughput[], queue_wait[], "
    "gpu_hours_by_user[], outcomes[]}",
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
        table.add_column("Free/Total GPUs", style="green")
        table.add_column("Nodes (healthy/total)", style="blue")

        for c in clusters:
            gpu_col = (
                f"{c.free_gpus}/{c.total_gpus}"
                if c.free_gpus is not None and c.total_gpus is not None
                else "N/A"
            )
            table.add_row(
                c.id,
                c.display_name,
                Text(c.status, style=status_color(c.status, SLURM_CLUSTER_STATUS_COLORS)),
                c.gpu_type or "N/A",
                gpu_col,
                f"{c.healthy_nodes}/{c.total_nodes}",
            )
        console.print(table)
        console.print(
            "\n[blue]Use 'prime slurm connect <cluster-id> <username>' to SSH in, "
            "or 'prime slurm get <cluster-id>' for details[/blue]"
        )

    except APIError as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {str(e)}")
        raise typer.Exit(1)


@app.command(name="get", no_args_is_help=True, epilog=CLUSTER_DETAIL_JSON_HELP)
def get_cluster(
    cluster_id: str,
    team_id: Optional[str] = _TEAM_ID_OPTION,
    output: str = typer.Option("table", "--output", "-o", help="Output format: table or json"),
) -> None:
    """Get connection info, health, and node details for a Slurm cluster."""
    validate_output_format(output, console)
    resolved_team_id = _resolve_team_id(team_id)

    try:
        cluster = _slurm_client().get(resolved_team_id, cluster_id)

        if output == "json":
            output_data_as_json(cluster.model_dump(), console)
            return

        table = Table(title=f"Cluster: {cluster.display_name}")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="white")
        table.add_row(
            "Status",
            Text(cluster.status, style=status_color(cluster.status, SLURM_CLUSTER_STATUS_COLORS)),
        )
        table.add_row("GPU Type", cluster.gpu_type or "N/A")
        table.add_row("GPU Count", str(cluster.gpu_count))
        if cluster.connectable and cluster.ssh_host and cluster.ssh_port:
            table.add_row(
                "SSH",
                f"ssh -p {cluster.ssh_port} <username>@{cluster.ssh_host}  "
                f"(or: prime slurm connect {cluster_id} <username>)",
            )
        else:
            table.add_row("SSH", "Not connectable (cluster is not RUNNING)")
        table.add_row(
            "Node health",
            f"{cluster.node_health.healthy_nodes}/{cluster.node_health.total_nodes} healthy",
        )
        console.print(table)

        if cluster.node_health.cordoned_nodes:
            cordoned_table = Table(title="Cordoned Nodes")
            cordoned_table.add_column("Name", style="cyan")
            cordoned_table.add_column("Reason", style="yellow")
            for n in cluster.node_health.cordoned_nodes:
                cordoned_table.add_row(n.name, n.reason)
            console.print("\n")
            console.print(cordoned_table)

        if cluster.nodes:
            nodes_table = Table(title="Nodes")
            nodes_table.add_column("Name", style="cyan")
            nodes_table.add_column("Ready", style="white")
            nodes_table.add_column("Free/Allocatable GPUs", style="green")
            nodes_table.add_column("Slurm State", style="blue")
            for n in cluster.nodes:
                nodes_table.add_row(
                    n.name,
                    "yes" if n.ready else "no",
                    f"{n.free_gpus}/{n.allocatable_gpus}",
                    ",".join(n.slurm_states) if n.slurm_states else "N/A",
                )
            console.print("\n")
            console.print(nodes_table)

        console.print(
            f"\n[blue]Use 'prime slurm members {cluster_id}' to see who has access[/blue]"
        )

    except APIError as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {str(e)}")
        raise typer.Exit(1)


@app.command(name="connect", no_args_is_help=True)
@app.command(name="ssh", no_args_is_help=True)
def connect(
    cluster_id: str,
    username: str,
    team_id: Optional[str] = _TEAM_ID_OPTION,
    identity_file: Optional[str] = typer.Option(
        None,
        "--identity",
        "-i",
        help="Path to SSH private key (uses configured ssh_key_path if not specified)",
    ),
) -> None:
    """SSH into a Slurm cluster's login node as the given member."""
    resolved_team_id = _resolve_team_id(team_id)

    try:
        cluster = _slurm_client().get(resolved_team_id, cluster_id)

        if not cluster.connectable or not cluster.ssh_host or not cluster.ssh_port:
            console.print(
                f"[red]Cluster {cluster_id} is not connectable (status: {cluster.status}).[/red]"
            )
            raise typer.Exit(1)

        ssh_key_path = identity_file or Config().ssh_key_path
        if not os.path.exists(ssh_key_path):
            console.print(f"[red]SSH key not found at {ssh_key_path}[/red]")
            raise typer.Exit(1)

        console.print(f"[blue]Connecting to {cluster.display_name} as {username}...[/blue]")
        ssh_command = [
            "ssh",
            "-i",
            ssh_key_path,
            "-o",
            "StrictHostKeyChecking=no",
            "-p",
            str(cluster.ssh_port),
            f"{username}@{cluster.ssh_host}",
        ]
        result = subprocess.run(ssh_command)
        if result.returncode != 0:
            raise typer.Exit(result.returncode)

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
        table.add_column("Sudo", style="white")
        table.add_column("Status", style="yellow")
        table.add_column("Linked User", style="blue")
        for m in members:
            table.add_row(
                m.username,
                "yes" if m.sudo else "no",
                m.status,
                m.linked_user_email or m.linked_user_name or "N/A",
            )
        console.print(table)

    except APIError as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {str(e)}")
        raise typer.Exit(1)


def _print_member(m: SlurmClusterMember) -> None:
    console.print(f"Username: {m.username}")
    console.print(f"Sudo: {'yes' if m.sudo else 'no'}")
    console.print(f"Status: {m.status}")


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


@app.command(name="sudo", no_args_is_help=True)
def set_sudo(
    cluster_id: str,
    username: str,
    on: bool = typer.Option(..., "--on/--off", help="Grant or revoke sudo"),
    team_id: Optional[str] = _TEAM_ID_OPTION,
) -> None:
    """Grant or revoke sudo for a cluster member. Requires team admin."""
    resolved_team_id = _resolve_team_id(team_id)

    try:
        with console.status("[bold blue]Updating sudo...", spinner="dots"):
            member = _slurm_client().set_sudo(resolved_team_id, cluster_id, username, on)
        verb = "Granted" if on else "Revoked"
        console.print(f"[green]{verb} sudo for {member.username} on cluster {cluster_id}[/green]")

    except APIError as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {str(e)}")
        raise typer.Exit(1)


@app.command(name="rename", no_args_is_help=True)
def rename(
    cluster_id: str,
    name: str,
    team_id: Optional[str] = _TEAM_ID_OPTION,
) -> None:
    """Rename a Slurm cluster. Requires team admin."""
    resolved_team_id = _resolve_team_id(team_id)

    try:
        with console.status("[bold blue]Renaming cluster...", spinner="dots"):
            display_name = _slurm_client().rename(resolved_team_id, cluster_id, name)
        console.print(f"[green]Cluster {cluster_id} renamed to '{display_name}'[/green]")

    except APIError as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {str(e)}")
        raise typer.Exit(1)


@app.command(name="delete", no_args_is_help=True)
def delete(
    cluster_id: str,
    team_id: Optional[str] = _TEAM_ID_OPTION,
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
    force: bool = typer.Option(
        False,
        "--force",
        help="Delete even if the cluster has active Slurm jobs (or accounting can't verify)",
    ),
) -> None:
    """Delete a Slurm cluster. Destructive — requires team admin."""
    resolved_team_id = _resolve_team_id(team_id)

    if not confirm_or_skip(
        f"Delete Slurm cluster {cluster_id}? This is destructive and permanently "
        "removes all data and any running jobs on it.",
        yes,
    ):
        console.print("Cancelled")
        raise typer.Exit(0)

    try:
        with console.status("[bold blue]Deleting cluster...", spinner="dots"):
            _slurm_client().delete(resolved_team_id, cluster_id, force=force)
        console.print(f"[green]Delete started for cluster {cluster_id}[/green]")

    except APIError as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        if "active" in str(e).lower() or "force" in str(e).lower():
            console.print("[yellow]Retry with --force to delete anyway.[/yellow]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {str(e)}")
        raise typer.Exit(1)


@app.command(name="accounting", no_args_is_help=True, epilog=ACCOUNTING_JSON_HELP)
def accounting(
    cluster_id: str,
    days: int = typer.Option(30, "--days", help="Number of days to include (1-90)"),
    team_id: Optional[str] = _TEAM_ID_OPTION,
    output: str = typer.Option("table", "--output", "-o", help="Output format: table or json"),
) -> None:
    """Show accounting rollups (throughput, queue wait, GPU-hours) for a Slurm cluster."""
    validate_output_format(output, console)
    resolved_team_id = _resolve_team_id(team_id)

    try:
        rollup = _slurm_client().accounting(resolved_team_id, cluster_id, days=days)

        if output == "json":
            output_data_as_json(rollup.model_dump(), console)
            return

        if not rollup.available:
            console.print("[yellow]Accounting is currently unavailable for this cluster.[/yellow]")
            return

        console.print(f"[bold]Accounting for the last {rollup.days} day(s)[/bold]")
        console.print(f"Total jobs: {rollup.total_jobs}")

        if rollup.throughput:
            table = Table(title="Throughput")
            table.add_column("Day", style="cyan")
            table.add_column("Completed", style="green")
            table.add_column("Failed", style="red")
            table.add_column("Cancelled", style="yellow")
            for row in rollup.throughput:
                table.add_row(row.day, str(row.completed), str(row.failed), str(row.cancelled))
            console.print(table)

        if rollup.queue_wait:
            table = Table(title="Queue Wait")
            table.add_column("Day", style="cyan")
            table.add_column("Median Wait", style="green")
            for row in rollup.queue_wait:
                table.add_row(row.day, f"{row.median_seconds:.0f}s")
            console.print(table)

        if rollup.gpu_hours_by_user:
            table = Table(title="GPU-Hours by User")
            table.add_column("User", style="cyan")
            table.add_column("GPU-Hours", style="green")
            for row in rollup.gpu_hours_by_user:
                table.add_row(row.user, f"{row.gpu_hours:.2f}")
            console.print(table)

        if rollup.outcomes:
            table = Table(title="Outcomes")
            table.add_column("Outcome", style="cyan")
            table.add_column("Count", style="white")
            for row in rollup.outcomes:
                table.add_row(row.outcome, str(row.count))
            console.print(table)

    except APIError as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {str(e)}")
        raise typer.Exit(1)


@app.command(name="utilization", no_args_is_help=True)
def utilization(
    cluster_id: str,
    range_seconds: int = typer.Option(
        21_600, "--range-seconds", help="Time range to query, in seconds"
    ),
    team_id: Optional[str] = _TEAM_ID_OPTION,
    output: str = typer.Option("table", "--output", "-o", help="Output format: table or json"),
) -> None:
    """Show GPU utilization over time for a Slurm cluster."""
    validate_output_format(output, console)
    resolved_team_id = _resolve_team_id(team_id)

    try:
        points = _slurm_client().utilization(
            resolved_team_id, cluster_id, range_seconds=range_seconds
        )

        if output == "json":
            output_data_as_json({"gpu_util": [p.model_dump() for p in points]}, console)
            return

        if not points:
            console.print("[yellow]No utilization data available for this range.[/yellow]")
            return

        table = Table(title=f"GPU Utilization (last {range_seconds}s)")
        table.add_column("Time", style="cyan")
        table.add_column("Utilization %", style="green")
        for p in points:
            timestamp = datetime.fromtimestamp(p.timestamp, tz=timezone.utc).strftime(
                "%Y-%m-%d %H:%M:%S UTC"
            )
            table.add_row(timestamp, f"{p.value:.1f}")
        console.print(table)

    except APIError as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {str(e)}")
        raise typer.Exit(1)
