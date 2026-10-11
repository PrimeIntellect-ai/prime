"""`prime factory slurm` — the team's Slurm deployment roster."""

from typing import Optional

import typer
from rich.markup import escape
from rich.table import Table
from rich.text import Text

from ..api.factory_slurm import FactorySlurmClient, FactorySlurmCluster, FactorySlurmMember
from ..client import APIClient, APIError
from ..core import Config
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

app = PlainTyper(help="Manage your team's Slurm cluster deployments", no_args_is_help=True)
console = get_console()

TEAM_OPTION = typer.Option(
    None, "--team", "-t", help="Team ID override (defaults to the selected account context)"
)

LIST_JSON_HELP = json_output_help(
    ". = {data[]}",
    ".data[] = {id, primeClusterId, displayName, status, gpuType,",
    "            gpuCount, createdAt, startedAt?}",
)
MEMBERS_JSON_HELP = json_output_help(
    ". = {data[]}",
    ".data[] = {username, uid, sshAuthorizedKeys[], sudo, status,",
    "            linkedUserId?, linkedUserName?, linkedUserEmail?}",
)


def _resolve_team_id(team: Optional[str], err_console) -> Optional[str]:
    resolved = team or Config().team_id
    if not resolved:
        # Diagnostics never pollute stdout in --json mode.
        err_console.print(
            "No team selected in the current account context. "
            "`prime factory slurm` manages your team's Slurm deployments."
        )
        err_console.print(
            "[dim]Run `prime switch` to select a team, or pass --team <team_id>.[/dim]"
        )
        return None
    return resolved


def _client() -> FactorySlurmClient:
    return FactorySlurmClient(APIClient())


def _plain_error(e: Exception, err_console) -> None:
    # Escape upstream error text: raw brackets would crash Rich markup.
    err_console.print(f"[red]Error:[/red] {escape(str(e))}")
    raise typer.Exit(1)


def _stderr_console_for(output: str):
    # JSON mode keeps stdout strictly data: diagnostics go to stderr.
    return get_console(stderr=True) if output == "json" else console


def _envelope_rows(envelope, model):
    """Validate the documented {data: [...]} envelope and parse its rows.

    A missing or non-list `data` field is schema drift, never an
    authoritative empty roster — fail closed as a malformed response.
    """
    if not isinstance(envelope, dict) or not isinstance(envelope.get("data"), list):
        raise APIError("Slurm cluster API returned a malformed response body.")
    try:
        return [model.model_validate(row) for row in envelope["data"]]
    except Exception as e:  # pydantic validation drift
        raise APIError("Slurm cluster API returned a malformed response body.") from e


def _cluster_rows(envelope) -> "list[FactorySlurmCluster]":
    return _envelope_rows(envelope, FactorySlurmCluster)


def _member_rows(envelope) -> "list[FactorySlurmMember]":
    return _envelope_rows(envelope, FactorySlurmMember)


@app.command(name="list", epilog=LIST_JSON_HELP)
def list_clusters(
    team: Optional[str] = TEAM_OPTION,
    json_output: bool = typer.Option(
        False, "--json", help="Print the API response as JSON (same as --output json)"
    ),
    output: str = typer.Option("table", "--output", "-o", help="Output format: table or json"),
) -> None:
    """List your team's Slurm deployments."""
    if json_output:
        output = "json"
    validate_output_format(output, console)
    err_console = _stderr_console_for(output)
    team_id = _resolve_team_id(team, err_console)
    if not team_id:
        return

    try:
        envelope = _client().list_clusters(team_id)
        clusters = _cluster_rows(envelope)
    except (APIError, ValueError) as e:
        # Parsing stays inside the caught path: schema drift surfaces as
        # the clean error, never an unhandled traceback. (pydantic
        # ValidationError is a ValueError subclass.)
        _plain_error(e, err_console)

    if output == "json":
        output_data_as_json(envelope, console)
        return

    if not clusters:
        console.print("No Slurm deployments found.")
        return

    table = Table(show_header=True, header_style="bold", show_lines=False)
    table.add_column("ID", style="cyan")
    table.add_column("NAME")
    table.add_column("STATUS")
    table.add_column("GPU TYPE")
    table.add_column("GPUS", justify="right")
    for c in clusters:
        table.add_row(
            Text(c.id),
            Text(c.display_name),
            Text(c.status, style=status_color(c.status, SLURM_CLUSTER_STATUS_COLORS)),
            Text(c.gpu_type or "—"),
            Text(str(c.gpu_count)),
        )
    console.print(table)


def _find_cluster(team_id: str, cluster_id: str) -> FactorySlurmCluster:
    envelope = _client().list_clusters(team_id)
    for cluster in _cluster_rows(envelope):
        if cluster.id == cluster_id:
            return cluster
    console.print(f"[red]Error:[/red] No Slurm deployment matched '{escape(cluster_id)}'.")
    raise typer.Exit(1)


@app.command(name="get")
def get_cluster(
    cluster_id: str,
    team: Optional[str] = TEAM_OPTION,
) -> None:
    """Show one Slurm deployment's details."""
    err_console = _stderr_console_for("table")
    team_id = _resolve_team_id(team, err_console)
    if not team_id:
        return

    try:
        cluster = _find_cluster(team_id, cluster_id)
    except (APIError, ValueError) as e:
        _plain_error(e, err_console)

    console.print(f"Name: {escape(cluster.display_name)}")
    console.print(f"Status: {escape(cluster.status)}")
    console.print(f"GPU type: {escape(cluster.gpu_type) if cluster.gpu_type else '—'}")
    console.print(f"GPUs: {cluster.gpu_count}")
    if cluster.started_at:
        console.print(f"Started: {escape(str(cluster.started_at))}")


@app.command(name="members", no_args_is_help=True, epilog=MEMBERS_JSON_HELP)
def list_members(
    cluster_id: str,
    team: Optional[str] = TEAM_OPTION,
    json_output: bool = typer.Option(
        False, "--json", help="Print the API response as JSON (same as --output json)"
    ),
    output: str = typer.Option("table", "--output", "-o", help="Output format: table or json"),
) -> None:
    """List members with SSH access to a Slurm deployment."""
    if json_output:
        output = "json"
    validate_output_format(output, console)
    err_console = _stderr_console_for(output)
    team_id = _resolve_team_id(team, err_console)
    if not team_id:
        return

    try:
        envelope = _client().list_members(team_id, cluster_id)
        members = _member_rows(envelope)
    except (APIError, ValueError) as e:
        _plain_error(e, err_console)

    if output == "json":
        output_data_as_json(envelope, console)
        return

    if not members:
        console.print("No members found.")
        return

    table = Table(show_header=True, header_style="bold", show_lines=False)
    table.add_column("USERNAME", style="cyan")
    table.add_column("UID", justify="right")
    table.add_column("SSH KEYS")
    table.add_column("SUDO")
    table.add_column("STATUS")
    table.add_column("LINKED USER")
    for m in members:
        keys = Text("\n").join(Text(_truncate_ssh_key(k)) for k in m.ssh_authorized_keys)
        linked = m.linked_user_name or m.linked_user_email or "—"
        table.add_row(
            Text(m.username),
            Text(str(m.uid)),
            keys,
            Text("yes" if m.sudo else "no"),
            Text(m.status),
            Text(linked),
        )
    console.print(table)


def _truncate_ssh_key(key: str, max_len: int = 60) -> str:
    """Shorten a key line for table display; --json carries the full value."""
    return key if len(key) <= max_len else f"{key[:max_len]}..."


def _print_member(m: FactorySlurmMember) -> None:
    console.print(f"Username: {escape(m.username)}")
    console.print(f"UID: {m.uid}")
    console.print(f"Sudo: {'yes' if m.sudo else 'no'}")
    console.print(f"Status: {escape(m.status)}")
    for key in m.ssh_authorized_keys:
        console.print(f"SSH Key: {escape(key)}")


@app.command(name="add-member", no_args_is_help=True)
def add_member(
    cluster_id: str,
    username: str,
    ssh_key: list[str] = typer.Option(
        ..., "--ssh-key", help="Authorized SSH public key line. Repeatable."
    ),
    link_user: Optional[str] = typer.Option(
        None, "--link-user", help="Prime user ID to link this member to"
    ),
    team: Optional[str] = TEAM_OPTION,
) -> None:
    """Add a member with SSH access to a Slurm deployment. Requires team admin."""
    err_console = _stderr_console_for("table")
    team_id = _resolve_team_id(team, err_console)
    if not team_id:
        return

    try:
        member = _client().add_member(team_id, cluster_id, username, ssh_key, link_user)
        parsed = FactorySlurmMember.model_validate(member)
    except (APIError, ValueError) as e:
        _plain_error(e, err_console)

    console.print(
        f"[green]Successfully added {escape(parsed.username)} to {escape(cluster_id)}[/green]"
    )
    _print_member(parsed)


@app.command(name="remove-member", no_args_is_help=True)
def remove_member(
    cluster_id: str,
    username: str,
    team: Optional[str] = TEAM_OPTION,
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
) -> None:
    """Remove a member's SSH access from a Slurm deployment. Requires team admin."""
    err_console = _stderr_console_for("table")
    team_id = _resolve_team_id(team, err_console)
    if not team_id:
        return

    if not confirm_or_skip(f"Remove {username}'s access to {cluster_id}?", yes):
        console.print("Cancelled")
        return

    try:
        _client().remove_member(team_id, cluster_id, username)
    except (APIError, ValueError) as e:
        _plain_error(e, err_console)

    console.print(
        f"[green]Successfully removed {escape(username)} from {escape(cluster_id)}[/green]"
    )
