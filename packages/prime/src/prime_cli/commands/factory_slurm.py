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

# Hidden: the team resolves automatically from the selected account
# context; --team is the documented override only.
TEAM_OPTION = typer.Option(None, "--team", "-t", hidden=True)

LIST_JSON_HELP = json_output_help(
    ". = {data[]}",
    ".data[] = {id, primeClusterId, displayName, status, gpuType,",
    "            gpuCount, createdAt, startedAt?}",
    "--team: use a different team than the selected account context",
)
MEMBERS_JSON_HELP = json_output_help(
    ". = {data[]}",
    ".data[] = {username, uid, sshAuthorizedKeys[], sudo, status,",
    "            linkedUserId?, linkedUserName?, linkedUserEmail?}",
    "--team: use a different team than the selected account context",
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
        # --link-user operates on IDs: an ID-only link is a real link,
        # never render it as absent.
        linked = m.linked_user_name or m.linked_user_email or m.linked_user_id or "—"
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
        # Invalid JSON or a mistyped field normalizes to the same coarse
        # malformed-response error as list/members — never raw
        # JSON/pydantic diagnostics. Raised INSIDE the try so the
        # except APIError path renders it cleanly.
        try:
            parsed = FactorySlurmMember.model_validate(member)
        except Exception as exc:
            raise APIError("Slurm cluster API returned a malformed response body.") from exc
    except APIError as e:
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


RESIZE_JSON_HELP = json_output_help(
    ". = {numNodes, added[], removed[]}",
    "numNodes: new total node count after resize",
    "added[]: node names that were added",
    "removed[]: node names that were removed",
    "--team: use a different team than the selected account context",
)


def _format_error_message(error: APIError) -> str:
    """Format API error messages with helpful context based on status code."""
    error_str = str(error)
    
    # Extract status code if present
    if "HTTP 400" in error_str:
        return f"{error_str}\n[dim]Ensure numNodes is >= 1 and <= max(64, current size) with sufficient resources.[/dim]"
    elif "HTTP 403" in error_str:
        return f"{error_str}\n[dim]This operation requires team admin privileges and the cluster must have self-service enabled.[/dim]"
    elif "HTTP 404" in error_str:
        return f"{error_str}\n[dim]Cluster not found. Check the cluster ID and team context.[/dim]"
    elif "HTTP 409" in error_str:
        return f"{error_str}\n[dim]The cluster may not be running, another operation is in progress, or there are insufficient idle nodes.[/dim]"
    elif "HTTP 502" in error_str:
        return f"{error_str}\n[dim]Node labels were changed but verification failed. Check cluster status.[/dim]"
    elif "HTTP 503" in error_str:
        return f"{error_str}\n[dim]Slurm state unavailable. This is likely temporary - please retry in a moment.[/dim]"
    
    return error_str


@app.command(name="resize", no_args_is_help=True, epilog=RESIZE_JSON_HELP)
def resize_cluster(
    cluster_id: str = typer.Argument(..., help="Cluster/job ID to resize"),
    nodes: int = typer.Option(
        ...,
        "--nodes",
        "-n",
        help="New total node count (absolute target, not delta; must be >= 1 and <= max(64, current))",
    ),
    team: Optional[str] = TEAM_OPTION,
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip confirmation prompt",
    ),
    wait: bool = typer.Option(
        False,
        "--wait",
        "-w",
        help="Wait until the resize operation completes (polls every 5 seconds, 5 minute timeout)",
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Print the API response as JSON (same as --output json)"
    ),
    output: str = typer.Option("table", "--output", "-o", help="Output format: table or json"),
) -> None:
    """Resize a Slurm cluster to the specified node count.
    
    Requires team admin privileges and cluster must have self-service enabled.
    The resize is asynchronous - nodes are added/removed by the reconciler.
    """
    if json_output:
        output = "json"
    validate_output_format(output, console)
    err_console = _stderr_console_for(output)

    if nodes <= 0:
        err_console.print(f"[red]Error:[/red] --nodes must be a positive integer (got {nodes}).")
        raise typer.Exit(1)

    team_id = _resolve_team_id(team, err_console)
    if not team_id:
        return

    # Find the cluster to get current size
    try:
        cluster = _find_cluster(team_id, cluster_id)
    except (APIError, ValueError) as e:
        err_console.print(f"[red]Error:[/red] {escape(str(e))}")
        raise typer.Exit(1)

    # Calculate current nodes from GPU count (assume gpus_per_node, or just show GPUs)
    # The API works with nodes, but we display what we know
    current_display = f"{cluster.gpu_count} GPUs"
    
    # Show confirmation unless --yes
    if not yes:
        console.print(f"[yellow]Resize Slurm cluster '{escape(cluster.display_name)}':[/yellow]")
        console.print(f"  Cluster ID:   {escape(cluster.id)}")
        console.print(f"  Current size: {current_display}")
        console.print(f"  New nodes:    {nodes} nodes")
        console.print()
        confirm = typer.confirm("Proceed with resize?", default=False)
        if not confirm:
            console.print("Resize cancelled.")
            raise typer.Exit(0)

    # Execute the resize
    try:
        response = _client().resize_cluster(team_id, cluster_id, nodes)
    except APIError as e:
        formatted_error = _format_error_message(e)
        err_console.print(f"[red]Error:[/red] {escape(formatted_error)}")
        raise typer.Exit(1)

    if output == "json":
        output_data_as_json(response, console)
        return

    # Display the result
    num_nodes = response.get("numNodes", nodes)
    added = response.get("added", [])
    removed = response.get("removed", [])

    console.print(
        f"[green]Resize initiated for cluster '{escape(cluster.display_name)}'[/green]"
    )
    console.print(f"Target nodes: {num_nodes}")
    
    if added:
        console.print(f"Nodes to add: {', '.join(escape(n) for n in added)}")
    if removed:
        console.print(f"Nodes to remove: {', '.join(escape(n) for n in removed)}")
    
    if not added and not removed:
        console.print("[dim]No immediate node changes (target matches current)[/dim]")

    if wait:
        import time

        console.print()
        console.print("Waiting for resize to complete (reconciliation is asynchronous)...")
        max_wait = 300  # 5 minutes
        poll_interval = 5  # seconds
        elapsed = 0

        while elapsed < max_wait:
            time.sleep(poll_interval)
            elapsed += poll_interval

            try:
                # Poll the cluster list to check GPU count
                updated_cluster = _find_cluster(team_id, cluster_id)
                
                # We can't directly check node count from the API, but we can check status
                # and GPU count as a proxy
                console.print(
                    f"  Status: {escape(updated_cluster.status)} | "
                    f"GPUs: {updated_cluster.gpu_count}"
                )
                
                # If status is running/ready, assume reconciliation is done
                # (The exact completion criteria depends on the status values)
                if updated_cluster.status.lower() in ["running", "ready", "online"]:
                    # Give it a bit more time for the reconciler
                    if elapsed >= 15:  # At least 15 seconds
                        console.print(f"[green]✓[/green] Cluster is {escape(updated_cluster.status)}")
                        break
                        
            except (APIError, ValueError) as e:
                err_console.print(f"[yellow]Warning:[/yellow] Status check failed: {escape(str(e))}")

        if elapsed >= max_wait:
            err_console.print(
                f"[yellow]Timeout:[/yellow] Resize did not complete within {max_wait} seconds."
            )
            err_console.print(
                "The resize may still be in progress. "
                "Check status with: prime factory slurm list"
            )
