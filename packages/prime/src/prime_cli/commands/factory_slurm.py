"""`prime factory slurm` — the team's Slurm deployment roster."""

import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer
from rich.console import Console
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


def _find_cluster(
    team_id: str, cluster_id: str, err_console: Optional[Console] = None
) -> FactorySlurmCluster:
    envelope = _client().list_clusters(team_id)
    for cluster in _cluster_rows(envelope):
        if cluster.id == cluster_id:
            return cluster
    (err_console or console).print(
        f"[red]Error:[/red] No Slurm deployment matched '{escape(cluster_id)}'."
    )
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


def _create_error_hint(error: APIError) -> Optional[str]:
    """Status-code-specific guidance for a failed create request."""
    error_str = str(error)
    for status, hint in CREATE_ERROR_HINTS.items():
        if status in error_str:
            return hint
    return None


def _get_platform_user() -> Optional[Dict[str, str]]:
    """Get the logged-in platform user from the API.

    Returns dict with 'id' and 'username' (slug sanitized to regex), or None if unavailable.
    """
    try:
        client = APIClient()
        response = client.get("/user/whoami")
        data = response.get("data") if isinstance(response, dict) else None
        if not isinstance(data, dict):
            return None

        user_id = data.get("id")
        slug = data.get("slug")

        if not user_id or not slug:
            return None

        # Sanitize slug to match username regex
        username = _sanitize_username(slug)
        if not username:
            return None

        return {"id": user_id, "username": username}
    except Exception:
        return None


def _sanitize_username(slug: str) -> Optional[str]:
    """Sanitize a platform slug to match Slurm username requirements.

    Converts to lowercase, replaces invalid chars with hyphens, ensures it matches
    ^[a-z][a-z0-9-]{0,30}[a-z0-9]$ with length 2-32.
    """
    if not slug:
        return None

    # Convert to lowercase
    sanitized = slug.lower()

    # Replace invalid characters with hyphens
    sanitized = re.sub(r"[^a-z0-9-]", "-", sanitized)

    # Ensure it starts with a letter
    if not sanitized or not sanitized[0].isalpha():
        sanitized = "u" + sanitized

    # Ensure it doesn't end with hyphen
    sanitized = sanitized.rstrip("-")
    if not sanitized or not sanitized[-1].isalnum():
        sanitized = sanitized + "0"

    # Truncate to 32 chars max
    if len(sanitized) > 32:
        sanitized = (
            sanitized[:31] + sanitized[-1] if sanitized[-1].isalnum() else sanitized[:31] + "0"
        )

    # Ensure minimum length
    if len(sanitized) < 2:
        sanitized = sanitized + "0"

    # Validate it matches the regex
    if _validate_username(sanitized):
        return sanitized

    return None


def _read_ssh_keys(key_file: Optional[str] = None) -> List[str]:
    """Read SSH public keys from a file or ~/.ssh/*.pub.

    Args:
        key_file: Specific key file to read. If None, reads all ~/.ssh/*.pub files.

    Returns:
        List of SSH public key strings.
    """
    keys = []

    if key_file:
        # Read from specified file
        path = Path(key_file).expanduser()
        if not path.exists():
            console.print(f"[yellow]Warning:[/yellow] SSH key file not found: {key_file}")
            return []
        try:
            content = path.read_text().strip()
            if content:
                keys.append(content)
        except Exception as e:
            console.print(f"[yellow]Warning:[/yellow] Could not read {key_file}: {escape(str(e))}")
            return []
    else:
        # Default: read from ~/.ssh/*.pub
        ssh_dir = Path.home() / ".ssh"
        if ssh_dir.exists():
            for pub_file in sorted(ssh_dir.glob("*.pub")):
                try:
                    content = pub_file.read_text().strip()
                    if content:
                        keys.append(content)
                except Exception:
                    # Silently skip unreadable files
                    pass

    return keys


def _validate_username(username: str) -> bool:
    """Validate username against Slurm cluster requirements."""
    if not (2 <= len(username) <= 32):
        return False
    return USERNAME_REGEX.match(username) is not None


def _parse_member_spec(spec: str, platform_user: Optional[Dict[str, str]]) -> Dict[str, Any]:
    """Parse a member specification in format USER[:KEYFILE].

    Args:
        spec: Member specification string
        platform_user: Platform user dict for validation of default member

    Returns:
        Dict with 'username' and 'keyfile' (optional)

    Raises:
        ValueError: If the spec is invalid
    """
    if ":" in spec:
        username, keyfile = spec.split(":", 1)
        return {"username": username, "keyfile": keyfile}
    else:
        # Bare username - only valid if it's the caller's own username
        if platform_user and spec == platform_user["username"]:
            return {"username": spec, "keyfile": None}
        else:
            raise ValueError(
                f"Member '{spec}' requires a key file. Use --member {spec}:KEYFILE. "
                f"Only the default member (your platform username) can omit the key file."
            )


def _wait_for_status(
    team_id: str, cluster_id: str, err_console: Console, timeout_s: int = 300
) -> None:
    """Poll the cluster list until RUNNING or FAILED; exit 1 on timeout or failure."""
    err_console.print("Waiting for cluster to reach RUNNING status...")
    elapsed = 0
    poll_interval_s = 5
    while True:
        try:
            cluster = _find_cluster(team_id, cluster_id, err_console)
            err_console.print(f"  Status: {escape(cluster.status)}")
            if cluster.status == "RUNNING":
                err_console.print("[green]✓[/green] Cluster is RUNNING")
                return
            if cluster.status == "FAILED":
                err_console.print("[red]✗[/red] Cluster deployment FAILED")
                raise typer.Exit(1)
        except (APIError, ValueError) as e:
            err_console.print(f"[yellow]Warning:[/yellow] Status check failed: {escape(str(e))}")
        if elapsed >= timeout_s:
            break
        time.sleep(poll_interval_s)
        elapsed += poll_interval_s

    err_console.print(
        f"[yellow]Timeout:[/yellow] Cluster did not reach RUNNING status "
        f"within {timeout_s} seconds."
    )
    err_console.print(
        "The deployment may still be in progress. Check status with: prime factory slurm list"
    )
    raise typer.Exit(1)


CREATE_JSON_HELP = json_output_help(
    ". = {id, primeClusterId, status, createdAt}",
    "id: cluster/job identifier for subsequent commands",
    "primeClusterId: target Model Factory or dedicated cluster",
    "status: deployment status (starts PENDING)",
    "createdAt: timestamp when the cluster was created",
    "--team: use a different team than the selected account context",
)

RESIZE_JSON_HELP = json_output_help(
    ". = {numNodes, added[], removed[]}",
    "numNodes: new total node count after resize",
    "added[]: node names that were added",
    "removed[]: node names that were removed",
    "--team: use a different team than the selected account context",
)


USERNAME_REGEX = re.compile(r"^[a-z][a-z0-9-]{0,30}[a-z0-9]$")

CREATE_ERROR_HINTS = {
    "HTTP 403": (
        "Creating clusters requires an API key with the slurm_clusters:write scope "
        "and team admin privileges."
    ),
    "HTTP 404": "Prime cluster not found. Check the cluster ID.",
    "HTTP 409": (
        "Conflict: A Slurm cluster with this name may already exist, not enough nodes "
        "are available, or there is a port conflict."
    ),
    "HTTP 422": (
        "Validation error: Check that all fields are valid. Username must match "
        "^[a-z][a-z0-9-]{0,30}[a-z0-9]$ with length 2-32, and at least one SSH key "
        "and member are required."
    ),
}

RESIZE_ERROR_HINTS = {
    "HTTP 400": ("Ensure numNodes is >= 1 and <= max(64, current size) with sufficient resources."),
    "HTTP 403": (
        "Resizing requires an API key with the factory:write scope, team admin "
        "privileges, and a cluster with self-service enabled."
    ),
    "HTTP 404": "Cluster not found. Check the cluster ID and team context.",
    "HTTP 409": (
        "The cluster may not be running, another operation is in progress, "
        "or there are insufficient idle nodes."
    ),
    "HTTP 502": "Node labels were changed but verification failed. Check cluster status.",
    "HTTP 503": ("Slurm state unavailable. This is likely temporary - please retry in a moment."),
}
LOOKUP_403_HINT = "Looking up the cluster requires an API key with the slurm_clusters:read scope."

CREATE_WAIT_TIMEOUT_S = 300
RESIZE_WAIT_TIMEOUT_S = 300
RESIZE_POLL_INTERVAL_S = 5


@app.command(name="create", no_args_is_help=True, epilog=CREATE_JSON_HELP)
def create_cluster(
    prime_cluster_id: str = typer.Argument(
        ..., help="Target Model Factory or dedicated cluster ID"
    ),
    name: str = typer.Option(..., "--name", "-n", help="Display name for the Slurm cluster"),
    nodes: int = typer.Option(..., "--nodes", help="Number of nodes to deploy"),
    member: List[str] = typer.Option(
        [],
        "--member",
        help=(
            "Member in format USER[:KEYFILE] (repeatable). "
            "Defaults to your platform username with ~/.ssh/*.pub keys if not specified. "
            "Other members require :KEYFILE."
        ),
    ),
    sudo: bool = typer.Option(
        False,
        "--sudo",
        help="Grant sudo privileges to the default member (your platform user)",
    ),
    gpus_per_node: Optional[int] = typer.Option(
        None,
        "--gpus-per-node",
        help="Number of GPUs per node",
    ),
    home_size: Optional[str] = typer.Option(
        None,
        "--home-size",
        help="Home directory size (e.g., '100Gi')",
    ),
    containers: bool = typer.Option(
        False,
        "--containers",
        help="Enable container support",
    ),
    tailscale_auth_key: Optional[str] = typer.Option(
        None,
        "--tailscale-auth-key",
        envvar="TAILSCALE_AUTH_KEY",
        help="Tailscale auth key (or set TAILSCALE_AUTH_KEY env var)",
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
        help=(
            f"Wait until RUNNING or FAILED status "
            f"(polls every 5s; exits 1 after {CREATE_WAIT_TIMEOUT_S}s or on failure)"
        ),
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Print the API response as JSON (same as --output json)"
    ),
    output: str = typer.Option("table", "--output", "-o", help="Output format: table or json"),
) -> None:
    """Create a new Slurm cluster deployment.

    Deploys a Slurm cluster on the specified Model Factory or dedicated cluster.
    Requires team admin privileges and an API key with the slurm_clusters:write scope
    (plus slurm_clusters:read for --wait polling).

    At least one member with SSH access is required. Members are specified as USER[:KEYFILE].
    If no --member is given, defaults to your platform username with SSH keys from ~/.ssh/*.pub.
    Other members must specify a key file (e.g., --member alice:~/.ssh/alice.pub).
    """
    if json_output:
        output = "json"
    validate_output_format(output, console)
    err_console = _stderr_console_for(output)

    # Validate nodes
    if nodes <= 0:
        err_console.print(f"[red]Error:[/red] --nodes must be positive (got {nodes}).")
        raise typer.Exit(1)

    team_id = _resolve_team_id(team, err_console)
    if not team_id:
        return

    # Get platform user for default member
    platform_user = _get_platform_user()

    # Build members list
    members_list: List[Dict[str, Any]] = []
    member_specs = member if member else []

    if not member_specs:
        # Default to platform user
        if not platform_user:
            err_console.print(
                "[red]Error:[/red] Could not determine your platform username. "
                "Please specify members explicitly with --member USER:KEYFILE."
            )
            raise typer.Exit(1)

        # Read default SSH keys for platform user
        ssh_keys = _read_ssh_keys()
        if not ssh_keys:
            err_console.print(
                "[red]Error:[/red] No SSH keys found in ~/.ssh/*.pub. "
                "Specify a key file with --member USER:KEYFILE."
            )
            raise typer.Exit(1)

        member_dict: Dict[str, Any] = {
            "username": platform_user["username"],
            "sshAuthorizedKeys": ssh_keys,
            "sudo": sudo,
            "linkedUserId": platform_user["id"],
        }
        members_list.append(member_dict)
    else:
        # Parse member specifications
        for spec in member_specs:
            try:
                parsed = _parse_member_spec(spec, platform_user)
            except ValueError as e:
                err_console.print(f"[red]Error:[/red] {escape(str(e))}")
                raise typer.Exit(1)

            username = parsed["username"]
            keyfile = parsed.get("keyfile")

            # Validate username
            if not _validate_username(username):
                err_console.print(
                    f"[red]Error:[/red] Invalid username '{escape(username)}'. "
                    "Must match ^[a-z][a-z0-9-]{0,30}[a-z0-9]$ with length 2-32."
                )
                raise typer.Exit(1)

            # Read SSH keys for this member
            ssh_keys = _read_ssh_keys(keyfile)
            if not ssh_keys:
                err_console.print(
                    f"[red]Error:[/red] No SSH keys found for member '{escape(username)}'. "
                    f"Specify a valid key file."
                )
                raise typer.Exit(1)

            member_dict: Dict[str, Any] = {
                "username": username,
                "sshAuthorizedKeys": ssh_keys,
                "sudo": False,  # Only default member gets sudo
            }

            # Set linkedUserId and sudo for platform user
            if platform_user and username == platform_user["username"]:
                member_dict["linkedUserId"] = platform_user["id"]
                member_dict["sudo"] = sudo

            members_list.append(member_dict)

    # Show confirmation prompt unless --yes
    if not yes:
        err_console.print("[yellow]Create Slurm cluster:[/yellow]")
        err_console.print(f"  Name:           {escape(name)}")
        err_console.print(f"  Prime Cluster:  {escape(prime_cluster_id)}")
        err_console.print(f"  Nodes:          {nodes}")
        if gpus_per_node:
            err_console.print(f"  GPUs per node:  {gpus_per_node}")
        err_console.print(f"  Members:        {', '.join(m['username'] for m in members_list)}")
        if any(m.get("sudo") for m in members_list):
            sudo_members = [m["username"] for m in members_list if m.get("sudo")]
            err_console.print(f"  Sudo:           {', '.join(sudo_members)}")
        if containers:
            err_console.print("  Containers:     enabled")
        if home_size:
            err_console.print(f"  Home size:      {home_size}")
        err_console.print()
    if not confirm_or_skip("Proceed with deployment?", yes, err=output == "json"):
        err_console.print("Deployment cancelled.")
        raise typer.Exit(0)

    try:
        response = _client().create_cluster(
            team_id=team_id,
            prime_cluster_id=prime_cluster_id,
            slurm_cluster_name=name,
            num_nodes=nodes,
            members=members_list,
            gpus_per_node=gpus_per_node,
            home_size=home_size,
            tailscale_auth_key=tailscale_auth_key,
            containers_enabled=containers,
        )
    except APIError as e:
        err_console.print(f"[red]Error:[/red] {escape(str(e))}")
        hint = _create_error_hint(e)
        if hint:
            err_console.print(f"[dim]{hint}[/dim]")
        raise typer.Exit(1)

    # Parse the response to get cluster ID
    if not isinstance(response, dict) or "id" not in response:
        err_console.print("[red]Error:[/red] Slurm cluster API returned a malformed response body.")
        raise typer.Exit(1)

    cluster_id = response["id"]
    status = response.get("status", "UNKNOWN")

    if output != "json":
        console.print(f"[green]Slurm cluster '{escape(name)}' created[/green]")
        console.print(f"Cluster ID: {escape(cluster_id)}")
        console.print(f"Status: {escape(status)}")
        console.print(f"Prime Cluster: {escape(response.get('primeClusterId', prime_cluster_id))}")

    if wait:
        err_console.print()
        _wait_for_status(team_id, cluster_id, err_console, CREATE_WAIT_TIMEOUT_S)

    if output == "json":
        output_data_as_json(response, console)


def _resize_error_hint(error: APIError) -> Optional[str]:
    """Status-code-specific guidance for a failed resize request."""
    error_str = str(error)
    for status, hint in RESIZE_ERROR_HINTS.items():
        if status in error_str:
            return hint
    return None


def _target_gpu_count(
    gpus_before: int, num_nodes: int, added: List[str], removed: List[str]
) -> Optional[int]:
    """GPU count the cluster list reports once the resize has reconciled.

    The list endpoint exposes GPUs, not nodes: derive GPUs per node from the
    pre-resize size, recovered from the response's target and node delta.
    None when the pre-resize size cannot be split evenly into nodes.
    """
    nodes_before = num_nodes - len(added) + len(removed)
    if nodes_before <= 0 or gpus_before % nodes_before:
        return None
    return num_nodes * (gpus_before // nodes_before)


def _wait_for_gpu_count(
    team_id: str, cluster_id: str, target_gpus: int, err_console: Console
) -> None:
    """Poll the cluster list until it reports target_gpus; exit 1 on timeout."""
    err_console.print(f"Waiting for the cluster to reach {target_gpus} GPUs...")
    elapsed = 0
    while True:
        try:
            cluster = _find_cluster(team_id, cluster_id, err_console)
            err_console.print(
                f"  Status: {escape(cluster.status)} | GPUs: {cluster.gpu_count}/{target_gpus}"
            )
            if cluster.gpu_count == target_gpus:
                err_console.print(f"[green]✓[/green] Resize complete: {cluster.gpu_count} GPUs")
                return
        except (APIError, ValueError) as e:
            err_console.print(f"[yellow]Warning:[/yellow] Status check failed: {escape(str(e))}")
        if elapsed >= RESIZE_WAIT_TIMEOUT_S:
            break
        time.sleep(RESIZE_POLL_INTERVAL_S)
        elapsed += RESIZE_POLL_INTERVAL_S

    err_console.print(
        f"[yellow]Timeout:[/yellow] Resize did not complete within {RESIZE_WAIT_TIMEOUT_S} seconds."
    )
    err_console.print(
        "The resize may still be in progress. Check status with: prime factory slurm list"
    )
    raise typer.Exit(1)


@app.command(name="resize", no_args_is_help=True, epilog=RESIZE_JSON_HELP)
def resize_cluster(
    cluster_id: str = typer.Argument(..., help="Cluster/job ID to resize"),
    nodes: int = typer.Option(
        ...,
        "--nodes",
        "-n",
        help=(
            "New total node count (absolute target, not delta; "
            "must be >= 1 and <= max(64, current))"
        ),
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
        help=(
            "Wait until the cluster reports the new GPU count "
            f"(polls every {RESIZE_POLL_INTERVAL_S}s; exits 1 after {RESIZE_WAIT_TIMEOUT_S}s)"
        ),
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Print the API response as JSON (same as --output json)"
    ),
    output: str = typer.Option("table", "--output", "-o", help="Output format: table or json"),
) -> None:
    """Resize a Slurm cluster to the specified node count.

    Requires team admin privileges, an API key with the factory:write scope
    (plus slurm_clusters:read for the cluster lookup and --wait polling), and a
    cluster with self-service enabled. The resize is asynchronous - nodes are
    added/removed by the reconciler.
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

    try:
        cluster = _find_cluster(team_id, cluster_id, err_console)
    except (APIError, ValueError) as e:
        err_console.print(f"[red]Error:[/red] {escape(str(e))}")
        if "HTTP 403" in str(e):
            err_console.print(f"[dim]{LOOKUP_403_HINT}[/dim]")
        raise typer.Exit(1)
    gpus_before = cluster.gpu_count

    if not yes:
        err_console.print(
            f"[yellow]Resize Slurm cluster '{escape(cluster.display_name)}':[/yellow]"
        )
        err_console.print(f"  Cluster ID:   {escape(cluster.id)}")
        err_console.print(f"  Current size: {gpus_before} GPUs")
        err_console.print(f"  New nodes:    {nodes} nodes")
        err_console.print()
    if not confirm_or_skip("Proceed with resize?", yes, err=output == "json"):
        err_console.print("Resize cancelled.")
        raise typer.Exit(0)

    try:
        response = _client().resize_cluster(team_id, cluster_id, nodes)
    except APIError as e:
        err_console.print(f"[red]Error:[/red] {escape(str(e))}")
        hint = _resize_error_hint(e)
        if hint:
            err_console.print(f"[dim]{hint}[/dim]")
        raise typer.Exit(1)

    num_nodes = response.get("numNodes", nodes)
    added = response.get("added", [])
    removed = response.get("removed", [])

    if output != "json":
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
        target_gpus = _target_gpu_count(gpus_before, num_nodes, added, removed)
        if target_gpus is None:
            err_console.print(
                f"[red]Error:[/red] Cannot derive the target GPU count from {gpus_before} GPUs "
                f"and the resize response; check progress with: prime factory slurm list"
            )
            raise typer.Exit(1)
        err_console.print()
        _wait_for_gpu_count(team_id, cluster_id, target_gpus, err_console)

    if output == "json":
        output_data_as_json(response, console)
