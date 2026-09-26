"""`prime volumes`: named volumes FFT runs write their outputs to.

A volume is a PVC in your team's (or personal) namespace on the cluster it
was created on. `prime train config.toml --volume <name>` makes the run
write under `runs/<runId>/` on it, and it outlives every run.
"""

import os
import re
import shlex
import subprocess
import tempfile
import time

import typer
from rich.table import Table

from prime_cli.api.training import HostedTrainingClient
from prime_cli.core import APIClient, APIError, Config

from ..utils import (
    PlainTyper,
    confirm_or_skip,
    get_console,
    output_data_as_json,
    validate_output_format,
)

app = PlainTyper(
    help="Manage volumes for full-FT run outputs (closed beta)",
    no_args_is_help=True,
)
console = get_console()


def _client() -> tuple[HostedTrainingClient, str | None]:
    return HostedTrainingClient(APIClient()), Config().team_id


@app.command()
def create(
    name: str = typer.Argument(..., help="Volume name (lowercase letters, digits, '-')"),
    size: str = typer.Option("1Ti", "--size", help="Size, e.g. 500Gi or 2Ti. Can grow later."),
    output: str = typer.Option("table", "--output", "-o", help="Output format: table or json"),
) -> None:
    """Create a volume on your team's (or personal) cluster."""
    validate_output_format(output, console)
    client, team_id = _client()
    try:
        volume = client.create_volume(name, size, team_id=team_id)
    except APIError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    if output == "json":
        output_data_as_json(volume.model_dump(by_alias=True), console)
        return
    console.print(f"[green]Creating volume {volume.name} ({volume.size}).[/green]")
    console.print(f"Use it with: prime train config.toml --volume {volume.name}")


@app.command("list")
def list_volumes(
    output: str = typer.Option("table", "--output", "-o", help="Output format: table or json"),
) -> None:
    """List your volumes."""
    validate_output_format(output, console)
    client, team_id = _client()
    try:
        volumes = client.list_volumes(team_id=team_id)
    except APIError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    if output == "json":
        output_data_as_json([v.model_dump(by_alias=True) for v in volumes], console)
        return
    table = Table("Name", "Size", "Status", "Namespace", "Created")
    for v in volumes:
        table.add_row(v.name, v.size or "-", v.status, v.namespace, v.created_at or "-")
    console.print(table)


@app.command()
def resize(
    name: str = typer.Argument(..., help="Volume name"),
    size: str = typer.Option(..., "--size", help="New size, larger than the current one"),
) -> None:
    """Grow a volume in place. Running pods see the new size; volumes can't shrink."""
    client, team_id = _client()
    try:
        volume = client.resize_volume(name, size, team_id=team_id)
    except APIError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    console.print(f"[green]Volume {volume.name} is now {volume.size}.[/green]")


@app.command()
def delete(
    name: str = typer.Argument(..., help="Volume name"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
) -> None:
    """Delete a volume and everything on it."""
    if not confirm_or_skip(f"Delete volume {name} and all run data on it?", yes):
        raise typer.Exit(0)
    client, team_id = _client()
    try:
        client.delete_volume(name, team_id=team_id)
    except APIError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    console.print(f"[green]Deleting volume {name}.[/green]")


# Explicitly parse the backend endpoint instead of passing an untrusted string
# to a shell. The same host/key/port can be used for sftp, scp and rsync.
_CONNECTION = re.compile(
    r"(?P<user>[a-zA-Z_][a-zA-Z0-9_-]*)@(?P<host>[a-zA-Z0-9.-]+)(?: -p (?P<port>[0-9]{1,5}))?"
)


def _pin_known_hosts(session, hostname: str, port: str) -> list[str]:
    """ssh options (separate argv items) pinning the session pod's host key.

    The platform returns the per-session sshd host key with the endpoint,
    so the CLI writes a scoped known_hosts file instead of disabling host
    key checking (`StrictHostKeyChecking=no` is never sent). A missing
    host key falls back to the user's own known_hosts verification.
    """
    if not getattr(session, "host_public_key", None):
        return []
    bracket = f"[{hostname}]:{port}" if port != "22" else hostname
    path = os.path.join(tempfile.mkdtemp(prefix="prime-volume-"), "known_hosts")
    with open(path, "w") as fh:
        fh.write(f"{bracket} {session.host_public_key}\n")
    return [
        "-o",
        f"UserKnownHostsFile={path}",
        "-o",
        "StrictHostKeyChecking=yes",
    ]


@app.command(name="ssh", no_args_is_help=True)
def ssh(
    name: str = typer.Argument(..., help="Volume name"),
    read_only: bool = typer.Option(
        False, "--read-only", "--read", help="Mount root read-only (default)"
    ),
    read_write: bool = typer.Option(False, "--read-write", "--write", help="Mount root read-write"),
) -> None:
    """SSH into a corporate-tailnet session mounting the volume."""
    if read_only and read_write:
        console.print("[red]Choose either --read-only or --read-write.[/red]")
        raise typer.Exit(2)
    key = Config().ssh_key_path
    if not key or not os.path.isfile(os.path.expanduser(key)):
        console.print("[red]SSH key not found; use prime config set-ssh-key-path.[/red]")
        raise typer.Exit(1)
    key = os.path.expanduser(key)
    client, team_id = _client()
    try:
        session = client.create_volume_session(name, read_only=not read_write, team_id=team_id)
        console.print(
            f"Session {session.id} ({'read-only' if session.read_only else 'read-write'})."
        )
        console.print(f"Stop later with: prime volumes stop {name} {session.id}")
        # Match `prime pods ssh`: poll until a connection is published,
        # then invoke local ssh with the configured key. Bound the wait so
        # a failed provision does not spin forever; stopping remains an
        # explicit action (transfers may outlive this shell).
        with console.status("Waiting for SSH connection to become available...", spinner="dots"):
            deadline = time.monotonic() + 120
            while not session.ssh_connection and time.monotonic() < deadline:
                if session.status in (
                    "FAILED",
                    "STOPPED",
                    "COMPLETED",
                    "UNKNOWN",
                    "TERMINATING",
                    "TOMBSTONED",
                ):
                    console.print(f"[red]Session is {session.status}.[/red]")
                    raise typer.Exit(1)
                time.sleep(5)
                session = client.get_volume_session(name, session.id, team_id=team_id)
        if not session.ssh_connection:
            console.print("[red]Timed out waiting for SSH. Stop the session when done.[/red]")
            raise typer.Exit(1)
    except APIError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc
    console.print(f"[blue]Using SSH key:[/blue] {key}")
    console.print("[dim]To change SSH key path, use: prime config set-ssh-key-path[/dim]")
    match = _CONNECTION.fullmatch(session.ssh_connection)
    if not match or not 1 <= int(match.group("port") or 22) <= 65535:
        console.print("[red]Invalid SSH endpoint returned by server.[/red]")
        raise typer.Exit(1)
    host = f"{match.group('user')}@{match.group('host')}"
    port = match.group("port") or "22"
    known_hosts_opts = _pin_known_hosts(session, match.group("host"), port)
    base = ["ssh", *known_hosts_opts, "-i", key, "-p", port, host]
    # The same endpoint carries shell, sftp/scp and rsync; print copyable
    # examples using the pinned host key, never "trust anything".
    # markup=False: Rich must not parse [..] in paths; soft_wrap: it must
    # not insert line breaks into copyable commands. Read-only sessions
    # get download-direction examples (uploads would fail on the RO
    # mount); read-write sessions get uploads.
    if session.read_only:
        scp_cmd = ["scp", *known_hosts_opts, "-i", key, "-P", port, f"{host}:/volume/FILE", "."]
        rsync_cmd = ["rsync", "-av", "-e", shlex.join(base[:-1]), f"{host}:/volume/FILE", "."]
    else:
        scp_cmd = ["scp", *known_hosts_opts, "-i", key, "-P", port, "FILE", f"{host}:/volume/"]
        rsync_cmd = ["rsync", "-av", "-e", shlex.join(base[:-1]), "FILE", f"{host}:/volume/"]
    examples = [
        shlex.join(["sftp", *known_hosts_opts, "-i", key, "-P", port, host]),
        shlex.join(scp_cmd),
        shlex.join(rsync_cmd),
    ]
    for example in examples:
        console.print(example, soft_wrap=True, markup=False)
    try:
        code = subprocess.run(base, check=False).returncode
    except OSError as exc:
        console.print(f"[red]Could not start SSH:[/red] {exc}")
        raise typer.Exit(1) from exc
    if code:
        raise typer.Exit(code)


@app.command()
def stop(
    name: str = typer.Argument(..., help="Volume name"),
    session_id: str = typer.Argument(..., help="Session ID"),
) -> None:
    """Stop a volume SSH session without deleting the volume."""
    client, team_id = _client()
    try:
        client.stop_volume_session(name, session_id, team_id=team_id)
    except APIError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc
    console.print(f"Stopping session {session_id}.")
