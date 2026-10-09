"""`prime volumes`: named volumes for dedicated training runs (full-FT and SFT).

A volume is a PVC owned by your team (or you) on the cluster it was
created on. `prime train config.toml --volume <name>` makes the run
write under `runs/<runId>/` on it, and it outlives every run.

Hosted SFT runs also read their dataset from the volume: the training
container mounts the volume read-only at `/volume`, so an SFT config's
`[data] name` must point at a path on the volume (e.g.
`/volume/datasets/<name>`, or the relative `datasets/<name>` — the
platform resolves it) before launch. This command manages the
volume lifecycle (create/list/expand/delete, plus `prime volumes ssh` for
read-write sessions); put datasets there yourself with
`prime volumes ssh <name> --read-write` and the huggingface CLI inside
that session (`hf download <repo> --repo-type dataset --local-dir
datasets/<name>`) — the trainer never downloads.
"""

import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import typer
from rich.markup import escape
from rich.table import Table

from prime_cli.api.training import HostedTrainingClient, VolumeTransfer
from prime_cli.core import APIClient, APIError, Config, NotFoundError
from prime_cli.volume_gateway import GatewayError, relay

from ..utils import (
    PlainTyper,
    confirm_or_skip,
    get_console,
    human_age,
    output_data_as_json,
    validate_output_format,
)
from .rl import VOLUME_DEFAULT_SIZE

app = PlainTyper(
    help="Manage volumes for dedicated run outputs and SFT datasets (closed beta)",
    no_args_is_help=True,
)
console = get_console()
err_console = get_console(stderr=True)


def _client() -> tuple[HostedTrainingClient, str | None]:
    return HostedTrainingClient(APIClient()), Config().team_id


# The API reports a volume's lifecycle in job terms (RUNNING etc.); say what
# it means for a volume. Unknown values pass through.
_STATUS_LABELS = {
    "RUNNING": "CREATED",
    "TERMINATING": "DELETING",
    "TOMBSTONED": "DELETE FAILED",
}


def status_label(status: str) -> str:
    return _STATUS_LABELS.get(status, status)


@app.command()
def create(
    name: str = typer.Argument(..., help="Volume name (lowercase letters, digits, '-')"),
    size: str = typer.Option(
        VOLUME_DEFAULT_SIZE, "--size", help="Size cap, e.g. 500Gi or 5Ti. Can grow later."
    ),
    cluster: str | None = typer.Option(
        None,
        "--cluster",
        help="(deprecated) Volumes are no longer tied to a cluster",
    ),
    warm: bool = typer.Option(
        True,
        "--warm/--no-warm",
        help="Start a read-write session in the background (stops after 30 min idle)",
    ),
    output: str = typer.Option("table", "--output", "-o", help="Output format: table or json"),
) -> None:
    """Create a volume for your team (or yourself if you have no team)."""
    validate_output_format(output, console)
    if cluster:
        err_console.print(
            "[yellow]Warning:[/yellow] --cluster is deprecated: "
            "volumes are no longer tied to a cluster."
        )
    client, team_id = _client()
    try:
        volume = client.create_volume(name, size, team_id=team_id, cluster=cluster, warm=warm)
    except APIError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    if output == "json":
        output_data_as_json(volume.model_dump(by_alias=True), console)
        return
    console.print(
        f"[green]Volume {volume.name} ({volume.size}) is {status_label(volume.status)}.[/green]"
    )
    console.print(f"Use it with: prime train config.toml --volume {volume.name}")
    if warm:
        console.print("Starting a session in the background so the volume is ready for put/ssh.")


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
    table = Table("Name", "Size", "Status", "Created")
    for v in volumes:
        table.add_row(v.name, v.size or "-", status_label(v.status), v.created_at or "-")
    console.print(table)


@app.command()
def expand(
    name: str = typer.Argument(..., help="Volume name"),
    size: str = typer.Option(..., "--size", help="New size cap, larger than the current one"),
) -> None:
    """Raise a volume's size cap. Volumes can't shrink; runs already going
    keep the size they started with."""
    client, team_id = _client()
    try:
        volume = client.expand_volume(name, size, team_id=team_id)
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


def _session_dir() -> Path:
    """Where the CLI keeps volume-session ssh config and pinned host keys."""
    path = Path(Config().config_dir) / "volume-ssh"
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    return path


def _replace_block(text: str, alias: str, block: str) -> str:
    """ssh config text with the `Host <alias>` block replaced by `block`."""
    kept, skipping = [], False
    for line in text.splitlines():
        if line.startswith("Host "):
            skipping = line.split(None, 1)[1].strip() == alias
        if not skipping:
            kept.append(line)
    body = "\n".join(kept).strip()
    return (body + "\n\n" if body else "") + block


_DIRECT_HELP = "Connect over the tailnet even if the server offers a public gateway"
_GATEWAY_HOST = re.compile(r"[A-Za-z0-9]([A-Za-z0-9-]*[A-Za-z0-9])?(\.[A-Za-z0-9-]+)*")
_SHA256_HEX = re.compile(r"[0-9a-fA-F]{64}")


def _gateway_of(session, direct: bool):
    """The session's gateway as (host:port, sha256), or None to connect over
    the tailnet: `--direct`, no gateway offered, or an offer that does not look
    like a host, port and digest (it is written into the ssh config)."""
    gateway = getattr(session, "gateway", None)
    if direct or gateway is None:
        return None
    if not (
        _GATEWAY_HOST.fullmatch(gateway.host)
        and 0 < gateway.port < 65536
        and _SHA256_HEX.fullmatch(gateway.cert_sha256)
    ):
        console.print("[yellow]Ignoring an invalid gateway from the server.[/yellow]")
        return None
    return f"{gateway.host}:{gateway.port}", gateway.cert_sha256.lower()


def _proxy_command(gateway: tuple[str, str], sni: str) -> str:
    """ProxyCommand running `prime volumes proxy` with the interpreter that runs
    this CLI (works from a pipx install, `uv run` or a venv). ssh expands `%`
    tokens in it; %h would be the tailnet name and the gateway routes on the
    short one, so the SNI is written out."""
    argv = [sys.executable, "-m", "prime_cli.main", "volumes", "proxy"]
    argv += ["--gateway", gateway[0], "--cert-sha256", gateway[1], sni]
    command = subprocess.list2cmdline(argv) if os.name == "nt" else shlex.join(argv)
    return command.replace("%", "%%")


def _write_ssh_config(
    session, alias: str, host: str, user: str, port: str, key: str, gateway=None
) -> Path:
    """Write the session's options ONCE into ~/.prime/volume-ssh/config under a
    short Host alias, so ssh, scp, sftp and rsync only need `-F <file> <alias>`.

    The platform returns the session pod's sshd host key, so it is pinned in
    a CLI-owned known_hosts with StrictHostKeyChecking=yes (host-key checking
    is never disabled; with no host key, ssh falls back to the user's own
    ~/.ssh/known_hosts). IdentitiesOnly: offer only the configured key, so a
    loaded ssh-agent can't exhaust MaxAuthTries first. `-F` also keeps the
    user's ~/.ssh/config (e.g. ControlMaster) out of these connections.

    With a `gateway` (from _gateway_of), ssh reaches HostName through
    `prime volumes proxy` instead of the tailnet; the pinned host key still
    matches because HostName stays the tailnet name.

    ponytail: blocks for ended sessions accumulate (a few lines each); prune
    them if the file ever gets noisy.
    """
    folder = _session_dir()
    lines = [
        f"Host {alias}",
        f"  HostName {host}",
        f"  User {user}",
        f"  Port {port}",
        f'  IdentityFile "{Path(key).as_posix()}"',
        "  IdentitiesOnly yes",
    ]
    if gateway:
        lines.append(f"  ProxyCommand {_proxy_command(gateway, alias)}")
    if getattr(session, "host_public_key", None):
        known_hosts = folder / "known_hosts"
        entry = f"[{host}]:{port}" if port != "22" else host
        old = known_hosts.read_text().splitlines() if known_hosts.exists() else []
        pinned = [line for line in old if line.split(" ", 1)[0] != entry]
        pinned.append(f"{entry} {session.host_public_key}")
        known_hosts.write_text("\n".join(pinned) + "\n")
        known_hosts.chmod(0o600)
        lines += [f'  UserKnownHostsFile "{known_hosts.as_posix()}"', "  StrictHostKeyChecking yes"]
    config = folder / "config"
    existing = config.read_text() if config.exists() else ""
    config.write_text(_replace_block(existing, alias, "\n".join(lines) + "\n"))
    config.chmod(0o600)
    return config


def _shell_path(path: Path, home_var: str) -> str:
    """`path` as the user would type it: `~/...` (or `$HOME/...` inside
    double quotes) when it's under their home, else shell-quoted. On Windows
    there is no `~`/`$HOME` in cmd/PowerShell: forward-slash path in quotes."""
    if os.name == "nt":
        return f'"{path.as_posix()}"'
    try:
        rel = path.relative_to(Path.home())
    except ValueError:
        return shlex.quote(str(path))
    if any(c.isspace() for c in str(rel)) or any(c.isspace() for c in str(Path.home())):
        return shlex.quote(str(path))
    return f"{home_var}/{rel}"


# Terminal session states: the wait stops polling and reports them.
_DEAD_SESSION_STATES = (
    "FAILED",
    "STOPPED",
    "COMPLETED",
    "UNKNOWN",
    "TERMINATING",
    "TOMBSTONED",
)
# Session phases (newer backends) that will never publish an endpoint.
_DEAD_SESSION_PHASES = ("finalizing", "stopped", "failed")
# Consecutive failed status polls tolerated before giving up (5s apart).
_MAX_POLL_ERRORS = 6
# The platform allows a session up to 60m to come up: staging copies the
# volume's data from the bucket onto the session's disk first.
_SESSION_WAIT_SECONDS = 3600


def _waiting_text(session) -> str:
    """The spinner text for the session's phase (absent on older backends)."""
    phase = getattr(session, "phase", None)
    if phase == "creating":
        return "Creating SSH container..."
    if phase == "staging":
        progress = getattr(session, "progress", None)
        return "Copying data from bucket..." + (f" {escape(progress)}" if progress else "")
    return "Waiting for SSH connection to become available..."


def _wait_for_connection(client, name: str, session, team_id):
    """Poll until the session publishes its SSH endpoint. Raises typer.Exit
    on a dead session, a timeout, or a status API that keeps failing; a
    single failed poll (network blip, 5xx) is retried.

    Match `prime pods ssh`: poll, then invoke local ssh. The spinner follows
    the session's phase (creating, then staging with rclone's progress).
    """
    errors = 0
    shown = _waiting_text(session)
    # No spinner off a terminal (console.status gives None): print each
    # phase change as a line instead.
    with console.status(shown, spinner="dots") as spinner:
        deadline = time.monotonic() + _SESSION_WAIT_SECONDS
        while not session.ssh_connection and time.monotonic() < deadline:
            phase = getattr(session, "phase", None)
            if session.status in _DEAD_SESSION_STATES or phase in _DEAD_SESSION_PHASES:
                state = session.status if session.status in _DEAD_SESSION_STATES else str(phase)
                detail = f": {session.error_message}" if session.error_message else "."
                console.print(f"[red]Session is {state.upper()}{escape(detail)}[/red]")
                raise typer.Exit(1)
            time.sleep(5)
            try:
                session = client.get_volume_session(name, session.id, team_id=team_id)
                errors = 0
            except APIError as exc:
                errors += 1
                if errors >= _MAX_POLL_ERRORS:
                    console.print(f"[red]Error:[/red] {escape(str(exc))}")
                    raise typer.Exit(1) from exc
            text = _waiting_text(session)
            if text != shown:
                shown = text
                if spinner:
                    spinner.update(text)
                else:
                    console.print(text)
    if not session.ssh_connection:
        console.print("[red]Timed out waiting for SSH.[/red]")
        raise typer.Exit(1)
    return session


def _stop_quietly(client, name: str, session_id: str, team_id) -> None:
    """Best-effort stop of a session this command created but never used."""
    try:
        client.stop_volume_session(name, session_id, team_id=team_id)
        console.print(f"Stopped session {session_id}.")
    except Exception:
        console.print(
            f"[yellow]Could not stop session {session_id}; run: "
            f"prime volumes stop {escape(name)} {session_id}[/yellow]"
        )


def _open_session(
    name: str,
    read_only: bool,
    direct: bool = False,
    allow_writable: bool = False,
    existing=None,
    from_ssh: bool = False,
):
    """Create or reuse a session, wait for its endpoint and write the ssh
    config block. Returns (session, alias, key, config, via_gateway).

    `allow_writable` lets a read-only request reuse the caller's live
    read-write session. `existing` is a session the server already chose
    (the transfer route): it is used as is, and never stopped here.
    `from_ssh` adds the --read-only hint to a single-writer 409."""
    key = Config().ssh_key_path
    if not key or not os.path.isfile(os.path.expanduser(key)):
        console.print("[red]SSH key not found; use prime config set-ssh-key-path.[/red]")
        raise typer.Exit(1)
    key = os.path.expanduser(key)
    client, team_id = _client()
    if existing is not None:
        session = existing
    else:
        try:
            session = client.create_volume_session(
                name, read_only=read_only, allow_writable=allow_writable, team_id=team_id
            )
        except APIError as exc:
            # 409: one read-write session per volume (someone else's is live,
            # or an upload is in progress). Reported as is, never retried.
            detail = str(exc).removeprefix("HTTP 409: ")
            console.print(f"[red]Error:[/red] {escape(detail)}")
            if from_ssh and not read_only and "active read-write session" in detail:
                console.print(
                    f"Tip: prime volumes ssh {name} --read-only opens a read-only session",
                    markup=False,
                )
            raise typer.Exit(1) from exc
    mode = "read-only" if session.read_only else "read-write"
    reused = existing is not None or (allow_writable and not session.read_only)
    label = "Reusing session" if reused else "Session"
    console.print(
        f"{label} {session.id} ({mode}). Stop with: prime volumes stop {escape(name)} {session.id}"
    )
    connected = False
    try:
        session = _wait_for_connection(client, name, session, team_id)
        connected = True
    finally:
        # Never leave a session behind that this command created but never
        # connected to (a poll that kept failing, a timeout, Ctrl-C). The
        # platform's idle watchdog would reap it after 30 minutes anyway;
        # stopping it here is immediate. Best-effort.
        if not connected and existing is None:
            _stop_quietly(client, name, session.id, team_id)
    match = _CONNECTION.fullmatch(session.ssh_connection)
    if not match or not 1 <= int(match.group("port") or 22) <= 65535:
        console.print("[red]Invalid SSH endpoint returned by server.[/red]")
        raise typer.Exit(1)
    host = match.group("host")
    alias = host.split(".", 1)[0]
    gateway = _gateway_of(session, direct)
    config = _write_ssh_config(
        session, alias, host, match.group("user"), match.group("port") or "22", key, gateway
    )
    return session, alias, key, config, gateway is not None


@app.command(name="ssh", no_args_is_help=True)
def ssh(
    name: str = typer.Argument(..., help="Volume name"),
    read_only: bool = typer.Option(False, "--read-only", "--read", help="Mount root read-only"),
    read_write: bool = typer.Option(
        False, "--read-write", "--write", help="Mount root read-write (default)"
    ),
    direct: bool = typer.Option(False, "--direct", help=_DIRECT_HELP),
) -> None:
    """SSH into a session mounting the volume."""
    if read_only and read_write:
        console.print("[red]Choose either --read-only or --read-write.[/red]")
        raise typer.Exit(2)
    session, alias, key, config, _ = _open_session(
        name, read_only=read_only, direct=direct, from_ssh=True
    )
    base = ["ssh", "-F", str(config), alias]
    console.print(
        f"[blue]Using SSH key:[/blue] {escape(_shell_path(Path(key), '~'))} "
        "[dim](change with: prime config set-ssh-key-path)[/dim]"
    )
    # Copyable examples. markup=False: Rich must not parse [..] in paths;
    # soft_wrap: it must not insert line breaks into commands. Read-only
    # sessions get downloads (uploads would fail on the RO mount),
    # read-write sessions get uploads.
    cfg_in_quotes = _shell_path(config, "$HOME")
    if session.read_only:
        src, dst = f"{alias}:/volume/FILE", "."
        easy = f"prime volumes get {name} FILE ."
    else:
        src, dst = "FILE", f"{alias}:/volume/"
        easy = f"prime volumes put {name} FILE /"
    if direct:
        easy += " --direct"
    console.print("Copy files (sftp works too):")
    console.print(f"  {easy}", soft_wrap=True, markup=False)
    console.print("Or raw (power users):")
    console.print(
        f'  rsync -av -e "ssh -F {cfg_in_quotes}" {src} {dst}', soft_wrap=True, markup=False
    )
    try:
        code = subprocess.run(base, check=False).returncode
    except OSError as exc:
        console.print(f"[red]Could not start SSH:[/red] {exc}")
        raise typer.Exit(1) from exc
    if not session.read_only:
        console.print("Changes sync to the volume every minute.")
    console.print(
        "The session stops after 30 minutes idle. Stop it now with: "
        f"prime volumes stop {escape(name)} {session.id}"
    )
    if code:
        raise typer.Exit(code)


# Remote paths are passed to rsync/scp unquoted. Quoting isn't portable: GNU
# rsync >= 3.2.4 protects remote args itself (a quoted path would keep its
# quotes), openrsync and older rsync don't. So only characters that need no
# quoting on any remote shell are allowed.
_SAFE_REMOTE_SEGMENT = re.compile(r"[A-Za-z0-9._@%+=,:-]+")


def _remote_path(path: str) -> str:
    """Path under the volume root (/volume on the pod), normalized like
    posixpath: a leading "/" means the root, "." and empty segments are
    dropped and ".." climbs one level. A trailing "/" (or a final "." or
    "..") is kept as a trailing "/". Rejects a ".." that would climb out of
    the root, and any character that would need shell quoting (spaces, *,
    $, quotes, ...)."""
    parts: list[str] = []
    for p in path.split("/"):
        if p == "..":
            if not parts:
                console.print(
                    f"[red]Invalid remote path {escape(repr(path))}: '..' climbs above "
                    "the volume root.[/red]"
                )
                raise typer.Exit(2)
            parts.pop()
        elif p not in ("", "."):
            parts.append(p)
    if not all(_SAFE_REMOTE_SEGMENT.fullmatch(p) for p in parts):
        console.print(
            f"[red]Invalid remote path {escape(repr(path))}: use letters, digits and "
            "._-@%+=,: only (no spaces or shell characters). For other names, use "
            "`prime volumes ssh`.[/red]"
        )
        raise typer.Exit(2)
    into = path.endswith("/") or path.rsplit("/", 1)[-1] in (".", "..")
    return "/volume/" + "/".join(parts) + ("/" if parts and into else "")


def _transfer_failed(alias: str, code: int, via_gateway: bool) -> None:
    """The connectivity hint a failed transfer (or symlink check) reports."""
    if via_gateway:
        reach = "that you can reach the gateway (outbound TCP 443)"
    else:
        reach = f"that you are on the tailnet (host {alias} must resolve)"
    console.print(f"[red]Transfer failed.[/red] Check {reach} and the path exists.")
    raise typer.Exit(code)


def _local_tree_has_symlink(path: str) -> bool:
    """True if `path` itself or anything under it is a symlink. os.walk
    (followlinks=False) lists symlinked directories but never enters them,
    so every entry in the tree is checked and no link is followed."""
    if os.path.islink(path):
        return True
    if not os.path.isdir(path):
        return False
    return any(
        os.path.islink(os.path.join(root, name))
        for root, dirs, files in os.walk(path, followlinks=False)
        for name in dirs + files
    )


def _remote_tree_has_symlink(alias: str, config: Path, remote: str, via_gateway: bool) -> bool:
    """True if `remote` (on the session pod) or anything under it is a
    symlink. One ssh call: the pod's BusyBox find supports -type l and
    tests the starting point too, and `head` caps the output and makes the
    pipeline report 0 even when find hits many links. `remote` is limited
    to shell-safe characters by _remote_path, so it needs no quoting."""
    check = subprocess.run(
        ["ssh", "-F", str(config), alias, f"find {remote} -type l | head -n 1"],
        check=False,
        capture_output=True,
        text=True,
    )
    if check.returncode:
        _transfer_failed(alias, check.returncode, via_gateway)
    return bool(check.stdout.strip())


# One TCP stream tops out well below the path's capacity (~14 MB/s single vs
# ~33 MB/s over 4 streams on rft-telus, ENG-6450), so a directory's files are
# split across this many concurrent rsyncs.
# ponytail: files are the unit of split, so one huge file stays single-stream;
# chunk large files if that ever matters.
_STREAMS = 4


def _split(entries: list[tuple[int, str]], n: int) -> list[list[str]]:
    """Largest first onto the least-loaded bucket, so buckets end up about
    the same size in bytes. Empty buckets are dropped."""
    buckets: list[list[str]] = [[] for _ in range(n)]
    loads = [0] * n
    for size, path in sorted(entries, reverse=True):
        i = loads.index(min(loads))
        buckets[i].append(path)
        loads[i] += size
    return [b for b in buckets if b]


def _rsync_base(path: str) -> str:
    """The directory a --files-from list is relative to, keeping rsync's
    source layout: "dir/" copies the contents of dir, "dir" copies dir
    itself, so its entries are listed as "dir/..." under dir's parent."""
    if path.endswith("/"):
        return path
    return (os.path.dirname(path) or ".") + "/"


def _local_entries(path: str) -> list[tuple[int, str]] | None:
    """(size, file) for everything but directories under a local directory,
    relative to its rsync base. Symlinks count as files (rsync -a copies them
    as links). None when `path` is not a directory or a name has a newline,
    which a --files-from list cannot carry."""
    if os.path.islink(path) or not os.path.isdir(path):
        return None
    base = _rsync_base(path)
    files = []
    for root, subdirs, names in os.walk(path, followlinks=False):
        for entry in subdirs + names:
            full = os.path.join(root, entry)
            rel = os.path.relpath(full, base)
            if "\n" in rel:
                return None
            if os.path.islink(full) or not os.path.isdir(full):
                files.append((os.lstat(full).st_size, rel))
    return files


def _remote_entries(alias: str, config: Path, remote: str) -> list[tuple[int, str]] | None:
    """Like _local_entries, for a directory on the session pod. One ssh call:
    BusyBox find + stat (stat does not follow links). None if the listing
    fails or `remote` is not a directory; the caller then falls back to one
    rsync, which reports the error itself. `remote` is shell-safe
    (_remote_path)."""
    listing = subprocess.run(
        [
            "ssh",
            "-F",
            str(config),
            alias,
            f"find {remote} -mindepth 1 -exec stat -c '%s|%F|%n' {{}} +",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if listing.returncode or not listing.stdout:
        return None
    base = _rsync_base(remote)
    root = remote.rstrip("/") + "/"
    files = []
    for line in listing.stdout.splitlines():
        size, kind, full = line.split("|", 2)
        if not full.startswith(root):
            return None  # a name with a newline split across lines
        if kind != "directory":
            files.append((int(size), os.path.relpath(full, base)))
    return files


def _parallel_rsync(
    cmd: list[str], tree: list[str], src: str, dst: str, files: list[tuple[int, str]]
) -> int | None:
    """Run the transfer as up to _STREAMS rsyncs, each given its share of the
    files with --files-from (on macOS openrsync and GNU rsync alike). First,
    one rsync copies only the directory tree (`tree` is the transfer's own
    source and destination, filtered to directories), so the parallel ones
    never race to mkdir the same path (GNU rsync fails one of them with code
    11), and empty directories are kept. Not a "." entry in --files-from:
    openrsync recurses into it and copies every file single-stream. None
    when there are too few files to split. Returns the first non-zero exit
    code, else 0."""
    buckets = _split(files, _STREAMS)
    if len(buckets) < 2:
        return None
    console.print(f"Transferring {len(files)} files over {len(buckets)} parallel streams")
    with tempfile.TemporaryDirectory() as tmp:

        def listed(i: int, paths: list[str]) -> list[str]:
            listfile = Path(tmp) / f"files-{i}"
            listfile.write_text("".join(f"{p}\n" for p in paths))
            return [*cmd, f"--files-from={listfile}", src, dst]

        dirs_only = [*cmd, "--include=*/", "--exclude=*", *tree]
        if code := subprocess.run(dirs_only, check=False).returncode:
            return code
        procs = [subprocess.Popen(listed(i, b)) for i, b in enumerate(buckets)]
        codes = [p.wait() for p in procs]
    return next((c for c in codes if c), 0)


def _refuse_runs(local: str, rel: str) -> None:
    """Exit if a put would write under runs/: run outputs live there, and a
    session's sync and staging leave runs/ alone, so such a put would be lost
    or clobber a run. `rel` is the volume-relative destination."""
    if rel:
        top = {rel.split("/", 1)[0]}
    elif os.path.isdir(local) and _copies_contents(local):
        top = set(os.listdir(local))
    else:
        top = {os.path.basename(os.path.normpath(local))}
    if "runs" in top:
        console.print("[red]runs/ holds run outputs and is read-only; put elsewhere.[/red]")
        raise typer.Exit(2)


def _copies_contents(local: str) -> bool:
    """rsync's rule: "dir/" (and "." or "..") copies a directory's contents,
    "dir" copies the directory itself."""
    return local.endswith(("/", os.sep)) or os.path.basename(os.path.normpath(local)) in (
        ".",
        "..",
    )


def _transfer(
    name: str, read_only: bool, remote: str, local: str, upload: bool, direct: bool
) -> None:
    remote = _remote_path(remote)
    rel = remote.removeprefix("/volume/")
    if upload:
        # A top-level link would be followed by the direct upload (and by
        # scp), but copied as a link by rsync; refuse it on both routes.
        if os.path.islink(local):
            console.print(f"[red]{escape(local)} is a symlink; pass the path it points to.[/red]")
            raise typer.Exit(1)
        _refuse_runs(local, rel)
    client, team_id = _client()
    try:
        route = client.route_volume_transfer(name, "put" if upload else "get", rel, team_id=team_id)
    except NotFoundError:
        route = None  # an older backend: always through a session
    except APIError as exc:
        # 409 (single writer: another read-write session or an upload is
        # live, or yours uses an old key): report it; only a 404 falls back.
        console.print(f"[red]Error:[/red] {escape(str(exc).removeprefix('HTTP 409: '))}")
        raise typer.Exit(1) from exc
    if route is not None and route.via == "r2":
        _r2_transfer(client, name, team_id, route, rel, local, upload)
        return
    existing = route.session if route is not None else None
    rsync = shutil.which("rsync")
    if not (shutil.which("ssh") and (rsync or shutil.which("scp"))):
        console.print("[red]ssh and scp (or rsync) are required; install the OpenSSH client.[/red]")
        raise typer.Exit(1)
    if not rsync:
        console.print(
            "rsync not found, using scp (full copy; install rsync for incremental transfers)"
        )
    # A relative local path starting with "-" would be parsed as an option, and
    # one containing ":" as a HOST:PATH remote operand, by rsync and scp alike.
    # A "./" prefix makes it a plain local path for every implementation
    # (more portable than relying on each tool's "--").
    if not os.path.isabs(local) and (local.startswith("-") or ":" in local):
        local = "./" + local
    session, alias, _key, config, via_gateway = _open_session(
        name,
        read_only=read_only,
        direct=direct,
        allow_writable=read_only,
        existing=existing,
    )
    if rsync:
        ssh_cmd = shlex.join(["ssh", "-F", str(config)])
        # A flag subset both sides take: macOS's openrsync (the laptop) and the
        # pod's Alpine GNU rsync. --partial-dir (implies --partial) parks an
        # interrupted file in <dest>/.rsync-partial/ instead of under its final
        # name, and a rerun resumes from the parked file in either direction.
        cmd = [rsync, "-a", "-v", "--partial-dir=.rsync-partial", "-e", ssh_cmd]
    else:
        # scp -r FOLLOWS symlinks and copies their TARGETS; rsync -a copies
        # them as links. Refuse a source containing one instead of letting
        # what gets copied depend on which tool is installed (on upload, a
        # link could copy files from outside the tree, e.g. ~/.ssh).
        if upload:
            linked = _local_tree_has_symlink(local)
        else:
            linked = _remote_tree_has_symlink(alias, config, remote, via_gateway)
        if linked:
            console.print(
                "[red]The source contains symbolic links, which scp would follow "
                "(it copies their targets). Install rsync, which copies links as "
                "links, or remove the links.[/red]"
            )
            raise typer.Exit(1)
        cmd = ["scp", "-r", "-F", str(config)]
        # rsync copies a directory's CONTENTS when the source ends in "/";
        # scp would copy the directory itself. "dir/." gives scp rsync's
        # layout, so the result doesn't depend on which tool is installed.
        if upload and local.endswith(("/", os.sep)):
            local += "."
        elif not upload and remote.endswith("/"):
            remote += "."
    remote_arg = f"{alias}:{remote}"
    try:
        if rsync:
            if upload:
                entries = _local_entries(local)
                src, dst = _rsync_base(local), remote_arg
            else:
                entries = _remote_entries(alias, config, remote)
                src, dst = f"{alias}:{_rsync_base(remote)}", local
            tree = [local, remote_arg] if upload else [remote_arg, local]
            code = _parallel_rsync(cmd, tree, src, dst, entries) if entries else None
            if code is not None:
                if code:
                    _transfer_failed(alias, code, via_gateway)
                return
        cmd += [local, remote_arg] if upload else [remote_arg, local]
        code = subprocess.run(cmd, check=False).returncode
    except OSError as exc:
        console.print(f"[red]Could not start transfer:[/red] {exc}")
        raise typer.Exit(1) from exc
    if code:
        _transfer_failed(alias, code, via_gateway)


# --- Direct get/put against the volume's R2 prefix (ENG-6585) --------------
# With no live session of the caller's on the volume, the platform hands out
# credentials scoped to the volume's prefix and the CLI talks to R2 itself:
# no pod to wait for. Path layout mirrors the rsync commands above.

# Files in flight at once, and parts in flight per file (multipart).
_R2_FILE_WORKERS = 8
_R2_PART_WORKERS = 4
_R2_PART_SIZE = 64 * 1024 * 1024


def _r2_client(refresh, route):
    """An S3 client for R2 with the route's credentials. botocore refreshes
    them through `refresh` (a new route call) before `expiresAt`, so a
    transfer longer than the credential TTL keeps going."""
    import boto3
    from botocore.config import Config as BotoConfig
    from botocore.credentials import RefreshableCredentials
    from botocore.session import get_session

    def metadata(r) -> dict:
        return {
            "access_key": r.access_key_id,
            "secret_key": r.secret_access_key,
            "token": r.session_token,
            "expiry_time": r.expires_at,
        }

    def fetch() -> dict:
        # A refresh can be refused too (409: a teammate started a read-write
        # session); RuntimeError is what _r2_transfer reports as a failure.
        try:
            r = refresh()
        except APIError as exc:
            raise RuntimeError(str(exc).removeprefix("HTTP 409: ")) from exc
        if r.via != "r2":
            raise RuntimeError("a session started on the volume; rerun the command")
        return metadata(r)

    core = get_session()
    if route.expires_at:
        # botocore's default windows (refresh 15 min before expiry) would
        # refresh 15-minute put credentials on every request.
        core._credentials = RefreshableCredentials.create_from_metadata(
            metadata(route),
            fetch,
            "prime-volumes-transfer",
            advisory_timeout=5 * 60,
            mandatory_timeout=2 * 60,
        )
    else:
        core.set_credentials(route.access_key_id, route.secret_access_key, route.session_token)
    return boto3.Session(botocore_session=core).client(
        "s3",
        endpoint_url=route.endpoint,
        region_name="auto",
        config=BotoConfig(
            signature_version="s3v4",
            max_pool_connections=_R2_FILE_WORKERS * _R2_PART_WORKERS,
        ),
    )


def _r2_keys(s3, bucket: str, prefix: str):
    """(key, size) of every object under `prefix`."""
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            yield obj["Key"], obj["Size"]


def _r2_is_dir(s3, bucket: str, prefix: str) -> bool:
    page = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1)
    return bool(page.get("Contents"))


def _r2_head(s3, bucket: str, key: str) -> int | None:
    """The object's size, or None if there is no such object."""
    from botocore.exceptions import ClientError

    try:
        return s3.head_object(Bucket=bucket, Key=key)["ContentLength"]
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") in ("404", "NoSuchKey", "NotFound"):
            return None
        raise


def _put_plan(s3, bucket: str, prefix: str, local: str, rel: str):
    """(local file, key, size) for `put LOCAL REL`, laid out like
    `rsync -a LOCAL host:/volume/REL`: a file lands at REL, or inside it when
    REL is the root, ends in "/" or is an existing directory; a directory
    lands inside REL as itself ("dir") or as its contents ("dir/").
    Symlinks inside a directory are skipped (R2 has no links, and following
    them could upload files from outside the tree). Returns (plan, skipped)."""
    if not os.path.isdir(local):
        if not os.path.isfile(local):
            console.print(f"[red]No such file or directory: {escape(local)}[/red]")
            raise typer.Exit(1)
        base = rel.rstrip("/")
        into = not base or rel.endswith("/") or _r2_is_dir(s3, bucket, f"{prefix}{base}/")
        key = f"{base}/{os.path.basename(local)}".lstrip("/") if into else base
        return [(local, prefix + key, os.path.getsize(local))], 0
    dest = rel.rstrip("/") + "/" if rel.rstrip("/") else ""
    if not _copies_contents(local):
        dest += os.path.basename(os.path.normpath(local)) + "/"
    plan, skipped = [], 0
    for root, dirs, files in os.walk(local, followlinks=False):
        for entry in dirs + files:
            full = os.path.join(root, entry)
            if os.path.islink(full):
                skipped += 1
            elif os.path.isfile(full):
                key = dest + Path(os.path.relpath(full, local)).as_posix()
                plan.append((full, prefix + key, os.path.getsize(full)))
    return plan, skipped


def _get_plan(s3, bucket: str, prefix: str, rel: str, local: str):
    """(key, local file, size) for `get REL LOCAL`, laid out like
    `rsync -a host:/volume/REL LOCAL`: a file lands at LOCAL, or inside it
    when LOCAL is a directory or ends in "/"; a directory lands inside LOCAL
    as itself ("dir") or as its contents ("dir/", or the root). Keys that
    would escape LOCAL ("..", empty segments) and the sessions' sync markers
    are skipped. Returns (plan, skipped)."""
    base = rel.rstrip("/")
    if base and not rel.endswith("/"):
        size = _r2_head(s3, bucket, prefix + base)
        if size is not None:
            if os.path.isdir(local) or local.endswith(("/", os.sep)):
                local = os.path.join(local, base.rsplit("/", 1)[-1])
            return [(prefix + base, local, size)], 0
    folder = f"{prefix}{base}/" if base else prefix
    into = local if not base or rel.endswith("/") else os.path.join(local, base.rsplit("/", 1)[-1])
    plan, skipped = [], 0
    for key, size in _r2_keys(s3, bucket, folder):
        sub = key[len(folder) :]
        if key.endswith("/"):
            continue  # a "directory" marker object
        if f"{base}/{sub}".lstrip("/").startswith("runs/.sessions/"):
            continue
        parts = sub.split("/")
        if any(p in ("", ".", "..") for p in parts) or any("\\" in p for p in parts):
            skipped += 1
            continue
        plan.append((key, os.path.join(into, *parts), size))
    if not plan and not skipped:
        console.print(f"[red]No such file or directory on the volume: /{escape(base)}[/red]")
        raise typer.Exit(1)
    return plan, skipped


def _run_r2_jobs(jobs, verb: str) -> None:
    """Run `jobs` ((size, fn(callback)) pairs) on a thread pool with one
    progress line: files done, bytes, rate, ETA. The first failure cancels
    what has not started and is re-raised."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from rich.filesize import decimal
    from rich.progress import (
        BarColumn,
        DownloadColumn,
        Progress,
        TextColumn,
        TimeRemainingColumn,
        TransferSpeedColumn,
    )

    total = sum(size for size, _ in jobs)
    console.print(f"{verb} {len(jobs)} files ({decimal(total)})")
    columns = (
        TextColumn("{task.fields[files]}"),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
    )
    with Progress(*columns, console=console) as progress:
        task = progress.add_task("", total=total, files=f"0/{len(jobs)} files")

        def advance(n: int) -> None:
            progress.advance(task, n)

        pool = ThreadPoolExecutor(_R2_FILE_WORKERS)
        try:
            futures = [pool.submit(fn, advance) for _, fn in jobs]
            for done, future in enumerate(as_completed(futures), 1):
                future.result()
                progress.update(task, files=f"{done}/{len(jobs)} files")
        finally:
            pool.shutdown(wait=True, cancel_futures=True)


def _r2_transfer(client, name: str, team_id, route, rel: str, local: str, upload: bool) -> None:
    from boto3.exceptions import Boto3Error
    from boto3.s3.transfer import TransferConfig
    from botocore.exceptions import BotoCoreError, ClientError

    mode = "put" if upload else "get"
    s3 = _r2_client(lambda: client.route_volume_transfer(name, mode, rel, team_id=team_id), route)
    config = TransferConfig(
        multipart_threshold=_R2_PART_SIZE,
        multipart_chunksize=_R2_PART_SIZE,
        max_concurrency=_R2_PART_WORKERS,
    )
    bucket, prefix = route.bucket, route.prefix
    try:
        if upload:
            plan, skipped = _put_plan(s3, bucket, prefix, local, rel)

            def up(path, key):
                return lambda cb: s3.upload_file(path, bucket, key, Config=config, Callback=cb)

            jobs = [(size, up(path, key)) for path, key, size in plan]
        else:
            plan, skipped = _get_plan(s3, bucket, prefix, rel, local)

            def down(key, path):
                def run(cb):
                    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
                    s3.download_file(bucket, key, path, Config=config, Callback=cb)

                return run

            jobs = [(size, down(key, path)) for key, path, size in plan]
        if skipped:
            what = "symbolic links" if upload else "objects with unsafe names"
            console.print(f"[yellow]Skipping {skipped} {what}.[/yellow]")
        if not jobs:
            console.print("Nothing to transfer.")
            return
        verb = f"Uploading to {escape(name)}:" if upload else f"Downloading from {escape(name)}:"
        _run_r2_jobs(jobs, verb)
    # upload_file/download_file wrap a ClientError in S3UploadFailedError
    # (a Boto3Error), so catch that family too.
    except (Boto3Error, BotoCoreError, ClientError, OSError, RuntimeError) as exc:
        console.print(f"[red]Transfer failed:[/red] {escape(str(exc))}")
        raise typer.Exit(1) from exc


@app.command(no_args_is_help=True)
def get(
    name: str = typer.Argument(..., help="Volume name"),
    remote_path: str = typer.Argument(..., help="Path in the volume; / is the volume root"),
    local_dest: str = typer.Argument(".", help="Local destination"),
    direct: bool = typer.Option(False, "--direct", help=_DIRECT_HELP),
) -> None:
    """Download from a volume: straight from storage, or through your live
    read-write session if you have one (rsync, else scp)."""
    _transfer(name, True, remote_path, local_dest, upload=False, direct=direct)


@app.command(no_args_is_help=True)
def put(
    name: str = typer.Argument(..., help="Volume name"),
    local_path: str = typer.Argument(..., help="Local file or directory"),
    remote_path: str = typer.Argument("/", help="Path in the volume; trailing / = into directory"),
    direct: bool = typer.Option(False, "--direct", help=_DIRECT_HELP),
) -> None:
    """Upload to a volume (not under runs/): straight to storage, or through
    your live read-write session if you have one (rsync, else scp)."""
    _transfer(name, False, remote_path, local_path, upload=True, direct=direct)


# --- S3 import/export (ENG-6450) ------------------------------------------
# rclone runs on the volume's cluster, so the S3 leg uses the cluster's
# egress instead of the caller's laptop. Commands are async: create returns
# a transfer id at once, and `transfers list --follow` polls it to the end.

_TRANSFER_TERMINAL = ("succeeded", "failed", "cancelled")
_TRANSFER_POLL_SECONDS = 5


def _aws_credentials() -> dict:
    """S3 credentials from the standard AWS environment variables.

    Prime never takes credentials as flags: they travel in the create
    request body only and are stored in a per-transfer k8s Secret that is
    deleted when the transfer ends.
    """
    access_key = os.environ.get("AWS_ACCESS_KEY_ID")
    secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
    if not access_key or not secret_key:
        missing = "AWS_ACCESS_KEY_ID" if not access_key else "AWS_SECRET_ACCESS_KEY"
        console.print(
            f"[red]Error:[/red] {missing} is not set. S3 credentials are read from the "
            "AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_SESSION_TOKEN environment "
            "variables."
        )
        raise typer.Exit(1)
    return {
        "accessKeyId": access_key,
        "secretAccessKey": secret_key,
        "sessionToken": os.environ.get("AWS_SESSION_TOKEN"),
    }


def _short_number(value: str) -> str:
    """`412.345 GiB` -> `412.3 GiB`: rclone's third decimal is noise here."""
    number, _, unit = value.partition(" ")
    try:
        rounded = f"{float(number):.1f}"
    except ValueError:
        return value
    return f"{rounded} {unit}".strip()


def _short_eta(eta: str) -> str:
    """`1h48m0s` -> `1h48m`; rclone always writes the seconds."""
    return eta[:-2] if eta.endswith("m0s") else eta


def _duration(started_at: str | None, completed_at: str | None) -> str | None:
    """`2h21m` between two ISO timestamps, or None if either is unusable."""
    if not started_at or not completed_at:
        return None
    try:
        start = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
        end = datetime.fromisoformat(completed_at.replace("Z", "+00:00"))
    except ValueError:
        return None
    seconds = int((end - start).total_seconds())
    if seconds < 0:
        return None
    hours, rem = divmod(seconds, 3600)
    minutes = rem // 60
    if hours:
        return f"{hours}h{minutes}m"
    if minutes:
        return f"{minutes}m"
    return f"{seconds}s"


def _progress_cell(transfer: VolumeTransfer) -> str:
    """The transfer's PROGRESS cell / follow line: rclone stats while
    running, the total (with duration) when done, the error when failed."""
    if transfer.status == "succeeded":
        progress = transfer.progress
        size = _short_number(progress.total) if progress and progress.total else "done"
        took = _duration(transfer.started_at, transfer.completed_at)
        return f"{size} in {took}" if took else size
    if transfer.status in ("failed", "cancelled"):
        return transfer.error_message or "-"
    progress = transfer.progress
    if progress is None:
        return "-"
    bits = []
    if progress.transferred or progress.total:
        size = _short_number(progress.transferred) if progress.transferred else "?"
        if progress.total:
            size += f" / {_short_number(progress.total)}"
        if progress.percentage is not None:
            size += f" ({progress.percentage}%)"
        bits.append(size)
    if progress.rate:
        bits.append(_short_number(progress.rate))
    if progress.eta:
        bits.append(f"ETA {_short_eta(progress.eta)}")
    return ", ".join(bits) if bits else "-"


def _transfers_table(transfers: list[VolumeTransfer]) -> Table:
    table = Table("ID", "DIRECTION", "PATH", "URL", "STATUS", "PROGRESS")
    for transfer in transfers:
        table.add_row(
            transfer.id,
            transfer.direction,
            escape(transfer.path or "/"),
            escape(transfer.url),
            transfer.status,
            escape(_progress_cell(transfer)),
        )
    return table


def _follow_transfer(client, name: str, transfer_id: str, team_id: str | None) -> None:
    """Poll one transfer and print each new state line until it is terminal.
    One failed poll (network blip, 5xx) is retried, like `volumes ssh`."""
    errors = 0
    shown = None
    while True:
        try:
            transfer = client.get_volume_transfer(name, transfer_id, team_id=team_id)
            errors = 0
        except APIError as exc:
            errors += 1
            if errors >= _MAX_POLL_ERRORS:
                console.print(f"[red]Error:[/red] {escape(str(exc))}")
                raise typer.Exit(1) from exc
            time.sleep(_TRANSFER_POLL_SECONDS)
            continue
        line = f"{transfer.status:<11}{_progress_cell(transfer)}"
        if line != shown:
            console.print(escape(line))
            shown = line
        if transfer.status in _TRANSFER_TERMINAL:
            return
        time.sleep(_TRANSFER_POLL_SECONDS)


def _start_transfer(
    name: str,
    direction: str,
    path: str,
    url: str,
    region: str | None,
    endpoint_url: str | None,
) -> None:
    credentials = _aws_credentials()
    client, team_id = _client()
    try:
        transfer = client.create_volume_transfer(
            name,
            direction=direction,
            path=path,
            url=url,
            credentials=credentials,
            region=region,
            endpoint_url=endpoint_url,
            team_id=team_id,
        )
    except APIError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    where = f"{escape(name)}:{escape(path or '/')}"
    if direction == "export":
        summary = f"Started export of {where} to {escape(url)}"
    else:
        summary = f"Started import of {where} from {escape(url)}"
    console.print(f"[green]{summary}[/green]")
    check = f"prime volumes transfers list {escape(name)} --id {transfer.id}"
    console.print(f"Transfer id: {transfer.id}")
    console.print(f"Check status: {check}")
    console.print(f"Follow:       {check} --follow")


@app.command(no_args_is_help=True)
def export(
    name: str = typer.Argument(..., help="Volume name"),
    path: str = typer.Argument(..., help="Volume-relative directory to copy (its contents)"),
    url: str = typer.Argument(..., help="Destination s3://bucket/prefix"),
    region: str | None = typer.Option(None, "--region", help="S3 region (default: us-east-1)"),
    endpoint_url: str | None = typer.Option(
        None, "--endpoint-url", help="Endpoint for S3-compatible stores (MinIO, R2, ...)"
    ),
) -> None:
    """Copy a volume directory to S3, with rclone on the volume's cluster."""
    _start_transfer(name, "export", path, url, region, endpoint_url)


@app.command("import", no_args_is_help=True)
def import_(
    name: str = typer.Argument(..., help="Volume name"),
    url: str = typer.Argument(..., help="Source s3://bucket/prefix"),
    path: str = typer.Argument(..., help="Volume-relative directory to copy into"),
    region: str | None = typer.Option(None, "--region", help="S3 region (default: us-east-1)"),
    endpoint_url: str | None = typer.Option(
        None, "--endpoint-url", help="Endpoint for S3-compatible stores (MinIO, R2, ...)"
    ),
) -> None:
    """Copy an S3 prefix onto a volume, with rclone on the volume's cluster."""
    _start_transfer(name, "import", path, url, region, endpoint_url)


transfers_app = PlainTyper(help="List or cancel a volume's S3 transfers", no_args_is_help=True)


@transfers_app.command("list")
def transfers_list(
    name: str = typer.Argument(..., help="Volume name"),
    transfer_id: str | None = typer.Option(None, "--id", help="Show one transfer by id"),
    follow: bool = typer.Option(False, "--follow", help="Poll one transfer until it finishes"),
    output: str = typer.Option("table", "--output", "-o", help="Output format: table or json"),
) -> None:
    """List a volume's transfers, newest first, or one with --id."""
    validate_output_format(output, console)
    if follow:
        if not transfer_id:
            console.print("[red]Error:[/red] --follow needs --id (one transfer).")
            raise typer.Exit(2)
        if output == "json":
            console.print("[red]Error:[/red] --follow cannot be combined with --output json.")
            raise typer.Exit(2)
        client, team_id = _client()
        _follow_transfer(client, name, transfer_id, team_id)
        return
    client, team_id = _client()
    try:
        if transfer_id:
            transfers = [client.get_volume_transfer(name, transfer_id, team_id=team_id)]
        else:
            transfers = client.list_volume_transfers(name, team_id=team_id)
    except APIError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    if output == "json":
        data = (
            transfers[0].model_dump(by_alias=True)
            if transfer_id
            else [t.model_dump(by_alias=True) for t in transfers]
        )
        output_data_as_json(data, console)
        return
    console.print(_transfers_table(transfers))


@transfers_app.command("cancel")
def transfers_cancel(
    name: str = typer.Argument(..., help="Volume name"),
    transfer_id: str = typer.Argument(..., help="Transfer ID"),
) -> None:
    """Cancel a transfer; the volume and the files already moved stay."""
    client, team_id = _client()
    try:
        client.cancel_volume_transfer(name, transfer_id, team_id=team_id)
    except APIError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    console.print(f"Cancelling transfer {transfer_id}.")


app.add_typer(transfers_app, name="transfers")


@app.command(hidden=True)
def proxy(
    sni: str = typer.Argument(..., help="Session hostname (vol-ssh-<hex>) the gateway routes on"),
    gateway: str = typer.Option(..., "--gateway", help="Gateway host:port"),
    cert_sha256: str = typer.Option(
        ..., "--cert-sha256", help="Pinned SHA-256 of the gateway cert"
    ),
) -> None:
    """ssh ProxyCommand: relay stdin/stdout to a session through the gateway."""
    try:
        relay(gateway, cert_sha256, sni, sys.stdin.fileno(), sys.stdout.fileno())
    except GatewayError as exc:
        typer.echo(f"prime volumes proxy: {exc}", err=True)
        raise typer.Exit(1) from exc


def _session_age(created_at: str | None) -> str:
    """The CREATED cell: kubectl-style age, "-" without a parsable createdAt."""
    if not created_at:
        return "-"
    try:
        return human_age(datetime.fromisoformat(created_at.replace("Z", "+00:00")))
    except ValueError:
        return "-"


@app.command("sessions")
def sessions(
    name: str | None = typer.Argument(None, help="Volume name (default: all your volumes)"),
    output: str = typer.Option("table", "--output", "-o", help="Output format: table or json"),
) -> None:
    """List your volume SSH sessions (stop them with `prime volumes stop`)."""
    validate_output_format(output, console)
    client, team_id = _client()
    try:
        if name:
            per_volume = [(name, client.list_volume_sessions(name, team_id=team_id))]
        else:
            per_volume = [
                (v.name, client.list_volume_sessions(v.name, team_id=team_id))
                for v in client.list_volumes(team_id=team_id)
            ]
    except APIError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    rows = [(volume, s) for volume, found in per_volume for s in found]
    if output == "json":
        output_data_as_json([s.model_dump(by_alias=True) for _, s in rows], console)
        return
    table = Table("VOLUME", "SESSION ID", "STATUS", "MODE", "CREATED")
    for volume, s in rows:
        mode = "ro" if s.read_only else "rw"
        table.add_row(volume, s.id, s.status, mode, _session_age(s.created_at))
    console.print(table)


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
