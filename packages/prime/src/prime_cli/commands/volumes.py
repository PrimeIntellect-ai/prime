"""`prime volumes`: named volumes for hosted training data and outputs.

A volume is a PVC owned by your team (or you) on the cluster it was
created on. `prime train config.toml --volume <name>` makes the run
write under `runs/<runId>/` on it, and it outlives every run. For hosted
SFT, `prime volumes stage` puts an HF dataset on it under
`datasets/<name>` so the run can read `[data] name = "/datasets/<name>"`.
"""

import os
import re
import shlex
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional

import typer
from rich.markup import escape
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
from ..utils.env_vars import EnvParseError
from .volumes_stage import StageError, print_stage_output, resolve_hf_token, stage_dataset

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
    table = Table("Name", "Size", "Status", "Created")
    for v in volumes:
        table.add_row(v.name, v.size or "-", v.status, v.created_at or "-")
    console.print(table)


@app.command()
def stage(
    source: str = typer.Argument(
        ...,
        help=(
            "HF dataset repository ID, 'name' or 'owner/name' (no URLs, "
            "hf:// aliases, or local paths - local upload is deferred; publish "
            "a dataset repo first)"
        ),
    ),
    volume: str = typer.Option(..., "--volume", help="Existing volume to stage onto."),
    path: Optional[str] = typer.Option(
        None,
        "--path",
        help=(
            "Dataset directory name under datasets/ (default: repository "
            "basename, kept exactly). One directory component; never sanitized."
        ),
    ),
    namespace: str = typer.Option(
        "auto",
        "--namespace",
        help="Must be 'auto' (use the volume's API-returned namespace) or match it.",
    ),
    kube_context: str = typer.Option(
        None,
        "--kube-context",
        help=(
            "Kubeconfig context to use. Defaults to the current context; the "
            "context must point at the volume's cluster."
        ),
    ),
    revision: str = typer.Option(
        "main", "--revision", help="HF revision (branch, tag, or SHA) to stage."
    ),
    env_file: list[str] = typer.Option(
        [], "--env-file", help="Env file that may contain HF_TOKEN (other entries ignored)."
    ),
    env_var: list[str] = typer.Option(
        [], "--env-var", "-e", help="Set HF_TOKEN (e.g. -e HF_TOKEN=hf_...)."
    ),
    timeout_seconds: int = typer.Option(
        3600, "--timeout-seconds", help="Overall staging deadline in seconds."
    ),
    output: str = typer.Option(
        "table", "--output", "-o", help="Output format: table or json (json: progress on stderr)."
    ),
) -> None:
    """Stage an HF dataset repository onto a volume for hosted SFT.

    Downloads the dataset snapshot inside a short-lived CPU pod on the
    volume's cluster, verifies it offline in a fresh process, then
    publishes it atomically under datasets/<name>. Requires kubectl and
    an authorized kubeconfig context for the volume's namespace (pods,
    pod logs, PVC read; plus Secrets for private datasets).

    Recipe:

    \b
        prime volumes create research
        prime volumes stage PrimeIntellect/INTELLECT-3-SFT-10K --volume research
        # then in your SFT TOML:  [data]
        #     type = "sft"
        #     name = "/datasets/intellect-3-sft-10k"
        #     splits = ["math"]
        prime train sft.toml --volume research

    Staged datasets are immutable: re-staging the same revision is
    idempotent ('already staged'); changed upstream content needs a new
    --path. Local laptop uploads are deferred in v1.
    """
    validate_output_format(output, console)
    client, team_id = _client()
    try:
        token = resolve_hf_token(list(env_var), list(env_file))
    except EnvParseError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    except StageError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    json_mode = output == "json"
    try:
        result = stage_dataset(
            client=client,
            team_id=team_id,
            source=source,
            volume_name=volume,
            path=path,
            namespace=namespace,
            kube_context=kube_context,
            revision=revision,
            token=token,
            timeout_seconds=timeout_seconds,
            json_mode=json_mode,
        )
    except StageError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    except KeyboardInterrupt:
        raise typer.Exit(130)

    print_stage_output(result, json_mode)


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


def _write_ssh_config(session, alias: str, host: str, user: str, port: str, key: str) -> Path:
    """Write the session's options ONCE into ~/.prime/volume-ssh/config under a
    short Host alias, so ssh, scp, sftp and rsync only need `-F <file> <alias>`.

    The platform returns the session pod's sshd host key, so it is pinned in
    a CLI-owned known_hosts with StrictHostKeyChecking=yes (host-key checking
    is never disabled; with no host key, ssh falls back to the user's own
    ~/.ssh/known_hosts). IdentitiesOnly: offer only the configured key, so a
    loaded ssh-agent can't exhaust MaxAuthTries first. `-F` also keeps the
    user's ~/.ssh/config (e.g. ControlMaster) out of these connections.

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
# Consecutive failed status polls tolerated before giving up (5s apart).
_MAX_POLL_ERRORS = 6


def _wait_for_connection(client, name: str, session, team_id):
    """Poll until the session publishes its SSH endpoint. Raises typer.Exit
    on a dead session, a timeout, or a status API that keeps failing; a
    single failed poll (network blip, 5xx) is retried.

    Match `prime pods ssh`: poll, then invoke local ssh. The platform fails
    a session deploy at 5m plus a 2m helm buffer (7m, measured 7m12s); 8
    minutes leaves margin so a failed deploy surfaces as FAILED, not a
    timeout.
    """
    errors = 0
    with console.status("Waiting for SSH connection to become available...", spinner="dots"):
        deadline = time.monotonic() + 480
        while not session.ssh_connection and time.monotonic() < deadline:
            if session.status in _DEAD_SESSION_STATES:
                detail = f": {session.error_message}" if session.error_message else "."
                console.print(f"[red]Session is {session.status}{escape(detail)}[/red]")
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


def _open_session(name: str, read_only: bool):
    """Create or reuse a session, wait for its endpoint and write the ssh
    config block. Returns (session, alias, key, config)."""
    key = Config().ssh_key_path
    if not key or not os.path.isfile(os.path.expanduser(key)):
        console.print("[red]SSH key not found; use prime config set-ssh-key-path.[/red]")
        raise typer.Exit(1)
    key = os.path.expanduser(key)
    client, team_id = _client()
    try:
        session = client.create_volume_session(name, read_only=read_only, team_id=team_id)
    except APIError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc
    mode = "read-only" if session.read_only else "read-write"
    console.print(
        f"Session {session.id} ({mode}). Stop with: prime volumes stop {escape(name)} {session.id}"
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
        if not connected:
            _stop_quietly(client, name, session.id, team_id)
    match = _CONNECTION.fullmatch(session.ssh_connection)
    if not match or not 1 <= int(match.group("port") or 22) <= 65535:
        console.print("[red]Invalid SSH endpoint returned by server.[/red]")
        raise typer.Exit(1)
    host = match.group("host")
    alias = host.split(".", 1)[0]
    config = _write_ssh_config(
        session, alias, host, match.group("user"), match.group("port") or "22", key
    )
    return session, alias, key, config


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
    session, alias, key, config = _open_session(name, read_only=not read_write)
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
    if code:
        raise typer.Exit(code)


# Remote paths are passed to rsync/scp unquoted. Quoting isn't portable: GNU
# rsync >= 3.2.4 protects remote args itself (a quoted path would keep its
# quotes), openrsync and older rsync don't. So only characters that need no
# quoting on any remote shell are allowed.
_SAFE_REMOTE_SEGMENT = re.compile(r"[A-Za-z0-9._@%+=,:-]+")


def _remote_path(path: str) -> str:
    """Path under the volume root (/volume on the pod); a leading "/" means the
    root. A trailing "/" is kept. Rejects empty and ".." segments, and any
    character that would need shell quoting (spaces, *, $, quotes, ...)."""
    rel = path[1:] if path.startswith("/") else path
    parts = rel.removesuffix("/").split("/") if rel else []
    if any(p in ("", "..") for p in parts):
        console.print(
            f"[red]Invalid remote path {escape(repr(path))}: no '..' or empty segments.[/red]"
        )
        raise typer.Exit(2)
    if not all(_SAFE_REMOTE_SEGMENT.fullmatch(p) for p in parts):
        console.print(
            f"[red]Invalid remote path {escape(repr(path))}: use letters, digits and "
            "._-@%+=,: only (no spaces or shell characters). For other names, use "
            "`prime volumes ssh`.[/red]"
        )
        raise typer.Exit(2)
    return "/volume/" + "/".join(parts) + ("/" if parts and rel.endswith("/") else "")


def _transfer_failed(alias: str, code: int) -> None:
    """The tailnet hint a failed transfer (or symlink check) reports."""
    console.print(
        "[red]Transfer failed.[/red] Check that you are on the tailnet "
        f"(host {alias} must resolve) and the path exists."
    )
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


def _remote_tree_has_symlink(alias: str, config: Path, remote: str) -> bool:
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
        _transfer_failed(alias, check.returncode)
    return bool(check.stdout.strip())


def _transfer(name: str, read_only: bool, remote: str, local: str, upload: bool) -> None:
    remote = _remote_path(remote)
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
    session, alias, _key, config = _open_session(name, read_only=read_only)
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
            linked = _remote_tree_has_symlink(alias, config, remote)
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
    cmd += [local, remote_arg] if upload else [remote_arg, local]
    try:
        code = subprocess.run(cmd, check=False).returncode
    except OSError as exc:
        console.print(f"[red]Could not start transfer:[/red] {exc}")
        raise typer.Exit(1) from exc
    if code:
        _transfer_failed(alias, code)


@app.command(no_args_is_help=True)
def get(
    name: str = typer.Argument(..., help="Volume name"),
    remote_path: str = typer.Argument(..., help="Path in the volume; / is the volume root"),
    local_dest: str = typer.Argument(".", help="Local destination"),
) -> None:
    """Download from a volume over a read-only session (rsync, else scp)."""
    _transfer(name, True, remote_path, local_dest, upload=False)


@app.command(no_args_is_help=True)
def put(
    name: str = typer.Argument(..., help="Volume name"),
    local_path: str = typer.Argument(..., help="Local file or directory"),
    remote_path: str = typer.Argument("/", help="Path in the volume; trailing / = into directory"),
) -> None:
    """Upload to a volume over a read-write session (rsync, else scp)."""
    _transfer(name, False, remote_path, local_path, upload=True)


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
