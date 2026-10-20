"""Sandbox volumes: persistent storage attached to VM sandboxes."""

import random
import shlex
import shutil
import string
import subprocess
from pathlib import Path
from typing import List, Optional

import typer
from prime_sandboxes import (
    APIClient,
    APIError,
    Config,
    CreateSandboxRequest,
    PaymentRequiredError,
    Sandbox,
    SandboxClient,
    SandboxNotRunningError,
    UnauthorizedError,
    Volume,
    VolumeMount,
)
from rich.markup import escape

from ..utils import (
    PlainTyper,
    build_table,
    confirm_or_skip,
    get_console,
    human_age,
    json_output_help,
    output_data_as_json,
)
from .sandbox_ssh import SandboxSshTransport, ssh_transport

app = PlainTyper(help="Manage sandbox volumes", no_args_is_help=True)
console = get_console()


VOLUMES_LIST_JSON_HELP = json_output_help(
    ".volumes[] = {id, name, user_id?, team_id?, created_at}",
    ".total = number",
    ".page = number",
    ".per_page = number",
    ".has_next = boolean",
)

# Public platform images (org-less, published via `prime images push
# --platform-image`); per-channel publication is a separate A7 gate.
_IMPORTER_IMAGE_DEV = "volume-import:dev"
_IMPORTER_IMAGE_PROD = "volume-import:prod"
_IMPORT_MOUNT_PATH = "/volume"
_IMPORTER_CPU_CORES = 1.0
_IMPORTER_MEMORY_GB = 1.0
_IMPORTER_DISK_SIZE_GB = 5.0
# No lifetime timeout: an import may stream for hours; the idle timeout below
# still reaps a sandbox nobody transfers through.
_IMPORTER_TIMEOUT_MINUTES = -1
_IMPORTER_IDLE_TIMEOUT_MINUTES = 30
_IMPORTER_LABEL = "prime:volume-import"


def _invalid_volume_mount(spec: str, problem: str) -> None:
    console.print(f"[red]Error:[/red] Invalid volume '{spec}': {problem}")
    raise typer.Exit(1)


def parse_volume_mount_specs(specs: List[str]) -> List[VolumeMount]:
    """Parse repeatable ``NAME:/mount/path[:ro]`` values into volume mounts.

    Only the syntax is split here; name, mount-path, and count rules are
    enforced once by the API.
    """
    mounts: List[VolumeMount] = []
    for spec in specs:
        name, sep, remainder = spec.partition(":")
        if not sep or not name or not remainder:
            _invalid_volume_mount(spec, "expected NAME:/mount/path[:ro]")
        mount_path, read_only = remainder, False
        if mount_path.endswith(":ro"):
            mount_path, read_only = mount_path[:-3], True
        if not mount_path:
            _invalid_volume_mount(spec, "empty mount path after ':ro'")
        mounts.append(VolumeMount(name=name, mount_path=mount_path, read_only=read_only))
    return mounts


def _volume_scope(volume: Volume) -> str:
    if volume.team_id:
        return f"team: {volume.team_id}"
    return "Personal"


def _print_volume(volume: Volume) -> None:
    console.print(f"Name: [cyan]{escape(volume.name)}[/cyan]")
    console.print(f"ID: {escape(volume.id)}")
    console.print(f"Scope: {_volume_scope(volume)}")
    # None until the first publish reaches DURABLE, then the newest one.
    head = volume.head_checkpoint_id
    console.print(f"Head checkpoint: {escape(head) if head else 'none (never published)'}")
    console.print(f"Created: {human_age(volume.created_at)}")


def _sandbox_client() -> SandboxClient:
    """One client per command; its volumes namespace rides the same transport."""
    return SandboxClient(APIClient())


def _importer_image() -> str:
    """Resolve the volume-import helper image for the API environment in use.

    The image is platform-owned and public; whether a channel's image has
    actually been published is enforced by the API at sandbox-create time,
    not here.
    """
    if Config().base_url == Config.DEFAULT_BASE_URL:
        return _IMPORTER_IMAGE_PROD
    return _IMPORTER_IMAGE_DEV


def _rsync_command(transport: SandboxSshTransport, path: Path) -> List[str]:
    remote_shell = shlex.join(
        [
            "ssh",
            "-p",
            str(transport.port),
            "-i",
            transport.key_path,
            "-o",
            f"ProxyCommand={transport.proxy_command}",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
        ]
    )
    return [
        "rsync",
        "-az",
        "--partial",
        "-e",
        remote_shell,
        f"{path}/",
        f"{transport.user}@{transport.host}:{_IMPORT_MOUNT_PATH}/",
    ]


def _print_retained_importer(
    sandbox_id: str, name: str, checkpoint_id: Optional[str], team_id: Optional[str]
) -> None:
    """Recovery guidance for an importer kept after an unsettled publish."""
    scope = f" --team-id {shlex.quote(team_id)}" if team_id else ""
    console.print(
        f"[yellow]Importer sandbox {escape(sandbox_id)} kept: the publish did "
        "not reach DURABLE in this run.[/yellow]"
    )
    if checkpoint_id:
        console.print(
            f"Checkpoint {escape(checkpoint_id)}: check its state with "
            f"'prime sandbox checkpoint get {shlex.quote(checkpoint_id)}'."
        )
    else:
        # A volume checkpoint never appears in the sandbox's checkpoint list
        # (rootfs-only filter), so discovery runs through the volume head.
        console.print(
            "If a checkpoint was created, its id is not known to this run; "
            "once it reaches DURABLE it becomes the volume's head, shown by "
            f"'prime sandbox volumes get {shlex.quote(name)}{scope}'. That head may be "
            "older or from another publisher and does not by itself "
            "confirm this publish. Check any checkpoint id with "
            "'prime sandbox checkpoint get <checkpoint-id>'."
        )
    console.print(
        "The imported data is private to the importer until a publish "
        "reaches DURABLE."
    )
    # The importer is not kept indefinitely: the idle monitor reaps it after
    # _IMPORTER_IDLE_TIMEOUT_MINUTES without transfer activity, and the
    # publish itself does not reset that clock.
    console.print(
        "The sandbox is reaped automatically after "
        f"{_IMPORTER_IDLE_TIMEOUT_MINUTES} minutes without transfer activity. "
        "Delete it manually with "
        f"'prime sandbox delete {shlex.quote(sandbox_id)}' once you no longer need it."
    )


def _import_volume(name: str, team_id: Optional[str], from_path: str) -> None:
    # Absolute before rsync: a relative source would be ambiguous in the rsync
    # argv (rsync reads bare words as remote paths, not only the dest).
    path = Path(from_path).expanduser().resolve()
    if not path.is_dir():
        console.print(f"[red]Error:[/red] --from path is not a local directory: {from_path}")
        raise typer.Exit(1)
    if not shutil.which("rsync"):
        console.print("[red]Error:[/red] rsync is required to import a volume.")
        raise typer.Exit(1)

    image = _importer_image()

    # One client for the whole flow: its volumes namespace shares this
    # transport, and publish waits go through its checkpoint waiter.
    sandbox_client = SandboxClient(APIClient())
    # The retention hint needs the scope the API resolves team_id to; read
    # it from the client's own config, the same source the requests use.
    effective_team_id = (
        team_id if team_id is not None else sandbox_client.client.config.team_id
    )

    try:
        sandbox_client.volumes.create(name, team_id=team_id)
    except APIError as error:
        # Prefix match on the SDK's exact format ("HTTP <status>: <body>"):
        # a detail string that merely mentions "HTTP 409" is not a conflict.
        if not str(error).startswith("HTTP 409:"):
            raise
        # 409 from create is C3's name-taken: the volume already exists in
        # this scope. The importer attaches at the volume's current durable
        # head, so this is a fresh copy, not a resume of a failed transfer;
        # auth, not-found, and server errors are not reuse signals.
        console.print(
            f"Volume {escape(name)} already exists in this scope; copying the "
            "local directory into its current published contents"
        )
    else:
        console.print(f"[green]✓[/green] Created volume {escape(name)}")

    console.print("[blue]Importing files into volume...[/blue]")

    sandbox: Optional[Sandbox] = None
    checkpoint_id: Optional[str] = None
    # The importer is deleted only on a pre-publish failure or after a
    # confirmed DURABLE publish; every unsettled publish keeps it.
    publish_requested = False
    durable = False
    try:
        suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=4))
        sandbox = sandbox_client.create(
            CreateSandboxRequest(
                name=f"volume-import-{name}-{suffix}",
                docker_image=image,
                cpu_cores=_IMPORTER_CPU_CORES,
                memory_gb=_IMPORTER_MEMORY_GB,
                disk_size_gb=_IMPORTER_DISK_SIZE_GB,
                timeout_minutes=_IMPORTER_TIMEOUT_MINUTES,
                idle_timeout_minutes=_IMPORTER_IDLE_TIMEOUT_MINUTES,
                labels=[_IMPORTER_LABEL],
                team_id=team_id,
                volumes=[VolumeMount(name=name, mount_path=_IMPORT_MOUNT_PATH, read_only=False)],
            )
        )
        sandbox_client.wait_for_creation(sandbox.id)
        with ssh_transport(sandbox_client, sandbox.id) as transport:
            # rsync progress is success-path noise; its errors go to stderr,
            # which stays attached to the terminal.
            result = subprocess.run(_rsync_command(transport, path), stdout=subprocess.DEVNULL)
        if result.returncode != 0:
            console.print(
                "[red]Error:[/red] rsync failed. Nothing was published by this "
                "attempt; the volume keeps its previously published contents "
                "(a never-published volume stays empty)."
            )
            raise typer.Exit(1)
        # From here the publish outcome is uncertain on any failure or
        # interrupt; the handlers below and the retention guidance in the
        # finally carry the full IDs/recovery output, not the success path.
        console.print("[blue]Publishing volume...[/blue]")
        publish_requested = True
        checkpoint = sandbox_client.volumes.publish(sandbox.id, name, wait=False)
        checkpoint_id = checkpoint.id
        try:
            sandbox_client.wait_for_checkpoint(checkpoint_id)
        except TimeoutError as error:
            # The wait budget expiring is not a publish failure; the upload
            # may still complete after the CLI exits.
            console.print(
                f"[red]Error:[/red] {escape(str(error))} The publish may still "
                "complete."
            )
            raise typer.Exit(1)
        except RuntimeError as error:
            console.print(f"[red]Error:[/red] {escape(str(error))} The publish failed.")
            raise typer.Exit(1)
        durable = True
        console.print(
            f"[green]✓[/green] Published volume {escape(name)}: checkpoint "
            f"{escape(checkpoint_id)} is DURABLE"
        )
    except KeyboardInterrupt:
        if publish_requested:
            console.print(
                "\n[yellow]Interrupted while publishing:[/yellow] the publish "
                "state is unknown."
            )
        else:
            console.print(
                "\n[yellow]Interrupted before publishing:[/yellow] nothing was "
                "published by this attempt; the volume keeps its previously "
                "published contents."
            )
        raise typer.Exit(130)
    finally:
        # No return here: a pre-create failure (sandbox is None) must keep
        # propagating its exception; there is nothing to clean up yet.
        if sandbox is not None:
            if durable or not publish_requested:
                try:
                    sandbox_client.delete(sandbox.id)
                # Broad on purpose: this finally must never mask the in-flight
                # exception (e.g. the rsync failure) with a cleanup failure.
                except Exception as error:
                    warning = (
                        f"[yellow]Warning:[/yellow] failed to delete importer sandbox "
                        f"{sandbox.id}: {escape(str(error))} "
                        f"Remove it manually with 'prime sandbox delete {sandbox.id}'."
                    )
                    if not durable:
                        warning += " It may still hold the unpublished import data."
                    console.print(warning)
            else:
                _print_retained_importer(sandbox.id, name, checkpoint_id, effective_team_id)


@app.command("create")
def create(
    name: str = typer.Argument(..., help="Volume name, unique within its scope"),
    team_id: Optional[str] = typer.Option(
        None, help="Team ID (uses config team_id if not specified)"
    ),
    from_path: Optional[str] = typer.Option(
        None, "--from", help="Import this local directory into the new volume"
    ),
) -> None:
    """Create a volume, optionally importing a local directory."""
    try:
        if from_path is None:
            volume = _sandbox_client().volumes.create(name, team_id=team_id)
            console.print(f"[green]✓[/green] Created volume {escape(volume.name)}")
            _print_volume(volume)
            return
        _import_volume(name, team_id, from_path)
    except typer.Exit:
        raise
    except KeyboardInterrupt:
        # Reached from the plain create or from the import flow's
        # volume-create POST, which runs before its own phase handlers;
        # either way the request may or may not have completed.
        console.print(
            "\n[yellow]Interrupted.[/yellow] The create request may or may not "
            "have completed; check with 'prime sandbox volumes list'."
        )
        raise typer.Exit(130)
    except SandboxNotRunningError as e:
        console.print(f"[red]Error:[/red] {escape(str(e))}")
        raise typer.Exit(1)
    except UnauthorizedError as e:
        console.print(f"[red]Unauthorized:[/red] {escape(str(e))}")
        raise typer.Exit(1)
    except PaymentRequiredError as e:
        console.print(f"[red]Payment Required:[/red] {escape(str(e))}")
        raise typer.Exit(1)
    except APIError as e:
        console.print(f"[red]Error:[/red] {escape(str(e))}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {escape(str(e))}")
        console.print_exception(show_locals=False)
        raise typer.Exit(1)


@app.command("list", epilog=VOLUMES_LIST_JSON_HELP)
def list_volumes(
    team_id: Optional[str] = typer.Option(
        None, help="Team ID (uses config team_id if not specified)"
    ),
    page: int = typer.Option(1, "--page", "-p", help="Page number"),
    num: int = typer.Option(50, "--num", "-n", help="Items per page"),
    json_output: bool = typer.Option(
        False, "--json", help="Output volumes as JSON instead of a table"
    ),
) -> None:
    """List volumes."""
    try:
        response = _sandbox_client().volumes.list(team_id=team_id, page=page, per_page=num)
    except UnauthorizedError as e:
        console.print(f"[red]Unauthorized:[/red] {escape(str(e))}")
        raise typer.Exit(1)
    except PaymentRequiredError as e:
        console.print(f"[red]Payment Required:[/red] {escape(str(e))}")
        raise typer.Exit(1)
    except APIError as e:
        console.print(f"[red]Error:[/red] {escape(str(e))}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {escape(str(e))}")
        console.print_exception(show_locals=False)
        raise typer.Exit(1)

    if json_output:
        output_data_as_json(response.model_dump(mode="json"), console)
        return
    if not response.volumes:
        console.print("No volumes found.")
        return
    table = build_table(
        "Volumes",
        [("Name", "cyan"), ("Scope", "white"), ("Created", "white"), ("ID", "dim")],
    )
    for volume in response.volumes:
        table.add_row(
            escape(volume.name),
            _volume_scope(volume),
            human_age(volume.created_at),
            escape(volume.id),
        )
    console.print(table)


@app.command(no_args_is_help=True)
def get(
    name: str = typer.Argument(..., help="Volume name"),
    team_id: Optional[str] = typer.Option(
        None, help="Team ID (uses config team_id if not specified)"
    ),
) -> None:
    """Show a volume's details."""
    try:
        volume = _sandbox_client().volumes.get(name, team_id=team_id)
    except UnauthorizedError as e:
        console.print(f"[red]Unauthorized:[/red] {escape(str(e))}")
        raise typer.Exit(1)
    except PaymentRequiredError as e:
        console.print(f"[red]Payment Required:[/red] {escape(str(e))}")
        raise typer.Exit(1)
    except APIError as e:
        console.print(f"[red]Error:[/red] {escape(str(e))}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {escape(str(e))}")
        console.print_exception(show_locals=False)
        raise typer.Exit(1)
    _print_volume(volume)


@app.command()
def delete(
    name: str = typer.Argument(..., help="Volume name"),
    team_id: Optional[str] = typer.Option(
        None, help="Team ID (uses config team_id if not specified)"
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
) -> None:
    """Delete a volume. Fails while any active sandbox has it attached."""
    scope = f" (team: {team_id})" if team_id else ""
    if not confirm_or_skip(f"Are you sure you want to delete volume '{name}'{scope}?", yes):
        console.print("[yellow]Cancelled[/yellow]")
        return
    try:
        _sandbox_client().volumes.delete(name, team_id=team_id)
    except UnauthorizedError as e:
        console.print(f"[red]Unauthorized:[/red] {escape(str(e))}")
        raise typer.Exit(1)
    except PaymentRequiredError as e:
        console.print(f"[red]Payment Required:[/red] {escape(str(e))}")
        raise typer.Exit(1)
    except APIError as e:
        console.print(f"[red]Error:[/red] {escape(str(e))}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {escape(str(e))}")
        console.print_exception(show_locals=False)
        raise typer.Exit(1)
    console.print(f"[green]✓[/green] Deleted volume {escape(name)}")


@app.command()
def publish(
    name: str = typer.Argument(..., help="Volume name to publish"),
    sandbox_id: str = typer.Option(
        ..., "--sandbox", help="ID of the sandbox that publishes the volume"
    ),
    force: bool = typer.Option(
        False, "--force", help="Publish even if the volume's head moved since attach"
    ),
    no_wait: bool = typer.Option(
        False, "--no-wait", help="Do not wait for the publish to become durable"
    ),
) -> None:
    """Publish a sandbox's private volume writes as a durable checkpoint."""
    try:
        checkpoint = _sandbox_client().volumes.publish(
            sandbox_id, name, force=force, wait=not no_wait
        )
    except UnauthorizedError as e:
        console.print(f"[red]Unauthorized:[/red] {escape(str(e))}")
        raise typer.Exit(1)
    except PaymentRequiredError as e:
        console.print(f"[red]Payment Required:[/red] {escape(str(e))}")
        raise typer.Exit(1)
    except APIError as e:
        # 400 read-only, 404 not attached, 409 stale head or another
        # sandbox's publish in flight (also with --force).
        console.print(f"[red]Error:[/red] {escape(str(e))}")
        raise typer.Exit(1)
    except TimeoutError as e:
        # The SDK wait budget expired; the publish upload may still finish.
        console.print(f"[red]Error:[/red] {escape(str(e))}")
        console.print(
            "The publish may still complete. Check it with "
            "'prime sandbox checkpoint get <checkpoint-id>' (the id is in "
            "the message above)."
        )
        raise typer.Exit(1)
    except RuntimeError as e:
        console.print(f"[red]Error:[/red] {escape(str(e))}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {escape(str(e))}")
        console.print_exception(show_locals=False)
        raise typer.Exit(1)

    console.print(f"Checkpoint ID: [cyan]{escape(checkpoint.id)}[/cyan]")
    console.print(f"State: {escape(checkpoint.state)}")
    if checkpoint.state == "DURABLE":
        console.print(
            f"[green]✓[/green] Published volume {escape(name)}; new sandboxes "
            "that attach it see this data"
        )
    else:
        console.print(
            "[yellow]The publish is not yet DURABLE; the volume's published "
            "contents do not include this data until it is. Check with "
            f"'prime sandbox checkpoint get {escape(checkpoint.id)}'."
        )
