import json
import random
import shlex
import string
import time
from typing import Any, Dict, List, Optional

import typer
from prime_sandboxes import (
    APIClient,
    APIError,
    BulkDeleteSandboxResponse,
    CommandTimeoutError,
    Config,
    CreateSandboxRequest,
    PaymentRequiredError,
    Sandbox,
    SandboxClient,
    SandboxNotRunningError,
    UnauthorizedError,
)
from rich.console import Console
from rich.markup import escape
from rich.table import Table
from rich.text import Text

from ..utils import (
    build_table,
    confirm_or_skip,
    format_resources,
    human_age,
    iso_timestamp,
    obfuscate_env_vars,
    obfuscate_secrets,
    output_data_as_json,
    sort_by_created,
    status_color,
    validate_output_format,
)
from ..utils.display import SANDBOX_STATUS_COLORS

app = typer.Typer(help="Manage code sandboxes", no_args_is_help=True)
console = Console()


config = Config()

BULK_DELETE_BATCH_SIZE = 100


def _bulk_delete_sandboxes(
    sandbox_client: SandboxClient,
    sandbox_ids: List[str],
    show_progress: bool = True,
) -> tuple[List[str], List[Dict[str, Any]]]:
    """Delete sandboxes in batches and return results."""
    all_succeeded: List[str] = []
    all_failed: List[Dict[str, Any]] = []

    for i in range(0, len(sandbox_ids), BULK_DELETE_BATCH_SIZE):
        batch = sandbox_ids[i : i + BULK_DELETE_BATCH_SIZE]

        if show_progress and len(sandbox_ids) > BULK_DELETE_BATCH_SIZE:
            batch_num = (i // BULK_DELETE_BATCH_SIZE) + 1
            total_batches = (
                len(sandbox_ids) + BULK_DELETE_BATCH_SIZE - 1
            ) // BULK_DELETE_BATCH_SIZE
            console.print(
                f"[dim]Processing batch {batch_num}/{total_batches} "
                f"({len(batch)} sandboxes)...[/dim]"
            )

        result: BulkDeleteSandboxResponse = sandbox_client.bulk_delete(sandbox_ids=batch)

        if result.succeeded:
            all_succeeded.extend(result.succeeded)
        if result.failed:
            all_failed.extend(result.failed)

    return all_succeeded, all_failed


def _print_bulk_delete_results(
    succeeded: List[str],
    failed: List[Dict[str, Any]],
    success_message: str,
) -> None:
    """Print results of a bulk delete operation."""
    if succeeded:
        console.print(f"\n[bold green]{success_message}[/bold green]")
        for sandbox_id in succeeded:
            console.print(f"  ✓ {sandbox_id}")

    if failed:
        console.print(f"\n[bold red]Failed to delete {len(failed)} sandbox(es):[/bold red]")
        for failure in failed:
            sandbox_id = failure.get("sandbox_id", "unknown")
            error = failure.get("error", "unknown error")
            console.print(f"  ✗ {sandbox_id}: {error}")


def _format_sandbox_for_list(sandbox: Sandbox) -> Dict[str, Any]:
    """Format sandbox data for list display (both table and JSON)"""
    return {
        "id": sandbox.id,
        "name": sandbox.name,
        "image": sandbox.docker_image,
        "status": sandbox.status,
        "resources": format_resources(sandbox.cpu_cores, sandbox.memory_gb, sandbox.gpu_count),
        "labels": ", ".join(sandbox.labels) if sandbox.labels else "-",  # For table output
        "labels_list": sandbox.labels,  # For JSON output
        "created_at": iso_timestamp(sandbox.created_at),  # For JSON output
        "age": human_age(sandbox.created_at),  # For table output
    }


def _format_sandbox_for_details(sandbox: Sandbox) -> Dict[str, Any]:
    """Format sandbox data for details display (both table and JSON)"""
    data: Dict[str, Any] = {
        "id": sandbox.id,
        "name": sandbox.name,
        "docker_image": sandbox.docker_image,
        "start_command": sandbox.start_command,
        "status": sandbox.status,
        "cpu_cores": sandbox.cpu_cores,
        "memory_gb": sandbox.memory_gb,
        "disk_size_gb": sandbox.disk_size_gb,
        "disk_mount_path": sandbox.disk_mount_path,
        "gpu_count": sandbox.gpu_count,
        "network_access": sandbox.network_access,
        "timeout_minutes": sandbox.timeout_minutes,
        "labels": sandbox.labels,
        "created_at": iso_timestamp(sandbox.created_at),
        "user_id": sandbox.user_id,
        "team_id": sandbox.team_id,
        "registry_credentials_id": getattr(sandbox, "registry_credentials_id", None),
    }

    if sandbox.started_at:
        data["started_at"] = iso_timestamp(sandbox.started_at)
    if sandbox.terminated_at:
        data["terminated_at"] = iso_timestamp(sandbox.terminated_at)
    if sandbox.exit_code is not None:
        data["exit_code"] = sandbox.exit_code
    if sandbox.environment_vars:
        data["environment_vars"] = obfuscate_env_vars(sandbox.environment_vars)
    if sandbox.secrets:
        data["secrets"] = obfuscate_secrets(sandbox.secrets)
    if sandbox.advanced_configs:
        data["advanced_configs"] = sandbox.advanced_configs.model_dump()

    return data


@app.command("list")
@app.command("ls", hidden=True)
def list_sandboxes_cmd(
    team_id: Optional[str] = typer.Option(
        None, help="Filter by team ID (uses config team_id if not specified)"
    ),
    status: Optional[str] = typer.Option(None, help="Filter by status"),
    labels: Optional[List[str]] = typer.Option(
        None,
        "--label",
        "-l",
        help="Filter by labels (can specify multiple, sandboxes must have ALL)",
    ),
    page: int = typer.Option(1, help="Page number"),
    per_page: int = typer.Option(50, help="Items per page"),
    all: bool = typer.Option(False, "--all", help="Show all sandboxes including terminated ones"),
    output: str = typer.Option("table", "--output", "-o", help="Output format: table or json"),
) -> None:
    """List your sandboxes (shortcut: ls)"""
    validate_output_format(output, console)

    try:
        base_client = APIClient()
        sandbox_client = SandboxClient(base_client)

        # Always exclude terminated sandboxes unless --all is specified or status filter is used
        exclude_terminated = not all and status is None

        sandbox_list = sandbox_client.list(
            team_id=team_id,
            status=status,
            labels=labels,
            page=page,
            per_page=per_page,
            exclude_terminated=exclude_terminated,
        )

        table = build_table(
            f"Code Sandboxes (Total: {sandbox_list.total})",
            [
                ("ID", "cyan"),
                ("Name", "blue"),
                ("Image", "green"),
                ("Status", "yellow"),
                ("Resources", "magenta"),
                ("Labels", "white"),
                ("Age", "blue"),
            ],
        )

        # Sort sandboxes by created_at (oldest first)
        sorted_sandboxes = sort_by_created(sandbox_list.sandboxes)

        if output == "json":
            # Output as JSON with timestamp (for automation)
            sandboxes_data = []
            for sandbox in sorted_sandboxes:
                sandbox_data = _format_sandbox_for_list(sandbox)
                # For JSON, use timestamp instead of age
                json_sandbox = {
                    "id": sandbox_data["id"],
                    "name": sandbox_data["name"],
                    "image": sandbox_data["image"],
                    "status": sandbox_data["status"],
                    "resources": sandbox_data["resources"],
                    "labels": sandbox_data["labels_list"],
                    "created_at": sandbox_data["created_at"],
                }
                sandboxes_data.append(json_sandbox)

            output_data = {
                "sandboxes": sandboxes_data,
                "total": sandbox_list.total,
                "page": sandbox_list.page,
                "per_page": sandbox_list.per_page,
                "has_next": sandbox_list.has_next,
            }
            output_data_as_json(output_data, console)
        else:
            # Output as table using shared formatting
            for sandbox in sorted_sandboxes:
                sandbox_data = _format_sandbox_for_list(sandbox)

                color = status_color(sandbox_data["status"], SANDBOX_STATUS_COLORS)

                table.add_row(
                    sandbox_data["id"],
                    sandbox_data["name"],
                    sandbox_data["image"],
                    Text(sandbox_data["status"], style=color),
                    sandbox_data["resources"],
                    sandbox_data["labels"],
                    sandbox_data["age"],
                )

            console.print(table)

            if sandbox_list.has_next:
                console.print(
                    f"\n[yellow]Showing page {page} of results. "
                    f"Use --page {page + 1} to see more.[/yellow]"
                )

    except typer.Exit:
        raise
    except UnauthorizedError as e:
        console.print(f"[red]Unauthorized:[/red] {str(e)}")
        raise typer.Exit(1)
    except PaymentRequiredError as e:
        console.print(f"[red]Payment Required:[/red] {str(e)}")
        raise typer.Exit(1)
    except APIError as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {escape(str(e))}")
        console.print_exception(show_locals=True)
        raise typer.Exit(1)


@app.command(no_args_is_help=True)
def get(
    sandbox_id: str,
    output: str = typer.Option("table", "--output", "-o", help="Output format: table or json"),
) -> None:
    """Get detailed information about a specific sandbox"""
    validate_output_format(output, console)

    try:
        base_client = APIClient()
        sandbox_client = SandboxClient(base_client)

        sandbox = sandbox_client.get(sandbox_id)

        if output == "json":
            # Output as JSON using shared formatting
            sandbox_data = _format_sandbox_for_details(sandbox)
            output_data_as_json(sandbox_data, console)
        else:
            # Output as table using shared formatting
            sandbox_data = _format_sandbox_for_details(sandbox)

            table = Table(title=f"Sandbox Details: {sandbox_id}")
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="white")

            table.add_row("ID", sandbox_data["id"])
            table.add_row("Name", sandbox_data["name"])
            table.add_row("Docker Image", sandbox_data["docker_image"])
            table.add_row("Start Command", sandbox_data["start_command"] or "N/A")

            sandbox_status_color = status_color(sandbox_data["status"], SANDBOX_STATUS_COLORS)
            table.add_row("Status", Text(sandbox_data["status"], style=sandbox_status_color))

            table.add_row("CPU Cores", str(sandbox_data["cpu_cores"]))
            table.add_row("Memory (GB)", str(sandbox_data["memory_gb"]))
            table.add_row("Disk Size (GB)", str(sandbox_data["disk_size_gb"]))
            table.add_row("Disk Mount Path", sandbox_data["disk_mount_path"])
            table.add_row("GPU Count", str(sandbox_data["gpu_count"]))
            network_display = Text(
                "Enabled" if sandbox_data["network_access"] else "Disabled",
                style="green" if sandbox_data["network_access"] else "yellow",
            )
            table.add_row("Network Access", network_display)
            table.add_row("Timeout (minutes)", str(sandbox_data["timeout_minutes"]))

            # Show labels
            labels_display = ", ".join(sandbox_data["labels"]) if sandbox_data["labels"] else "None"
            table.add_row("Labels", labels_display)

            table.add_row("Created", sandbox_data["created_at"])
            if "started_at" in sandbox_data:
                table.add_row("Started", sandbox_data["started_at"])
            if "terminated_at" in sandbox_data:
                table.add_row("Terminated", sandbox_data["terminated_at"])
            if "exit_code" in sandbox_data:
                table.add_row("Exit Code", str(sandbox_data["exit_code"]))

            table.add_row("User ID", sandbox_data["user_id"] or "N/A")
            table.add_row("Team ID", sandbox_data["team_id"] or "Personal")
            if sandbox_data.get("registry_credentials_id"):
                table.add_row(
                    "Registry Credentials",
                    sandbox_data["registry_credentials_id"],
                )

            if "environment_vars" in sandbox_data:
                env_vars = json.dumps(sandbox_data["environment_vars"], indent=2)
                table.add_row("Environment Variables", env_vars)

            if "secrets" in sandbox_data:
                secrets = json.dumps(sandbox_data["secrets"], indent=2)
                table.add_row("Secrets", secrets)

            if "advanced_configs" in sandbox_data:
                advanced_configs = json.dumps(sandbox_data["advanced_configs"], indent=2)
                table.add_row("Advanced Configs", advanced_configs)

            console.print(table)

    except typer.Exit:
        raise
    except UnauthorizedError as e:
        console.print(f"[red]Unauthorized:[/red] {str(e)}")
        raise typer.Exit(1)
    except PaymentRequiredError as e:
        console.print(f"[red]Payment Required:[/red] {str(e)}")
        raise typer.Exit(1)
    except APIError as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {escape(str(e))}")
        console.print_exception(show_locals=True)
        raise typer.Exit(1)


@app.command(no_args_is_help=True)
def create(
    docker_image: str = typer.Argument(..., help="Docker image to run"),
    name: Optional[str] = typer.Option(
        None, help="Name for the sandbox (auto-generated if not provided)"
    ),
    start_command: Optional[str] = typer.Option(
        "tail -f /dev/null", help="Command to run in the container"
    ),
    cpu_cores: int = typer.Option(1, help="Number of CPU cores"),
    memory_gb: int = typer.Option(2, help="Memory in GB"),
    disk_size_gb: int = typer.Option(10, help="Disk size in GB"),
    gpu_count: int = typer.Option(0, help="Number of GPUs"),
    network_access: bool = typer.Option(
        True,
        "--network-access/--no-network-access",
        help="Allow outbound internet access (enabled by default)",
    ),
    timeout_minutes: int = typer.Option(60, help="Timeout in minutes"),
    team_id: Optional[str] = typer.Option(
        None, help="Team ID (uses config team_id if not specified)"
    ),
    registry_credentials_id: Optional[str] = typer.Option(
        None,
        "--registry-credentials-id",
        help="Registry credentials ID for pulling private images",
    ),
    env: Optional[List[str]] = typer.Option(
        None,
        help="Environment variables in KEY=VALUE format. Can be specified multiple times.",
    ),
    secret: Optional[List[str]] = typer.Option(
        None,
        help="Secrets in KEY=VALUE format. Can be specified multiple times.",
    ),
    labels: Optional[List[str]] = typer.Option(
        None,
        "--label",
        "-l",
        help="Labels/tags for the sandbox. Can be specified multiple times.",
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
) -> None:
    """Create a new sandbox"""
    try:
        base_client = APIClient()
        sandbox_client = SandboxClient(base_client)

        # Parse environment variables
        env_vars = {}
        if env:
            for env_var in env:
                if "=" not in env_var:
                    console.print("[red]Environment variables must be in KEY=VALUE format[/red]")
                    raise typer.Exit(1)
                key, value = env_var.split("=", 1)
                env_vars[key] = value

        secrets_vars = {}
        if secret:
            for secret_var in secret:
                if "=" not in secret_var:
                    console.print("[red]Secrets must be in KEY=VALUE format[/red]")
                    raise typer.Exit(1)
                key, value = secret_var.split("=", 1)
                secrets_vars[key] = value

        # Auto-generate name if not provided
        if not name:
            # Extract image name without tag/registry
            image_parts = docker_image.split("/")[-1].split(":")[0]
            # Replace underscores and dots with dashes, keep only alphanumeric and dashes
            clean_image = "".join(
                c if c.isalnum() or c == "-" else "-" for c in image_parts.lower()
            )
            # Remove multiple consecutive dashes and trim
            clean_image = "-".join(filter(None, clean_image.split("-")))

            suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=4))
            name = f"{clean_image}-{suffix}"

        request = CreateSandboxRequest(
            name=name,
            docker_image=docker_image,
            start_command=start_command,
            cpu_cores=cpu_cores,
            memory_gb=memory_gb,
            disk_size_gb=disk_size_gb,
            gpu_count=gpu_count,
            network_access=network_access,
            timeout_minutes=timeout_minutes,
            environment_vars=env_vars if env_vars else None,
            secrets=secrets_vars if secrets_vars else None,
            labels=labels if labels else [],
            team_id=team_id,
            registry_credentials_id=registry_credentials_id,
        )

        # Show configuration summary
        console.print("\n[bold]Sandbox Configuration:[/bold]")
        console.print(f"Name: {name}")
        console.print(f"Docker Image: {docker_image}")
        console.print(f"Start Command: {start_command or 'N/A'}")
        console.print(f"Resources: {cpu_cores} CPU, {memory_gb}GB RAM, {disk_size_gb}GB disk")
        if gpu_count > 0:
            console.print(f"GPUs: {gpu_count}")
        network_status = "[green]Enabled[/green]" if network_access else "[yellow]Disabled[/yellow]"
        console.print(f"Network Access: {network_status}")
        console.print(f"Timeout: {timeout_minutes} minutes")
        console.print(f"Team: {team_id or 'Personal'}")
        if registry_credentials_id:
            console.print(f"Registry Credentials: {registry_credentials_id}")
        if labels:
            console.print(f"Labels: {', '.join(labels)}")
        if env_vars:
            obfuscated_env = obfuscate_env_vars(env_vars)
            console.print(f"Environment Variables: {obfuscated_env}")
        if secrets_vars:
            obfuscated_secrets = obfuscate_secrets(secrets_vars)
            console.print(f"Secrets: {obfuscated_secrets}")

        if confirm_or_skip("\nDo you want to create this sandbox?", yes, default=True):
            with console.status("[bold blue]Creating sandbox...", spinner="dots"):
                sandbox = sandbox_client.create(request)

            console.print(f"\n[green]Successfully created sandbox {sandbox.id}[/green]")
            console.print(
                f"[blue]Use 'prime sandbox get {sandbox.id}' to check the sandbox status[/blue]"
            )
        else:
            console.print("\nSandbox creation cancelled")
            return

    except typer.Exit:
        raise
    except UnauthorizedError as e:
        console.print(f"[red]Unauthorized:[/red] {str(e)}")
        raise typer.Exit(1)
    except PaymentRequiredError as e:
        console.print(f"[red]Payment Required:[/red] {str(e)}")
        raise typer.Exit(1)
    except APIError as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {escape(str(e))}")
        console.print_exception(show_locals=True)
        raise typer.Exit(1)


@app.command(no_args_is_help=True)
def delete(
    sandbox_ids: Optional[List[str]] = typer.Argument(
        None, help="Sandbox ID(s) to delete (space or comma-separated)"
    ),
    all: bool = typer.Option(False, "--all", help="Delete all sandboxes"),
    labels: Optional[List[str]] = typer.Option(
        None, "--label", "-l", help="Delete all sandboxes with ALL these labels"
    ),
    name: Optional[str] = typer.Option(
        None, "--name", "-n", help="Delete all sandboxes with this name"
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
    only_mine: bool = typer.Option(
        True,
        "--only-mine/--all-users",
        help="Restrict to only your sandboxes when using --all, --name, or --label",
        show_default=True,
    ),
) -> None:
    """Delete one or more sandboxes by ID, by label, by name, or all sandboxes with --all

    --only-mine controls whether deletes will restrict to your sandboxes or delete for all users.
    This applies to --all, --name, and --label options.
    """
    try:
        base_client = APIClient()
        sandbox_client = SandboxClient(base_client)

        if sum([bool(all), bool(sandbox_ids), bool(labels), bool(name)]) > 1:
            console.print(
                "[red]Error:[/red] Cannot specify more than one of: "
                "sandbox IDs, --all, --label, or --name"
            )
            raise typer.Exit(1)

        if not all and not sandbox_ids and not labels and not name:
            console.print(
                "[red]Error:[/red] Must specify either sandbox IDs, --all flag, --label, or --name"
            )
            raise typer.Exit(1)

        if all:
            with console.status("[bold blue]Fetching all sandboxes...", spinner="dots"):
                all_sandboxes = []
                page = 1
                while True:
                    list_response = sandbox_client.list(
                        per_page=100, page=page, exclude_terminated=True
                    )
                    all_sandboxes.extend(list_response.sandboxes)
                    if not list_response.has_next:
                        break
                    page += 1

                if only_mine:
                    current_user_id = config.user_id
                    if not current_user_id:
                        console.print(
                            "[red]Error:[/red] Cannot filter by user - no user_id configured. "
                            "Use --all-users to delete all sandboxes, or configure your user_id."
                        )
                        raise typer.Exit(1)
                    sandboxes_to_delete = [s for s in all_sandboxes if s.user_id == current_user_id]
                else:
                    sandboxes_to_delete = all_sandboxes

                sandbox_ids = [s.id for s in sandboxes_to_delete]

                if not sandbox_ids:
                    console.print("[yellow]No sandboxes to delete[/yellow]")
                    if only_mine and all_sandboxes:
                        console.print(
                            "\n[dim]Note: --all only deletes your own sandboxes by default. "
                            "Use --all-users to delete sandboxes from all team members.[/dim]"
                        )
                    return
        elif name:
            with console.status("[bold blue]Fetching sandboxes...", spinner="dots"):
                all_sandboxes = []
                page = 1
                while True:
                    list_response = sandbox_client.list(
                        per_page=100, page=page, exclude_terminated=True
                    )
                    all_sandboxes.extend(list_response.sandboxes)
                    if not list_response.has_next:
                        break
                    page += 1

                # Filter by exact name match
                name_matched_sandboxes = [s for s in all_sandboxes if s.name == name]

                # Apply only_mine filter if set
                if only_mine:
                    current_user_id = config.user_id
                    if not current_user_id:
                        console.print(
                            "[red]Error:[/red] Cannot filter by user - no user_id configured. "
                            "Use --all-users to delete sandboxes from all users, "
                            "or configure your user_id."
                        )
                        raise typer.Exit(1)
                    matching_sandboxes = [
                        s for s in name_matched_sandboxes if s.user_id == current_user_id
                    ]
                else:
                    matching_sandboxes = name_matched_sandboxes

            if not matching_sandboxes:
                console.print(f"[yellow]No sandboxes found with name '{name}'[/yellow]")
                if only_mine and name_matched_sandboxes:
                    console.print(
                        "\n[dim]Note: --name only matches your own sandboxes by default. "
                        "Use --all-users to match sandboxes from all team members.[/dim]"
                    )
                return

            # Show warning table when multiple sandboxes match
            if len(matching_sandboxes) > 1:
                console.print(
                    f"\n[yellow]Warning:[/yellow] Multiple sandboxes found with name '{name}':\n"
                )
                table = build_table(
                    "",
                    [
                        ("ID", "cyan"),
                        ("Status", "yellow"),
                        ("Age", "blue"),
                    ],
                )
                for sandbox in sort_by_created(matching_sandboxes):
                    color = status_color(sandbox.status, SANDBOX_STATUS_COLORS)
                    table.add_row(
                        sandbox.id,
                        Text(sandbox.status, style=color),
                        human_age(sandbox.created_at),
                    )
                console.print(table)
                console.print(
                    "\n[dim]Tip: To delete a specific sandbox, use its ID: "
                    "prime sandbox delete <sandbox-id>[/dim]\n"
                )
                confirmation_msg = (
                    f"Are you sure you want to delete ALL {len(matching_sandboxes)} "
                    f"sandboxes named '{name}'? This action cannot be undone."
                )
            else:
                sandbox = matching_sandboxes[0]
                confirmation_msg = (
                    f"Are you sure you want to delete sandbox '{name}' ({sandbox.id})?"
                )

            if not confirm_or_skip(confirmation_msg, yes):
                console.print("Delete cancelled")
                return

            sandbox_ids = [s.id for s in matching_sandboxes]

            if len(sandbox_ids) == 1:
                with console.status("[bold blue]Deleting sandbox...", spinner="dots"):
                    sandbox_client.delete(sandbox_ids[0])
                console.print(
                    f"[green]Successfully deleted sandbox '{name}' ({sandbox_ids[0]})[/green]"
                )
            else:
                with console.status("[bold blue]Deleting sandboxes...", spinner="dots"):
                    succeeded, failed = _bulk_delete_sandboxes(sandbox_client, sandbox_ids)

                _print_bulk_delete_results(
                    succeeded,
                    failed,
                    f"Successfully deleted {len(succeeded)} sandbox(es) named '{name}':",
                )
            return
        elif labels:
            labels_str = ", ".join(labels)

            # Fetch sandboxes with matching labels
            with console.status("[bold blue]Fetching sandboxes...", spinner="dots"):
                all_sandboxes = []
                page = 1
                while True:
                    list_response = sandbox_client.list(
                        per_page=100, page=page, labels=labels, exclude_terminated=True
                    )
                    all_sandboxes.extend(list_response.sandboxes)
                    if not list_response.has_next:
                        break
                    page += 1

                # Apply only_mine filter if set
                if only_mine:
                    current_user_id = config.user_id
                    if not current_user_id:
                        console.print(
                            "[red]Error:[/red] Cannot filter by user - no user_id configured. "
                            "Use --all-users to delete sandboxes from all users, "
                            "or configure your user_id."
                        )
                        raise typer.Exit(1)
                    matching_sandboxes = [s for s in all_sandboxes if s.user_id == current_user_id]
                else:
                    matching_sandboxes = all_sandboxes

            if not matching_sandboxes:
                console.print(f"[yellow]No sandboxes found with labels: {labels_str}[/yellow]")
                if only_mine and all_sandboxes:
                    console.print(
                        "\n[dim]Note: --label only matches your own sandboxes by default. "
                        "Use --all-users to match sandboxes from all team members.[/dim]"
                    )
                return

            # Show sandboxes that will be deleted
            if len(matching_sandboxes) > 1:
                console.print(
                    f"\n[yellow]Found {len(matching_sandboxes)} sandbox(es) "
                    f"with labels: {labels_str}[/yellow]\n"
                )
                table = build_table(
                    "",
                    [
                        ("ID", "cyan"),
                        ("Name", "blue"),
                        ("Status", "yellow"),
                        ("Age", "blue"),
                    ],
                )
                for sandbox in sort_by_created(matching_sandboxes):
                    color = status_color(sandbox.status, SANDBOX_STATUS_COLORS)
                    table.add_row(
                        sandbox.id,
                        sandbox.name,
                        Text(sandbox.status, style=color),
                        human_age(sandbox.created_at),
                    )
                console.print(table)
                console.print(
                    "\n[dim]Tip: To delete a specific sandbox, use its ID: "
                    "prime sandbox delete <sandbox-id>[/dim]\n"
                )

            confirmation_msg = (
                f"Are you sure you want to delete {len(matching_sandboxes)} sandbox(es) "
                f"with labels: {labels_str}? This action cannot be undone."
            )

            if not confirm_or_skip(confirmation_msg, yes):
                console.print("Delete cancelled")
                return

            sandbox_ids = [s.id for s in matching_sandboxes]

            if len(sandbox_ids) == 1:
                with console.status("[bold blue]Deleting sandbox...", spinner="dots"):
                    sandbox_client.delete(sandbox_ids[0])
                console.print(
                    f"[green]Successfully deleted sandbox with labels '{labels_str}' "
                    f"({sandbox_ids[0]})[/green]"
                )
            else:
                with console.status("[bold blue]Deleting sandboxes...", spinner="dots"):
                    succeeded, failed = _bulk_delete_sandboxes(sandbox_client, sandbox_ids)

                success_msg = (
                    f"Successfully deleted {len(succeeded)} sandbox(es) with labels '{labels_str}':"
                )
                _print_bulk_delete_results(succeeded, failed, success_msg)
            return
        else:
            # Parse direct sandbox IDs
            parsed_ids = []
            for id_string in sandbox_ids or []:
                if "," in id_string:
                    parsed_ids.extend([id.strip() for id in id_string.split(",") if id.strip()])
                else:
                    parsed_ids.append(id_string.strip())

            cleaned_ids = []
            seen = set()
            for id in parsed_ids:
                if id and id not in seen:
                    cleaned_ids.append(id)
                    seen.add(id)
            sandbox_ids = cleaned_ids

        if len(sandbox_ids) == 1 and not all:
            sandbox_id = sandbox_ids[0]
            if not confirm_or_skip(f"Are you sure you want to delete sandbox {sandbox_id}?", yes):
                console.print("Delete cancelled")
                return

            with console.status("[bold blue]Deleting sandbox...", spinner="dots"):
                sandbox_client.delete(sandbox_id)

            console.print(f"[green]Successfully deleted sandbox {sandbox_id}[/green]")

        else:
            if all:
                confirmation_msg = (
                    f"Are you sure you want to delete ALL {len(sandbox_ids)} "
                    f"sandbox(es)? This action cannot be undone."
                )
                cancel_msg = "Delete all cancelled"
            else:
                confirmation_msg = (
                    f"Are you sure you want to delete {len(sandbox_ids)} sandbox(es)?"
                )
                cancel_msg = "Bulk delete cancelled"

            if not confirm_or_skip(confirmation_msg, yes):
                console.print(cancel_msg)
                return

            with console.status("[bold blue]Deleting sandboxes...", spinner="dots"):
                succeeded, failed = _bulk_delete_sandboxes(sandbox_client, sandbox_ids)

            # Display combined results
            total_processed = len(succeeded) + len(failed)
            console.print(f"\n[green]Processed {total_processed} sandbox(es)[/green]")

            _print_bulk_delete_results(
                succeeded,
                failed,
                f"Successfully deleted {len(succeeded)} sandbox(es):",
            )

    except typer.Exit:
        raise
    except UnauthorizedError as e:
        console.print(f"[red]Unauthorized:[/red] {str(e)}")
        raise typer.Exit(1)
    except PaymentRequiredError as e:
        console.print(f"[red]Payment Required:[/red] {str(e)}")
        raise typer.Exit(1)
    except APIError as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {escape(str(e))}")
        console.print_exception(show_locals=True)
        raise typer.Exit(1)


@app.command(no_args_is_help=True)
def logs(sandbox_id: str) -> None:
    """Get logs from a sandbox"""
    try:
        base_client = APIClient()
        sandbox_client = SandboxClient(base_client)

        with console.status("[bold blue]Fetching logs...", spinner="dots"):
            logs = sandbox_client.get_logs(sandbox_id)

        if logs:
            console.print(f"\n[bold]Logs for sandbox {sandbox_id}:[/bold]")
            console.print(f"{escape(logs)}")
        else:
            console.print(f"[yellow]No logs available for sandbox {sandbox_id}[/yellow]")

    except typer.Exit:
        raise
    except UnauthorizedError as e:
        console.print(f"[red]Unauthorized:[/red] {str(e)}")
        raise typer.Exit(1)
    except PaymentRequiredError as e:
        console.print(f"[red]Payment Required:[/red] {str(e)}")
        raise typer.Exit(1)
    except APIError as e:
        console.print(f"[red]Error:[/red] {escape(str(e))}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {escape(str(e))}")
        console.print_exception(show_locals=True)
        raise typer.Exit(1)


@app.command(no_args_is_help=True)
def run(
    sandbox_id: str,
    command: List[str] = typer.Argument(
        ...,
        help="Command to execute. Use -- before commands with options "
        "(e.g., -- bash -c 'echo hello')",
    ),
    working_dir: Optional[str] = typer.Option(
        None, "-w", "--working-dir", help="Working directory"
    ),
    env: Optional[List[str]] = typer.Option(
        None,
        "-e",
        "--env",
        help="Environment variables in KEY=VALUE format. Can be specified multiple times.",
    ),
    timeout: Optional[int] = typer.Option(
        None,
        "--timeout",
        help="Timeout for the command in seconds",
    ),
) -> None:
    """Execute a command in a sandbox.

    Use -- to separate sandbox run options from the command arguments when
    the command has its own options (starting with -). Example:

        prime sandbox run <id> -- bash -c "echo hello"
    """
    try:
        base_client = APIClient()
        sandbox_client = SandboxClient(base_client)

        # Parse environment variables
        env_vars = {}
        if env:
            for env_var in env:
                if "=" not in env_var:
                    console.print("[red]Environment variables must be in KEY=VALUE format[/red]")
                    raise typer.Exit(1)
                key, value = env_var.split("=", 1)
                env_vars[key] = value

        # Handle case where user passes entire command as a quoted string (e.g., "ls /home")
        # We need to parse it properly to handle both:
        # - "ls /home" -> should become: ls /home
        # - "./my script.sh" -> should become: './my script.sh' (properly quoted)
        if len(command) == 1:
            # Parse the single string as shell tokens, then re-join properly
            command_str = shlex.join(shlex.split(command[0]))
        else:
            command_str = shlex.join(command)

        console.print(f"[bold blue]Executing command:[/bold blue] {command_str}")
        if working_dir:
            console.print(f"[bold blue]Working directory:[/bold blue] {working_dir}")
        if env_vars:
            obfuscated_env = obfuscate_env_vars(env_vars)
            console.print(f"[bold blue]Environment:[/bold blue] {obfuscated_env}")
        if timeout is not None:
            console.print(f"[bold blue]Timeout:[/bold blue] {timeout}s")

        start_time = time.perf_counter()

        with console.status("[bold blue]Running command...", spinner="dots"):
            result = sandbox_client.execute_command(
                sandbox_id,
                command_str,
                working_dir,
                env_vars if env_vars else None,
                timeout=timeout,
            )

        # End timing
        end_time = time.perf_counter()
        execution_time_ms = (end_time - start_time) * 1000

        # Display output
        if result.stdout:
            console.print("\n[bold green]stdout:[/bold green]")
            console.print(result.stdout)

        if result.stderr:
            console.print("\n[bold red]stderr:[/bold red]")
            console.print(result.stderr)

        console.print(f"\n[dim]Execution time: {execution_time_ms:.1f}ms[/dim]")

        if result.exit_code != 0:
            console.print(f"\n[bold yellow]Exit code:[/bold yellow] {result.exit_code}")
            raise typer.Exit(result.exit_code)

    except typer.Exit:
        raise
    except SandboxNotRunningError as e:
        console.print(f"[red]Sandbox Not Running:[/red] {str(e)}")
        console.print(
            f"[yellow]Tip:[/yellow] Check sandbox status with: prime sandbox get {sandbox_id}"
        )
        raise typer.Exit(1)
    except CommandTimeoutError as e:
        console.print(f"[red]Command Timeout:[/red] {str(e)}")
        raise typer.Exit(1)
    except UnauthorizedError as e:
        console.print(f"[red]Unauthorized:[/red] {str(e)}")
        raise typer.Exit(1)
    except PaymentRequiredError as e:
        console.print(f"[red]Payment Required:[/red] {str(e)}")
        raise typer.Exit(1)
    except APIError as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {escape(str(e))}")
        console.print_exception(show_locals=True)
        raise typer.Exit(1)


@app.command("upload", no_args_is_help=True)
def upload_file(
    sandbox_id: str = typer.Argument(..., help="Sandbox ID to upload file to"),
    local_file: str = typer.Argument(..., help="Path to local file to upload"),
    remote_path: str = typer.Argument(..., help="Path where file should be stored in sandbox"),
) -> None:
    """Upload a file to a sandbox"""
    import os

    try:
        # Check if local file exists
        if not os.path.exists(local_file):
            console.print(f"[red]Error:[/red] Local file not found: {local_file}")
            raise typer.Exit(1)

        # Get file size for display
        file_size = os.path.getsize(local_file)
        filename = os.path.basename(local_file)

        # If remote_path ends with '/', treat as directory
        if remote_path.endswith("/"):
            remote_path = remote_path.rstrip("/") + "/" + filename

        base_client = APIClient()
        sandbox_client = SandboxClient(base_client)

        console.print(f"[bold blue]Uploading file:[/bold blue] {filename}")
        console.print(f"[bold blue]From:[/bold blue] {local_file}")
        console.print(f"[bold blue]To:[/bold blue] {remote_path}")
        console.print(f"[bold blue]Size:[/bold blue] {file_size:,} bytes")

        with console.status("[bold blue]Uploading...", spinner="dots"):
            response = sandbox_client.upload_file(
                sandbox_id=sandbox_id, file_path=remote_path, local_file_path=local_file
            )

        console.print("[green]✓[/green] File uploaded successfully!")
        console.print(f"[bold green]Remote path:[/bold green] {response.path}")
        console.print(f"[bold green]Size:[/bold green] {response.size:,} bytes")
        console.print(f"[bold green]Timestamp:[/bold green] {response.timestamp}")

    except typer.Exit:
        raise
    except UnauthorizedError as e:
        console.print(f"[red]Unauthorized:[/red] {str(e)}")
        raise typer.Exit(1)
    except PaymentRequiredError as e:
        console.print(f"[red]Payment Required:[/red] {str(e)}")
        raise typer.Exit(1)
    except APIError as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {escape(str(e))}")
        console.print_exception(show_locals=True)
        raise typer.Exit(1)


@app.command("download", no_args_is_help=True)
def download_file(
    sandbox_id: str = typer.Argument(..., help="Sandbox ID to download file from"),
    remote_path: str = typer.Argument(..., help="Path to file in sandbox"),
    local_file: str = typer.Argument(..., help="Path where file should be saved locally"),
) -> None:
    """Download a file from a sandbox"""
    import os

    try:
        # Check if local directory exists
        local_dir = os.path.dirname(local_file)
        if local_dir and not os.path.exists(local_dir):
            console.print(f"[yellow]Warning:[/yellow] Creating directory: {local_dir}")
            os.makedirs(local_dir, exist_ok=True)

        # Check if local file already exists
        if os.path.exists(local_file):
            if not typer.confirm(f"File {local_file} already exists. Overwrite?"):
                console.print("Download cancelled.")
                return

        base_client = APIClient()
        sandbox_client = SandboxClient(base_client)

        console.print(f"[bold blue]Downloading file from sandbox:[/bold blue] {sandbox_id}")
        console.print(f"[bold blue]From:[/bold blue] {remote_path}")
        console.print(f"[bold blue]To:[/bold blue] {local_file}")

        with console.status("[bold blue]Downloading...", spinner="dots"):
            sandbox_client.download_file(
                sandbox_id=sandbox_id, file_path=remote_path, local_file_path=local_file
            )

        # Get downloaded file size for display
        file_size = os.path.getsize(local_file) if os.path.exists(local_file) else 0

        console.print("[green]✓[/green] File downloaded successfully!")
        console.print(f"[bold green]Local path:[/bold green] {local_file}")
        console.print(f"[bold green]Size:[/bold green] {file_size:,} bytes")

    except typer.Exit:
        raise
    except FileNotFoundError as e:
        console.print(f"[red]File not found:[/red] {str(e)}")
        raise typer.Exit(1)
    except UnauthorizedError as e:
        console.print(f"[red]Unauthorized:[/red] {str(e)}")
        raise typer.Exit(1)
    except PaymentRequiredError as e:
        console.print(f"[red]Payment Required:[/red] {str(e)}")
        raise typer.Exit(1)
    except APIError as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {escape(str(e))}")
        console.print_exception(show_locals=True)
        raise typer.Exit(1)


@app.command("reset-cache")
def reset_cache(
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
) -> None:
    """Reset sandbox authentication cache"""
    if yes or typer.confirm("Are you sure you want to clear the sandbox auth cache?"):
        try:
            client = APIClient()
            sandbox_client = SandboxClient(client)
            sandbox_client.clear_auth_cache()
            console.print("[green]Sandbox authentication cache cleared successfully![/green]")
        except Exception as e:
            console.print(f"[red]Error clearing cache: {e}[/red]")
            raise typer.Exit(1)


@app.command("expose", no_args_is_help=True)
def expose_port(
    sandbox_id: str = typer.Argument(..., help="Sandbox ID to expose port from"),
    port: int = typer.Argument(..., help="Port number to expose"),
    name: Optional[str] = typer.Option(None, help="Optional name for the exposed port"),
    output: str = typer.Option("table", "--output", "-o", help="Output format: table or json"),
) -> None:
    """Expose an HTTP port from a sandbox.

    Currently only HTTP is supported. TCP, UDP, and SSH support coming soon.
    """
    validate_output_format(output, console)

    try:
        base_client = APIClient()
        sandbox_client = SandboxClient(base_client)

        with console.status("[bold blue]Exposing port...", spinner="dots"):
            exposed = sandbox_client.expose(sandbox_id, port, name)

        if output == "json":
            output_data_as_json(exposed.model_dump(), console)
        else:
            console.print("[green]✓[/green] Port exposed successfully!")
            console.print(f"[bold green]Exposure ID:[/bold green] {exposed.exposure_id}")
            console.print(f"[bold green]Port:[/bold green] {exposed.port}")
            if exposed.name:
                console.print(f"[bold green]Name:[/bold green] {exposed.name}")
            console.print(f"[bold green]URL:[/bold green] {exposed.url}")
            console.print(f"[bold green]TLS Socket:[/bold green] {exposed.tls_socket}")

    except APIError as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {escape(str(e))}")
        console.print_exception(show_locals=True)
        raise typer.Exit(1)


@app.command("unexpose", no_args_is_help=True)
def unexpose_port(
    sandbox_id: str = typer.Argument(..., help="Sandbox ID"),
    exposure_id: str = typer.Argument(..., help="Exposure ID to remove"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
) -> None:
    """Unexpose a port from a sandbox"""
    try:
        if not confirm_or_skip(
            f"Are you sure you want to unexpose {exposure_id}?", yes, default=True
        ):
            console.print("Unexpose cancelled")
            raise typer.Exit(0)

        base_client = APIClient()
        sandbox_client = SandboxClient(base_client)

        with console.status("[bold blue]Unexposing port...", spinner="dots"):
            sandbox_client.unexpose(sandbox_id, exposure_id)

        console.print(f"[green]✓ Successfully unexposed {exposure_id}[/green]")

    except APIError as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {escape(str(e))}")
        console.print_exception(show_locals=True)
        raise typer.Exit(1)


@app.command("list-ports", no_args_is_help=True)
def list_ports(
    sandbox_id: str = typer.Argument(..., help="Sandbox ID"),
    output: str = typer.Option("table", "--output", "-o", help="Output format: table or json"),
) -> None:
    """List all exposed ports for a sandbox"""
    validate_output_format(output, console)

    try:
        base_client = APIClient()
        sandbox_client = SandboxClient(base_client)

        with console.status("[bold blue]Fetching exposed ports...", spinner="dots"):
            response = sandbox_client.list_exposed_ports(sandbox_id)

        if output == "json":
            output_data_as_json(
                {"exposures": [exp.model_dump() for exp in response.exposures]}, console
            )
        else:
            if not response.exposures:
                console.print(f"[yellow]No exposed ports for sandbox {sandbox_id}[/yellow]")
            else:
                table = build_table(
                    f"Exposed Ports for Sandbox {sandbox_id}",
                    [
                        ("Exposure ID", "cyan"),
                        ("Port", "blue"),
                        ("Name", "green"),
                        ("URL", "magenta"),
                        ("TLS Socket", "yellow"),
                    ],
                )

                for exp in response.exposures:
                    table.add_row(
                        exp.exposure_id,
                        str(exp.port),
                        exp.name or "N/A",
                        exp.url,
                        exp.tls_socket,
                    )

                console.print(table)

    except APIError as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {escape(str(e))}")
        console.print_exception(show_locals=True)
        raise typer.Exit(1)
