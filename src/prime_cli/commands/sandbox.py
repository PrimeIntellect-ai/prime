import json
import random
import string
import time
from typing import Any, Dict, List, Optional

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table
from rich.text import Text

from ..api.client import APIClient, APIError
from ..api.sandbox import BulkDeleteSandboxResponse, CreateSandboxRequest, Sandbox, SandboxClient
from ..config import Config
from ..utils import (
    build_table,
    confirm_or_skip,
    format_resources,
    human_age,
    iso_timestamp,
    obfuscate_env_vars,
    output_data_as_json,
    sort_by_created,
    status_color,
    validate_output_format,
)
from ..utils.display import SANDBOX_STATUS_COLORS

app = typer.Typer(help="Manage code sandboxes", no_args_is_help=True)
console = Console()


config = Config()


def _format_sandbox_for_list(sandbox: Sandbox) -> Dict[str, Any]:
    """Format sandbox data for list display (both table and JSON)"""
    return {
        "id": sandbox.id,
        "name": sandbox.name,
        "image": sandbox.docker_image,
        "status": sandbox.status,
        "resources": format_resources(sandbox.cpu_cores, sandbox.memory_gb, sandbox.gpu_count),
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
        "timeout_minutes": sandbox.timeout_minutes,
        "created_at": iso_timestamp(sandbox.created_at),
        "user_id": sandbox.user_id,
        "team_id": sandbox.team_id,
    }

    if sandbox.started_at:
        data["started_at"] = iso_timestamp(sandbox.started_at)
    if sandbox.terminated_at:
        data["terminated_at"] = iso_timestamp(sandbox.terminated_at)
    if sandbox.exit_code is not None:
        data["exit_code"] = sandbox.exit_code
    if sandbox.environment_vars:
        data["environment_vars"] = obfuscate_env_vars(sandbox.environment_vars)
    if sandbox.advanced_configs:
        data["advanced_configs"] = sandbox.advanced_configs.model_dump()

    return data


@app.command("list")
def list_sandboxes_cmd(
    team_id: Optional[str] = typer.Option(
        None, help="Filter by team ID (uses config team_id if not specified)"
    ),
    status: Optional[str] = typer.Option(None, help="Filter by status"),
    page: int = typer.Option(1, help="Page number"),
    per_page: int = typer.Option(50, help="Items per page"),
    all: bool = typer.Option(False, "--all", help="Show all sandboxes including terminated ones"),
    output: str = typer.Option("table", "--output", "-o", help="Output format: table or json"),
) -> None:
    """List your sandboxes (excludes terminated by default)"""
    validate_output_format(output, console)

    try:
        base_client = APIClient()
        sandbox_client = SandboxClient(base_client)

        # Always exclude terminated sandboxes unless --all is specified or status filter is used
        exclude_terminated = not all and status is None

        sandbox_list = sandbox_client.list(
            team_id=team_id,
            status=status,
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
                    sandbox_data["age"],
                )

            console.print(table)

            if sandbox_list.has_next:
                console.print(
                    f"\n[yellow]Showing page {page} of results. "
                    f"Use --page {page + 1} to see more.[/yellow]"
                )

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
            table.add_row("Timeout (minutes)", str(sandbox_data["timeout_minutes"]))

            table.add_row("Created", sandbox_data["created_at"])
            if "started_at" in sandbox_data:
                table.add_row("Started", sandbox_data["started_at"])
            if "terminated_at" in sandbox_data:
                table.add_row("Terminated", sandbox_data["terminated_at"])
            if "exit_code" in sandbox_data:
                table.add_row("Exit Code", str(sandbox_data["exit_code"]))

            table.add_row("User ID", sandbox_data["user_id"] or "N/A")
            table.add_row("Team ID", sandbox_data["team_id"] or "Personal")

            if "environment_vars" in sandbox_data:
                env_vars = json.dumps(sandbox_data["environment_vars"], indent=2)
                table.add_row("Environment Variables", env_vars)

            if "advanced_configs" in sandbox_data:
                advanced_configs = json.dumps(sandbox_data["advanced_configs"], indent=2)
                table.add_row("Advanced Configs", advanced_configs)

            console.print(table)

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
    timeout_minutes: int = typer.Option(60, help="Timeout in minutes"),
    team_id: Optional[str] = typer.Option(
        None, help="Team ID (uses config team_id if not specified)"
    ),
    env: Optional[List[str]] = typer.Option(
        None,
        help="Environment variables in KEY=VALUE format. Can be specified multiple times.",
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
            timeout_minutes=timeout_minutes,
            environment_vars=env_vars if env_vars else None,
            team_id=team_id,
        )

        # Show configuration summary
        console.print("\n[bold]Sandbox Configuration:[/bold]")
        console.print(f"Name: {name}")
        console.print(f"Docker Image: {docker_image}")
        console.print(f"Start Command: {start_command or 'N/A'}")
        console.print(f"Resources: {cpu_cores} CPU, {memory_gb}GB RAM, {disk_size_gb}GB disk")
        if gpu_count > 0:
            console.print(f"GPUs: {gpu_count}")
        console.print(f"Timeout: {timeout_minutes} minutes")
        console.print(f"Team: {team_id or 'Personal'}")
        if env_vars:
            obfuscated_env = obfuscate_env_vars(env_vars)
            console.print(f"Environment Variables: {obfuscated_env}")

        if confirm_or_skip("\nDo you want to create this sandbox?", yes, default=True):
            with console.status("[bold blue]Creating sandbox...", spinner="dots"):
                sandbox = sandbox_client.create(request)

            console.print(f"\n[green]Successfully created sandbox {sandbox.id}[/green]")
            console.print(
                f"[blue]Use 'prime sandbox get {sandbox.id}' to check the sandbox status[/blue]"
            )
        else:
            console.print("\nSandbox creation cancelled")
            raise typer.Exit(0)

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
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
) -> None:
    """Delete one or more sandboxes, or all sandboxes with --all"""
    try:
        base_client = APIClient()
        sandbox_client = SandboxClient(base_client)

        # Validate arguments
        if all and sandbox_ids:
            console.print("[red]Error:[/red] Cannot specify both sandbox IDs and --all flag")
            raise typer.Exit(1)

        if not all and not sandbox_ids:
            console.print("[red]Error:[/red] Must specify either sandbox IDs or --all flag")
            raise typer.Exit(1)

        if all:
            # Get all sandboxes to delete
            with console.status("[bold blue]Fetching all sandboxes...", spinner="dots"):
                all_sandboxes = []
                page = 1
                while True:
                    list_response = sandbox_client.list(
                        per_page=100, page=page, exclude_terminated=False
                    )
                    all_sandboxes.extend(list_response.sandboxes)
                    if not list_response.has_next:
                        break
                    page += 1

                # Filter out already terminated sandboxes
                active_sandboxes = [
                    s for s in all_sandboxes if s.status not in {"TERMINATED", "TIMEOUT"}
                ]
                sandbox_ids = [s.id for s in active_sandboxes]

                if not sandbox_ids:
                    console.print("[yellow]No sandboxes to delete[/yellow]")
                    raise typer.Exit(0)
        else:
            # Handle comma-separated IDs by splitting them
            parsed_ids = []
            for id_string in sandbox_ids or []:
                # Split by comma and strip whitespace
                if "," in id_string:
                    parsed_ids.extend([id.strip() for id in id_string.split(",") if id.strip()])
                else:
                    parsed_ids.append(id_string.strip())

            # Remove any empty strings and duplicates while preserving order
            cleaned_ids = []
            seen = set()
            for id in parsed_ids:
                if id and id not in seen:
                    cleaned_ids.append(id)
                    seen.add(id)

            sandbox_ids = cleaned_ids

        # Handle single vs multiple sandbox IDs
        if len(sandbox_ids) == 1 and not all:
            # Single sandbox deletion
            sandbox_id = sandbox_ids[0]
            if not confirm_or_skip(f"Are you sure you want to delete sandbox {sandbox_id}?", yes):
                console.print("Delete cancelled")
                raise typer.Exit(0)

            with console.status("[bold blue]Deleting sandbox...", spinner="dots"):
                sandbox_client.delete(sandbox_id)

            console.print(f"[green]Successfully deleted sandbox {sandbox_id}[/green]")

        else:
            # Bulk sandbox deletion
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
                raise typer.Exit(0)

            # Batch the deletion into chunks of 100 to respect API limits
            batch_size = 100
            all_succeeded = []
            all_failed = []

            with console.status("[bold blue]Deleting sandboxes...", spinner="dots"):
                for i in range(0, len(sandbox_ids), batch_size):
                    batch = sandbox_ids[i : i + batch_size]
                    batch_num = (i // batch_size) + 1
                    total_batches = (len(sandbox_ids) + batch_size - 1) // batch_size

                    console.print(
                        f"[dim]Processing batch {batch_num}/{total_batches} "
                        f"({len(batch)} sandboxes)...[/dim]"
                    )

                    result: BulkDeleteSandboxResponse = sandbox_client.bulk_delete(batch)

                    if result.succeeded:
                        all_succeeded.extend(result.succeeded)
                    if result.failed:
                        all_failed.extend(result.failed)

            # Display combined results
            total_processed = len(all_succeeded) + len(all_failed)
            console.print(f"\n[green]Processed {total_processed} sandbox(es)[/green]")

            if all_succeeded:
                console.print(
                    f"\n[bold green]Successfully deleted {len(all_succeeded)} "
                    f"sandbox(es):[/bold green]"
                )
                for sandbox_id in all_succeeded:
                    console.print(f"  ✓ {sandbox_id}")

            if all_failed:
                console.print(
                    f"\n[bold red]Failed to delete {len(all_failed)} sandbox(es):[/bold red]"
                )
                for failure in all_failed:
                    sandbox_id = failure.get("sandbox_id", "unknown")
                    error = failure.get("error", "unknown error")
                    console.print(f"  ✗ {sandbox_id}: {error}")

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

    except APIError as e:
        console.print(f"[red]Error:[/red] {escape(str(e))}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {escape(str(e))}")
        console.print_exception(show_locals=True)
        raise typer.Exit(1)


@app.command(no_args_is_help=True)
def status(sandbox_id: str) -> None:
    """Update and get the current status of a sandbox from Kubernetes"""
    try:
        base_client = APIClient()
        sandbox_client = SandboxClient(base_client)

        with console.status("[bold blue]Updating status from Kubernetes...", spinner="dots"):
            sandbox = sandbox_client.update_status(sandbox_id)

        console.print(f"[green]Status updated for sandbox {sandbox.id}[/green]")
        console.print(f"Current status: [bold]{sandbox.status}[/bold]")

    except APIError as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {escape(str(e))}")
        console.print_exception(show_locals=True)
        raise typer.Exit(1)


@app.command(no_args_is_help=True)
def run(
    sandbox_id: str,
    command: List[str] = typer.Argument(..., help="Command to execute"),
    working_dir: Optional[str] = typer.Option(
        None, "-w", "--working-dir", help="Working directory"
    ),
    env: Optional[List[str]] = typer.Option(
        None,
        "-e",
        "--env",
        help="Environment variables in KEY=VALUE format. Can be specified multiple times.",
    ),
) -> None:
    """Execute a command in a sandbox"""
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

        # Join command list into a single string
        command_str = " ".join(command)

        console.print(f"[bold blue]Executing command:[/bold blue] {command_str}")
        if working_dir:
            console.print(f"[bold blue]Working directory:[/bold blue] {working_dir}")
        if env_vars:
            obfuscated_env = obfuscate_env_vars(env_vars)
            console.print(f"[bold blue]Environment:[/bold blue] {obfuscated_env}")

        # Start timing
        start_time = time.perf_counter()

        with console.status("[bold blue]Running command...", spinner="dots"):
            result = sandbox_client.execute_command(
                sandbox_id, command_str, working_dir, env_vars if env_vars else None
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
                raise typer.Exit(0)

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

    except APIError as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        raise typer.Exit(1)
    except FileNotFoundError as e:
        console.print(f"[red]File not found:[/red] {str(e)}")
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
