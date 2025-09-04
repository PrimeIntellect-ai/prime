import json
import random
import string
import time
from typing import Any, Dict, List, Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

from ..api.client import APIClient, APIError
from ..api.sandbox import CreateSandboxRequest, Sandbox, SandboxClient
from ..config import Config
from ..utils.debug import debug_log, set_debug_enabled
from ..utils import (
    build_table,
    confirm_or_skip,
    obfuscate_env_vars,
    output_data_as_json,
    sort_by_created,
    status_color,
    validate_output_format,
    expand_home_in_path,
    parse_cp_arg,
    format_sandbox_for_list,
    format_sandbox_for_details,
)
from ..utils.display import SANDBOX_STATUS_COLORS

app = typer.Typer(help="Manage code sandboxes")
console = Console()
config = Config()


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
                sandbox_data = format_sandbox_for_list(sandbox)
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
                sandbox_data = format_sandbox_for_list(sandbox)

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
        console.print(f"[red]Unexpected error:[/red] {str(e)}")
        raise typer.Exit(1)


@app.command()
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
            sandbox_data = format_sandbox_for_details(sandbox)
            output_data_as_json(sandbox_data, console)
        else:
            # Output as table using shared formatting
            sandbox_data = format_sandbox_for_details(sandbox)

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
        console.print(f"[red]Unexpected error:[/red] {str(e)}")
        raise typer.Exit(1)


@app.command()
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
        console.print(f"[red]Unexpected error:[/red] {str(e)}")
        raise typer.Exit(1)


@app.command()
def delete(
    sandbox_id: str,
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
) -> None:
    """Delete a sandbox"""
    try:
        base_client = APIClient()
        sandbox_client = SandboxClient(base_client)

        if not confirm_or_skip(f"Are you sure you want to delete sandbox {sandbox_id}?", yes):
            console.print("Delete cancelled")
            raise typer.Exit(0)

        with console.status("[bold blue]Deleting sandbox...", spinner="dots"):
            sandbox_client.delete(sandbox_id)

        console.print(f"[green]Successfully deleted sandbox {sandbox_id}[/green]")

    except APIError as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {str(e)}")
        raise typer.Exit(1)


@app.command()
def logs(sandbox_id: str) -> None:
    """Get logs from a sandbox"""
    try:
        base_client = APIClient()
        sandbox_client = SandboxClient(base_client)

        with console.status("[bold blue]Fetching logs...", spinner="dots"):
            logs = sandbox_client.get_logs(sandbox_id)

        if logs:
            console.print(f"\n[bold]Logs for sandbox {sandbox_id}:[/bold]")
            console.print(logs)
        else:
            console.print(f"[yellow]No logs available for sandbox {sandbox_id}[/yellow]")

    except APIError as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {str(e)}")
        raise typer.Exit(1)


@app.command()
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
        console.print(f"[red]Unexpected error:[/red] {str(e)}")
        raise typer.Exit(1)


@app.command()
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
        console.print(f"[red]Unexpected error:[/red] {str(e)}")
        raise typer.Exit(1)


@app.command("cp")
def cp(
    ctx: typer.Context,
    source: str = typer.Argument(
        ..., help="Source path or <sandbox-id>:<path>", show_default=False
    ),
    destination: str = typer.Argument(
        ..., help="Destination path or <sandbox-id>:<path>", show_default=False
    ),
    working_dir: Optional[str] = typer.Option(
        None, "-w", "--working-dir", help="Sandbox working dir"
    ),
) -> None:
    """Copy files or folders between local and a sandbox.

    Examples:
      prime sandbox cp ./localdir <sandbox>:work
      prime sandbox cp <sandbox>:/work/out ./localdir
    """
    try:
        # Set debug flag from context
        debug_enabled = ctx.obj.get("debug", False) if ctx.obj else False
        set_debug_enabled(debug_enabled)

        debug_log(f"cp command called with source={source}, destination={destination}")
        base_client = APIClient()
        sandbox_client = SandboxClient(base_client)

        src_sid, src_path = parse_cp_arg(source)
        dst_sid, dst_path = parse_cp_arg(destination)
        debug_log(
            f"Parsed args: src_sid={src_sid}, src_path={src_path}, "
            f"dst_sid={dst_sid}, dst_path={dst_path}"
        )

        # Expand $HOME in sandbox paths for user convenience
        if src_sid is not None:
            src_path = expand_home_in_path(src_path)
        if dst_sid is not None:
            dst_path = expand_home_in_path(dst_path)

        if (src_sid is None) == (dst_sid is None):
            console.print("[red]One side must be a sandbox, the other must be local.[/red]")
            raise typer.Exit(1)

        # Local -> Sandbox
        if src_sid is None and dst_sid is not None:
            sandbox_client.handle_local_to_sandbox_copy(src_path, dst_sid, dst_path, working_dir, console)
            return

        # Sandbox -> Local
        if src_sid is not None and dst_sid is None:
            sandbox_client.handle_sandbox_to_local_copy(src_sid, src_path, dst_path, working_dir, console)
            return

        console.print("[red]Unsupported copy direction[/red]")
        raise typer.Exit(1)

    except APIError as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {str(e)}")
        raise typer.Exit(1)
