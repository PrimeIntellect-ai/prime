import json
from typing import List, Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

from ..api.client import APIClient, APIError
from ..api.sandbox import CreateSandboxRequest, SandboxClient
from ..config import Config

app = typer.Typer(help="Manage code sandboxes")
console = Console()
config = Config()


@app.command()
def list(
    team_id: Optional[str] = typer.Option(None, help="Filter by team ID"),
    status: Optional[str] = typer.Option(None, help="Filter by status"),
    page: int = typer.Option(1, help="Page number"),
    per_page: int = typer.Option(50, help="Items per page"),
    all: bool = typer.Option(False, "--all", help="Show all sandboxes including terminated ones"),
) -> None:
    """List your sandboxes (excludes terminated by default)"""
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

        table = Table(
            title=f"Code Sandboxes (Total: {sandbox_list.total})",
            show_lines=True,
        )
        table.add_column("ID", style="cyan", no_wrap=True)
        table.add_column("Name", style="blue")
        table.add_column("Image", style="green")
        table.add_column("Status", style="yellow")
        table.add_column("Resources", style="magenta")
        table.add_column("Created", style="blue")

        for sandbox in sandbox_list.sandboxes:
            status_color = {
                "PENDING": "yellow",
                "PROVISIONING": "yellow",
                "RUNNING": "green",
                "STOPPED": "blue",
                "ERROR": "red",
                "TERMINATED": "red",
            }.get(sandbox.status, "white")

            created_at = sandbox.created_at.strftime("%Y-%m-%d %H:%M:%S")

            resources = f"{sandbox.cpu_cores}CPU/{sandbox.memory_gb}GB"
            if sandbox.gpu_count > 0:
                resources += f"/{sandbox.gpu_count}GPU"

            table.add_row(
                sandbox.id,
                sandbox.name,
                sandbox.docker_image,
                Text(sandbox.status, style=status_color),
                resources,
                created_at,
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
def get(sandbox_id: str) -> None:
    """Get detailed information about a specific sandbox"""
    try:
        base_client = APIClient()
        sandbox_client = SandboxClient(base_client)

        sandbox = sandbox_client.get(sandbox_id)

        table = Table(title=f"Sandbox Details: {sandbox_id}")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="white")

        table.add_row("ID", sandbox.id)
        table.add_row("Name", sandbox.name)
        table.add_row("Docker Image", sandbox.docker_image)
        table.add_row("Start Command", sandbox.start_command or "N/A")
        table.add_row("Working Directory", sandbox.working_dir)

        status_color = {
            "PENDING": "yellow",
            "PROVISIONING": "yellow",
            "RUNNING": "green",
            "STOPPED": "blue",
            "ERROR": "red",
            "TERMINATED": "red",
        }.get(sandbox.status, "white")
        table.add_row("Status", Text(sandbox.status, style=status_color))

        table.add_row("CPU Cores", str(sandbox.cpu_cores))
        table.add_row("Memory (GB)", str(sandbox.memory_gb))
        table.add_row("Disk Size (GB)", str(sandbox.disk_size_gb))
        table.add_row("GPU Count", str(sandbox.gpu_count))
        table.add_row("Timeout (minutes)", str(sandbox.timeout_minutes))

        table.add_row("Created", sandbox.created_at.strftime("%Y-%m-%d %H:%M:%S UTC"))
        if sandbox.started_at:
            table.add_row("Started", sandbox.started_at.strftime("%Y-%m-%d %H:%M:%S UTC"))
        if sandbox.terminated_at:
            table.add_row("Terminated", sandbox.terminated_at.strftime("%Y-%m-%d %H:%M:%S UTC"))

        table.add_row("User ID", sandbox.user_id or "N/A")
        table.add_row("Team ID", sandbox.team_id or "Personal")
        table.add_row("Kubernetes Job ID", sandbox.kubernetes_job_id or "N/A")

        if sandbox.environment_vars:
            env_vars = json.dumps(sandbox.environment_vars, indent=2)
            table.add_row("Environment Variables", env_vars)

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
    start_command: Optional[str] = typer.Option(None, help="Command to run in the container"),
    cpu_cores: int = typer.Option(1, help="Number of CPU cores"),
    memory_gb: int = typer.Option(2, help="Memory in GB"),
    disk_size_gb: int = typer.Option(10, help="Disk size in GB"),
    gpu_count: int = typer.Option(0, help="Number of GPUs"),
    timeout_minutes: int = typer.Option(60, help="Timeout in minutes"),
    working_dir: str = typer.Option("/workspace", help="Working directory"),
    team_id: Optional[str] = typer.Option(None, help="Team ID (optional)"),
    env: Optional[List[str]] = typer.Option(
        None,
        help="Environment variables in KEY=VALUE format. Can be specified multiple times.",
    ),
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

            import random
            import string

            suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=4))
            name = f"{clean_image}-{suffix}"

        # Get team ID from config if not provided
        if not team_id:
            team_id = config.team_id

        # Ensure empty string is converted to None
        if team_id == "":
            team_id = None

        request = CreateSandboxRequest(
            name=name,
            docker_image=docker_image,
            start_command=start_command,
            cpu_cores=cpu_cores,
            memory_gb=memory_gb,
            disk_size_gb=disk_size_gb,
            gpu_count=gpu_count,
            timeout_minutes=timeout_minutes,
            working_dir=working_dir,
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
        console.print(f"Working Directory: {working_dir}")
        console.print(f"Team: {team_id or 'Personal'}")
        if env_vars:
            console.print(f"Environment Variables: {env_vars}")

        if typer.confirm("\nDo you want to create this sandbox?", default=True):
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
def delete(sandbox_id: str) -> None:
    """Delete a sandbox"""
    try:
        base_client = APIClient()
        sandbox_client = SandboxClient(base_client)

        if not typer.confirm(f"Are you sure you want to delete sandbox {sandbox_id}?"):
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
            console.print(f"[bold blue]Environment:[/bold blue] {env_vars}")

        with console.status("[bold blue]Running command...", spinner="dots"):
            result = sandbox_client.execute_command(
                sandbox_id, command_str, working_dir, env_vars if env_vars else None
            )

        # Display output
        if result.stdout:
            console.print("\n[bold green]stdout:[/bold green]")
            console.print(result.stdout)

        if result.stderr:
            console.print("\n[bold red]stderr:[/bold red]")
            console.print(result.stderr)

        if result.exit_code != 0:
            console.print(f"\n[bold yellow]Exit code:[/bold yellow] {result.exit_code}")
            raise typer.Exit(result.exit_code)

    except APIError as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {str(e)}")
        raise typer.Exit(1)
