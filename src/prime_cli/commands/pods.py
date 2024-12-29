import typer
from typing import Optional, List, Union
from rich.console import Console
from rich.table import Table
from rich.text import Text
from datetime import datetime
import subprocess
import os

from ..api.client import APIClient, APIError
from ..api.pods import PodsClient

app = typer.Typer(help="Manage compute pods")
console = Console()


def format_ip_display(ip: Optional[Union[str, List[str]]]) -> str:
    """Format IP address(es) for display, handling both single and list cases"""
    if not ip:
        return "N/A"
    # Handle both list and single IP cases by always converting to list
    ip_list = ip if hasattr(ip, "__iter__") and not isinstance(ip, str) else [ip]
    return ", ".join(ip_list)


@app.command()
def list(
    limit: int = typer.Option(100, help="Maximum number of pods to list"),
    offset: int = typer.Option(0, help="Number of pods to skip"),
):
    """List your running pods"""
    try:
        # Create API clients
        base_client = APIClient()
        pods_client = PodsClient(base_client)

        # Get pods list
        pods_list = pods_client.list(offset=offset, limit=limit)

        # If we have pods, get their detailed status
        if pods_list.data:
            pod_statuses = pods_client.get_status([pod.id for pod in pods_list.data])
            # Create a lookup dict for quick access
            status_lookup = {status.pod_id: status for status in pod_statuses}
        else:
            status_lookup = {}

        # Create display table
        table = Table(title=f"Compute Pods (Total: {pods_list.total_count})")
        table.add_column("ID", style="cyan")
        table.add_column("Name", style="blue")
        table.add_column("GPU", style="green")
        table.add_column("Status", style="yellow")
        table.add_column("IP", style="white")
        table.add_column("Created", style="blue")
        table.add_column("Team", style="blue")

        # Add rows for each pod
        for pod in pods_list.data:
            status = status_lookup.get(pod.id)

            # Format status with color
            display_status = pod.status
            if pod.status == "ACTIVE" and pod.installation_status != "FINISHED":
                display_status = "INSTALLING"
            
            status_color = {
                "ACTIVE": "green", 
                "PENDING": "yellow", 
                "ERROR": "red",
                "INSTALLING": "yellow"
            }.get(display_status, "white")

            # Format created time
            created_at = datetime.fromisoformat(pod.created_at.replace("Z", "+00:00"))
            created_str = created_at.strftime("%Y-%m-%d %H:%M:%S UTC")

            # Get IP display using helper function
            ip_display = format_ip_display(status.ip if status else None)

            table.add_row(
                pod.id,
                pod.name or "N/A",
                f"{pod.gpu_type} x{pod.gpu_count}",
                Text(display_status, style=status_color),
                ip_display,
                created_str,
                pod.team_id or "Personal",
            )

        console.print(table)

        # If there are more pods, show a message
        if pods_list.total_count > offset + limit:
            remaining = pods_list.total_count - (offset + limit)
            console.print(
                f"\n[yellow]Showing {limit} of {pods_list.total_count} pods. "
                f"Use --offset {offset + limit} to see the next {min(limit, remaining)} pods.[/yellow]"
            )

    except APIError as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {str(e)}")
        import traceback

        traceback.print_exc()
        raise typer.Exit(1)


@app.command()
def status(pod_id: str):
    """Get detailed status of a specific pod"""
    try:
        base_client = APIClient()
        pods_client = PodsClient(base_client)

        # Get pod status
        statuses = pods_client.get_status([pod_id])
        if not statuses:
            console.print(f"[red]No status found for pod {pod_id}[/red]")
            raise typer.Exit(1)

        status = statuses[0]

        # Create display table
        table = Table(title=f"Pod Status: {pod_id}")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="white")

        # Display status with installation state consideration
        display_status = status.status
        if status.status == "ACTIVE" and status.installation_status != "FINISHED":
            display_status = "INSTALLING"

        table.add_row(
            "Status",
            Text(
                display_status, 
                style="green" if display_status == "ACTIVE" else "yellow"
            ),
        )
        table.add_row("Team", status.team_id or "Personal")

        # Use helper function for IP display
        table.add_row("IP", format_ip_display(status.ip))

        # Handle SSH connection display for both single and list cases
        ssh_display = format_ip_display(status.ssh_connection)
        table.add_row("SSH", ssh_display)

        if status.installation_progress is not None:
            table.add_row("Installation Progress", f"{status.installation_progress}%")

        if status.installation_failure:
            table.add_row("Error", Text(status.installation_failure, style="red"))

        if status.prime_port_mapping:
            ports = "\n".join(
                [
                    f"{port.protocol}:{port.external}->{port.internal} ({port.used_by or 'unknown'})"
                    for port in status.prime_port_mapping
                ]
            )
            table.add_row("Ports", ports)

        console.print(table)

    except APIError as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {str(e)}")
        import traceback

        traceback.print_exc()
        raise typer.Exit(1)
