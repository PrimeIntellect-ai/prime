import hashlib
import json
import os
import time
from datetime import datetime
from typing import Any, Dict, Optional

import typer
from prime_core import APIClient, APIError, Config
from rich.console import Console
from rich.table import Table
from rich.text import Text

from prime_cli.api.availability import AvailabilityClient
from prime_cli.helper.short_id import generate_short_id_disk

from ..api.disks import Disk, DisksClient
from ..utils import (
    confirm_or_skip,
    human_age,
    iso_timestamp,
    output_data_as_json,
    status_color,
    validate_output_format,
)
from ..utils.display import DISK_STATUS_COLORS

app = typer.Typer(help="Manage storage", no_args_is_help=True)
console = Console()
config = Config()


def _format_disk_for_list(disk: Disk) -> Dict[str, Any]:
    """Format disk data for list display (both table and JSON)"""
    created_at = datetime.fromisoformat(disk.created_at.replace("Z", "+00:00"))
    created_timestamp = iso_timestamp(created_at)
    age = human_age(created_at)

    country = disk.info.get("country", "N/A") if disk.info else "N/A"
    data_center = disk.info.get("dataCenterId", "N/A") if disk.info else "N/A"

    return {
        "id": disk.id,
        "name": disk.name,
        "size": disk.size,
        "status": disk.status,
        "provider": disk.provider_type,
        "location": f"{country} ({data_center})",
        "created_at": created_timestamp,  # For JSON output
        "age": age,  # For table output
        "price_hr": disk.price_hr,
        "pods": disk.pods,
        "clusters": disk.clusters,
    }


def _format_disk_for_detail(disk: Disk) -> Dict[str, Any]:
    """Format disk data for detailed display (both table and JSON)"""
    created_at = datetime.fromisoformat(disk.created_at.replace("Z", "+00:00"))
    created_timestamp = iso_timestamp(created_at)

    updated_at = datetime.fromisoformat(disk.updated_at.replace("Z", "+00:00"))
    updated_timestamp = iso_timestamp(updated_at)

    terminated_at = None
    terminated_timestamp = None
    if disk.terminated_at:
        terminated_at = datetime.fromisoformat(disk.terminated_at.replace("Z", "+00:00"))
        terminated_timestamp = iso_timestamp(terminated_at)

    return {
        "id": disk.id,
        "name": disk.name,
        "size": disk.size,
        "status": disk.status,
        "provider": disk.provider_type,
        "created_at": created_timestamp,
        "updated_at": updated_timestamp,
        "terminated_at": terminated_timestamp,
        "price_hr": disk.price_hr,
        "stopped_price_hr": disk.stopped_price_hr,
        "user_id": disk.user_id,
        "team_id": disk.team_id,
        "wallet_id": disk.wallet_id,
        "info": disk.info,
        "pods": disk.pods,
        "clusters": disk.clusters,
    }


@app.command()
def list(
    limit: int = typer.Option(100, help="Maximum number of disks to list"),
    offset: int = typer.Option(0, help="Number of disks to skip"),
    watch: bool = typer.Option(False, "--watch", "-w", help="Watch disks list in real-time"),
    output: str = typer.Option("table", "--output", "-o", help="Output format: table or json"),
) -> None:
    """List your persistent disks"""
    validate_output_format(output, console)

    if watch and output == "json":
        console.print("[red]Error: --watch mode is not compatible with --output=json[/red]")
        raise typer.Exit(1)

    try:
        # Create API clients
        base_client = APIClient()
        disks_client = DisksClient(base_client)

        last_disks_hash = None

        while True:
            disks_list = disks_client.list(offset=offset, limit=limit)

            current_disks_hash = hashlib.md5(
                json.dumps([disk.model_dump() for disk in disks_list.data], sort_keys=True).encode()
            ).hexdigest()

            # Only update display if data changed or first run
            if current_disks_hash != last_disks_hash:
                # Clear screen if watching
                if watch:
                    os.system("cls" if os.name == "nt" else "clear")

                # Sort disks by created_at (oldest first)
                sorted_disks = sorted(
                    disks_list.data,
                    key=lambda disk: datetime.fromisoformat(disk.created_at.replace("Z", "+00:00")),
                )

                if output == "json":
                    # Output as JSON with timestamp (for automation)
                    disks_data = []
                    for disk in sorted_disks:
                        disk_data = _format_disk_for_list(disk)
                        json_disk = {
                            "id": disk_data["id"],
                            "name": disk_data["name"],
                            "size": disk_data["size"],
                            "status": disk_data["status"],
                            "provider": disk_data["provider"],
                            "location": disk_data["location"],
                            "created_at": disk_data["created_at"],
                            "price_hr": disk_data["price_hr"],
                        }
                        disks_data.append(json_disk)

                    output_data = {
                        "disks": disks_data,
                        "total_count": disks_list.total_count,
                        "offset": offset,
                        "limit": limit,
                    }
                    output_data_as_json(output_data, console)
                else:
                    # Create display table
                    table = Table(
                        title=f"Persistent Disks (Total: {disks_list.total_count})",
                        show_lines=True,
                    )
                    table.add_column("ID", style="cyan", no_wrap=True)
                    table.add_column("Name", style="blue")
                    table.add_column("Size", style="green")
                    table.add_column("Status", style="yellow")
                    table.add_column("Provider", style="magenta")
                    table.add_column("Location", style="blue")
                    table.add_column("Age", style="blue")
                    table.add_column("Price/hr", style="green")
                    table.add_column("Pods", style="blue")
                    table.add_column("Clusters", style="blue")

                    # Add rows for each disk using shared formatting
                    for disk in sorted_disks:
                        disk_data = _format_disk_for_list(disk)

                        disk_status_color = status_color(disk_data["status"], DISK_STATUS_COLORS)
                        price_display = (
                            f"${disk_data['price_hr']:.3f}" if disk_data["price_hr"] else "N/A"
                        )

                        table.add_row(
                            disk_data["id"],
                            disk_data["name"] or "N/A",
                            f"{disk_data['size']}GB",
                            Text(disk_data["status"], style=disk_status_color),
                            disk_data["provider"],
                            disk_data["location"],
                            disk_data["age"],
                            price_display,
                            f"{len(disk_data['pods'])}",
                            f"{len(disk_data['clusters'])}",
                        )

                    console.print(table)

                    if not watch:
                        console.print(
                            "\n[blue]Use 'prime disks get <disk-id>' to "
                            "see detailed information about a specific disk[/blue]"
                        )

                        # If there are more disks, show a message
                        if disks_list.total_count > offset + limit:
                            remaining = disks_list.total_count - (offset + limit)
                            console.print(
                                f"\n[yellow]Showing {limit} of {disks_list.total_count} disks. "
                                f"Use --offset {offset + limit} to see the next "
                                f"{min(limit, remaining)} disks.[/yellow]"
                            )
            if not watch:
                break
            else:
                # Only print the message when we're not repeating due to unchanged data
                if current_disks_hash != last_disks_hash or last_disks_hash is None:
                    console.print("\n[dim]Press Ctrl+C to exit watch mode[/dim]")
                last_disks_hash = current_disks_hash
                try:
                    # Wait before refreshing
                    time.sleep(5)
                except KeyboardInterrupt:
                    # Clear the progress dots on exit
                    if current_disks_hash == last_disks_hash:
                        console.print("\n")
                    break

    except APIError as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {str(e)}")
        import traceback

        traceback.print_exc()
        raise typer.Exit(1)


@app.command(no_args_is_help=True)
def get(
    disk_id: str,
    output: str = typer.Option("table", "--output", "-o", help="Output format: table or json"),
) -> None:
    """Get detailed information about a specific disk"""
    validate_output_format(output, console)

    try:
        base_client = APIClient()
        disks_client = DisksClient(base_client)

        disk = disks_client.get(disk_id)

        if output == "json":
            # Output as JSON using shared formatting
            disk_data = _format_disk_for_detail(disk)
            output_data_as_json(disk_data, console)
        else:
            # Create display table
            disk_data = _format_disk_for_detail(disk)

            table = Table(title=f"Disk Details: {disk_id}")
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="white")

            table.add_row(
                "Status",
                Text(
                    disk_data["status"],
                    style=status_color(disk_data["status"], DISK_STATUS_COLORS),
                ),
            )

            # Basic disk info
            table.add_row("Name", disk_data["name"] or "N/A")
            table.add_row("Size", Text(f"{disk_data['size']}GB", style="magenta"))
            table.add_row("Provider", disk_data["provider"])
            table.add_row("Team", disk_data["team_id"] or "Personal")

            # Pricing info
            if disk_data["price_hr"]:
                table.add_row("Cost per Hour", f"${disk_data['price_hr']:.3f}")

            # Timestamps
            table.add_row("Created", disk_data["created_at"])
            table.add_row("Updated", disk_data["updated_at"])
            if disk_data["terminated_at"]:
                table.add_row("Terminated", disk_data["terminated_at"])

            # Additional info
            if disk_data["info"]:
                info_str = json.dumps(disk_data["info"], indent=2)
                table.add_row("Info", info_str)

            console.print(table)

            # Display attached pods/clusters if they exist
            if disk_data["pods"] or disk_data["clusters"]:
                attachment_table = Table(title="Attachments")
                attachment_table.add_column("Type", style="cyan")
                attachment_table.add_column("ID", style="white")

                for pod_id in disk_data["pods"]:
                    attachment_table.add_row("Pod", pod_id)

                for cluster_id in disk_data["clusters"]:
                    attachment_table.add_row("Cluster", cluster_id)

                console.print("\n")  # Add spacing between tables
                console.print(attachment_table)

    except APIError as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {str(e)}")
        import traceback

        traceback.print_exc()
        raise typer.Exit(1)


@app.command()
def create(
    id: Optional[str] = typer.Option(None, help="Short ID from availability list"),
    size: int = typer.Option(..., help="Size of the disk in GB"),
    name: Optional[str] = typer.Option(None, help="Name for the disk"),
    country: Optional[str] = typer.Option(None, help="Country location"),
    cloud_id: Optional[str] = typer.Option(None, help="Cloud ID from availability"),
    data_center_id: Optional[str] = typer.Option(None, help="Data center ID"),
    team_id: Optional[str] = typer.Option(
        None, help="Team ID to use for the disk (uses config team_id if not specified)"
    ),
    provider_type: Optional[str] = typer.Option(None, help="Provider type (e.g., lambda, runpod)"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
) -> None:
    """Create a new storage disk"""
    try:
        base_client = APIClient()
        availability_client = AvailabilityClient(base_client)
        disks_client = DisksClient(base_client)

        disk_config = None

        # Validate size
        if size <= 0:
            console.print("[red]Error: Disk size must be greater than 0[/red]")
            raise typer.Exit(1)

        with console.status("[bold blue]Loading available disks...", spinner="dots"):
            available_disks = availability_client.get_disks()

        if id:
            for disk in available_disks:
                if generate_short_id_disk(disk) == id:
                    disk_config = {
                        "disk": {
                            "size": size,
                            "name": name,
                            "country": disk.country,
                            "cloudId": disk.cloud_id,
                            "dataCenterId": disk.data_center,
                        },
                        "provider": {"type": disk.provider},
                        "team": {"teamId": team_id} if team_id else None,
                    }
                    break
        else:
            # Build disk configuration
            disk_config = {
                "disk": {
                    "size": size,
                    "name": name,
                    "country": country,
                    "cloudId": cloud_id,
                    "dataCenterId": data_center_id,
                },
                "provider": {"type": provider_type} if provider_type else {},
                "team": {"teamId": team_id} if team_id else None,
            }

        if not disk_config or not disk_config.get("disk") or not disk_config.get("provider"):
            console.print("[red]Error: Invalid disk configuration[/red]")
            raise typer.Exit(1)

        # Show configuration summary
        console.print("\n[bold]Disk Configuration Summary:[/bold]")
        console.print(f"Size: {size}GB")
        if name:
            console.print(f"Name: {name}")

        disk_data = disk_config.get("disk", {})
        if isinstance(disk_data, dict):
            if disk_data.get("country"):
                console.print(f"Country: {disk_data.get('country')}")
            if disk_data.get("cloudId"):
                console.print(f"Cloud ID: {disk_data.get('cloudId')}")
            if disk_data.get("dataCenterId"):
                console.print(f"Data Center ID: {disk_data.get('dataCenterId')}")

        provider_data = disk_config.get("provider", {})
        if isinstance(provider_data, dict) and provider_data.get("type"):
            console.print(f"Provider: {provider_data.get('type')}")

        if team_id:
            console.print(f"Team: {team_id}")

        if confirm_or_skip("\nDo you want to create this disk?", yes, default=True):
            try:
                # Create the disk with loading animation
                with console.status("[bold blue]Creating disk...", spinner="dots"):
                    disk = disks_client.create(disk_config)

                console.print(f"\n[green]Successfully created disk {disk.id}[/green]")
                console.print(
                    f"\n[blue]Use 'prime disks get {disk.id}' to check the disk status[/blue]"
                )
            except AttributeError:
                console.print(
                    "[red]Error: Failed to create disk - invalid API client configuration[/red]"
                )
                raise typer.Exit(1)
        else:
            console.print("\nDisk creation cancelled")
            raise typer.Exit(0)

    except APIError as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {str(e)}")
        import traceback

        traceback.print_exc()
        raise typer.Exit(1)


@app.command(no_args_is_help=True)
def update(
    disk_id: str,
    name: str = typer.Option(..., help="New name for the disk"),
) -> None:
    """Update a disk's name"""
    try:
        base_client = APIClient()
        disks_client = DisksClient(base_client)

        with console.status("[bold blue]Updating disk...", spinner="dots"):
            result = disks_client.update(disk_id, name)

        console.print(f"[green]Successfully updated disk {disk_id}[/green]")
        console.print(f"New name: {result.get('name', name)}")

    except APIError as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {str(e)}")
        import traceback

        traceback.print_exc()
        raise typer.Exit(1)


@app.command(no_args_is_help=True)
def terminate(
    disk_id: str,
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
) -> None:
    """Terminate a disk"""
    try:
        base_client = APIClient()
        disks_client = DisksClient(base_client)

        # Confirm deletion
        if not confirm_or_skip(f"Are you sure you want to delete disk {disk_id}?", yes):
            console.print("Deletion cancelled")
            raise typer.Exit(0)

        with console.status("[bold blue]Deleting disk...", spinner="dots"):
            response = disks_client.delete(disk_id)

        console.print(f"[green]Successfully deleted disk {disk_id}[/green]")
        console.print(f"Status: {response.status}")

    except APIError as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {str(e)}")
        import traceback

        traceback.print_exc()
        raise typer.Exit(1)
