import typer
from typing import Optional, List
from rich.console import Console
from rich.table import Table
from rich.text import Text
from ..helper.short_id import generate_short_id

from ..api.client import APIClient, APIError
from ..api.availability import AvailabilityClient

app = typer.Typer(help="Check GPU availability and pricing")
console = Console()


@app.command()
def list(
    gpu_type: Optional[str] = typer.Option(None, help="GPU type (e.g., H100_80GB)"),
    gpu_count: Optional[int] = typer.Option(None, help="Number of GPUs required"),
    regions: Optional[List[str]] = typer.Option(
        None, help="Filter by regions (e.g., united_states)"
    ),
    socket: Optional[str] = typer.Option(
        None, help="Filter by socket type (e.g., PCIe, SXM5, SXM4)"
    ),
):
    """List available GPU resources"""
    try:
        # Create API clients
        base_client = APIClient()
        availability_client = AvailabilityClient(base_client)

        # Get availability data
        availability_data = availability_client.get(
            gpu_type=gpu_type, gpu_count=gpu_count, regions=regions
        )

        # Create display table
        table = Table(title="Available GPU Resources")
        table.add_column("ID", style="cyan", no_wrap=True)  # Short ID column
        table.add_column("GPU Type", style="cyan")
        table.add_column("Socket", style="blue")
        table.add_column("Provider", style="blue")
        table.add_column("Location", style="green")
        table.add_column("Stock", style="yellow")
        table.add_column("Price/Hr", style="magenta")
        table.add_column("Memory (GB)", style="blue")
        table.add_column("Security", style="white")
        table.add_column("vCPUs", style="blue")
        table.add_column("RAM (GB)", style="blue")

        all_gpus = []
        for gpu_type, gpus in availability_data.items():
            for gpu in gpus:
                if socket and gpu.socket != socket:
                    continue

                # Get price based on provider
                if gpu.security == "community_cloud":
                    price = gpu.prices.community_price
                    price_str = f"${price:.2f}" if price else "N/A"
                else:
                    price = gpu.prices.on_demand
                    price_str = f"${price:.2f}" if price else "N/A"

                stock_color = {"High": "green", "Medium": "yellow", "Low": "red"}.get(
                    gpu.stock_status, "white"
                )

                location = (
                    f"{gpu.country or 'N/A'} - {gpu.data_center or 'N/A'}"
                    if gpu.country or gpu.data_center
                    else "N/A"
                )

                short_id = generate_short_id(gpu)
                gpu_data = {
                    "short_id": short_id,
                    "cloud_id": gpu.cloud_id,
                    "gpu_type": gpu_type,
                    "socket": gpu.socket or "N/A",
                    "provider": gpu.provider or "N/A",
                    "location": location,
                    "stock_status": Text(gpu.stock_status, style=stock_color),
                    "price": price_str,
                    "price_value": price or float("inf"),  # For sorting
                    "gpu_memory": gpu.gpu_memory,
                    "security": gpu.security or "N/A",
                    "vcpu": gpu.vcpu.default_count,
                    "memory": gpu.memory.default_count,
                }
                all_gpus.append(gpu_data)

        # Sort by price and remove duplicates based on short_id
        seen_ids = set()
        filtered_gpus = []
        for gpu in sorted(all_gpus, key=lambda x: x["price_value"]):
            if gpu["short_id"] not in seen_ids:
                seen_ids.add(gpu["short_id"])
                filtered_gpus.append(gpu)

        for gpu in filtered_gpus:
            table.add_row(
                gpu["short_id"],
                gpu["gpu_type"],
                gpu["socket"],
                gpu["provider"],
                gpu["location"],
                gpu["stock_status"],
                gpu["price"],
                str(gpu["gpu_memory"]),
                gpu["security"],
                str(gpu["vcpu"]),
                str(gpu["memory"]),
            )

        console.print(table)

        # Add deployment instructions
        console.print(
            "\n[bold blue]To deploy a pod with one of these configurations:[/bold blue]"
        )
        console.print("1. Copy either the ID or Cloud ID of your desired configuration")
        console.print("2. Run one of the following commands:")
        console.print("   [green]prime pods create --id <ID>[/green]")
        console.print(
            "\nThe command will guide you through an interactive setup process to configure the pod."
        )

    except APIError as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {str(e)}")
        import traceback

        traceback.print_exc()
        raise typer.Exit(1)
