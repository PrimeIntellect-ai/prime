from typing import Any, Dict, List, Optional

import typer
from prime_core import APIClient, APIError
from rich.console import Console
from rich.table import Table
from rich.text import Text

from ..api.availability import AvailabilityClient, GPUAvailability
from ..helper.short_id import generate_short_id
from ..utils import output_data_as_json, status_color, validate_output_format
from ..utils.display import STOCK_STATUS_COLORS

app = typer.Typer(help="Check GPU availability and pricing", no_args_is_help=True)
console = Console()


def _format_availability_for_display(gpu_entry: Dict[str, Any]) -> Dict[str, Any]:
    """Format availability data for display (both table and JSON)"""
    gpu_type_display = (
        f"{gpu_entry['gpu_type']} (Spot)" if gpu_entry["is_spot"] else gpu_entry["gpu_type"]
    ).replace("_", " ")

    return {
        "id": gpu_entry["short_id"],
        "cloud_id": gpu_entry["cloud_id"],
        "gpu_type": gpu_type_display,
        "gpu_count": gpu_entry["gpu_count"],
        "socket": gpu_entry["socket"],
        "provider": gpu_entry["provider"],
        "location": gpu_entry["location"],
        "stock_status": gpu_entry["stock_status"]
        if isinstance(gpu_entry["stock_status"], str)
        else gpu_entry["stock_status"].plain,
        "price_per_hour": gpu_entry["price"],
        "price_value": gpu_entry["price_value"],
        "security": "community" if gpu_entry["security"] == "community_cloud" else "datacenter",
        "vcpus": str(gpu_entry["vcpu"]),
        "memory_gb": str(gpu_entry["memory"]),
        "disk_gb": str(gpu_entry["disk"]),
        "gpu_memory": gpu_entry["gpu_memory"],
        "is_spot": gpu_entry["is_spot"],
    }


@app.command()
def gpu_types() -> None:
    """List available GPU types"""
    try:
        # Create API clients
        base_client = APIClient()
        availability_client = AvailabilityClient(base_client)

        # Get availability data
        availability_data = availability_client.get()

        # Create display table
        table = Table(title="Available GPU Types")
        table.add_column("GPU Type", style="cyan")

        # Get unique GPU types
        gpu_types = sorted(availability_data.keys())

        for gpu_type in gpu_types:
            table.add_row(gpu_type.replace("_", " "))

        console.print(table)

    except APIError as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {str(e)}")
        import traceback

        traceback.print_exc()
        raise typer.Exit(1)


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
    provider: Optional[str] = typer.Option(
        None, help="Filter by provider (e.g., aws, azure, google)"
    ),
    group_similar: bool = typer.Option(
        True, help="Group similar configurations from same provider"
    ),
    output: str = typer.Option("table", "--output", "-o", help="Output format: table or json"),
) -> None:
    """List available GPU resources"""
    validate_output_format(output, console)

    try:
        # Create API clients
        base_client = APIClient()
        availability_client = AvailabilityClient(base_client)

        # Get availability data
        availability_data: Dict[str, List[GPUAvailability]] = availability_client.get(
            gpu_type=gpu_type, gpu_count=gpu_count, regions=regions
        )

        # Filter by provider if provided
        if provider:
            availability_data = {
                gpu_type: [gpu for gpu in gpus if gpu.provider == provider]
                for gpu_type, gpus in availability_data.items()
            }

        # Create display table
        table = Table(title="Available GPU Resources")
        table.add_column("ID", style="cyan", no_wrap=True)
        table.add_column("GPU Type", style="cyan")
        table.add_column("GPUs", style="cyan")
        table.add_column("Socket", style="blue")
        table.add_column("Provider", style="blue")
        table.add_column("Location", style="green")
        table.add_column("Stock", style="yellow")
        table.add_column("Price/Hr", style="magenta")
        table.add_column("Security", style="white")
        table.add_column("vCPUs", style="blue")
        table.add_column("RAM (GB)", style="blue")
        table.add_column("Disk (GB)", style="blue")

        all_gpus: List[Dict[str, Any]] = []
        for gpu_type, gpus in availability_data.items():
            for gpu in gpus:
                if socket and gpu.socket != socket:
                    continue

                price = gpu.prices.price
                price_str = f"${price:.2f}" if price != float("inf") else "N/A"

                stock_status_color = status_color(gpu.stock_status, STOCK_STATUS_COLORS)

                location = f"{gpu.country or 'N/A'}"

                short_id = generate_short_id(gpu)

                disk_info: str = str(gpu.disk.default_count)
                if gpu.disk.max_count is not None and gpu.disk.max_count != gpu.disk.default_count:
                    disk_info = f"{gpu.disk.default_count}+"

                gpu_data = {
                    "short_id": short_id,
                    "cloud_id": gpu.cloud_id,
                    "gpu_type": gpu_type,
                    "gpu_count": gpu.gpu_count,
                    "socket": gpu.socket or "N/A",
                    "provider": gpu.provider or "N/A",
                    "location": location,
                    "stock_status": Text(gpu.stock_status, style=stock_status_color),
                    "price": price_str,
                    "price_value": price,
                    "gpu_memory": gpu.gpu_memory,
                    "security": gpu.security or "N/A",
                    "vcpu": gpu.vcpu.default_count,
                    "memory": gpu.memory.default_count,
                    "is_spot": gpu.is_spot,
                    "disk": disk_info,
                }
                all_gpus.append(gpu_data)

        # Sort by price and remove duplicates based on short_id
        seen_ids = set()
        filtered_gpus: List[Dict[str, Any]] = []

        if group_similar:
            grouped_gpus: Dict[str, List[Dict[str, Any]]] = {}
            for gpu_config in sorted(all_gpus, key=lambda x: (x["price_value"], x["short_id"])):
                key = (
                    f"{gpu_config['provider']}_{gpu_config['gpu_type']}_{gpu_config['gpu_count']}_"
                    f"{gpu_config['socket']}_{gpu_config['location']}_{gpu_config['security']}_{gpu_config['price']}"
                )
                if key not in grouped_gpus:
                    grouped_gpus[key] = []
                grouped_gpus[key].append(gpu_config)

            # For each group, select representative configuration
            for group in grouped_gpus.values():
                if len(group) > 1:
                    # Use first ID but show ranges for variable specs
                    base = group[0].copy()
                    min_vcpu = min(g["vcpu"] for g in group)
                    max_vcpu = max(g["vcpu"] for g in group)
                    min_mem = min(g["memory"] for g in group)
                    max_mem = max(g["memory"] for g in group)
                    vcpu_range = f"{min_vcpu}-{max_vcpu}" if min_vcpu != max_vcpu else str(min_vcpu)
                    memory_range = f"{min_mem}-{max_mem}" if min_mem != max_mem else str(min_mem)
                    base["vcpu"] = vcpu_range
                    base["memory"] = memory_range
                    filtered_gpus.append(base)
                else:
                    filtered_gpus.append(group[0])
        else:
            for gpu_config in sorted(all_gpus, key=lambda x: (x["price_value"], x["short_id"])):
                if gpu_config["short_id"] not in seen_ids:
                    seen_ids.add(gpu_config["short_id"])
                    filtered_gpus.append(gpu_config)

        if output == "json":
            # Output as JSON using shared formatting
            json_data = [_format_availability_for_display(gpu_entry) for gpu_entry in filtered_gpus]
            output_data = {
                "gpu_resources": json_data,
                "total_count": len(json_data),
                "filters": {
                    "gpu_type": gpu_type,
                    "gpu_count": gpu_count,
                    "regions": regions,
                    "socket": socket,
                    "provider": provider,
                    "group_similar": group_similar,
                },
            }
            output_data_as_json(output_data, console)
        else:
            # Table output
            for gpu_entry in filtered_gpus:
                display_data = _format_availability_for_display(gpu_entry)

                # Convert stock_status back to Text for table display
                stock_status_color = status_color(display_data["stock_status"], STOCK_STATUS_COLORS)
                stock_text = Text(display_data["stock_status"], style=stock_status_color)

                table.add_row(
                    display_data["id"],
                    display_data["gpu_type"],
                    str(display_data["gpu_count"]),
                    display_data["socket"],
                    display_data["provider"],
                    display_data["location"],
                    stock_text,
                    display_data["price_per_hour"],
                    display_data["security"],
                    display_data["vcpus"],
                    display_data["memory_gb"],
                    display_data["disk_gb"],
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
                "\nThe command will guide you through an interactive ",
                "setup process to configure the pod.",
            )

    except APIError as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {str(e)}")
        import traceback

        traceback.print_exc()
        raise typer.Exit(1)
