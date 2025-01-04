import typer
from typing import Optional, List, Union
from rich.console import Console
from rich.table import Table
from rich.text import Text
from datetime import datetime
from ..config import Config
from ..api.client import APIClient, APIError
from ..api.pods import PodsClient
from ..api.availability import AvailabilityClient
from ..helper.short_id import generate_short_id

app = typer.Typer(help="Manage compute pods")
console = Console()
config = Config()


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

        # Create display table
        table = Table(
            title=f"Compute Pods (Total: {pods_list.total_count})", show_lines=True
        )
        table.add_column("ID", style="cyan", no_wrap=True)
        table.add_column("Name", style="blue")
        table.add_column("GPU", style="green")
        table.add_column("Status", style="yellow")
        table.add_column("Created", style="blue")

        # Add rows for each pod
        for pod in pods_list.data:
            # Format status with color
            display_status = pod.status
            if pod.status == "ACTIVE" and pod.installation_status != "FINISHED":
                display_status = "INSTALLING"

            status_color = {
                "ACTIVE": "green",
                "PENDING": "yellow",
                "ERROR": "red",
                "INSTALLING": "yellow",
            }.get(display_status, "white")

            # Format created time
            created_at = datetime.fromisoformat(pod.created_at.replace("Z", "+00:00"))
            created_str = created_at.strftime("%Y-%m-%d %H:%M:%S UTC")

            table.add_row(
                pod.id,
                pod.name or "N/A",
                f"{pod.gpu_type} x{pod.gpu_count}",
                Text(display_status, style=status_color),
                created_str,
            )

        console.print(table)
        console.print(
            "\n[blue]Use 'prime pods status <pod-id>' to see detailed information about a specific pod[/blue]"
        )

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

        # Get pod details for additional info
        pod_details = pods_client.get(pod_id)

        # Create display table
        table = Table(title=f"Pod Status: {pod_id}")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="white")

        # Display status with installation state consideration
        display_status = status.status
        if (
            status.status == "ACTIVE"
            and status.installation_status
            and status.installation_status != "FINISHED"
        ):
            display_status = "INSTALLING"

        table.add_row(
            "Status",
            Text(
                display_status,
                style="green" if display_status == "ACTIVE" else "yellow",
            ),
        )

        # Basic pod info
        table.add_row("Name", pod_details.name or "N/A")
        table.add_row("Team", status.team_id or "Personal")
        table.add_row("Provider", status.provider_type)
        table.add_row("GPU", f"{pod_details.gpu_type} x{pod_details.gpu_count}")

        # Cost info if available
        if status.cost_per_hr:
            table.add_row("Cost per Hour", f"${status.cost_per_hr:.3f}")

        # Created time
        created_at = datetime.fromisoformat(
            pod_details.created_at.replace("Z", "+00:00")
        )
        table.add_row("Created", created_at.strftime("%Y-%m-%d %H:%M:%S UTC"))

        # Connection details
        table.add_row("IP", format_ip_display(status.ip))
        ssh_display = format_ip_display(status.ssh_connection)
        table.add_row("SSH", ssh_display)

        # Installation status
        if status.installation_status:
            table.add_row("Installation Status", status.installation_status)
        if status.installation_progress is not None:
            table.add_row("Installation Progress", f"{status.installation_progress}%")
        if status.installation_failure:
            table.add_row(
                "Installation Error", Text(status.installation_failure, style="red")
            )

        # Port mappings
        if status.prime_port_mapping:
            ports = "\n".join(
                [
                    f"{port.protocol}:{port.external}->{port.internal} "
                    f"({port.description + ' - ' if port.description else ''}"
                    f"{port.used_by or 'unknown'})"
                    for port in status.prime_port_mapping
                ]
            )
            table.add_row("Port Mappings", ports)

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
def create(
    id: Optional[str] = typer.Option(None, help="Short ID from availability list"),
    cloud_id: Optional[str] = typer.Option(None, help="Cloud ID from cloud provider"),
    gpu_type: Optional[str] = typer.Option(None, help="GPU type (e.g. A100, V100)"),
    gpu_count: Optional[int] = typer.Option(None, help="Number of GPUs"),
    name: Optional[str] = typer.Option(None, help="Name for the pod"),
    disk_size: Optional[int] = typer.Option(None, help="Disk size in GB"),
    vcpus: Optional[int] = typer.Option(None, help="Number of vCPUs"),
    memory: Optional[int] = typer.Option(None, help="Memory in GB"),
    image: Optional[str] = typer.Option(None, help="Custom image"),
    team_id: Optional[str] = typer.Option(None, help="Team ID to use for the pod"),
):
    """Create a new pod with an interactive setup process"""
    try:
        base_client = APIClient()
        availability_client = AvailabilityClient(base_client)
        pods_client = PodsClient(base_client)

        selected_gpu = None

        # Get availability info
        with console.status(
            "[bold blue]Loading available GPU configurations...", spinner="dots"
        ):
            availabilities = availability_client.get()

        if id or cloud_id:
            # Find the matching GPU configuration by ID or cloud_id
            for gpu_type_key, gpus in availabilities.items():
                for gpu in gpus:
                    if id and generate_short_id(gpu) == id:
                        selected_gpu = gpu
                        cloud_id = gpu.cloud_id
                        break
                    elif gpu.cloud_id == cloud_id:
                        selected_gpu = gpu
                        break
                if selected_gpu:
                    break

        else:
            # Interactive GPU selection if no ID provided
            if not gpu_type:
                # Show available GPU types
                console.print("\n[bold]Available GPU Types:[/bold]")
                gpu_types = sorted(availabilities.keys())
                for idx, gpu_type_option in enumerate(gpu_types, 1):
                    console.print(f"{idx}. {gpu_type_option}")

                # Show loading status after displaying options
                gpu_type_idx = typer.prompt(
                    "Select GPU type number", type=int, default=1
                )
                if gpu_type_idx < 1 or gpu_type_idx > len(gpu_types):
                    console.print("[red]Invalid GPU type selection[/red]")
                    raise typer.Exit(1)
                gpu_type = gpu_types[gpu_type_idx - 1]

            if not gpu_count:
                # Show available configurations for selected GPU type
                console.print(f"\n[bold]Available {gpu_type} Configurations:[/bold]")
                gpu_configs = availabilities.get(gpu_type, [])

                # Get unique GPU counts
                unique_configs = {}
                for gpu in gpu_configs:
                    if gpu.gpu_count not in unique_configs:
                        unique_configs[gpu.gpu_count] = gpu

                # Display unique configurations
                config_list = sorted(unique_configs.values(), key=lambda x: x.gpu_count)
                for idx, gpu in enumerate(config_list, 1):
                    price = (
                        gpu.prices.on_demand
                        if gpu.prices and gpu.prices.on_demand
                        else "N/A"
                    )
                    console.print(
                        f"{idx}. {gpu.gpu_count}x {gpu_type} (${round(float(price), 2) if price != 'N/A' else price}/hr)"
                    )

                config_idx = typer.prompt(
                    "Select configuration number", type=int, default=1
                )
                if config_idx < 1 or config_idx > len(config_list):
                    console.print("[red]Invalid configuration selection[/red]")
                    raise typer.Exit(1)

                # Find the best provider for selected configuration
                selected_count = config_list[config_idx - 1].gpu_count
                matching_configs = [
                    gpu for gpu in gpu_configs if gpu.gpu_count == selected_count
                ]

                # Sort by price and select cheapest
                selected_gpu = sorted(
                    matching_configs,
                    key=lambda x: x.prices.on_demand
                    if x.prices and x.prices.on_demand
                    else float("inf"),
                )[0]
                cloud_id = selected_gpu.cloud_id
            else:
                # Find configuration matching GPU type and count
                matching_configs = [
                    gpu
                    for gpu in availabilities.get(gpu_type, [])
                    if gpu.gpu_count == gpu_count
                ]
                if not matching_configs:
                    console.print(
                        f"[red]No configuration found for {gpu_count}x {gpu_type}[/red]"
                    )
                    raise typer.Exit(1)

                # Sort by price and select cheapest
                selected_gpu = sorted(
                    matching_configs,
                    key=lambda x: x.prices.on_demand
                    if x.prices and x.prices.on_demand
                    else float("inf"),
                )[0]
                cloud_id = selected_gpu.cloud_id

        if not selected_gpu:
            console.print("[red]No valid GPU configuration found[/red]")
            raise typer.Exit(1)

        # Interactive configuration
        if not name:
            while True:
                name = typer.prompt(
                    "Pod name (alphanumeric and dashes only, must contain at least 1 letter)",
                )
                if (
                    name
                    and any(c.isalpha() for c in name)
                    and all(c.isalnum() or c == "-" for c in name)
                ):
                    break
                console.print(
                    "[red]Invalid name format. Use only letters, numbers and dashes. Must contain at least 1 letter.[/red]"
                )

        gpu_count = selected_gpu.gpu_count

        if not disk_size:
            min_disk = selected_gpu.disk.min_count
            max_disk = selected_gpu.disk.max_count
            default_disk = selected_gpu.disk.default_count

            if min_disk is None or max_disk is None:
                disk_size = default_disk
            else:
                disk_size = typer.prompt(
                    f"Disk size in GB (min: {min_disk}, max: {max_disk})",
                    default=default_disk or min_disk,
                    type=int,
                )
                if disk_size < min_disk or disk_size > max_disk:
                    console.print(
                        f"[red]Disk size must be between {min_disk}GB and {max_disk}GB[/red]"
                    )
                    raise typer.Exit(1)

        if not vcpus:
            min_vcpus = selected_gpu.vcpu.min_count
            max_vcpus = selected_gpu.vcpu.max_count
            default_vcpus = selected_gpu.vcpu.default_count

            if min_vcpus is None or max_vcpus is None:
                vcpus = default_vcpus
            else:
                vcpus = typer.prompt(
                    f"Number of vCPUs (min: {min_vcpus}, max: {max_vcpus})",
                    default=default_vcpus,
                    type=int,
                )
                if vcpus < min_vcpus or vcpus > max_vcpus:
                    console.print(
                        f"[red]vCPU count must be between {min_vcpus} and {max_vcpus}[/red]"
                    )
                    raise typer.Exit(1)

        if not memory:
            min_memory = selected_gpu.memory.min_count
            max_memory = selected_gpu.memory.max_count
            default_memory = selected_gpu.memory.default_count

            if min_memory is None or max_memory is None:
                memory = default_memory
            else:
                memory = typer.prompt(
                    f"Memory in GB (min: {min_memory}, max: {max_memory})",
                    default=default_memory,
                    type=int,
                )
                if memory < min_memory or memory > max_memory:
                    console.print(
                        f"[red]Memory must be between {min_memory}GB and {max_memory}GB[/red]"
                    )
                    raise typer.Exit(1)

        if not image and selected_gpu.images:
            # Show available images
            console.print("\n[bold]Available Images:[/bold]")
            for idx, img in enumerate(selected_gpu.images):
                console.print(f"{idx + 1}. {img}")

            # Prompt for image selection
            image_idx = typer.prompt(
                "Select image number", type=int, default=1, show_default=False
            )

            if image_idx < 1 or image_idx > len(selected_gpu.images):
                console.print("[red]Invalid image selection[/red]")
                raise typer.Exit(1)

            image = selected_gpu.images[image_idx - 1]

        # Get team ID from config if not provided
        if not team_id:
            default_team_id = config.team_id
            options = ["Personal Account", "Custom Team ID"]
            if default_team_id:
                options.insert(1, f"Pre-selected Team ({default_team_id})")

            console.print("\n[bold]Select Team:[/bold]")
            for idx, opt in enumerate(options, 1):
                console.print(f"{idx}. {opt}")

            choice = typer.prompt("Enter choice", type=int, default=1)

            if choice < 1 or choice > len(options):
                console.print("[red]Invalid selection[/red]")
                raise typer.Exit(1)

            if options[choice - 1] == "Personal Account":
                team_id = None
            elif "Pre-selected Team" in options[choice - 1]:
                team_id = default_team_id
            else:
                team_id = typer.prompt("Enter team ID")

        # Create pod configuration
        pod_config = {
            "pod": {
                "name": name or None,
                "cloudId": cloud_id,
                "gpuType": selected_gpu.gpu_type,
                "socket": selected_gpu.socket,
                "gpuCount": gpu_count,
                "diskSize": disk_size,
                "vcpus": vcpus,
                "memory": memory,
                "image": image,
                "dataCenterId": selected_gpu.data_center,
                "maxPrice": None,
                "customTemplateId": None,
                "country": None,
                "security": None,
                "jupyterPassword": None,
                "autoRestart": False,
            },
            "provider": {"type": selected_gpu.provider}
            if selected_gpu.provider
            else {},
            "team": {
                "teamId": team_id,
            }
            if team_id
            else None,
        }

        # Show configuration summary
        console.print("\n[bold]Pod Configuration Summary:[/bold]")
        for key, value in pod_config["pod"].items():
            if value is not None:
                if key == "provider":
                    continue
                console.print(f"{key}: {value}")
        if pod_config["provider"]:
            console.print(f"provider: {pod_config['provider']['type']}")
        console.print(f"team: {team_id}")

        if typer.confirm("\nDo you want to create this pod?", default=True):
            try:
                # Create the pod with loading animation
                with console.status("[bold blue]Creating pod...", spinner="dots"):
                    pod = pods_client.create(pod_config)

                console.print(f"\n[green]Successfully created pod {pod.id}[/green]")
                console.print(
                    f"\n[blue]Use 'prime pods status {pod.id}' to check the pod status[/blue]"
                )
            except AttributeError:
                console.print(
                    "[red]Error: Failed to create pod - invalid API client configuration[/red]"
                )
                raise typer.Exit(1)
        else:
            console.print("\nPod creation cancelled")
            raise typer.Exit(0)

    except APIError as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {str(e)}")
        import traceback

        traceback.print_exc()
        raise typer.Exit(1)


@app.command()
def terminate(pod_id: str):
    """Terminate a pod"""
    try:
        base_client = APIClient()
        pods_client = PodsClient(base_client)

        # Confirm termination
        if not typer.confirm(f"Are you sure you want to terminate pod {pod_id}?"):
            console.print("Termination cancelled")
            raise typer.Exit(0)

        # Delete the pod
        pods_client.delete(pod_id)
        console.print(f"[green]Successfully terminated pod {pod_id}[/green]")

    except APIError as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {str(e)}")
        import traceback

        traceback.print_exc()
        raise typer.Exit(1)
