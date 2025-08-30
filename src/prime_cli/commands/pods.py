import hashlib
import json
import os
import subprocess
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

from ..api.availability import AvailabilityClient, GPUAvailability
from ..api.client import APIClient, APIError
from ..api.pods import Pod, PodsClient, PodStatus
from ..config import Config
from ..helper.short_id import generate_short_id
from ..utils import (
    confirm_or_skip,
    format_ip_display,
    human_age,
    output_data_as_json,
    status_color,
    validate_output_format,
)
from ..utils.display import POD_STATUS_COLORS

app = typer.Typer(help="Manage compute pods")
console = Console()
config = Config()




def _format_pod_for_status(status: PodStatus, pod_details: Pod) -> Dict[str, Any]:
    """Format pod status data for display (both table and JSON)"""
    # Display status with installation state consideration
    display_status = status.status
    if (
        status.status == "ACTIVE"
        and status.installation_progress is not None
        and status.installation_progress < 100
    ):
        display_status = "INSTALLING"

    created_at = datetime.fromisoformat(pod_details.created_at.replace("Z", "+00:00"))
    created_timestamp = created_at.strftime("%Y-%m-%d %H:%M:%S UTC")

    # Build basic status data
    status_data: Dict[str, Any] = {
        "id": pod_details.id,
        "status": display_status,
        "name": pod_details.name,
        "team_id": pod_details.team_id,
        "provider": status.provider_type,
        "gpu": f"{pod_details.gpu_type} x{pod_details.gpu_count}",
        "image": pod_details.environment_type,
        "created_at": created_timestamp,
        "ip": format_ip_display(status.ip),
        "ssh": format_ip_display(status.ssh_connection),
    }

    # Add optional fields
    if status.cost_per_hr:
        status_data["cost_per_hour"] = status.cost_per_hr

    if pod_details.installation_status:
        status_data["installation_status"] = pod_details.installation_status

    if status.installation_progress is not None:
        status_data["installation_progress"] = status.installation_progress

    if status.installation_failure:
        status_data["installation_error"] = status.installation_failure

    if status.prime_port_mapping:
        status_data["port_mappings"] = [
            {
                "protocol": port.protocol,
                "external": port.external,
                "internal": port.internal,
                "description": port.description,
                "used_by": port.used_by,
            }
            for port in status.prime_port_mapping
        ]

    if pod_details.attached_resources:
        status_data["attached_resources"] = [
            {
                "id": str(resource.id),
                "type": resource.type,
                "status": resource.status,
                "size": resource.size,
                "mount_path": resource.mount_path,
            }
            for resource in pod_details.attached_resources
        ]

    return status_data


def _format_pod_for_list(pod: Pod) -> Dict[str, Any]:
    """Format pod data for list display (both table and JSON)"""
    display_status = pod.status
    if pod.status == "ACTIVE" and pod.installation_status != "FINISHED":
        display_status = "INSTALLING"

    created_at = datetime.fromisoformat(pod.created_at.replace("Z", "+00:00"))
    created_timestamp = created_at.strftime("%Y-%m-%d %H:%M:%S UTC")
    age = human_age(created_at)

    return {
        "id": pod.id,
        "name": pod.name,
        "gpu": f"{pod.gpu_type} x{pod.gpu_count}",
        "status": display_status,
        "created_at": created_timestamp,  # For JSON output
        "age": age,  # For table output
    }


@app.command()
def list(
    limit: int = typer.Option(100, help="Maximum number of pods to list"),
    offset: int = typer.Option(0, help="Number of pods to skip"),
    watch: bool = typer.Option(False, "--watch", "-w", help="Watch pods list in real-time"),
    output: str = typer.Option("table", "--output", "-o", help="Output format: table or json"),
) -> None:
    """List your running pods"""
    validate_output_format(output, console)

    if watch and output == "json":
        console.print("[red]Error: --watch mode is not compatible with --output=json[/red]")
        raise typer.Exit(1)

    try:
        # Create API clients
        base_client = APIClient()
        pods_client = PodsClient(base_client)

        last_pods_hash = None

        while True:
            pods_list = pods_client.list(offset=offset, limit=limit)

            current_pods_hash = hashlib.md5(
                json.dumps([pod.model_dump() for pod in pods_list.data], sort_keys=True).encode()
            ).hexdigest()

            # Only update display if data changed or first run
            if current_pods_hash != last_pods_hash:
                # Clear screen if watching
                if watch:
                    os.system("cls" if os.name == "nt" else "clear")

                # Sort pods by created_at (oldest first, like sandbox list)
                sorted_pods = sorted(
                    pods_list.data,
                    key=lambda pod: datetime.fromisoformat(pod.created_at.replace("Z", "+00:00")),
                )

                if output == "json":
                    # Output as JSON with timestamp (for automation)
                    pods_data = []
                    for pod in sorted_pods:
                        pod_data = _format_pod_for_list(pod)
                        # For JSON, use timestamp instead of age
                        json_pod = {
                            "id": pod_data["id"],
                            "name": pod_data["name"],
                            "gpu": pod_data["gpu"],
                            "status": pod_data["status"],
                            "created_at": pod_data["created_at"],
                        }
                        pods_data.append(json_pod)

                    output_data = {
                        "pods": pods_data,
                        "total_count": pods_list.total_count,
                        "offset": offset,
                        "limit": limit,
                    }
                    output_data_as_json(output_data, console)
                else:
                    # Create display table
                    table = Table(
                        title=f"Compute Pods (Total: {pods_list.total_count})",
                        show_lines=True,
                    )
                    table.add_column("ID", style="cyan", no_wrap=True)
                    table.add_column("Name", style="blue")
                    table.add_column("GPU", style="green")
                    table.add_column("Status", style="yellow")
                    table.add_column("Age", style="blue")

                    # Add rows for each pod using shared formatting
                    for pod in sorted_pods:
                        pod_data = _format_pod_for_list(pod)

                        pod_status_color = status_color(pod_data["status"], POD_STATUS_COLORS)

                        table.add_row(
                            pod_data["id"],
                            pod_data["name"] or "N/A",
                            pod_data["gpu"],
                            Text(pod_data["status"], style=pod_status_color),
                            pod_data["age"],
                        )

                    console.print(table)

                # Update hash after displaying
            if not watch:
                console.print(
                    "\n[blue]Use 'prime pods status <pod-id>' to "
                    "see detailed information about a specific pod[/blue]"
                )

                # If there are more pods, show a message
                if pods_list.total_count > offset + limit:
                    remaining = pods_list.total_count - (offset + limit)
                    console.print(
                        f"\n[yellow]Showing {limit} of {pods_list.total_count} pods. "
                        f"Use --offset {offset + limit} to see the next "
                        f"{min(limit, remaining)} pods.[/yellow]"
                    )

                break
            else:
                # Only print the message when we're not repeating due to unchanged data
                if current_pods_hash != last_pods_hash or last_pods_hash is None:
                    console.print("\n[dim]Press Ctrl+C to exit watch mode[/dim]")
                last_pods_hash = current_pods_hash
                try:
                    # Wait before refreshing
                    import time

                    time.sleep(5)
                except KeyboardInterrupt:
                    # Clear the progress dots on exit
                    if current_pods_hash == last_pods_hash:
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


@app.command()
def status(
    pod_id: str,
    output: str = typer.Option("table", "--output", "-o", help="Output format: table or json"),
) -> None:
    """Get detailed status of a specific pod"""
    validate_output_format(output, console)

    try:
        base_client = APIClient()
        pods_client = PodsClient(base_client)

        # Get both pod status and details
        statuses = pods_client.get_status([pod_id])

        pod_details = pods_client.get(pod_id)

        if not statuses:
            console.print(f"[red]No status found for pod {pod_id}[/red]")
            raise typer.Exit(1)

        status = statuses[0]

        if output == "json":
            # Output as JSON using shared formatting
            status_data = _format_pod_for_status(status, pod_details)
            output_data_as_json(status_data, console)
        else:
            # Create display table
            status_data = _format_pod_for_status(status, pod_details)

            table = Table(title=f"Pod Status: {pod_id}")
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="white")

            table.add_row(
                "Status",
                Text(
                    status_data["status"],
                    style="green" if status_data["status"] == "ACTIVE" else "yellow",
                ),
            )

            # Basic pod info
            table.add_row("Name", status_data["name"] or "N/A")
            table.add_row("Team", status_data["team_id"] or "Personal")
            table.add_row("Provider", status_data["provider"])
            table.add_row("GPU", status_data["gpu"])
            table.add_row("Image", status_data["image"])

            # Cost info if available
            if "cost_per_hour" in status_data:
                table.add_row("Cost per Hour", f"${status_data['cost_per_hour']:.3f}")

            table.add_row("Created", status_data["created_at"])

            # Connection details
            table.add_row("IP", status_data["ip"])
            table.add_row("SSH", status_data["ssh"])

            # Installation status
            if "installation_status" in status_data:
                table.add_row("Installation Status", status_data["installation_status"])
            if "installation_progress" in status_data:
                table.add_row("Installation Progress", f"{status_data['installation_progress']}%")
            if "installation_error" in status_data:
                table.add_row(
                    "Installation Error", Text(status_data["installation_error"], style="red")
                )

            # Port mappings
            if "port_mappings" in status_data:
                ports = "\n".join(
                    [
                        f"{port['protocol']}:{port['external']}->{port['internal']} "
                        f"({port['description'] + ' - ' if port['description'] else ''}"
                        f"{port['used_by'] or 'unknown'})"
                        for port in status_data["port_mappings"]
                    ]
                )
                table.add_row("Port Mappings", ports)

            console.print(table)

            # Display attached resources in a separate table if they exist
            if "attached_resources" in status_data:
                resource_table = Table(title="Attached Resources")
                resource_table.add_column("ID", style="cyan")
                resource_table.add_column("Type", style="white")
                resource_table.add_column("Status", style="white")
                resource_table.add_column("Size", style="white")
                resource_table.add_column("Mount Path", style="white")

                for resource in status_data["attached_resources"]:
                    status_style = "green" if resource["status"] == "ACTIVE" else "yellow"
                    resource_table.add_row(
                        resource["id"],
                        resource["type"] or "N/A",
                        Text(resource["status"] or "N/A", style=status_style),
                        str(resource["size"]) + "GB" if resource["size"] else "N/A",
                        resource["mount_path"] or "N/A",
                    )

                console.print("\n")  # Add spacing between tables
                console.print(resource_table)

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
    image: Optional[str] = typer.Option(
        None, help="Image name or 'custom_template' when using custom template ID"
    ),
    custom_template_id: Optional[str] = typer.Option(None, help="Custom template ID"),
    team_id: Optional[str] = typer.Option(None, help="Team ID to use for the pod"),
    env: Optional[List[str]] = typer.Option(
        None,
        help="Environment variables to set in the pod. Can be specified multiple times "
        "using --env KEY=value --env KEY2=value2",
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
) -> None:
    """Create a new pod with an interactive setup process"""
    env_vars = []
    if env:
        for env_var in env:
            key, value = env_var.split("=")
            env_vars.append({"key": key, "value": value})

    try:
        # Validate custom template usage
        if custom_template_id and not image == "custom_template":
            console.print(
                "[red]Error: Must set image='custom_template' when using custom_template_id[/red]"  # noqa: E501
            )
            raise typer.Exit(1)
        if image == "custom_template" and not custom_template_id:
            console.print(
                "[red]Error: Must provide custom_template_id when image='custom_template'[/red]"  # noqa: E501
            )
            raise typer.Exit(1)

        base_client = APIClient()
        availability_client = AvailabilityClient(base_client)
        pods_client = PodsClient(base_client)

        selected_gpu = None

        # Get availability info
        with console.status("[bold blue]Loading available GPU configurations...", spinner="dots"):
            availabilities = availability_client.get()

        if env_vars:
            filtered_availabilities = {}
            for availability_type, gpus in availabilities.items():
                filtered_gpus = []
                for gpu in gpus:
                    if gpu.provider == "runpod":
                        continue
                    if gpu.images:
                        # Filter out ubuntu image
                        filtered_images = [img for img in gpu.images if img != "ubuntu_22_cuda_12"]
                        if len(filtered_images) > 0:
                            gpu.images = filtered_images
                            filtered_gpus.append(gpu)

                if filtered_gpus:
                    filtered_availabilities[availability_type] = filtered_gpus

            availabilities = filtered_availabilities

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
                gpu_types = sorted(
                    [gpu_type for gpu_type, gpus in availabilities.items() if len(gpus) > 0]
                )
                for idx, gpu_type_option in enumerate(gpu_types, 1):
                    console.print(f"{idx}. {gpu_type_option}")

                gpu_type_idx = typer.prompt("Select GPU type number", type=int, default=1)
                if gpu_type_idx < 1 or gpu_type_idx > len(gpu_types):
                    console.print("[red]Invalid GPU type selection[/red]")
                    raise typer.Exit(1)
                gpu_type = gpu_types[gpu_type_idx - 1]

            def select_provider_from_configs(
                matching_configs: List[GPUAvailability],
            ) -> GPUAvailability:
                if not matching_configs:
                    raise ValueError("No matching GPU configurations found")

                # Sort by price
                matching_configs.sort(key=lambda x: x.prices.price if x.prices else float("inf"))

                seen_provider_types = set()
                unique_configs = []
                for gpu in matching_configs:
                    # Create unique key combining provider and spot status
                    provider_type = (gpu.provider, gpu.is_spot)
                    if provider_type not in seen_provider_types:
                        seen_provider_types.add(provider_type)
                        unique_configs.append(gpu)

                if len(unique_configs) > 1:
                    console.print("\n[bold]Available Providers:[/bold]")
                    for idx, gpu in enumerate(unique_configs, 1):
                        price = gpu.prices.price if gpu.prices else float("inf")
                        price_display = (
                            f"${round(float(price), 2)}/hr" if price != float("inf") else "N/A"
                        )
                        spot_display = " (spot)" if gpu.is_spot else ""
                        console.print(f"{idx}. {gpu.provider}{spot_display} ({price_display})")

                    provider_idx = typer.prompt(
                        "Select provider number",
                        type=int,
                        default=1,
                        show_default=False,
                    )
                    if provider_idx < 1 or provider_idx > len(unique_configs):
                        console.print("[red]Invalid provider selection[/red]")
                        raise typer.Exit(1)
                    selected_gpu = unique_configs[provider_idx - 1]
                    if not isinstance(selected_gpu, GPUAvailability):
                        raise TypeError("Selected GPU is not of type GPUAvailability")
                    return selected_gpu

                selected_gpu = unique_configs[0]
                if not isinstance(selected_gpu, GPUAvailability):
                    raise TypeError("Selected GPU is not of type GPUAvailability")
                return selected_gpu

            if not gpu_count:
                console.print(f"\n[bold]Available {gpu_type} Configurations:[/bold]")
                gpu_configs = availabilities.get(str(gpu_type), [])

                # Get unique GPU counts and find cheapest price for each count
                unique_configs: Dict[int, Tuple[GPUAvailability, float]] = {}
                for gpu in gpu_configs:
                    gpu_count = gpu.gpu_count
                    price = gpu.prices.price if gpu.prices else float("inf")

                    if gpu_count not in unique_configs or price < unique_configs[gpu_count][1]:
                        unique_configs[gpu_count] = (gpu, price)

                # Display unique configurations with their cheapest prices
                config_list = sorted(
                    [(count, gpu, price) for count, (gpu, price) in unique_configs.items()],
                    key=lambda x: x[0],
                )

                for idx, (count, gpu, price) in enumerate(config_list, 1):
                    price_display = (
                        f"${round(float(price), 2)}/hr" if price != float("inf") else "N/A"
                    )
                    console.print(f"{idx}. {count}x {gpu_type} ({price_display})")

                config_idx = typer.prompt(
                    "Select configuration number",
                    type=int,
                    default=1,
                    show_default=False,
                )
                if config_idx < 1 or config_idx > len(config_list):
                    console.print("[red]Invalid configuration selection[/red]")
                    raise typer.Exit(1)

                # Find all providers for selected configuration
                selected_count = config_list[config_idx - 1][0]
                matching_configs = [gpu for gpu in gpu_configs if gpu.gpu_count == selected_count]

                selected_gpu = select_provider_from_configs(matching_configs)
                cloud_id = selected_gpu.cloud_id
            else:
                # Find configuration matching GPU type and count
                matching_configs = [
                    gpu
                    for gpu in availabilities.get(str(gpu_type), [])
                    if gpu.gpu_count == gpu_count
                ]
                if not matching_configs:
                    console.print(f"[red]No configuration found for {gpu_count}x {gpu_type}[/red]")
                    raise typer.Exit(1)

                selected_gpu = select_provider_from_configs(matching_configs)
                cloud_id = selected_gpu.cloud_id

        if not selected_gpu:
            console.print("[red]No valid GPU configuration found[/red]")
            raise typer.Exit(1)

        if not name:
            while True:
                gpu_name = selected_gpu.gpu_type.lower().split("_")[0]
                default_name = f"{gpu_name}-{selected_gpu.gpu_count}"
                name = typer.prompt(
                    "Pod name (alphanumeric and dashes only, must contain at least 1 letter)",
                    default=default_name,
                )
                if (
                    name
                    and any(c.isalpha() for c in name)
                    and all(c.isalnum() or c == "-" for c in name)
                ):
                    break
                console.print(
                    "[red]Invalid name format. Use only letters, numbers and dashes. "
                    "Must contain at least 1 letter.[/red]"
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
                if min_disk is not None and disk_size is not None and disk_size < min_disk:
                    console.print(f"[red]Disk size must be at least {min_disk}GB[/red]")
                    raise typer.Exit(1)
                if max_disk is not None and disk_size is not None and disk_size > max_disk:
                    console.print(f"[red]Disk size must be at most {max_disk}GB[/red]")
                    raise typer.Exit(1)

        if not vcpus:
            min_vcpus = selected_gpu.vcpu.min_count
            max_vcpus = selected_gpu.vcpu.max_count
            default_vcpus = selected_gpu.vcpu.default_count
            if min_vcpus is None or max_vcpus is None or default_vcpus is None:
                vcpus = default_vcpus
            else:
                vcpus = typer.prompt(
                    f"Number of vCPUs (min: {min_vcpus}, max: {max_vcpus})",
                    default=default_vcpus,
                    type=int,
                )
                if vcpus is None or vcpus < min_vcpus or vcpus > max_vcpus:
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
                if memory is None or memory < min_memory or memory > max_memory:
                    console.print(
                        f"[red]Memory must be between {min_memory}GB and {max_memory}GB[/red]"
                    )
                    raise typer.Exit(1)

        available_images = selected_gpu.images

        if not image and available_images:
            if len(available_images) == 1:
                # If only one image available, use it directly
                image = available_images[0]
            else:
                # Show available images
                console.print("\n[bold]Available Images:[/bold]")
                for idx, img in enumerate(available_images):
                    console.print(f"{idx + 1}. {img}")

                # Prompt for image selection
                image_idx = typer.prompt(
                    "Select image number", type=int, default=1, show_default=False
                )

                if image_idx < 1 or image_idx > len(available_images):
                    console.print("[red]Invalid image selection[/red]")
                    raise typer.Exit(1)

                image = available_images[image_idx - 1]

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
                "country": None,
                "security": None,
                "jupyterPassword": None,
                "autoRestart": False,
                "customTemplateId": custom_template_id,
                "envVars": env_vars,
            },
            "provider": {"type": selected_gpu.provider} if selected_gpu.provider else {},
            "team": {
                "teamId": team_id,
            }
            if team_id
            else None,
        }

        # Show configuration summary
        console.print("\n[bold]Pod Configuration Summary:[/bold]")
        pod_dict = pod_config.get("pod", {})
        if isinstance(pod_dict, dict):
            for key, value in pod_dict.items():
                if value is not None:
                    if key == "provider":
                        continue
                    console.print(f"{key}: {value}")
        if isinstance(pod_config["provider"], dict) and isinstance(
            pod_config["provider"].get("type"), str
        ):
            console.print(f"provider: {pod_config['provider']['type']}")
        console.print(f"team: {team_id}")

        if confirm_or_skip("\nDo you want to create this pod?", yes, default=True):
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
def terminate(
    pod_id: str,
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
) -> None:
    """Terminate a pod"""
    try:
        base_client = APIClient()
        pods_client = PodsClient(base_client)

        # Confirm termination
        if not confirm_or_skip(f"Are you sure you want to terminate pod {pod_id}?", yes):
            console.print("Termination cancelled")
            raise typer.Exit(0)

        with console.status("[bold blue]Terminating pod...", spinner="dots"):
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


@app.command(name="connect")
@app.command(name="ssh")
def connect(pod_id: str) -> None:
    """SSH / connect to a pod using configured SSH key"""
    try:
        base_client = APIClient()
        pods_client = PodsClient(base_client)

        # Keep trying until SSH connection is available
        status_message = "[bold blue]Waiting for SSH connection to become available..."
        with console.status(status_message, spinner="dots"):
            while True:
                # Get pod status to check SSH connection details
                statuses = pods_client.get_status([pod_id])
                if not statuses:
                    console.print(f"[red]No status found for pod {pod_id}[/red]")
                    raise typer.Exit(1)

                status = statuses[0]
                if status.ssh_connection:
                    break

                # Wait before retrying
                time.sleep(5)  # Wait 5 seconds before retrying

        # Get SSH key path from config
        ssh_key_path = config.ssh_key_path
        if not os.path.exists(ssh_key_path):
            console.print(f"[red]SSH key not found at {ssh_key_path}[/red]")
            raise typer.Exit(1)

        console.print(f"[blue]Using SSH key:[/blue] {ssh_key_path}")
        console.print("[dim]To change SSH key path, use: prime config set-ssh-key-path[/dim]")

        ssh_conn = status.ssh_connection
        # Handle ssh_conn being either a string or list of strings
        connections: List[str] = []
        if isinstance(ssh_conn, List):
            # Filter out None values and convert to strings
            connections = [str(conn) for conn in ssh_conn if conn is not None]
        else:
            connections = [str(ssh_conn)] if ssh_conn else []

        if not connections:
            console.print("[red]No valid SSH connections available[/red]")
            raise typer.Exit(1)

        # If multiple connections available, let user choose
        connection_str: str
        if len(connections) > 1:
            console.print("\nMultiple nodes available. Please select one:")
            for idx, conn in enumerate(connections):
                console.print(f"[blue]{idx + 1}[/blue]) {conn}")

            choice = typer.prompt("Enter node number", type=int, default=1, show_default=False)

            if choice < 1 or choice > len(connections):
                console.print("[red]Invalid selection[/red]")
                raise typer.Exit(1)

            connection_str = connections[choice - 1]
        else:
            connection_str = connections[0]

        connection_parts = connection_str.split(" -p ")
        host = connection_parts[0]
        port = connection_parts[1] if len(connection_parts) > 1 else "22"

        ssh_command = [
            "ssh",
            "-i",
            ssh_key_path,
            "-o",
            "StrictHostKeyChecking=no",
            "-p",
            port,
            host,
        ]

        # Execute SSH command
        try:
            subprocess.run(ssh_command)
        except subprocess.CalledProcessError as e:
            console.print(f"[red]SSH connection failed: {str(e)}[/red]")
            raise typer.Exit(1)

    except APIError as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {str(e)}")
        import traceback

        traceback.print_exc()
        raise typer.Exit(1)
