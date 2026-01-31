"""Commands for managing GitHub Actions runners on pods."""

from datetime import datetime
from typing import Any, Dict, List, Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

from prime_cli.core import Config

from ..api.availability import AvailabilityClient, GPUAvailability
from ..api.pods import Pod, PodsClient
from ..api.runners import RunnersClient
from ..client import APIClient, APIError
from ..helper.short_id import generate_short_id
from ..utils import (
    confirm_or_skip,
    human_age,
    iso_timestamp,
    output_data_as_json,
    status_color,
    validate_output_format,
)
from ..utils.display import POD_STATUS_COLORS

app = typer.Typer(help="Manage GitHub Actions runners", no_args_is_help=True)
console = Console()
config = Config()


def _format_runner_for_display(pod: Pod) -> Dict[str, Any]:
    """Format runner pod data for display."""
    display_status = pod.status
    if pod.status == "ACTIVE" and pod.installation_status != "FINISHED":
        display_status = "INSTALLING"

    created_at = datetime.fromisoformat(pod.created_at.replace("Z", "+00:00"))
    created_timestamp = iso_timestamp(created_at)
    age = human_age(created_at)

    return {
        "id": pod.id,
        "name": pod.name,
        "gpu": f"{pod.gpu_type} x{pod.gpu_count}",
        "status": display_status,
        "created_at": created_timestamp,
        "age": age,
        "provider": pod.provider_type,
    }


@app.command()
def list(
    limit: int = typer.Option(100, help="Maximum number of runners to list"),
    output: str = typer.Option("table", "--output", "-o", help="Output format: table or json"),
) -> None:
    """List all GitHub Actions runner pods."""
    validate_output_format(output, console)

    try:
        base_client = APIClient()
        runners_client = RunnersClient(base_client)

        runner_pods = runners_client.list_runner_pods(limit=limit)

        # Sort by created_at (oldest first)
        sorted_pods = sorted(
            runner_pods,
            key=lambda pod: datetime.fromisoformat(pod.created_at.replace("Z", "+00:00")),
        )

        if output == "json":
            runners_data = []
            for pod in sorted_pods:
                runner_data = _format_runner_for_display(pod)
                runners_data.append(
                    {
                        "id": runner_data["id"],
                        "name": runner_data["name"],
                        "gpu": runner_data["gpu"],
                        "status": runner_data["status"],
                        "created_at": runner_data["created_at"],
                        "provider": runner_data["provider"],
                    }
                )
            output_data_as_json({"runners": runners_data, "count": len(runners_data)}, console)
        else:
            table = Table(
                title=f"GitHub Actions Runners (Total: {len(sorted_pods)})",
                show_lines=True,
            )
            table.add_column("ID", style="cyan", no_wrap=True)
            table.add_column("Name", style="blue")
            table.add_column("GPU", style="green")
            table.add_column("Status", style="yellow")
            table.add_column("Age", style="blue")

            for pod in sorted_pods:
                runner_data = _format_runner_for_display(pod)
                pod_status_color = status_color(runner_data["status"], POD_STATUS_COLORS)

                table.add_row(
                    runner_data["id"],
                    runner_data["name"] or "N/A",
                    runner_data["gpu"],
                    Text(runner_data["status"], style=pod_status_color),
                    runner_data["age"],
                )

            console.print(table)

            if not sorted_pods:
                console.print(
                    "\n[dim]No GitHub Actions runners found. "
                    "Use 'prime runner up' to create one.[/dim]"
                )
            else:
                console.print(
                    "\n[blue]Use 'prime runner down <pod-id>' to terminate a runner[/blue]"
                )

    except APIError as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {str(e)}")
        raise typer.Exit(1)


@app.command()
def up(
    repo: Optional[str] = typer.Option(
        None,
        "--repo",
        "-r",
        help="GitHub repository in format 'owner/repo' for repo-level runner",
    ),
    org: Optional[str] = typer.Option(
        None,
        "--org",
        "-O",
        help="GitHub organization name for org-level runner",
    ),
    token: Optional[str] = typer.Option(
        None,
        "--token",
        "-t",
        help="GitHub Actions runner registration token",
    ),
    labels: Optional[List[str]] = typer.Option(
        None,
        "--label",
        "-l",
        help="Additional labels for the runner. Can be specified multiple times.",
    ),
    name: Optional[str] = typer.Option(None, "--name", "-n", help="Name for the runner pod"),
    id: Optional[str] = typer.Option(None, help="Short ID from availability list"),
    gpu_type: Optional[str] = typer.Option(None, help="GPU type (e.g. A100, H100)"),
    gpu_count: Optional[int] = typer.Option(None, help="Number of GPUs"),
    disk_size: Optional[int] = typer.Option(None, help="Disk size in GB"),
    vcpus: Optional[int] = typer.Option(None, help="Number of vCPUs"),
    memory: Optional[int] = typer.Option(None, help="Memory in GB"),
    image: Optional[str] = typer.Option(None, help="Image to use"),
    team_id: Optional[str] = typer.Option(
        None, help="Team ID (uses config team_id if not specified)"
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
) -> None:
    """Spin up a GitHub Actions runner on a new pod.

    The runner will be configured to register with GitHub using the provided token.
    You can create either a repository-level runner (--repo) or an organization-level
    runner (--org).

    To get a runner registration token:
    - For repo: Go to Settings > Actions > Runners > New self-hosted runner
    - For org: Go to Organization Settings > Actions > Runners > New runner

    Example:
        prime runner up --repo owner/repo --token AXXXX...
        prime runner up --org my-org --token AXXXX... --gpu-type H100
    """
    # Validate repo/org options
    if repo and org:
        console.print("[red]Error: Cannot specify both --repo and --org[/red]")
        raise typer.Exit(1)

    if not repo and not org:
        console.print("[red]Error: Must specify either --repo or --org[/red]")
        raise typer.Exit(1)

    try:
        base_client = APIClient()
        availability_client = AvailabilityClient(base_client)
        runners_client = RunnersClient(base_client)

        selected_gpu: Optional[GPUAvailability] = None
        cloud_id: Optional[str] = None

        # Get availability info
        with console.status("[bold blue]Loading available GPU configurations...", spinner="dots"):
            availabilities = availability_client.get()

        if id:
            # Find by short ID
            for gpu_type_key, gpus in availabilities.items():
                for gpu in gpus:
                    if generate_short_id(gpu) == id:
                        selected_gpu = gpu
                        cloud_id = gpu.cloud_id
                        break
                if selected_gpu:
                    break

            if not selected_gpu:
                console.print(f"[red]No GPU configuration found with ID '{id}'[/red]")
                raise typer.Exit(1)
        else:
            # Interactive GPU selection
            if not gpu_type:
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

            # Get GPU count options
            gpu_configs = availabilities.get(str(gpu_type), [])
            if not gpu_configs:
                console.print(f"[red]No configurations found for {gpu_type}[/red]")
                raise typer.Exit(1)

            if not gpu_count:
                # Get unique GPU counts
                unique_counts: Dict[int, GPUAvailability] = {}
                for gpu in gpu_configs:
                    count = gpu.gpu_count
                    price = gpu.prices.price if gpu.prices else float("inf")
                    if count not in unique_counts or price < (
                        unique_counts[count].prices.price
                        if unique_counts[count].prices
                        else float("inf")
                    ):
                        unique_counts[count] = gpu

                config_list = sorted(unique_counts.items(), key=lambda x: x[0])

                console.print(f"\n[bold]Available {gpu_type} Configurations:[/bold]")
                for idx, (count, gpu) in enumerate(config_list, 1):
                    price = gpu.prices.price if gpu.prices else float("inf")
                    price_display = f"${round(float(price), 2)}/hr" if price != float("inf") else "N/A"
                    console.print(f"{idx}. {count}x {gpu_type} ({price_display})")

                config_idx = typer.prompt("Select configuration number", type=int, default=1)
                if config_idx < 1 or config_idx > len(config_list):
                    console.print("[red]Invalid configuration selection[/red]")
                    raise typer.Exit(1)

                gpu_count = config_list[config_idx - 1][0]

            # Find matching configs for selected count
            matching_configs = [gpu for gpu in gpu_configs if gpu.gpu_count == gpu_count]

            if not matching_configs:
                console.print(f"[red]No configuration found for {gpu_count}x {gpu_type}[/red]")
                raise typer.Exit(1)

            # Select provider if multiple
            matching_configs.sort(key=lambda x: x.prices.price if x.prices else float("inf"))

            if len(matching_configs) > 1:
                seen_providers = set()
                unique_provider_configs = []
                for gpu in matching_configs:
                    provider_key = (gpu.provider, gpu.is_spot)
                    if provider_key not in seen_providers:
                        seen_providers.add(provider_key)
                        unique_provider_configs.append(gpu)

                if len(unique_provider_configs) > 1:
                    console.print("\n[bold]Available Providers:[/bold]")
                    for idx, gpu in enumerate(unique_provider_configs, 1):
                        price = gpu.prices.price if gpu.prices else float("inf")
                        price_display = f"${round(float(price), 2)}/hr" if price != float("inf") else "N/A"
                        spot_display = " (spot)" if gpu.is_spot else ""
                        console.print(f"{idx}. {gpu.provider}{spot_display} ({price_display})")

                    provider_idx = typer.prompt("Select provider number", type=int, default=1)
                    if provider_idx < 1 or provider_idx > len(unique_provider_configs):
                        console.print("[red]Invalid provider selection[/red]")
                        raise typer.Exit(1)
                    selected_gpu = unique_provider_configs[provider_idx - 1]
                else:
                    selected_gpu = unique_provider_configs[0]
            else:
                selected_gpu = matching_configs[0]

            cloud_id = selected_gpu.cloud_id

        if not selected_gpu:
            console.print("[red]No valid GPU configuration found[/red]")
            raise typer.Exit(1)

        # Generate runner name if not provided
        if not name:
            import time

            timestamp = int(time.time()) % 10000
            gpu_name = selected_gpu.gpu_type.lower().split("_")[0]
            target = repo.replace("/", "-") if repo else org
            name = f"gha-runner-{target}-{gpu_name}-{timestamp}"
            # Ensure valid pod name format
            name = "".join(c if c.isalnum() or c == "-" else "-" for c in name)[:50]

        # Ensure name starts with gha-runner- prefix
        if not name.startswith("gha-runner-"):
            name = f"gha-runner-{name}"

        # Get disk size
        if not disk_size:
            disk_size = selected_gpu.disk.default_count or 100

        # Get vcpus
        if not vcpus:
            vcpus = selected_gpu.vcpu.default_count

        # Get memory
        if not memory:
            memory = selected_gpu.memory.default_count

        # Get image
        available_images = selected_gpu.images or []
        if not image and available_images:
            if len(available_images) == 1:
                image = available_images[0]
            else:
                console.print("\n[bold]Available Images:[/bold]")
                for idx, img in enumerate(available_images, 1):
                    console.print(f"{idx}. {img}")

                image_idx = typer.prompt("Select image number", type=int, default=1)
                if image_idx < 1 or image_idx > len(available_images):
                    console.print("[red]Invalid image selection[/red]")
                    raise typer.Exit(1)
                image = available_images[image_idx - 1]

        image = image or "pytorch_24"

        # Build labels list
        runner_labels = list(labels) if labels else []
        # Add GPU-related labels
        runner_labels.extend([f"gpu-{selected_gpu.gpu_type.lower()}", f"gpu-count-{gpu_count}"])

        # Show configuration summary
        console.print("\n[bold]Runner Configuration Summary:[/bold]")
        console.print(f"Name: {name}")
        console.print(f"Target: {'repo:' + repo if repo else 'org:' + (org or '')}")
        console.print(f"GPU: {selected_gpu.gpu_type} x{gpu_count}")
        console.print(f"Provider: {selected_gpu.provider}")
        console.print(f"Disk: {disk_size}GB")
        console.print(f"vCPUs: {vcpus}")
        console.print(f"Memory: {memory}GB")
        console.print(f"Image: {image}")
        console.print(f"Labels: {', '.join(runner_labels)}")
        if selected_gpu.prices:
            console.print(f"Price: ${selected_gpu.prices.price:.3f}/hr")

        if not token:
            console.print(
                "\n[yellow]Note: No runner token provided. "
                "You will need to manually register the runner after the pod starts.[/yellow]"
            )
            console.print(
                "[dim]Get a token from GitHub: Settings > Actions > Runners > New self-hosted runner[/dim]"
            )

        if confirm_or_skip("\nDo you want to create this runner?", yes, default=True):
            with console.status("[bold blue]Creating runner pod...", spinner="dots"):
                pod = runners_client.create_runner_pod(
                    cloud_id=cloud_id or selected_gpu.cloud_id,
                    gpu_type=selected_gpu.gpu_type,
                    socket=selected_gpu.socket or "",
                    gpu_count=gpu_count or selected_gpu.gpu_count,
                    provider=selected_gpu.provider or "prime",
                    name=name,
                    disk_size=disk_size,
                    vcpus=vcpus,
                    memory=memory,
                    image=image,
                    data_center_id=selected_gpu.data_center,
                    team_id=team_id,
                    runner_token=token,
                    runner_repo=repo,
                    runner_org=org,
                    runner_labels=runner_labels,
                )

            console.print(f"\n[green]Successfully created runner pod {pod.id}[/green]")
            console.print(f"[blue]Name: {pod.name}[/blue]")
            console.print(f"\n[blue]Use 'prime pods status {pod.id}' to check the pod status[/blue]")
            console.print(f"[blue]Use 'prime pods ssh {pod.id}' to connect to the pod[/blue]")

            if not token:
                console.print(
                    "\n[yellow]Remember to register the runner manually once the pod is ready:[/yellow]"
                )
                if repo:
                    console.print(
                        f"[dim]Visit: https://github.com/{repo}/settings/actions/runners/new[/dim]"
                    )
                elif org:
                    console.print(
                        f"[dim]Visit: https://github.com/organizations/{org}/settings/actions/runners/new[/dim]"
                    )
        else:
            console.print("\nRunner creation cancelled")
            raise typer.Exit(0)

    except typer.Abort:
        console.print("\n[yellow]Operation cancelled[/yellow]")
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
def down(
    pod_id: str = typer.Argument(..., help="Pod ID of the runner to terminate"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
) -> None:
    """Spin down (terminate) a GitHub Actions runner pod.

    Example:
        prime runner down abc123
        prime runner down abc123 --yes
    """
    try:
        base_client = APIClient()
        runners_client = RunnersClient(base_client)
        pods_client = PodsClient(base_client)

        # Get pod details to confirm it's a runner
        try:
            pod = pods_client.get(pod_id)
        except APIError:
            console.print(f"[red]Error: Pod {pod_id} not found[/red]")
            raise typer.Exit(1)

        # Show pod info
        console.print(f"\n[bold]Runner to terminate:[/bold]")
        console.print(f"ID: {pod.id}")
        console.print(f"Name: {pod.name or 'N/A'}")
        console.print(f"GPU: {pod.gpu_type} x{pod.gpu_count}")
        console.print(f"Status: {pod.status}")

        if not confirm_or_skip(f"\nAre you sure you want to terminate runner {pod_id}?", yes):
            console.print("Termination cancelled")
            raise typer.Exit(0)

        with console.status("[bold blue]Terminating runner...", spinner="dots"):
            runners_client.terminate_runner(pod_id)

        console.print(f"\n[green]Successfully terminated runner {pod_id}[/green]")

    except typer.Abort:
        console.print("\n[yellow]Operation cancelled[/yellow]")
        raise typer.Exit(0)
    except APIError as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {str(e)}")
        raise typer.Exit(1)


@app.command()
def status(
    pod_id: str = typer.Argument(..., help="Pod ID of the runner"),
    output: str = typer.Option("table", "--output", "-o", help="Output format: table or json"),
) -> None:
    """Get status of a GitHub Actions runner pod.

    Example:
        prime runner status abc123
    """
    validate_output_format(output, console)

    try:
        base_client = APIClient()
        pods_client = PodsClient(base_client)

        pod = pods_client.get(pod_id)
        statuses = pods_client.get_status([pod_id])

        if not statuses:
            console.print(f"[red]No status found for pod {pod_id}[/red]")
            raise typer.Exit(1)

        pod_status = statuses[0]

        # Build status data
        display_status = pod_status.status
        if pod_status.status == "ACTIVE" and (
            pod_status.installation_progress is not None and pod_status.installation_progress < 100
        ):
            display_status = "INSTALLING"

        created_at = datetime.fromisoformat(pod.created_at.replace("Z", "+00:00"))

        status_data: Dict[str, Any] = {
            "id": pod.id,
            "name": pod.name,
            "status": display_status,
            "gpu": f"{pod.gpu_type} x{pod.gpu_count}",
            "provider": pod_status.provider_type,
            "created_at": iso_timestamp(created_at),
            "age": human_age(created_at),
            "ssh": pod_status.ssh_connection,
            "ip": pod_status.ip,
        }

        if pod_status.installation_progress is not None:
            status_data["installation_progress"] = pod_status.installation_progress

        if pod_status.installation_failure:
            status_data["installation_error"] = pod_status.installation_failure

        if pod_status.cost_per_hr:
            status_data["cost_per_hour"] = pod_status.cost_per_hr

        if output == "json":
            output_data_as_json(status_data, console)
        else:
            table = Table(title=f"Runner Status: {pod_id}")
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="white")

            status_style = "green" if display_status == "ACTIVE" else "yellow"
            table.add_row("Status", Text(display_status, style=status_style))
            table.add_row("Name", pod.name or "N/A")
            table.add_row("GPU", status_data["gpu"])
            table.add_row("Provider", status_data["provider"])
            table.add_row("Created", status_data["created_at"])
            table.add_row("Age", status_data["age"])

            if "installation_progress" in status_data:
                table.add_row("Installation Progress", f"{status_data['installation_progress']}%")

            if "installation_error" in status_data:
                table.add_row(
                    "Installation Error", Text(status_data["installation_error"], style="red")
                )

            if "cost_per_hour" in status_data:
                table.add_row("Cost per Hour", f"${status_data['cost_per_hour']:.3f}")

            # Format SSH connection
            ssh_conn = status_data.get("ssh")
            if ssh_conn:
                if isinstance(ssh_conn, list):
                    ssh_display = "\n".join(str(c) for c in ssh_conn if c)
                else:
                    ssh_display = str(ssh_conn)
                table.add_row("SSH", ssh_display or "N/A")
            else:
                table.add_row("SSH", "Pending...")

            console.print(table)

            if display_status == "ACTIVE":
                console.print(f"\n[blue]Use 'prime pods ssh {pod_id}' to connect[/blue]")

    except APIError as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {str(e)}")
        raise typer.Exit(1)
