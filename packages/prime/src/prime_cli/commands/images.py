"""Commands for managing Docker images in Prime Intellect registry."""

import json
import tarfile
import tempfile
from datetime import datetime
from pathlib import Path

import httpx
import typer
from prime_sandboxes import APIClient, APIError, Config, UnauthorizedError
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="Manage Docker images in Prime Intellect registry", no_args_is_help=True)
console = Console()

config = Config()


@app.command("push")
def push_image(
    image_reference: str = typer.Argument(
        ..., help="Image reference (e.g., 'myapp:v1.0.0' or 'myapp:latest')"
    ),
    dockerfile: str = typer.Option("Dockerfile", "--dockerfile", "-f", help="Path to Dockerfile"),
    context: str = typer.Option(".", "--context", "-c", help="Build context directory"),
    platform: str = typer.Option(
        "linux/amd64",
        "--platform",
        help="Target platform (defaults to linux/amd64 for Kubernetes compatibility)",
    ),
):
    """
    Build and push a Docker image to Prime Intellect registry.

    Examples:
        prime images push myapp:v1.0.0
        prime images push myapp:latest --dockerfile custom.Dockerfile
        prime images push myapp:v1 --platform linux/arm64
    """
    try:
        # Parse image reference
        if ":" in image_reference:
            image_name, image_tag = image_reference.rsplit(":", 1)
        else:
            image_name = image_reference
            image_tag = "latest"

        console.print(
            f"[bold blue]Building and pushing image:[/bold blue] {image_name}:{image_tag}"
        )
        console.print()

        # Initialize API client
        client = APIClient()

        # Check if Dockerfile exists
        dockerfile_path = Path(context) / dockerfile
        if not dockerfile_path.exists():
            console.print(f"[red]Error: Dockerfile not found at {dockerfile_path}[/red]")
            raise typer.Exit(1)

        # Create tar.gz of build context
        console.print("[cyan]Preparing build context...[/cyan]")
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp_file:
            tar_path = tmp_file.name

        try:
            with tarfile.open(tar_path, "w:gz") as tar:
                tar.add(context, arcname=".")

            tar_size_mb = Path(tar_path).stat().st_size / (1024 * 1024)
            console.print(f"[green]✓[/green] Build context packaged ({tar_size_mb:.2f} MB)")
            console.print()

            # Initialize build
            console.print("[cyan]Initiating build...[/cyan]")
            try:
                build_response = client.request(
                    "POST",
                    "/images/build",
                    json={
                        "image_name": image_name,
                        "image_tag": image_tag,
                        "dockerfile_path": dockerfile,
                        "platform": platform,
                    },
                )
            except UnauthorizedError:
                console.print(
                    "[red]Error: Not authenticated. Please run 'prime login' first.[/red]"
                )
                raise typer.Exit(1)
            except APIError as e:
                console.print(f"[red]Error: Failed to initiate build: {e}[/red]")
                raise typer.Exit(1)

            build_id = build_response.get("build_id")
            upload_url = build_response.get("upload_url")
            if not build_id or not upload_url:
                console.print(
                    "[red]Error: Invalid response from server "
                    "(missing build_id or upload_url)[/red]"
                )
                raise typer.Exit(1)
            full_image_path = build_response.get("fullImagePath") or f"{image_name}:{image_tag}"

            console.print("[green]✓[/green] Build initiated")
            console.print()

            # Upload build context to GCS
            console.print("[cyan]Uploading build context...[/cyan]")
            try:
                with open(tar_path, "rb") as f:
                    upload_response = httpx.put(
                        upload_url,
                        content=f,
                        headers={"Content-Type": "application/octet-stream"},
                        timeout=300.0,
                    )
                    upload_response.raise_for_status()
            except httpx.HTTPError as e:
                console.print(f"[red]Upload failed: {e}[/red]")
                raise typer.Exit(1)

            console.print("[green]✓[/green] Build context uploaded")
            console.print()

            # Start the build
            console.print("[cyan]Starting build...[/cyan]")
            try:
                client.request(
                    "POST",
                    f"/images/build/{build_id}/start",
                    json={"context_uploaded": True},
                )
            except APIError as e:
                console.print(f"[red]Error: Failed to start build: {e}[/red]")
                raise typer.Exit(1)

            console.print("[green]✓[/green] Build started")
            console.print()

            console.print("[bold green]Build initiated successfully![/bold green]")
            console.print()
            console.print(f"[bold]Build ID:[/bold] {build_id}")
            console.print(f"[bold]Image:[/bold] {full_image_path}")
            console.print()
            console.print("[cyan]Your image is being built.[/cyan]")
            console.print()
            console.print("[bold]Check build status:[/bold]")
            console.print("  prime images list")
            console.print()
            console.print(
                "[dim]The build typically takes a few minutes depending on image complexity.[/dim]"
            )
            console.print(
                "[dim]Once completed, you can use it with: "
                f"prime sandbox create {full_image_path}[/dim]"
            )
            console.print()

        finally:
            # Clean up temporary tar file
            try:
                Path(tar_path).unlink()
            except Exception:
                pass

    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user[/yellow]")
        raise typer.Exit(1)


@app.command("list")
def list_images(
    output: str = typer.Option("table", "--output", "-o", help="Output format (table or json)"),
):
    """
    List all images you've pushed to Prime Intellect registry.

    Examples:
        prime images list
        prime images list --output json
    """
    try:
        client = APIClient()

        response = client.request("GET", "/images")
        data = response

        if not data["images"]:
            console.print("[yellow]No images or builds found.[/yellow]")
            console.print("Push an image with: [bold]prime images push <name>:<tag>[/bold]")
            return

        if output == "json":
            console.print(json.dumps(data, indent=2))
            return

        # Table output
        table = Table(title="Your Docker Images")
        table.add_column("Image Reference", style="cyan")
        table.add_column("Status", justify="center")
        table.add_column("Size", justify="right")
        table.add_column("Created", style="dim")

        for img in data["images"]:
            # Status with color coding
            status = img.get("status", "UNKNOWN")
            if status == "COMPLETED":
                status_display = "[green]Ready[/green]"
            elif status == "BUILDING":
                status_display = "[yellow]Building[/yellow]"
            elif status == "PENDING":
                status_display = "[blue]Pending[/blue]"
            elif status == "FAILED":
                status_display = "[red]Failed[/red]"
            elif status == "CANCELLED":
                status_display = "[dim]Cancelled[/dim]"
            else:
                status_display = f"[dim]{status}[/dim]"

            # Size
            size_mb = ""
            if img.get("sizeBytes"):
                size_mb = f"{img['sizeBytes'] / 1024 / 1024:.1f} MB"

            # Date - use pushedAt for completed images, createdAt for builds
            try:
                if img.get("pushedAt"):
                    date_dt = datetime.fromisoformat(img["pushedAt"].replace("Z", "+00:00"))
                else:
                    date_dt = datetime.fromisoformat(img["createdAt"].replace("Z", "+00:00"))
                date_str = date_dt.strftime("%Y-%m-%d %H:%M")
            except Exception:
                date_str = img.get("pushedAt") or img.get("createdAt", "")

            # Image reference
            image_ref = (
                img.get("fullImagePath")
                or f"{img.get('imageName', 'unknown')}:{img.get('imageTag', 'latest')}"
            )

            table.add_row(image_ref, status_display, size_mb, date_str)

        console.print()
        console.print(table)
        console.print()
        console.print(f"[dim]Total: {data['total']} image(s)[/dim]")
        console.print()

    except UnauthorizedError:
        console.print("[red]Error: Not authenticated. Please run 'prime login' first.[/red]")
        raise typer.Exit(1)
    except APIError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command("delete")
def delete_image(
    image_reference: str = typer.Argument(
        ..., help="Image reference to delete (e.g., 'myapp:v1.0.0')"
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
):
    """
    Delete an image from your registry.

    Note: This removes the database record but does not delete the actual
    image from Google Artifact Registry.

    Examples:
        prime images delete myapp:v1.0.0
        prime images delete myapp:latest --yes
    """
    try:
        # Parse image reference
        if ":" not in image_reference:
            console.print(
                "[red]Error: Image reference must include a tag (e.g., myapp:latest)[/red]"
            )
            raise typer.Exit(1)

        image_name, image_tag = image_reference.rsplit(":", 1)

        if not yes:
            confirm = typer.confirm(f"Are you sure you want to delete {image_name}:{image_tag}?")
            if not confirm:
                console.print("[yellow]Cancelled[/yellow]")
                raise typer.Exit(0)

        client = APIClient()

        client.request("DELETE", f"/images/{image_name}/{image_tag}")
        console.print(f"[green]✓[/green] Deleted {image_name}:{image_tag}")

    except UnauthorizedError:
        console.print("[red]Error: Not authenticated. Please run 'prime login' first.[/red]")
        raise typer.Exit(1)
    except APIError as e:
        if "404" in str(e):
            console.print(f"[red]Error: Image {image_reference} not found[/red]")
        else:
            console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)
