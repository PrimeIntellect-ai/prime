"""Commands for managing Docker images in Prime Intellect registry."""

import json
import subprocess
from pathlib import Path

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
    no_cache: bool = typer.Option(False, "--no-cache", help="Build without using cache"),
):
    """
    Build and push a Docker image to Prime Intellect registry.

    Examples:
        prime images push myapp:v1.0.0
        prime images push myapp:latest --dockerfile custom.Dockerfile
        prime images push myapp:v1 --platform linux/arm64
    """
    try:
        # Check if docker is installed
        try:
            subprocess.run(["docker", "--version"], check=True, capture_output=True, text=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            console.print("[red]Error: Docker is not installed or not in PATH[/red]")
            console.print("Please install Docker: https://docs.docker.com/get-docker/")
            raise typer.Exit(1)

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

        # Get push token from backend
        console.print("[cyan]🔐 Authenticating with Prime Intellect registry...[/cyan]")
        try:
            response = client.request("POST", "/images/push-token")
            token_data = response
        except UnauthorizedError:
            console.print("[red]Error: Not authenticated. Please run 'prime login' first.[/red]")
            raise typer.Exit(1)
        except APIError as e:
            console.print(f"[red]Error: Failed to get push token: {e}[/red]")
            raise typer.Exit(1)

        registry_url = token_data["registry_url"]
        access_token = token_data["access_token"]
        user_namespace = token_data["user_namespace"]

        # Construct full image path
        full_image_path = f"{user_namespace}/{image_name}:{image_tag}"

        console.print("[green]✓[/green] Authenticated")
        console.print(f"[dim]Registry:[/dim] {registry_url}")
        console.print(f"[dim]Full image path:[/dim] {full_image_path}")
        console.print()

        # Check if Dockerfile exists
        dockerfile_path = Path(context) / dockerfile
        if not dockerfile_path.exists():
            console.print(f"[red]Error: Dockerfile not found at {dockerfile_path}[/red]")
            raise typer.Exit(1)

        # Build image
        console.print("[cyan]📦 Building image...[/cyan]")
        build_cmd = [
            "docker",
            "build",
            "-t",
            full_image_path,
            "-f",
            str(dockerfile_path),
        ]

        # Always specify platform for consistent builds across different architectures
        build_cmd.extend(["--platform", platform])

        if no_cache:
            build_cmd.append("--no-cache")

        build_cmd.append(context)

        console.print(f"[dim]$ {' '.join(build_cmd)}[/dim]")
        console.print()

        try:
            result = subprocess.run(build_cmd, check=True)
            if result.returncode != 0:
                raise subprocess.CalledProcessError(result.returncode, build_cmd)
        except subprocess.CalledProcessError as e:
            console.print(f"[red]❌ Build failed with exit code {e.returncode}[/red]")
            raise typer.Exit(1)

        console.print()
        console.print("[green]✓[/green] Image built successfully")
        console.print()

        # Docker login
        console.print("[cyan]🔐 Logging in to registry...[/cyan]")
        login_cmd = ["docker", "login", "-u", "_token", "--password-stdin", registry_url]

        try:
            subprocess.run(login_cmd, input=access_token.encode(), check=True, capture_output=True)
        except subprocess.CalledProcessError:
            console.print("[red]❌ Docker login failed[/red]")
            raise typer.Exit(1)

        console.print("[green]✓[/green] Logged in to registry")
        console.print()

        # Push image
        console.print("[cyan]⬆️  Pushing image to registry...[/cyan]")
        console.print("[dim]This may take a few minutes depending on image size[/dim]")
        console.print()

        push_cmd = ["docker", "push", full_image_path]
        console.print(f"[dim]$ {' '.join(push_cmd)}[/dim]")
        console.print()

        try:
            subprocess.run(push_cmd, check=True)
        except subprocess.CalledProcessError:
            console.print("[red]❌ Push failed[/red]")
            raise typer.Exit(1)

        console.print()
        console.print("[green]✓[/green] Image pushed successfully")
        console.print()

        # Get image details
        console.print("[cyan]📝 Registering image with backend...[/cyan]")

        try:
            # Get image digest and size using docker inspect
            inspect_cmd = ["docker", "inspect", full_image_path]
            inspect_result = subprocess.run(inspect_cmd, check=True, capture_output=True, text=True)
            inspect_data = json.loads(inspect_result.stdout)

            if inspect_data:
                image_info = inspect_data[0]
                repo_digests = image_info.get("RepoDigests", [])
                digest = repo_digests[0] if repo_digests else None
                size_bytes = image_info.get("Size", 0)
            else:
                digest = None
                size_bytes = None

        except (subprocess.CalledProcessError, json.JSONDecodeError, IndexError):
            console.print("[yellow]⚠️  Could not get image details, continuing...[/yellow]")
            digest = None
            size_bytes = None

        # Register with backend
        try:
            register_response = client.request(
                "POST",
                "/images/register",
                json={
                    "image_name": image_name,
                    "image_tag": image_tag,
                    "digest": digest,
                    "size_bytes": size_bytes,
                },
            )
            result = register_response
        except APIError as e:
            console.print(
                f"[yellow]⚠️  Warning: Failed to register image with backend: {e}[/yellow]"
            )
            console.print(
                "[yellow]Image was pushed successfully but may not appear in "
                "'prime images list'[/yellow]"
            )
            raise typer.Exit(0)

        console.print("[green]✓[/green] Image registered")
        console.print()
        console.print("[bold green]✅ Success![/bold green]")
        console.print()
        console.print(f"[bold]Image:[/bold] {result['full_image_path']}")
        if size_bytes:
            size_mb = size_bytes / 1024 / 1024
            console.print(f"[bold]Size:[/bold] {size_mb:.2f} MB")
        console.print()
        console.print("[bold]To use in a sandbox:[/bold]")
        console.print(f"  prime sandbox create {result['full_image_path']}")
        console.print()

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
            console.print("[yellow]No images found.[/yellow]")
            console.print("Push an image with: [bold]prime images push <name>:<tag>[/bold]")
            return

        if output == "json":
            console.print(json.dumps(data, indent=2))
            return

        # Table output
        table = Table(title="Your Docker Images")
        table.add_column("Image Name", style="cyan")
        table.add_column("Tag", style="green")
        table.add_column("Size", justify="right")
        table.add_column("Pushed", style="dim")

        for img in data["images"]:
            size_mb = ""
            if img.get("size_bytes"):
                size_mb = f"{img['size_bytes'] / 1024 / 1024:.1f} MB"

            # Format timestamp
            from datetime import datetime

            try:
                pushed_dt = datetime.fromisoformat(img["pushed_at"].replace("Z", "+00:00"))
                pushed_str = pushed_dt.strftime("%Y-%m-%d %H:%M")
            except Exception:
                pushed_str = img["pushed_at"]

            table.add_row(img["image_name"], img["image_tag"], size_mb, pushed_str)

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
