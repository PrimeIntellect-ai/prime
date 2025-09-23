import hashlib
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import httpx
import toml
import typer
from rich.console import Console
from rich.table import Table

from ..api.client import APIClient, APIError
from ..utils import output_data_as_json, validate_output_format

app = typer.Typer(help="Manage verifiers environments")

# Force interactive mode for known terminal environments
force_interactive = bool(
    os.environ.get("TERM")
    or os.environ.get("TMUX")
    or os.environ.get("SSH_TTY")
    or os.environ.get("VSCODE_TERM_PROFILE")
)
console = Console(force_terminal=force_interactive if force_interactive else None)

# Constants
MAX_FILES_TO_SHOW = 10
DEFAULT_HASH_LENGTH = 8
DEFAULT_LIST_LIMIT = 20


def download_with_progress(
    url: str, output_file: str, description: str, headers: dict | None = None
) -> None:
    """Download a file with a progress indicator."""
    import sys

    with httpx.stream("GET", url, headers=headers or {}, timeout=120.0) as resp:
        resp.raise_for_status()

        total_header = resp.headers.get("Content-Length")
        total_bytes = int(total_header) if total_header and total_header.isdigit() else None

        with open(output_file, "wb") as f:
            if total_bytes:
                # Show size info
                size_mb = total_bytes / (1024 * 1024)
                console.print(f"{description} [dim]({size_mb:.1f} MB)[/dim]")

                # Download with progress bar using sys.stdout for immediate flush
                downloaded = 0

                for chunk in resp.iter_bytes(chunk_size=8192):
                    if not chunk:
                        continue
                    f.write(chunk)
                    downloaded += len(chunk)

                    # Calculate progress
                    progress = downloaded / total_bytes
                    bar_length = 30
                    filled = int(bar_length * progress)
                    bar = "█" * filled + "░" * (bar_length - filled)
                    pct = int(progress * 100)
                    downloaded_mb = downloaded / (1024 * 1024)

                    # Write progress directly to stdout with immediate flush
                    sys.stdout.write(
                        f"\r  [{bar}] {pct:3d}% ({downloaded_mb:.1f}/{size_mb:.1f} MB)"
                    )
                    sys.stdout.flush()

                sys.stdout.write("\n")
                sys.stdout.flush()
                console.print("[green]✓[/green] Download complete")
            else:
                # No content length, just show message
                console.print(f"{description}...")
                for chunk in resp.iter_bytes(chunk_size=8192):
                    if not chunk:
                        continue
                    f.write(chunk)
                console.print("[green]✓[/green] Download complete")


def upload_with_progress(
    url: str, content: bytes, description: str, headers: dict | None = None, timeout: float = 300.0
) -> httpx.Response:
    """Upload content with a progress indicator."""
    total_bytes = len(content)
    size_mb = total_bytes / (1024 * 1024)

    # Show upload info
    console.print(f"{description} [dim]({size_mb:.1f} MB)[/dim]")
    console.print("[dim]Processing on server...[/dim]", end="")

    response = httpx.put(url, content=content, headers=headers or {}, timeout=timeout)

    console.print("\r[green]✓[/green] Upload complete              ")
    return response


def should_include_file_in_archive(file_path: Path, base_path: Path) -> bool:
    """Determine if a file should be included in the archive based on filtering rules."""
    if not file_path.is_file():
        return False

    # Skip hidden files
    if file_path.name.startswith("."):
        return False

    # Skip files in __pycache__ directories
    if "__pycache__" in str(file_path.relative_to(base_path)):
        return False

    return True


def should_include_directory_in_archive(dir_path: Path) -> bool:
    """Determine if a directory should be included in the archive based on filtering rules."""
    if not dir_path.is_dir():
        return False

    # Skip hidden directories
    if dir_path.name.startswith("."):
        return False

    # Skip build artifacts and cache directories
    if dir_path.name in ["dist", "__pycache__", "build"]:
        return False

    # Skip egg-info directories
    if dir_path.name.endswith(".egg-info"):
        return False

    return True


def compute_content_hash(env_path: Path) -> str:
    """Compute deterministic, cross-platform content hash for environment files.

    Args:
        env_path: Path to the environment directory

    Returns:
        SHA256 hexdigest of the environment content
    """
    content_hasher = hashlib.sha256()

    # Collect all items to hash in a deterministic order
    items_to_hash = []

    # Add root-level files
    for pattern in ["pyproject.toml", "*.py", "README.md"]:
        for file_path in env_path.glob(pattern):
            if file_path.is_file():
                items_to_hash.append(("file", file_path))

    # Add subdirectory contents
    for subdir in sorted(env_path.iterdir(), key=lambda x: x.name):
        if should_include_directory_in_archive(subdir):
            # Add directory marker
            items_to_hash.append(("dir", subdir))

            # Add files in subdirectory
            for file_path in subdir.rglob("*"):
                if should_include_file_in_archive(file_path, env_path):
                    items_to_hash.append(("file", file_path))

    # Sort all items by their relative path for deterministic ordering
    items_to_hash.sort(key=lambda item: str(item[1].relative_to(env_path)).replace("\\", "/"))

    # Hash items in sorted order
    for item_type, item_path in items_to_hash:
        rel_path = item_path.relative_to(env_path)
        # Use forward slashes for cross-platform consistency
        normalized_path = str(rel_path).replace("\\", "/")

        if item_type == "dir":
            content_hasher.update(f"dir:{normalized_path}".encode("utf-8"))
        elif item_type == "file":
            content_hasher.update(f"file:{normalized_path}".encode("utf-8"))
            try:
                with open(item_path, "rb") as f:
                    content_hasher.update(f.read())
            except IOError:
                # Skip files that can't be read
                pass

    return content_hasher.hexdigest()


@app.command("list")
def list_cmd(
    limit: int = typer.Option(
        DEFAULT_LIST_LIMIT, "--limit", "-l", help="Number of environments to show"
    ),
    offset: int = typer.Option(0, "--offset", help="Number of environments to skip"),
    owner: Optional[str] = typer.Option(None, "--owner", help="Filter by owner name"),
    visibility: Optional[str] = typer.Option(
        None, "--visibility", help="Filter by visibility (PUBLIC/PRIVATE)"
    ),
    output: str = typer.Option("table", "--output", help="Output format: table or json"),
) -> None:
    """List available verifiers environments"""
    validate_output_format(output, console)

    try:
        client = APIClient(require_auth=False)

        params: Dict[str, Any] = {
            "include_teams": True,
            "limit": limit,
            "offset": offset,
        }
        if owner:
            params["owner"] = owner
        if visibility:
            params["visibility"] = visibility

        result = client.get("/environmentshub/", params=params)

        environments = result.get("data", result.get("environments", []))
        total = result.get("total_count", result.get("total", 0))

        if not environments:
            if output == "json":
                output_data_as_json(
                    {"environments": [], "total": 0, "offset": offset, "limit": limit}, console
                )
            else:
                console.print("No environments found.", style="yellow")
            return

        if output == "json":
            # Format environments for JSON output
            env_data = []
            for env in environments:
                owner_name = env["owner"]["name"]
                env_name = env["name"]
                env_data.append(
                    {
                        "environment": f"{owner_name}/{env_name}",
                        "description": env.get("description", ""),
                        "visibility": env.get("visibility", ""),
                    }
                )

            output_data = {
                "environments": env_data,
                "total": total,
                "offset": offset,
                "limit": limit,
            }
            output_data_as_json(output_data, console)
        else:
            # Table output
            table = Table(title=f"Environments (Total: {total})")
            table.add_column("Environment", style="cyan")
            table.add_column("Description", style="green")
            table.add_column("Visibility", style="magenta")

            for env in environments:
                owner_name = env["owner"]["name"]
                env_name = env["name"]
                env_id = f"{owner_name}/{env_name}"
                description = env.get("description", "")
                visibility = env.get("visibility", "")
                table.add_row(env_id, description, visibility)

            console.print(table)

            remaining = total - (offset + len(environments))
            if remaining > 0:
                next_offset = offset + limit
                console.print(
                    f"\n[dim]Use --offset {next_offset} to see the next environments.[/dim]"
                )

    except APIError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def push(
    path: str = typer.Option(".", "--path", "-p", help="Path to environment directory"),
    name: Optional[str] = typer.Option(
        None, "--name", "-n", help="Override environment name (defaults to pyproject.toml name)"
    ),
    team: Optional[str] = typer.Option(
        None,
        "--team",
        "-t",
        help="Team slug for team ownership (uses config team_id if not provided)",
    ),
    visibility: str = typer.Option(
        "PUBLIC", "--visibility", "-v", help="Environment visibility (PUBLIC/PRIVATE)"
    ),
    auto_bump: bool = typer.Option(
        False, "--auto-bump", help="Automatically bump patch version before push"
    ),
) -> None:
    """Push environment to registry"""

    try:
        env_path = Path(path).resolve()

        # Validate basic structure
        pyproject_path = env_path / "pyproject.toml"
        if not pyproject_path.exists():
            console.print("[red]Error: pyproject.toml not found[/red]")
            raise typer.Exit(1)

        try:
            pyproject_data = toml.load(pyproject_path)
            project_info = pyproject_data.get("project", {})

            env_name = name or project_info.get("name")
            if not env_name:
                console.print(
                    "[red]Error: No name found in pyproject.toml and no --name provided[/red]"
                )
                raise typer.Exit(1)

            # Auto-bump version if requested
            if auto_bump:
                current_version = project_info.get("version")
                if not current_version:
                    console.print(
                        "[red]Error: No version found in pyproject.toml for auto-bump[/red]"
                    )
                    raise typer.Exit(1)

                new_version = bump_version(current_version)
                console.print(f"Auto-bumping version: {current_version} → {new_version}")

                try:
                    update_pyproject_version(pyproject_path, new_version)
                    # Reload pyproject.toml with new version
                    pyproject_data = toml.load(pyproject_path)
                    project_info = pyproject_data.get("project", {})
                    console.print("[green]✓ Updated version in pyproject.toml[/green]")
                except Exception as e:
                    console.print(f"[red]Failed to update version in pyproject.toml: {e}[/red]")
                    raise typer.Exit(1)

            # Only show environment name if user overrode it
            if name:
                console.print(f"[dim]Using environment name: {env_name}[/dim]")

        except Exception as e:
            console.print(f"[red]Failed to parse pyproject.toml: {e}[/red]")
            raise typer.Exit(1)

        # Find any Python file in the environment
        has_env_file = False
        py_files = list(env_path.glob("*.py"))

        if py_files:
            has_env_file = True
        else:
            # Check for package structure with __init__.py
            for subdir in env_path.iterdir():
                if subdir.is_dir():
                    init_file = subdir / "__init__.py"
                    if init_file.exists():
                        has_env_file = True
                        break

        if not has_env_file:
            console.print("[red]Error: No environment Python file found[/red]")
            raise typer.Exit(1)

        console.print(f"\n[cyan]Building {env_name}...[/cyan]")

        # Clean dist directory to ensure fresh build
        dist_dir = env_path / "dist"
        if dist_dir.exists():
            shutil.rmtree(dist_dir)

        try:
            if shutil.which("uv"):
                subprocess.run(
                    ["uv", "build", "--wheel", "--out-dir", "dist"],
                    cwd=env_path,
                    capture_output=True,
                    text=True,
                    check=True,
                )
            else:
                subprocess.run(
                    [sys.executable, "-m", "build", "--wheel", str(env_path)],
                    capture_output=True,
                    text=True,
                    check=True,
                )
        except subprocess.CalledProcessError as e:
            console.print("[red]Build failed![/red]")
            console.print(e.stderr)
            raise typer.Exit(1)
        except FileNotFoundError:
            console.print("[red]Build tool not found. Please install 'uv' or 'build'.[/red]")
            raise typer.Exit(1)

        dist_dir = env_path / "dist"
        wheels = list(dist_dir.glob("*.whl"))
        if not wheels:
            console.print("[red]Error: No wheel file found after build[/red]")
            raise typer.Exit(1)

        wheel_path = wheels[0]
        wheel_size = wheel_path.stat().st_size
        console.print(f"[green]✓ Built package[/green] [dim]({wheel_size:,} bytes)[/dim]")

        try:
            client = APIClient()

            console.print("\n[cyan]Uploading to Prime Intellect Hub...[/cyan]")
            resolve_data = {"name": env_name, "visibility": visibility}
            if team:
                resolve_data["team_slug"] = team
            elif client.config.team_id:
                resolve_data["team_id"] = client.config.team_id

            try:
                response = client.post("/environmentshub/resolve", json=resolve_data)

                if "data" in response:
                    resolve_response = response["data"]
                else:
                    resolve_response = response

                env_id = resolve_response["id"]
                owner_info = resolve_response["owner"]

                if not resolve_response["created"]:
                    console.print(
                        f"[dim]Found existing environment: {owner_info['name']}/{env_name}[/dim]"
                    )
            except APIError as e:
                console.print(f"[red]Failed to resolve environment: {e}[/red]")
                raise typer.Exit(1)

            # Remove this line - progress bar is enough

            try:
                with open(wheel_path, "rb") as f:
                    wheel_sha256 = hashlib.sha256(f.read()).hexdigest()
            except IOError as e:
                console.print(f"[red]Failed to read wheel file: {e}[/red]")
                raise typer.Exit(1)

            project_metadata = project_info

            # Compute deterministic content hash
            content_hash = compute_content_hash(env_path)

            unique_wheel_name = wheel_path.name

            wheel_data = {
                "content_hash": content_hash,
                "filename": unique_wheel_name,
                "sha256": wheel_sha256,
                "size": wheel_path.stat().st_size,
                "semantic_version": project_metadata.get("version"),
                "metadata": {
                    "description": project_metadata.get("description", ""),
                    "tags": project_metadata.get("tags", []),
                    "license": project_metadata.get("license", ""),
                    "dependencies": project_metadata.get("dependencies", []),
                    "python_requires": project_metadata.get("requires-python", ">=3.8"),
                    "original_filename": wheel_path.name,
                },
            }

            try:
                response = client.post(f"/environmentshub/{env_id}/wheels", json=wheel_data)

                wheel_response = response["data"]

                wheel_id = wheel_response["wheel_id"]
                wheel_upload_url = wheel_response["upload_url"]

            except APIError as e:
                if "content hash" in str(e).lower() and "already exists" in str(e):
                    console.print(f"[red]Failed to prepare wheel upload: {e}[/red]")
                    console.print(
                        "[yellow]Tip: If you've made changes to your environment, "
                        "ensure the content has actually changed.[/yellow]"
                    )
                    console.print(
                        "[yellow]The content hash is based on your source files "
                        "(*.py, pyproject.toml, README.md).[/yellow]"
                    )
                else:
                    console.print(f"[red]Failed to prepare wheel upload: {e}[/red]")
                raise typer.Exit(1)

            if wheel_upload_url:
                try:
                    with open(wheel_path, "rb") as f:
                        wheel_content = f.read()

                    description = f"Uploading {wheel_path.name}"
                    upload_response = upload_with_progress(
                        wheel_upload_url,
                        wheel_content,
                        description,
                        headers={"Content-Type": "application/octet-stream"},
                    )
                    upload_response.raise_for_status()
                except httpx.RequestError as e:
                    console.print(f"[red]Failed to upload wheel: {e}[/red]")
                    raise typer.Exit(1)
                except IOError as e:
                    console.print(f"[red]Failed to read wheel file for upload: {e}[/red]")
                    raise typer.Exit(1)

                try:
                    client.post(f"/environmentshub/{env_id}/wheels/{wheel_id}/finalize")
                except APIError as e:
                    console.print(f"[red]Failed to finalize wheel upload: {e}[/red]")
                    raise typer.Exit(1)

            # Creating source archive silently
            temp_file_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
                    temp_file_path = tmp.name
                    with tarfile.open(tmp.name, "w:gz") as tar:
                        for pattern in ["README.md", "pyproject.toml", "*.py"]:
                            for file in env_path.glob(pattern):
                                if file.is_file():
                                    tar.add(file, arcname=file.name)

                        # Sort subdirectories for deterministic ordering and apply filtering
                        for subdir in sorted(env_path.iterdir(), key=lambda x: x.name):
                            if should_include_directory_in_archive(subdir):
                                # Add directory with custom filtering instead of entire subdirectory
                                for file in subdir.rglob("*"):
                                    if should_include_file_in_archive(file, env_path):
                                        # Calculate relative path from env_path for consistent
                                        # archive structure
                                        arcname = file.relative_to(env_path)
                                        tar.add(file, arcname=str(arcname))

                    with open(tmp.name, "rb") as f:
                        source_sha256 = hashlib.sha256(f.read()).hexdigest()

                    version = project_metadata.get("version")
                    unique_source_name = f"{env_name}-{version}-{content_hash[:8]}.tar.gz"

                    source_data = {
                        "content_hash": content_hash,
                        "filename": unique_source_name,
                        "sha256": source_sha256,
                        "semantic_version": version,
                        "metadata": {
                            **wheel_data["metadata"],
                            "original_filename": f"{env_name}-{version}.tar.gz",
                        },
                    }

                    try:
                        response = client.post(
                            f"/environmentshub/{env_id}/versions", json=source_data
                        )

                        version_response = response["data"]

                        version_id = version_response["version_id"]
                        source_upload_url = version_response["upload_url"]

                    except APIError as e:
                        if "content hash" in str(e).lower() and "already exists" in str(e):
                            console.print(f"[red]Failed to prepare source upload: {e}[/red]")
                            console.print(
                                "[yellow]Tip: If you've made changes to your environment, "
                                "ensure the content has actually changed.[/yellow]"
                            )
                            console.print(
                                "[yellow]The content hash is based on your source files "
                                "(*.py, pyproject.toml, README.md).[/yellow]"
                            )
                        else:
                            console.print(f"[red]Failed to prepare source upload: {e}[/red]")
                        raise typer.Exit(1)

                    try:
                        with open(tmp.name, "rb") as f:
                            source_content = f.read()

                        description = f"Uploading {unique_source_name}"
                        upload_response = upload_with_progress(
                            source_upload_url,
                            source_content,
                            description,
                            headers={"Content-Type": "application/octet-stream"},
                            timeout=300.0,
                        )
                        upload_response.raise_for_status()
                    except httpx.RequestError as e:
                        console.print(f"[red]Failed to upload source archive: {e}[/red]")
                        raise typer.Exit(1)
                    except IOError as e:
                        console.print(f"[red]Failed to read source archive for upload: {e}[/red]")
                        raise typer.Exit(1)

                    # Finalize
                    try:
                        response = client.post(
                            f"/environmentshub/{env_id}/versions/{version_id}/finalize"
                        )

                        finalize_response = response["data"]

                    except APIError as e:
                        console.print(f"[red]Failed to finalize source upload: {e}[/red]")
                        raise typer.Exit(1)

            except (tarfile.TarError, OSError) as e:
                console.print(f"[red]Failed to create source archive: {e}[/red]")
                raise typer.Exit(1)
            finally:
                # Clean up temporary file if it was created
                if temp_file_path and Path(temp_file_path).exists():
                    Path(temp_file_path).unlink()

            if finalize_response.get("success"):
                owner_name = owner_info["name"]
                console.print(f"\n[green]✓ Successfully pushed {owner_name}/{env_name}[/green]")

                # Show Hub page link for the environment
                frontend_url = client.config.frontend_url.rstrip("/")
                hub_url = f"{frontend_url}/dashboard/environments/{owner_name}/{env_name}"
                console.print(f"\n[dim]View:[/dim] [link={hub_url}]{hub_url}[/link]")
                console.print(f"[dim]Install:[/dim] prime env install {owner_name}/{env_name}")
            else:
                console.print(f"[red]Error finalizing: {finalize_response.get('message')}[/red]")
                raise typer.Exit(1)

        except APIError as e:
            console.print(f"[red]API Error: {e}[/red]")
            raise typer.Exit(1)
        except Exception as e:
            console.print(f"[red]Upload failed: {e}[/red]")
            raise typer.Exit(1)

    except subprocess.CalledProcessError as e:
        console.print(f"[red]Build error: {e}[/red]")
        raise typer.Exit(1)
    except FileNotFoundError as e:
        console.print(f"[red]File not found: {e}[/red]")
        raise typer.Exit(1)
    except PermissionError as e:
        console.print(f"[red]Permission error: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def init(
    name: str = typer.Argument(..., help="Name of the new environment"),
    path: str = typer.Option(
        "./environments", "--path", "-p", help="Path to environments directory"
    ),
    rewrite_readme: bool = typer.Option(
        False, "--rewrite-readme", help="Overwrite README.md with template if it already exists"
    ),
) -> None:
    """Initialize a new verifiers environment from template"""
    try:
        # this import is slow, so we do it inside the command
        from verifiers.scripts.init import init_environment

        created_path = init_environment(name, path, rewrite_readme)

        console.print(f"[green]✓ Created environment template in {created_path}/[/green]")
        console.print("\nNext steps:")
        console.print(f"  cd {created_path}")
        console.print("  # Edit the environment file to implement your verifier")
        console.print("  prime env push")

    except FileNotFoundError as e:
        console.print(f"[red]File not found: {e}[/red]")
        raise typer.Exit(1)
    except PermissionError as e:
        console.print(f"[red]Permission error: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def pull(
    env_id: str = typer.Argument(..., help="Environment ID (owner/name or owner/name@version)"),
    target: Optional[str] = typer.Option(None, "--target", "-t", help="Target directory"),
    version: str = typer.Option("latest", "--version", "-v", help="Version to pull"),
) -> None:
    """Pull environment for local inspection"""
    try:
        client = APIClient(require_auth=False)

        # Parse version from env_id if present (e.g., owner/name@version)
        if "@" in env_id:
            env_id_base, id_version = env_id.rsplit("@", 1)
            # Use the version from the env_id, overriding the --version flag
            version = id_version
            env_id = env_id_base

        parts = env_id.split("/")
        if len(parts) != 2:
            console.print("[red]Error: Invalid environment ID format. Expected: owner/name[/red]")
            raise typer.Exit(1)

        owner, name = parts

        console.print(f"\n[cyan]Pulling {env_id}@{version}...[/cyan]")

        try:
            response = client.get(f"/environmentshub/{owner}/{name}/@{version}")

            if "data" in response:
                details = response["data"]
            else:
                # Fallback for old format
                details = response
        except APIError as e:
            console.print(f"[red]Failed to get environment details: {e}[/red]")
            raise typer.Exit(1)

        download_url = details.get("package_url")
        if not download_url:
            console.print("[red]Error: No downloadable package found[/red]")
            raise typer.Exit(1)

        if target:
            target_dir = Path(target)
        else:
            target_dir = Path.cwd() / f"{owner}-{name}-{version}"

        try:
            target_dir.mkdir(parents=True, exist_ok=True)
        except PermissionError as e:
            console.print(f"[red]Permission error creating directory: {e}[/red]")
            raise typer.Exit(1)
        except OSError as e:
            console.print(f"[red]Error creating directory: {e}[/red]")
            raise typer.Exit(1)

        console.print(f"[dim]Target: {target_dir}[/dim]")

        temp_file_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
                temp_file_path = tmp.name
                try:
                    if is_valid_url(download_url):
                        headers = {}
                        if client.api_key:
                            headers["Authorization"] = f"Bearer {client.api_key}"

                        description = f"Downloading {owner}/{name}@{version}"
                        download_with_progress(download_url, tmp.name, description, headers)
                    else:
                        console.print(f"[red]Error: Invalid download URL: {download_url}[/red]")
                        raise typer.Exit(1)
                except httpx.RequestError as e:
                    console.print(f"[red]Download failed: {e}[/red]")
                    raise typer.Exit(1)
                except IOError as e:
                    console.print(f"[red]Failed to write downloaded file: {e}[/red]")
                    raise typer.Exit(1)

                try:
                    with tarfile.open(tmp.name, "r:gz") as tar:
                        tar.extractall(target_dir)
                except tarfile.TarError as e:
                    console.print(f"[red]Failed to extract archive: {e}[/red]")
                    raise typer.Exit(1)
                except IOError as e:
                    console.print(f"[red]Failed to extract files: {e}[/red]")
                    raise typer.Exit(1)
        except OSError as e:
            console.print(f"[red]Failed to create temporary file: {e}[/red]")
            raise typer.Exit(1)
        finally:
            # Clean up temporary file if it was created
            if temp_file_path and Path(temp_file_path).exists():
                Path(temp_file_path).unlink()

        console.print(f"\n[green]✓ Successfully pulled to {target_dir}[/green]")

        try:
            extracted_files = list(target_dir.iterdir())
            file_count = len(extracted_files)
            if file_count > 0:
                console.print(
                    f"[dim]{file_count} file{'s' if file_count != 1 else ''} extracted[/dim]"
                )
        except OSError:
            pass  # Silently skip if we can't list files

    except APIError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        raise typer.Exit(1)


def validate_env_id(env_id: str) -> Tuple[str, str]:
    """Validate and parse environment ID.

    Args:
        env_id: Environment ID in format 'owner/name' or 'owner/name@version'

    Returns:
        Tuple of (env_id_without_version, version)

    Raises:
        ValueError: If format is invalid
    """
    if not env_id or not env_id.strip():
        raise ValueError("Environment ID cannot be empty")

    # Handle version suffix
    version = "latest"
    if "@" in env_id:
        env_id, version = env_id.rsplit("@", 1)

    parts = env_id.split("/")
    if len(parts) != 2:
        raise ValueError(
            f"Invalid environment ID format: '{env_id}'. "
            f"Expected: 'owner/name' or 'owner/name@version'"
        )

    owner, name = parts
    if not owner or not name:
        raise ValueError("Owner and name cannot be empty")

    return env_id, version


def is_valid_url(url: str) -> bool:
    """Check if a string is a valid URL."""
    try:
        result = urlparse(url)
        return all([result.scheme in ("http", "https"), result.netloc])
    except Exception:
        return False


def normalize_package_name(name: str) -> str:
    """Normalize package name according to Python packaging standards."""
    return name.replace("-", "_").lower()


def bump_version(version: str) -> str:
    """Bump patch version (e.g., 1.2.3 -> 1.2.4)."""
    parts = version.split(".")
    if len(parts) >= 3:
        # Handle pre-release versions (e.g., 1.2.3-alpha -> 1.2.4)
        patch_part = parts[2]
        if "-" in patch_part:
            patch_num = patch_part.split("-")[0]
        elif "+" in patch_part:
            patch_num = patch_part.split("+")[0]
        else:
            patch_num = patch_part

        try:
            new_patch = str(int(patch_num) + 1)
            parts[2] = new_patch
            return ".".join(parts)
        except ValueError:
            # If patch is non-numeric, append .1
            return f"{version}.1"
    elif len(parts) == 2:
        return f"{version}.1"
    else:
        return f"{version}.0.1"


def update_pyproject_version(pyproject_path: Path, new_version: str) -> None:
    """Update version in pyproject.toml file."""
    with open(pyproject_path, "r") as f:
        content = f.read()

    # Find and replace version line (handles indentation)
    updated_content = re.sub(
        r'(\s*)version\s*=\s*["\'][^"\']*["\']',
        rf'\1version = "{new_version}"',
        content,
        flags=re.MULTILINE,
    )

    # Verify the replacement worked
    if updated_content == content:
        raise ValueError("Version line not found or updated in pyproject.toml")

    with open(pyproject_path, "w") as f:
        f.write(updated_content)


def get_install_command(tool: str, wheel_url: str) -> List[str]:
    """Generate install command for the specified tool."""
    if tool == "uv":
        return ["uv", "pip", "install", "--upgrade", wheel_url]
    elif tool == "pip":
        return ["pip", "install", "--upgrade", wheel_url]
    else:
        raise ValueError(f"Unsupported package manager: {tool}. Use 'uv' or 'pip'.")


@app.command()
def info(
    env_id: str = typer.Argument(..., help="Environment ID (owner/name)"),
    version: str = typer.Option("latest", "--version", "-v", help="Version to show"),
) -> None:
    """Show environment details and installation commands"""
    try:
        client = APIClient(require_auth=False)

        # Validate and parse environment ID
        try:
            env_id, parsed_version = validate_env_id(env_id)
            # Use parsed version if it was specified in the env_id, otherwise use the --version flag
            if parsed_version != "latest":
                target_version = parsed_version
            else:
                target_version = version
        except ValueError as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1)

        owner, name = env_id.split("/")

        console.print(f"Fetching {env_id}@{target_version}...")

        # Fetch environment details
        try:
            response = client.get(f"/environmentshub/{owner}/{name}/@{target_version}")
            details = response.get("data", response)
        except APIError as e:
            console.print(f"[red]Failed to get environment details: {e}[/red]")
            raise typer.Exit(1)

        # Process wheel URL
        wheel_url = process_wheel_url(details.get("wheel_url"))

        # Display basic info with nice formatting
        console.print()
        console.print(f"[bold cyan]{owner}/{name}[/bold cyan][dim]@{target_version}[/dim]")

        # Display metadata if available
        if metadata := details.get("metadata"):
            if desc := metadata.get("description"):
                console.print(f"[dim]{desc}[/dim]")

        console.print()

        # Display key installation commands based on availability
        simple_index_url = details.get("simple_index_url")
        if wheel_url or simple_index_url:
            normalized_name = normalize_package_name(name)

            console.print("[bold yellow]Install (choose one)[/bold yellow]")
            console.print(f"  [green]$[/green] prime env install {owner}/{name}@{target_version}")

            # Use simple index if available, otherwise fall back to wheel URL
            if simple_index_url:
                # For versioned installs, show package name with version specification
                if target_version and target_version != "latest":
                    console.print(
                        f"  [green]$[/green] uv pip install {normalized_name}=={target_version} "
                        f"--extra-index-url {simple_index_url}"
                    )
                    console.print(
                        f"  [green]$[/green] uv add {normalized_name}=={target_version} "
                        f"--index {simple_index_url}"
                    )
                    console.print(
                        f"  [green]$[/green] pip install {normalized_name}=={target_version} "
                        f"--extra-index-url {simple_index_url}"
                    )
                else:
                    console.print(
                        f"  [green]$[/green] uv pip install {normalized_name} "
                        f"--extra-index-url {simple_index_url}"
                    )
                    console.print(
                        f"  [green]$[/green] uv add {normalized_name} --index {simple_index_url}"
                    )
                    console.print(
                        f"  [green]$[/green] pip install {normalized_name} "
                        f"--extra-index-url {simple_index_url}"
                    )
            elif wheel_url:
                console.print(f"  [green]$[/green] uv pip install {wheel_url}")
                console.print(f"  [green]$[/green] uv add {normalized_name}@{wheel_url}")
                console.print(f"  [green]$[/green] pip install {wheel_url}")

            console.print()
            console.print("[bold yellow]Usage[/bold yellow]")
            console.print("  [blue]>>>[/blue] from verifiers import load_environment")
            console.print(f"  [blue]>>>[/blue] env = load_environment('{name}')")
        elif details.get("visibility") == "PRIVATE":
            console.print("[bold yellow]Install (private environment)[/bold yellow]")
            console.print(f"  [green]$[/green] prime env pull {owner}/{name}@{target_version}")
            console.print(
                "  [dim]Note: Direct UV/pip install not available for private environments[/dim]"
            )

            console.print()
            console.print("[bold yellow]After pulling[/bold yellow]")
            console.print("  [green]$[/green] cd <target_directory>")
            console.print("  [green]$[/green] uv pip install -e .")
        else:
            console.print("[yellow]No wheel available for this version[/yellow]")

        console.print()

    except APIError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        raise typer.Exit(1)


def fetch_environment_details(
    client: APIClient, owner: str, name: str, version: str
) -> Dict[str, Any]:
    """Fetch environment details from the API.

    Returns:
        Dictionary containing environment details

    Raises:
        APIError: If the API request fails
    """
    response = client.get(f"/environmentshub/{owner}/{name}/@{version}")
    details = response.get("data", response)
    # Ensure we return a dict
    if not isinstance(details, dict):
        raise ValueError(f"Invalid response format: expected dict, got {type(details)}")
    return details


def process_wheel_url(wheel_url: Optional[str]) -> Optional[str]:
    """Process and validate wheel URL.

    Args:
        wheel_url: The wheel URL from API (should be a full URL)

    Returns:
        Full wheel URL or None if not available
    """
    if not wheel_url:
        return None

    # Validate the URL
    if not is_valid_url(wheel_url):
        raise ValueError(f"Invalid wheel URL: {wheel_url}")

    return wheel_url


def execute_install_command(cmd: List[str], env_id: str, version: str, tool: str) -> None:
    """Execute the installation command with proper output handling.

    Args:
        cmd: Command to execute
        env_id: Environment ID for display
        version: Version for display
        tool: Tool name for display

    Raises:
        typer.Exit: If installation fails
    """
    console.print(f"[dim]Using {tool}: {' '.join(cmd)}[/dim]")

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        # Stream output line by line
        while True:
            output = process.stdout.readline() if process.stdout else ""
            if output == "" and process.poll() is not None:
                break
            if output:
                console.print(output.rstrip())

        return_code = process.poll()
        if return_code != 0:
            console.print(
                f"[red]Environment installation failed with exit code {return_code}[/red]"
            )
            raise typer.Exit(1)

        console.print(f"\n[green]✓ Successfully installed {env_id}@{version}[/green]")

    except FileNotFoundError:
        console.print(f"[red]Failed to run command. Is {cmd[0]} installed?[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Installation failed: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def install(
    env_id: str = typer.Argument(..., help="Environment ID to install (owner/name)"),
    with_tool: str = typer.Option(
        "uv",
        "--with",
        help="Package manager to use (uv or pip)",
    ),
) -> None:
    """Install a verifiers environment

    Examples:
        prime env install owner/environment
        prime env install owner/environment@0.2.3
        prime env install owner/environment --with pip
    """
    try:
        client = APIClient(require_auth=False)

        # Validate package manager
        if with_tool not in ["uv", "pip"]:
            console.print(
                f"[red]Error: Unsupported package manager '{with_tool}'. Use 'uv' or 'pip'.[/red]"
            )
            raise typer.Exit(1)

        # Validate and parse environment ID
        try:
            env_id, target_version = validate_env_id(env_id)
        except ValueError as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1)

        owner, name = env_id.split("/")

        console.print(f"\n[cyan]Installing {env_id}@{target_version}...[/cyan]")

        # Fetch environment details
        try:
            details = fetch_environment_details(client, owner, name, target_version)
        except APIError as e:
            console.print(f"[red]Failed to get environment details: {e}[/red]")
            raise typer.Exit(1)

        # Get both simple index URL and wheel URL
        simple_index_url = details.get("simple_index_url")
        wheel_url = process_wheel_url(details.get("wheel_url"))

        # Check if this is a private environment
        if not simple_index_url and not wheel_url and details.get("visibility") == "PRIVATE":
            console.print(
                "[yellow]Private environment detected. Using authenticated download.[/yellow]"
            )
            console.print(
                "[red]Direct installation not available for private environments.[/red]\n"
                "Please use one of these alternatives:\n"
                "  1. Use 'prime env pull' to download and install locally\n"
                "  2. Make the environment public to enable direct installation"
            )
            raise typer.Exit(1)
        elif not simple_index_url and not wheel_url:
            console.print(
                "[red]Error: No installation method available for this environment.[/red]"
            )
            console.print(
                "Use 'prime env info' to see available options or 'pull' to download source."
            )
            raise typer.Exit(1)

        # Remove this - we already said we're installing

        # Generate install command preferring simple index over wheel URL
        normalized_name = normalize_package_name(name)

        if simple_index_url:
            # Prefer simple index approach
            if with_tool == "uv":
                if target_version and target_version != "latest":
                    cmd_parts = [
                        "uv",
                        "pip",
                        "install",
                        f"{normalized_name}=={target_version}",
                        "--extra-index-url",
                        simple_index_url,
                    ]
                else:
                    cmd_parts = [
                        "uv",
                        "pip",
                        "install",
                        normalized_name,
                        "--extra-index-url",
                        simple_index_url,
                    ]
            else:  # pip
                if target_version and target_version != "latest":
                    cmd_parts = [
                        "pip",
                        "install",
                        f"{normalized_name}=={target_version}",
                        "--extra-index-url",
                        simple_index_url,
                    ]
                else:
                    cmd_parts = [
                        "pip",
                        "install",
                        normalized_name,
                        "--extra-index-url",
                        simple_index_url,
                    ]
        elif wheel_url:
            # Fall back to wheel URL if simple index not available
            try:
                cmd_parts = get_install_command(with_tool, wheel_url)
            except ValueError as e:
                console.print(f"[red]Error: {e}[/red]")
                raise typer.Exit(1)
        else:
            # Should not reach here due to earlier checks, but just in case
            console.print("[red]Error: No installation method available.[/red]")
            raise typer.Exit(1)

        # Check if tool is installed
        if not shutil.which(cmd_parts[0]):
            console.print(f"[red]Error: {cmd_parts[0]} is not installed.[/red]")
            raise typer.Exit(1)

        # Execute installation
        execute_install_command(cmd_parts, env_id, target_version, with_tool)

        # Display usage instructions
        console.print(
            f"\n[dim]Usage:[/dim] from verifiers import load_environment; "
            f"env = load_environment('{name}')"
        )

    except APIError as e:
        console.print(f"[red]API Error: {e}[/red]")
        raise typer.Exit(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Installation cancelled by user[/yellow]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        raise typer.Exit(1)


def execute_uninstall_command(cmd: List[str], env_name: str, tool: str) -> None:
    """Execute the uninstall command with proper output handling.

    Args:
        cmd: Command to execute
        env_name: Environment name for display
        tool: Tool name for display

    Raises:
        typer.Exit: If uninstall fails
    """

    console.print(f"\n[cyan]Uninstalling {env_name}...[/cyan]")
    console.print(f"[dim]Using {tool}: {' '.join(cmd)}[/dim]")

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        # Stream output line by line
        while True:
            output = process.stdout.readline() if process.stdout else ""
            if output == "" and process.poll() is not None:
                break
            if output:
                console.print(output.rstrip())

        return_code = process.poll()
        if return_code != 0:
            console.print(f"[red]Environment uninstall failed with exit code {return_code}[/red]")
            raise typer.Exit(1)

        console.print(f"\n[green]✓ Successfully uninstalled {env_name}[/green]")

    except FileNotFoundError:
        console.print(f"[red]Failed to run command. Is {cmd[0]} installed?[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Uninstall failed: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def uninstall(
    env_name: str = typer.Argument(..., help="Environment name to uninstall"),
    with_tool: str = typer.Option(
        "uv",
        "--with",
        help="Package manager to use (uv or pip)",
    ),
) -> None:
    """Uninstall a verifiers environment

    Examples:
        prime env uninstall environment
        prime env uninstall environment --with pip
    """
    try:
        # Validate package manager
        if with_tool not in ["uv", "pip"]:
            console.print(
                f"[red]Error: Unsupported package manager '{with_tool}'. Use 'uv' or 'pip'.[/red]"
            )
            raise typer.Exit(1)

        # Ignore owner if given
        if "/" in env_name:
            _, env_name = env_name.split("/", 1)

        normalized_name = normalize_package_name(env_name)

        # Generate uninstall command
        if with_tool == "uv":
            cmd_parts = [
                "uv",
                "pip",
                "uninstall",
                normalized_name,
            ]
        else:  # pip
            cmd_parts = [
                "pip",
                "uninstall",
                normalized_name,
            ]

        # Check if tool is installed
        if not shutil.which(cmd_parts[0]):
            console.print(f"[red]Error: {cmd_parts[0]} is not installed.[/red]")
            raise typer.Exit(1)

        # Execute uninstall
        execute_uninstall_command(cmd_parts, env_name, with_tool)

    except KeyboardInterrupt:
        console.print("\n[yellow]Uninstall cancelled by user[/yellow]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        raise typer.Exit(1)


version_app = typer.Typer(help="Manage environment versions")
app.add_typer(version_app, name="version")


@version_app.command("list")
def list_versions(
    env_id: str = typer.Argument(..., help="Environment ID (owner/name)"),
    full_hashes: bool = typer.Option(
        False, "--full-hashes", help="Show full content hashes instead of shortened ones"
    ),
) -> None:
    """List all versions of an environment"""
    try:
        client = APIClient(require_auth=False)

        parts = env_id.split("/")
        if len(parts) != 2:
            console.print("[red]Error: Invalid environment ID format. Expected: owner/name[/red]")
            raise typer.Exit(1)

        owner, name = parts

        console.print(f"Fetching versions for {env_id}...")

        try:
            response = client.get(f"/environmentshub/{owner}/{name}/versions")

            if "data" in response:
                versions_data = response["data"]
            else:
                versions_data = response

        except APIError as e:
            console.print(f"[red]Failed to get environment versions: {e}[/red]")
            raise typer.Exit(1)

        if not versions_data:
            console.print("No versions found.")
            return

        table = Table(title=f"Versions for {env_id}")
        table.add_column("Version", style="cyan")
        table.add_column("Created", style="green")
        table.add_column("Content Hash", style="yellow")
        table.add_column("Artifacts", style="magenta")

        # Sort versions by creation date (newest first)
        if isinstance(versions_data, list):
            versions_list = versions_data
        else:
            versions_list = versions_data.get("versions", [])

        for version in versions_list:
            version_display = version.get("version", "unknown")
            created_date = version.get("created_at", "")
            if created_date:
                # Format date nicely if it's a full timestamp
                try:
                    if "T" in created_date:
                        dt = datetime.fromisoformat(created_date.replace("Z", "+00:00"))
                        created_date = dt.strftime("%Y-%m-%d %H:%M")
                except Exception:
                    pass

            content_hash = version.get("sha256", "")
            if full_hashes or version_display == "unknown":
                content_hash_display = content_hash
            else:
                content_hash_display = content_hash[:DEFAULT_HASH_LENGTH] if content_hash else ""

            artifact_count = version.get("size", 0)
            artifacts_str = f"{artifact_count} artifact{'s' if artifact_count != 1 else ''}"

            table.add_row(version_display, created_date, content_hash_display, artifacts_str)

        console.print(table)

        if versions_list:
            latest = versions_list[0]  # Assuming first is latest
            console.print(f"\n[dim]Latest version: {latest.get('version', 'unknown')}[/dim]")
            install_cmd = f"prime env install {env_id}@{latest.get('version', 'latest')}"
            console.print(f"[dim]Install with: {install_cmd}[/dim]")

    except APIError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        raise typer.Exit(1)


@version_app.command("delete")
def delete_version(
    env_id: str = typer.Argument(..., help="Environment ID (owner/name)"),
    content_hash: str = typer.Argument(..., help="Content hash of the version to delete"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
) -> None:
    """Delete a specific environment version from the environments hub using its content hash"""
    try:
        # Validate that we have a proper content hash (basic validation)
        if len(content_hash) < 8:
            console.print(
                "[red]Error: Please provide a valid content hash (at least 8 characters)[/red]"
            )
            console.print(
                "[yellow]Use 'prime env version list' to see available content hashes[/yellow]"
            )
            raise typer.Exit(1)

        if not force:
            try:
                confirm_msg = (
                    f"Are you sure you want to permanently delete version with content "
                    f"hash '{content_hash}' from '{env_id}' on the environments hub?"
                )
                confirm = typer.confirm(confirm_msg)
                if not confirm:
                    console.print("Deletion cancelled.")
                    raise typer.Exit()
            except typer.Abort:
                console.print("Deletion cancelled.")
                raise typer.Exit()

        client = APIClient()

        parts = env_id.split("/")
        if len(parts) != 2:
            console.print("[red]Error: Invalid environment ID format. Expected: owner/name[/red]")
            raise typer.Exit(1)

        owner, name = parts
        console.print(f"Deleting version {content_hash} from {env_id}...")

        try:
            url = f"/environmentshub/{owner}/{name}/@{content_hash}"
            client.delete(url)
            console.print(
                f"[green]✓ Version {content_hash} deleted successfully from {env_id}[/green]"
            )
        except APIError as e:
            if "404" in str(e):
                console.print(
                    f"[red]Version with content hash '{content_hash}' "
                    f"not found in environment '{env_id}'[/red]"
                )
            else:
                console.print(f"[red]Failed to delete version: {e}[/red]")
            raise typer.Exit(1)

    except APIError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def delete(
    env_id: str = typer.Argument(..., help="Environment ID to delete"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
) -> None:
    """Delete an entire environment from the environments hub"""
    try:
        if not force:
            try:
                delete_msg = (
                    f"Are you sure you want to permanently delete entire environment "
                    f"'{env_id}' and ALL its versions from the environments hub?"
                )
                confirm = typer.confirm(delete_msg)
                if not confirm:
                    console.print("Deletion cancelled.")
                    raise typer.Exit()
            except typer.Abort:
                console.print("Deletion cancelled.")
                raise typer.Exit()

        client = APIClient()
        console.print(f"Deleting {env_id} from remote hub...")

        try:
            client.delete(f"/environmentshub/{env_id}")
            console.print(f"[green]✓ Environment {env_id} deleted successfully[/green]")
        except APIError as e:
            console.print(f"[red]Failed to delete environment: {e}[/red]")
            raise typer.Exit(1)

    except APIError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        raise typer.Exit(1)
