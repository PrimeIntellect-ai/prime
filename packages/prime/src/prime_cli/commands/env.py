import hashlib
import json
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
from prime_core import APIClient, APIError, Config
from rich.console import Console
from rich.table import Table

from ..api.inference import InferenceAPIError, InferenceClient
from ..utils import output_data_as_json, validate_output_format

app = typer.Typer(help="Manage verifiers environments", no_args_is_help=True)
console = Console()

# Constants
MAX_FILES_TO_SHOW = 10
DEFAULT_HASH_LENGTH = 8
DEFAULT_LIST_LIMIT = 20


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

            console.print(f"Environment name: {env_name}")

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

        console.print(f"Building environment package at {env_path}...")

        # Clean dist directory to ensure fresh build
        dist_dir = env_path / "dist"
        if dist_dir.exists():
            console.print("[dim]Cleaning existing dist directory...[/dim]")
            shutil.rmtree(dist_dir)

        console.print("Building wheel distribution...")

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
        console.print(f"[green]✓ Built {wheel_path.name} ({wheel_size:,} bytes)[/green]")

        console.print("\nUploading to Prime Intellect Hub...")

        try:
            client = APIClient()

            console.print("Resolving environment...")
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

                if resolve_response["created"]:
                    console.print(
                        f"[green]✓ Created environment: {owner_info['name']}/{env_name}[/green]"
                    )
                else:
                    console.print(
                        f"[green]✓ Found existing environment: "
                        f"{owner_info['name']}/{env_name}[/green]"
                    )
            except APIError as e:
                # Handle missing username (slug) by prompting user to set it and retrying
                err_msg = str(e)
                if "missing a username" in err_msg.lower():
                    console.print(
                        "[yellow]Your user profile is missing a username.[/yellow] "
                        "You must choose a username to publish environments."
                    )
                    console.print(
                        "[dim]Note: This username can only be chosen once and will be public.[/dim]"
                    )

                    while True:
                        try:
                            chosen = (
                                typer.prompt(
                                    "Enter your desired username",
                                )
                                .strip()
                                .lower()
                            )
                        except typer.Abort:
                            console.print("[red]Cancelled by user[/red]")
                            raise typer.Exit(1)

                        if not chosen:
                            console.print("[red]Username cannot be empty[/red]")
                            continue

                        if not re.match(r"^[a-z0-9-]{3,30}$", chosen):
                            console.print(
                                "[red]Invalid username.[/red] "
                                "Use 3-30 chars with lowercase letters, numbers, and '-' only."
                            )
                            continue

                        try:
                            client.patch("/user/slug", json={"slug": chosen})
                            console.print(f"[green]✓ Username set to {chosen}[/green]")
                            break
                        except APIError as se:
                            se_msg = str(se)
                            if "409" in se_msg or "already taken" in se_msg.lower():
                                console.print(
                                    "[red]That username is already taken.[/red] "
                                    "Please choose another."
                                )
                                continue
                            else:
                                console.print(f"[red]Failed to set username: {se}[/red]")
                                raise typer.Exit(1)

                    # Retry resolve after setting username
                    try:
                        response = client.post("/environmentshub/resolve", json=resolve_data)

                        if "data" in response:
                            resolve_response = response["data"]
                        else:
                            resolve_response = response

                        env_id = resolve_response["id"]
                        owner_info = resolve_response["owner"]

                        if resolve_response["created"]:
                            console.print(
                                f"[green]✓ Created environment: {owner_info['name']}/"
                                f"{env_name}[/green]"
                            )
                        else:
                            console.print(
                                f"[green]✓ Found existing environment: "
                                f"{owner_info['name']}/{env_name}[/green]"
                            )
                    except APIError as e2:
                        console.print(
                            f"[red]Failed to resolve environment after setting username: {e2}[/red]"
                        )
                        raise typer.Exit(1)
                else:
                    console.print(f"[red]Failed to resolve environment: {e}[/red]")
                    raise typer.Exit(1)

            console.print("Uploading wheel ...")

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
                        upload_response = httpx.put(
                            wheel_upload_url,
                            content=f.read(),
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

            console.print("Creating source archive...")
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
                            upload_response = httpx.put(
                                source_upload_url,
                                content=f.read(),
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
                console.print(f"Wheel: {wheel_path.name}")
                console.print(f"SHA256: {wheel_sha256}")

                # Save environment hub metadata for future eval pushes
                hub_metadata = {
                    "environment_id": env_id,
                    "version_id": version_id,
                    "owner": owner_name,
                    "name": env_name,
                    "pushed_at": datetime.now().isoformat(),
                }
                hub_metadata_path = env_path / ".prime-cli.json"
                try:
                    with open(hub_metadata_path, "w") as f:
                        json.dump(hub_metadata, f, indent=2)
                    console.print(f"[dim]Saved hub metadata to {hub_metadata_path.name}[/dim]")
                except Exception as e:
                    console.print(f"[yellow]Warning: Could not save hub metadata: {e}[/yellow]")

                # Show Hub page link for the environment
                frontend_url = client.config.frontend_url.rstrip("/")
                hub_url = f"{frontend_url}/dashboard/environments/{owner_name}/{env_name}"
                console.print("\n[cyan]View on Environments Hub:[/cyan]")
                console.print(f"  [link={hub_url}]{hub_url}[/link]")

                # Show install command
                console.print("\n[cyan]Install with:[/cyan]")
                console.print(f"  prime env install {owner_name}/{env_name}")
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


@app.command(no_args_is_help=True)
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
        filename = f"{name}.py".replace("-", "_")
        console.print(f"  # Edit the {filename} file to implement your environment")
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


@app.command(no_args_is_help=True)
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

        console.print(f"Pulling {env_id}@{version}...")

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

        console.print(f"Downloading to {target_dir}...")

        temp_file_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
                temp_file_path = tmp.name
                try:
                    if is_valid_url(download_url):
                        headers = {}
                        if client.api_key:
                            headers["Authorization"] = f"Bearer {client.api_key}"
                        with httpx.stream(
                            "GET", download_url, headers=headers, timeout=60.0
                        ) as resp:
                            resp.raise_for_status()
                            with open(tmp.name, "wb") as f:
                                for chunk in resp.iter_bytes(chunk_size=8192):
                                    f.write(chunk)
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

        console.print(f"[green]✓ Environment pulled to {target_dir}[/green]")

        try:
            extracted_files = list(target_dir.iterdir())
            if extracted_files:
                console.print("\nExtracted files:")
                for file in extracted_files[:MAX_FILES_TO_SHOW]:
                    console.print(f"  - {file.name}")
                if len(extracted_files) > MAX_FILES_TO_SHOW:
                    remaining = len(extracted_files) - MAX_FILES_TO_SHOW
                    console.print(f"  ... and {remaining} more files")
        except OSError as e:
            console.print(f"[yellow]Warning: Could not list extracted files: {e}[/yellow]")

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


@app.command(no_args_is_help=True)
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
    console.print(f"\n[cyan]Installing {env_id}@{version} with {tool}...[/cyan]")
    console.print(f"[dim]Command: {' '.join(cmd)}[/dim]")

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


@app.command(no_args_is_help=True)
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

        console.print(f"Resolving {env_id}@{target_version}...")

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

        console.print(f"[green]✓ Found {env_id}@{target_version}[/green]")

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
        console.print("\n[dim]Use in Python:[/dim]")
        console.print("  from verifiers import load_environment")
        console.print(f"  env = load_environment('{name}')")

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

    console.print(f"\n[cyan]Uninstalling {env_name} with {tool}...[/cyan]")
    console.print(f"[dim]Command: {' '.join(cmd)}[/dim]")

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


@app.command(no_args_is_help=True)
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


version_app = typer.Typer(help="Manage environment versions", no_args_is_help=True)
app.add_typer(version_app, name="version")


@version_app.command("list", no_args_is_help=True)
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


@version_app.command("delete", no_args_is_help=True)
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


@app.command(no_args_is_help=True)
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


@app.command(
    "eval",
    no_args_is_help=True,
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def eval_env(
    ctx: typer.Context,
    environment: str = typer.Argument(
        ...,
        help="Installed Verifiers environment name (e.g. 'wordle')",
    ),
    model: str = typer.Option(
        "meta-llama/llama-3.1-70b-instruct",
        "--model",
        "-m",
        help=(
            "Model to use (e.g. 'meta-llama/llama-3.1-70b-instruct', see 'prime inference models' "
            "for available models)"
        ),
    ),
    # --- vf-eval options ---
    num_examples: Optional[int] = typer.Option(
        5, "--num-examples", "-n", help="Number of examples"
    ),
    rollouts_per_example: Optional[int] = typer.Option(
        3, "--rollouts-per-example", "-r", help="Rollouts per example"
    ),
    max_concurrent: Optional[int] = typer.Option(
        32, "--max-concurrent", "-c", help="Max concurrent requests"
    ),
    max_tokens: Optional[int] = typer.Option(
        None, "--max-tokens", "-t", help="Max tokens to generate (unset → model default)"
    ),
    temperature: Optional[float] = typer.Option(None, "--temperature", "-T", help="Temperature"),
    sampling_args: Optional[str] = typer.Option(
        None,
        "--sampling-args",
        "-S",
        help='Sampling args as JSON, e.g. \'{"enable_thinking": false, "max_tokens": 256}\'',
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    save_dataset: bool = typer.Option(False, "--save-dataset", "-s", help="Save dataset to disk"),
    save_to_hf_hub: bool = typer.Option(False, "--save-to-hf-hub", "-H", help="Save to HF Hub"),
    hf_hub_dataset_name: Optional[str] = typer.Option(
        None, "--hf-hub-dataset-name", "-D", help="HF Hub dataset name"
    ),
    env_args: Optional[str] = typer.Option(
        None, "--env-args", "-a", help='Environment args as JSON, e.g. \'{"key":"value"}\''
    ),
    api_key_var: Optional[str] = typer.Option(
        None, "--api-key-var", "-k", help="override api key variable instead of using PRIME_API_KEY"
    ),
    api_base_url: Optional[str] = typer.Option(
        None,
        "--api-base-url",
        "-b",
        help=(
            "override api base url variable instead of using prime inference url, "
            "should end in '/v1'"
        ),
    ),
) -> None:
    """
    Run verifiers' vf-eval with Prime Inference (closed beta)
    (This feature in currently in closed beta and requires prime inference permissions.)

    Example:
       prime env eval meow -m meta-llama/llama-3.1-70b-instruct -n 2 -r 3 -t 1024 -T 0.7
       All extra args are forwarded unchanged to vf-eval.
    """
    config = Config()

    api_key = config.api_key
    inference_base_url = (config.inference_url or "").strip()

    if not api_key:
        console.print(
            "[red]No API key configured.[/red] "
            "Run [bold]prime login[/bold] or [bold]prime config set-api-key[/bold]."
        )
        raise typer.Exit(1)

    # Choose base from --api-base-url (if given) or config
    if api_base_url:
        chosen_base = api_base_url.rstrip("/")
    else:
        if not inference_base_url:
            console.print(
                "[red]Inference URL not configured.[/red] Check [bold]prime config view[/bold]."
            )
            raise typer.Exit(1)
        chosen_base = inference_base_url.rstrip("/")

    inference_url = chosen_base

    # Fast fail if the model doesn't exist (only for Prime Inference, not custom URLs)
    # Check if using Prime Inference URL (either from config or explicitly provided)
    if chosen_base == inference_base_url:
        client = InferenceClient()
        try:
            client.retrieve_model(model)
        except InferenceAPIError as e:
            console.print(
                f"[red]Invalid model:[/red] {e} \n\n"
                f"[b]Use 'prime inference models' to see available models.[/b]"
            )
            raise typer.Exit(1)

    cmd = ["uv", "run", "vf-eval", environment]

    # Add chosen inference url
    cmd += ["-b", inference_url]

    # Always pass the selected model (required option)
    cmd += ["-m", model]

    # Environment modification may be necessary for passing in API key
    env = os.environ.copy()

    # API key var: respect --api-key-var if provided to this command, else inject PRIME_API_KEY
    if api_key_var:
        cmd += ["-k", api_key_var]
    else:
        env["PRIME_API_KEY"] = api_key
        cmd += ["-k", "PRIME_API_KEY"]

    # Forward vf-eval options if provided here
    if env_args:
        cmd += ["-a", env_args]
    if num_examples is not None:
        cmd += ["-n", str(num_examples)]
    if rollouts_per_example is not None:
        cmd += ["-r", str(rollouts_per_example)]
    if max_concurrent is not None:
        cmd += ["-c", str(max_concurrent)]
    if max_tokens is not None:
        cmd += ["-t", str(max_tokens)]
    if temperature is not None:
        cmd += ["-T", str(temperature)]
    if sampling_args:
        cmd += ["-S", sampling_args]
    if verbose:
        cmd += ["-v"]
    if save_dataset:
        cmd += ["-s"]
    if save_to_hf_hub:
        cmd += ["-H"]
    if hf_hub_dataset_name:
        cmd += ["-D", hf_hub_dataset_name]

    # Execute; stream output directly
    try:
        result = subprocess.run(cmd, env=env)
        if result.returncode != 0:
            raise typer.Exit(result.returncode)
    except KeyboardInterrupt:
        raise typer.Exit(130)
    except FileNotFoundError:
        console.print("[red]Failed to start vf-eval process.[/red]")
        raise typer.Exit(1)
