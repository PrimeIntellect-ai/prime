import hashlib
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import requests
import toml
import typer
from rich.console import Console
from rich.table import Table
from verifiers.scripts.init import init_environment

from ..api.client import APIClient, APIError

app = typer.Typer(help="Manage verifier environments")
console = Console()

# Constants
MAX_FILES_TO_SHOW = 10
DEFAULT_HASH_LENGTH = 8
DEFAULT_LIST_LIMIT = 20


@app.command("list")
def list_cmd(
    limit: int = typer.Option(
        DEFAULT_LIST_LIMIT, "--limit", "-l", help="Number of environments to show"
    ),
    offset: int = typer.Option(0, "--offset", "-o", help="Number of environments to skip"),
    owner: Optional[str] = typer.Option(None, "--owner", help="Filter by owner name"),
    visibility: Optional[str] = typer.Option(
        None, "--visibility", help="Filter by visibility (PUBLIC/PRIVATE)"
    ),
) -> None:
    """List available verifier environments"""
    try:
        client = APIClient()

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
            console.print("No environments found.", style="yellow")
            return

        table = Table(title=f"Verifier Environments (Total: {total})")
        table.add_column("Environment", style="cyan")
        table.add_column("Description", style="green")
        table.add_column("Visibility", style="magenta")
        table.add_column("Latest Version", style="yellow")

        for env in environments:
            owner_name = env["owner"]["name"]
            env_name = env["name"]
            env_id = f"{owner_name}/{env_name}"
            description = env.get("description", "")
            visibility = env.get("visibility", "")

            latest_version = ""
            if env.get("latest_version"):
                latest = env["latest_version"]
                if isinstance(latest, dict) and "sha256" in latest:
                    latest_version = f"sha256:{latest['sha256'][:8]}"

            table.add_row(env_id, description, visibility, latest_version)

        console.print(table)

        remaining = total - (offset + len(environments))
        if remaining > 0:
            next_offset = offset + limit
            console.print(f"\n[dim]Use --offset {next_offset} to see the next environments.[/dim]")

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
    team: Optional[str] = typer.Option(None, "--team", "-t", help="Team slug for team ownership"),
    visibility: str = typer.Option(
        "PUBLIC", "--visibility", "-v", help="Environment visibility (PUBLIC/PRIVATE)"
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

            console.print(f"Environment name: {env_name}")

        except Exception as e:
            console.print(f"[red]Failed to parse pyproject.toml: {e}[/red]")
            raise typer.Exit(1)

        env_file = None
        for file in env_path.glob("*.py"):
            if file.name.startswith("vf_") or file.name == f"{env_name.replace('-', '_')}.py":
                env_file = file
                break

        if not env_file:
            console.print("[red]Error: No environment Python file found (should be vf_*.py)[/red]")
            raise typer.Exit(1)

        console.print(f"Building environment package at {env_path}...")

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
                console.print(f"[red]Failed to resolve environment: {e}[/red]")
                raise typer.Exit(1)

            console.print("Uploading wheel...")

            try:
                with open(wheel_path, "rb") as f:
                    wheel_sha256 = hashlib.sha256(f.read()).hexdigest()
            except IOError as e:
                console.print(f"[red]Failed to read wheel file: {e}[/red]")
                raise typer.Exit(1)

            project_metadata = project_info

            # Generate content hash based on actual file contents
            content_hasher = hashlib.sha256()

            # Hash key files that define the environment content
            files_to_hash = []

            if pyproject_path.exists():
                files_to_hash.append(pyproject_path)

            for py_file in env_path.glob("*.py"):
                files_to_hash.append(py_file)

            readme_path = env_path / "README.md"
            if readme_path.exists():
                files_to_hash.append(readme_path)

            # Sort files for consistent hashing
            files_to_hash.sort(key=lambda x: x.name)

            # Hash the content of each file
            for file_path in files_to_hash:
                try:
                    with open(file_path, "rb") as f:
                        content_hasher.update(f.read())
                except IOError:
                    pass

            content_hash = content_hasher.hexdigest()

            timestamp = int(time.time())
            unique_wheel_name = wheel_path.name

            wheel_data = {
                "content_hash": content_hash,
                "filename": unique_wheel_name,
                "sha256": wheel_sha256,
                "size": wheel_path.stat().st_size,
                "metadata": {
                    "version": project_metadata.get("version", "0.1.0"),
                    "description": project_metadata.get("description", ""),
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
                        upload_response = requests.put(
                            wheel_upload_url,
                            data=f.read(),
                            headers={"Content-Type": "application/octet-stream"},
                        )
                        upload_response.raise_for_status()
                except requests.RequestException as e:
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

                        if (env_path / "docs").exists():
                            tar.add(env_path / "docs", arcname="docs")

                    with open(tmp.name, "rb") as f:
                        source_sha256 = hashlib.sha256(f.read()).hexdigest()

                    version = project_metadata.get("version", "0.1.0")
                    unique_source_name = f"{env_name}-{version}-{timestamp}.tar.gz"

                    source_data = {
                        "content_hash": content_hash,
                        "filename": unique_source_name,
                        "sha256": source_sha256,
                        "metadata": {
                            **wheel_data["metadata"],  # type: ignore
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
                            upload_response = requests.put(
                                source_upload_url,
                                data=f.read(),
                                headers={"Content-Type": "application/octet-stream"},
                            )
                            upload_response.raise_for_status()
                    except requests.RequestException as e:
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
                console.print(
                    f"\n[green]✓ Successfully pushed {owner_info['name']}/{env_name}[/green]"
                )
                console.print(f"Wheel: {wheel_path.name}")
                console.print(f"SHA256: {wheel_sha256}")
                console.print("\n[cyan]Install with:[/cyan]")
                console.print(f"  prime env install {owner_info['name']}/{env_name}")
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
    skip_vf_prefix: bool = typer.Option(
        False, "--skip-vf-prefix", help="Skip the vf- prefix in the environment id"
    ),
) -> None:
    """Initialize a new verifier environment from template"""
    try:
        created_path = init_environment(name, path, skip_vf_prefix)

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
    env_id: str = typer.Argument(..., help="Environment ID (owner/name)"),
    target: Optional[str] = typer.Option(None, "--target", "-t", help="Target directory"),
    version: str = typer.Option("latest", "--version", "-v", help="Version to pull"),
) -> None:
    """Pull environment for local inspection"""
    try:
        client = APIClient()

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

        if not details.get("download_object_key"):
            console.print("[red]Error: No downloadable package found[/red]")
            raise typer.Exit(1)

        download_url = f"/environmentshub/{owner}/{name}/@{version}/download"

        if target:
            target_dir = Path(target)
        else:
            target_dir = Path.cwd() / f"{name}-{version}"

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
                    resp = client.session.get(client.base_url + download_url, stream=True)
                    resp.raise_for_status()

                    with open(tmp.name, "wb") as f:
                        for chunk in resp.iter_content(chunk_size=8192):
                            f.write(chunk)
                except requests.RequestException as e:
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


@app.command()
def install(
    env_id: str = typer.Argument(..., help="Environment ID to install (owner/name)"),
    version: str = typer.Option("latest", "--version", "-v", help="Version to install"),
    python: Optional[str] = typer.Option(
        None, "--python", "-p", help="Python version to use with uv"
    ),
    add_to_project: bool = typer.Option(
        True,
        "--add-to-project/--no-add-to-project",
        help="Add environment URL to pyproject.toml dependencies",
    ),
) -> None:
    """Install a verifier environment"""

    try:
        client = APIClient()

        if "@" in env_id:
            env_id, version = env_id.rsplit("@", 1)

        console.print(f"Resolving {env_id}@{version}...")

        parts = env_id.split("/")
        if len(parts) != 2:
            console.print("[red]Error: Invalid environment ID format. Expected: owner/name[/red]")
            raise typer.Exit(1)

        owner, name = parts

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

        wheel_url = details.get("wheel_url")
        if not wheel_url:
            console.print("[red]Error: No wheel file available for this environment.[/red]")
            console.print("For now, use the pull command to download and inspect the environment.")
            raise typer.Exit(1)

        # Construct full URL from relative path
        wheel_url = f"{client.base_url}{wheel_url}"

        console.print(f"[dim]Installing from: {wheel_url}[/dim]")

        cmd_parts = ["uv", "pip", "install", "--upgrade"]

        if python:
            cmd_parts.extend(["--python", python])
        else:
            cmd_parts.extend(["--python", "python"])

        cmd_parts.append(wheel_url)

        console.print(f"[green]✓ Found {env_id}@{version}[/green]")

        if not shutil.which("uv"):
            console.print("[red]Error: uv is not installed.[/red]")
            console.print("Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh")
            raise typer.Exit(1)

        console.print(f"\n[cyan]Installing {env_id}@{version}...[/cyan]")

        try:
            process = subprocess.Popen(
                cmd_parts,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Merge stderr into stdout
                text=True,
                bufsize=1,  # Line buffered
                universal_newlines=True,
            )

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

            if add_to_project:
                try:
                    pyproject_path = Path("pyproject.toml")
                    if pyproject_path.exists():
                        with open(pyproject_path, "r") as f:
                            pyproject_data = toml.load(f)

                        if "project" not in pyproject_data:
                            pyproject_data["project"] = {}
                        if "dependencies" not in pyproject_data["project"]:
                            pyproject_data["project"]["dependencies"] = []

                        # PEP 508 compliant direct URL dependency format
                        dependency_url = f"{name.replace('-', '_')} @ {wheel_url}"

                        normalized_name = name.replace("-", "_")
                        existing_deps = pyproject_data["project"]["dependencies"]
                        if not any(
                            dep.startswith(f"{normalized_name} @")
                            for dep in existing_deps
                            if isinstance(dep, str)
                        ):
                            existing_deps.append(dependency_url)

                            with open(pyproject_path, "w") as f:
                                toml.dump(pyproject_data, f)

                            console.print(
                                f"[green]✓ Added {normalized_name} to pyproject.toml "
                                f"dependencies[/green]"
                            )
                        else:
                            console.print(
                                f"[yellow]! {normalized_name} already exists in "
                                f"pyproject.toml[/yellow]"
                            )
                    else:
                        console.print(
                            "[yellow]Warning: No pyproject.toml found in current directory[/yellow]"
                        )

                except Exception as e:
                    console.print(f"[yellow]Warning: Failed to update pyproject.toml: {e}[/yellow]")

            console.print("\n[dim]Use in Python:[/dim]")
            console.print("  from verifiers import load_environment")
            console.print(f"  env = load_environment('{name}')")

        except FileNotFoundError:
            console.print("[red]Failed to run UV command. Is UV installed?[/red]")
            raise typer.Exit(1)
        except Exception as e:
            console.print(f"[red]Environment installation failed: {e}[/red]")
            raise typer.Exit(1)

    except APIError as e:
        console.print(f"[red]Error: {e}[/red]")
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
        client = APIClient()

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
    version: str = typer.Argument(..., help="Version to delete (content hash or tag)"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
) -> None:
    """Delete a specific version of an environment"""
    try:
        if version == "latest" and not force:
            console.print("[yellow]Warning: You're about to delete the 'latest' tag![/yellow]")
            msg = "This will remove the latest reference but keep the actual version data."
            console.print(f"[yellow]{msg}[/yellow]")
            advice = "Consider using a content hash to delete the specific version instead."
            console.print(f"[yellow]{advice}[/yellow]")

        if not force:
            try:
                confirm_msg = (
                    f"Are you sure you want to permanently delete version "
                    f"'{version}' from '{env_id}'?"
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
        console.print(f"Deleting version {version} from {env_id}...")

        try:
            client.delete(f"/environmentshub/{owner}/{name}/@{version}")
            console.print(f"[green]✓ Version {version} deleted successfully from {env_id}[/green]")
        except APIError as e:
            if "404" in str(e):
                console.print(f"[red]Version '{version}' not found in environment '{env_id}'[/red]")
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
    """Delete an entire environment"""
    try:
        if not force:
            try:
                delete_msg = (
                    f"Are you sure you want to permanently delete entire "
                    f"environment '{env_id}' and ALL its versions?"
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
