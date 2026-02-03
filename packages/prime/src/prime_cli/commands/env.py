import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import uuid
import zipfile
from datetime import datetime

# Wheel METADATA files use RFC 822 format (PEP 566), same as email headers
from email.parser import Parser as EmailParser
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import httpx
import toml
import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

from prime_cli.core import Config

from ..api.inference import InferenceAPIError, InferenceClient
from ..client import APIClient, APIError
from ..utils import output_data_as_json, validate_output_format
from ..utils.env_metadata import find_environment_metadata
from ..utils.eval_push import push_eval_results_to_hub
from ..utils.formatters import format_file_size
from ..utils.time_utils import format_time_ago, iso_timestamp

app = typer.Typer(help="Manage verifiers environments", no_args_is_help=True)
console = Console()

# Constants
MAX_FILES_TO_SHOW = 10
DEFAULT_HASH_LENGTH = 8
DEFAULT_LIST_LIMIT = 20
MAX_TARBALL_SIZE_LIMIT = 250 * 1024 * 1024  # 250MB

# Action subcommand app
action_app = typer.Typer(help="Manage environment actions (CI jobs)", no_args_is_help=True)
app.add_typer(action_app, name="action", rich_help_panel="Manage")

# Log cleaning pattern
ANSI_ESCAPE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape codes from text."""
    return ANSI_ESCAPE.sub("", text)


def _parse_environment_slug(environment: str) -> Tuple[str, str]:
    """Parse owner/name from an environment slug.

    Args:
        environment: Environment slug in format 'owner/name'

    Returns:
        Tuple of (owner, name)

    Raises:
        typer.Exit: If the slug is invalid
    """
    if "/" not in environment:
        console.print(f"[red]Invalid environment format: {environment}[/red]")
        console.print("[dim]Use format: owner/environment-name[/dim]")
        raise typer.Exit(1)

    parts = environment.split("/", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        console.print(f"[red]Invalid environment format: {environment}[/red]")
        console.print("[dim]Use format: owner/environment-name[/dim]")
        raise typer.Exit(1)

    return parts[0], parts[1]


@action_app.command("list")
def actions_list(
    environment: str = typer.Argument(
        ...,
        help="Environment slug (e.g., 'owner/environment-name')",
    ),
    version_id: Optional[str] = typer.Option(
        None,
        "--version-id",
        "-v",
        help="Filter by version ID",
    ),
    limit: int = typer.Option(
        20,
        "--limit",
        "-l",
        help="Maximum number of actions to show",
    ),
    offset: int = typer.Option(
        0,
        "--offset",
        help="Offset for pagination",
    ),
    output: str = typer.Option(
        "table",
        "--output",
        "-o",
        help="Output format: table or json",
    ),
) -> None:
    """List actions (CI jobs) for an environment."""
    validate_output_format(output, console)

    owner, env_name = _parse_environment_slug(environment)

    try:
        client = APIClient()
        params = {
            "limit": limit,
            "offset": offset,
        }
        if version_id:
            params["version_id"] = version_id

        response = client.get(f"/environmentshub/{owner}/{env_name}/actions", params=params)
        data = response.get("data", {})

        if output == "json":
            output_data_as_json(data, console)
            return

        actions = data.get("actions", [])
        total = data.get("total", 0)

        if not actions:
            console.print("[yellow]No actions found for this environment.[/yellow]")
            return

        table = Table(title=f"Actions for {owner}/{env_name}")
        table.add_column("ID", style="cyan", no_wrap=True)
        table.add_column("Name", style="blue")
        table.add_column("Status", style="yellow")
        table.add_column("Version", style="dim")
        table.add_column("Trigger", style="dim")
        table.add_column("Created", style="dim")

        for action in actions:
            action_id = action.get("id", "")
            name = action.get("name") or action.get("job_type", "")
            status = action.get("status", "")

            # Color the status
            status_color = {
                "SUCCESS": "[green]SUCCESS[/green]",
                "FAILED": "[red]FAILED[/red]",
                "RUNNING": "[yellow]RUNNING[/yellow]",
                "PENDING": "[dim]PENDING[/dim]",
                "CANCELLED": "[dim]CANCELLED[/dim]",
            }.get(status, status)

            version = action.get("version") or {}
            version_str = version.get("semantic_version") or (version.get("content_hash") or "")[:8]
            trigger = action.get("trigger", "")
            created = action.get("created_at", "")
            if created:
                created = format_time_ago(created)

            table.add_row(action_id, name, status_color, version_str, trigger, created)

        console.print(table)
        console.print(f"[dim]Showing {len(actions)} of {total} actions[/dim]")

    except APIError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@action_app.command("logs")
def actions_logs(
    environment: str = typer.Argument(
        ...,
        help="Environment slug (e.g., 'owner/environment-name')",
    ),
    action_id: str = typer.Argument(
        ...,
        help="Action/job ID to get logs for",
    ),
    tail: int = typer.Option(1000, "--tail", "-n", help="Number of lines to show"),
    follow: bool = typer.Option(False, "--follow", "-f", help="Follow log output"),
) -> None:
    """Get logs for a specific action."""
    owner, env_name = _parse_environment_slug(environment)

    try:
        client = APIClient()

        if follow:
            console.print(f"[dim]Watching logs for action {action_id}... (Ctrl+C to stop)[/dim]\n")
            last_logs = ""
            consecutive_errors = 0

            while True:
                try:
                    response = client.get(
                        f"/environmentshub/{owner}/{env_name}/actions/{action_id}/logs",
                        params={"tail_lines": tail},
                    )
                    data = response.get("data", {})
                    logs = _strip_ansi(data.get("logs") or "")
                    consecutive_errors = 0

                    if logs != last_logs:
                        old_lines = last_logs.splitlines() if last_logs else []
                        new_lines = logs.splitlines()

                        if not last_logs:
                            for line in new_lines:
                                console.print(line)
                        else:
                            overlap = 0
                            max_overlap = min(len(old_lines), len(new_lines))
                            for i in range(1, max_overlap + 1):
                                if old_lines[-i:] == new_lines[:i]:
                                    overlap = i
                            for line in new_lines[overlap:]:
                                console.print(line)

                        last_logs = logs
                except APIError as e:
                    consecutive_errors += 1
                    if "429" in str(e):
                        if consecutive_errors >= 3:
                            console.print("[yellow]Rate limited. Waiting 30s...[/yellow]")
                            time.sleep(30)
                        else:
                            time.sleep(10)
                        continue
                    raise

                time.sleep(5)
        else:
            response = client.get(
                f"/environmentshub/{owner}/{env_name}/actions/{action_id}/logs",
                params={"tail_lines": tail},
            )
            data = response.get("data", {})
            logs = _strip_ansi(data.get("logs") or "")

            if logs:
                console.print(logs)
            else:
                console.print("[yellow]No logs available yet.[/yellow]")

    except KeyboardInterrupt:
        console.print("\n[dim]Stopped watching logs.[/dim]")
    except APIError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@action_app.command("retry")
def actions_retry(
    environment: str = typer.Argument(
        ...,
        help="Environment slug (e.g., 'owner/environment-name')",
    ),
    action_id: Optional[str] = typer.Argument(
        None,
        help="Action ID to retry (retries latest action if not provided)",
    ),
    output: str = typer.Option(
        "table",
        "--output",
        "-o",
        help="Output format: table or json",
    ),
) -> None:
    """Retry an action (integration test) for an environment.

    If no action ID is provided, retries the latest action.
    """
    validate_output_format(output, console)

    owner, env_name = _parse_environment_slug(environment)

    try:
        client = APIClient()
        payload = {}
        if action_id:
            payload["action_id"] = action_id

        response = client.post(
            f"/environmentshub/{owner}/{env_name}/actions/retry",
            json=payload,
        )
        data = response.get("data", {})

        if output == "json":
            output_data_as_json(data, console)
            return

        if data.get("success"):
            console.print("[green]Successfully triggered retry[/green]")
            console.print(f"[dim]Job ID: {data.get('job_id')}[/dim]")
            console.print(f"[dim]Version: {data.get('version_id')}[/dim]")
            job_id = data.get('job_id')
            console.print(
                f"\n[dim]Use 'prime env action logs {environment} {job_id}' to view logs[/dim]"
            )
        else:
            console.print(f"[red]Retry failed:[/red] {data.get('message', 'Unknown error')}")
            raise typer.Exit(1)

    except APIError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


def display_upstream_environment_info(
    env_path: Optional[Path] = None, environment_name: Optional[str] = None
) -> bool:
    """Display the upstream environment name if metadata exists.

    Checks the provided path (or current directory) for environment metadata
    and displays "Using upstream environment {owner}/{name}" if found.

    If environment_name is provided, also checks ./environments/{module_name} as a fallback.

    Args:
        env_path: Path to check for metadata (defaults to current directory)
        environment_name: Optional environment name to check in ./environments/{module_name}
    """
    # Determine module_name if environment_name is provided
    module_name = None
    if environment_name:
        module_name = environment_name.replace("-", "_")

    # Search for environment metadata in common locations
    env_metadata = find_environment_metadata(
        env_name=environment_name,
        env_path=env_path,
        module_name=module_name,
    )

    if env_metadata and env_metadata.get("owner") and env_metadata.get("name"):
        owner = env_metadata.get("owner")
        env_name = env_metadata.get("name")
        console.print(f"[dim]Using upstream environment {owner}/{env_name}[/dim]\n")
        return True
    else:
        console.print("[dim]No upstream environment found.[/dim]\n")
        return False


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

    # Skip hidden directories (includes .prime/, .git/, etc.)
    if dir_path.name.startswith("."):
        return False

    # Skip build artifacts, cache directories, and outputs
    if dir_path.name in ["dist", "__pycache__", "build", "outputs"]:
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


def _format_action_status(status: Optional[str]) -> Text:
    """Format action status with color coding."""
    if not status:
        return Text("-", style="dim")
    status_colors = {
        "SUCCESS": "green",
        "FAILED": "red",
        "RUNNING": "yellow",
        "PENDING": "yellow",
        "CANCELLED": "dim",
    }
    color = status_colors.get(status.upper(), "white")
    return Text(status, style=color)


@app.command("list", rich_help_panel="Explore")
def list_cmd(
    limit: int = typer.Option(
        DEFAULT_LIST_LIMIT, "--num", "-n", help="Number of environments to show"
    ),
    offset: int = typer.Option(0, "--offset", help="Number of environments to skip"),
    owner: Optional[str] = typer.Option(None, "--owner", help="Filter by owner name"),
    visibility: Optional[str] = typer.Option(
        None, "--visibility", help="Filter by visibility (PUBLIC/PRIVATE)"
    ),
    output: str = typer.Option("table", "--output", help="Output format: table or json"),
    search: Optional[str] = typer.Option(
        None, "--search", "-s", help="Search by name or description"
    ),
    tag: Optional[List[str]] = typer.Option(
        None, "--tag", "-t", help="Filter by tag (repeatable)"
    ),
    action_status: Optional[str] = typer.Option(
        None, "--action-status", help="Filter by action status (SUCCESS/FAILED/RUNNING/PENDING)"
    ),
    sort: str = typer.Option(
        "created_at", "--sort", help="Sort by: name, created_at, updated_at, stars"
    ),
    order: str = typer.Option("desc", "--order", help="Sort order: asc, desc"),
    show_actions: bool = typer.Option(False, "--show-actions", help="Show action status column"),
    starred: bool = typer.Option(
        False, "--starred", help="Filter to only environments you have starred"
    ),
    mine: bool = typer.Option(
        False, "--mine", help="Filter to only your own environments (personal + team)"
    ),
) -> None:
    """List environments from the hub.

    By default, shows all public environments. If authenticated, also includes
    private environments you have access to. Use --starred or --mine to filter.

    \b
    Examples:
        prime env list                       # All public environments
        prime env list --starred             # Your starred environments
        prime env list --mine                # Your own environments
        prime env list --search "math"       # Search by name/description
        prime env list --sort stars          # Sort by most starred
    """
    validate_output_format(output, console)

    # Validate sort and order
    if sort not in ("name", "created_at", "updated_at", "stars"):
        console.print(
            "[red]Error: --sort must be one of: name, created_at, updated_at, stars[/red]"
        )
        raise typer.Exit(1)
    if order.lower() not in ("asc", "desc"):
        console.print("[red]Error: --order must be one of: asc, desc[/red]")
        raise typer.Exit(1)

    try:
        # Require auth if filtering by starred or mine
        require_auth = starred or mine
        client = APIClient(require_auth=require_auth)

        params: Dict[str, Any] = {
            "include_teams": True,
            "limit": limit,
            "offset": offset,
            "sort_by": sort,
            "sort_order": order,
        }
        if owner:
            params["owner"] = owner
        if visibility:
            params["visibility"] = visibility
        if search:
            params["search"] = search
        if tag:
            params["tags"] = tag
        if action_status:
            params["ci_status"] = action_status
        if show_actions or action_status:
            params["include_ci_status"] = True
        if starred:
            params["starred_only"] = True
        if mine:
            params["mine_only"] = True

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
                env_entry = {
                    "environment": f"{owner_name}/{env_name}",
                    "description": env.get("description", ""),
                    "visibility": env.get("visibility", ""),
                    "version": env.get("latest_version"),
                    "stars": env.get("stars", 0),
                    "updated_at": env.get("updated_at"),
                }
                if show_actions or action_status:
                    env_entry["action_status"] = env.get("latest_ci_status")
                if env.get("tags"):
                    env_entry["tags"] = env.get("tags")
                env_data.append(env_entry)

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
            table.add_column("Version", style="blue")
            table.add_column("Stars", style="yellow", justify="right")
            table.add_column("Updated", style="dim")
            if show_actions or action_status:
                table.add_column("Action Status")

            for env in environments:
                owner_name = env["owner"]["name"]
                env_name = env["name"]
                env_id = f"{owner_name}/{env_name}"
                description = env.get("description", "")
                version = env.get("latest_version") or "-"
                stars = str(env.get("stars", 0))
                updated_at = env.get("updated_at", "")
                if updated_at:
                    # Format as short date
                    try:
                        dt = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
                        updated_at = dt.strftime("%Y-%m-%d")
                    except (ValueError, AttributeError):
                        pass

                if show_actions or action_status:
                    action_text = _format_action_status(env.get("latest_ci_status"))
                    table.add_row(env_id, description, version, stars, updated_at, action_text)
                else:
                    table.add_row(env_id, description, version, stars, updated_at)

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


@app.command("status", rich_help_panel="Explore")
def status_cmd(
    env_id: str = typer.Argument(..., help="Environment ID (owner/name)"),
    output: str = typer.Option("table", "--output", help="Output format: table or json"),
) -> None:
    """Show action status for an environment.

    \b
    Examples:
        prime env status owner/my-env
        prime env status owner/my-env --output json
    """
    validate_output_format(output, console)

    # Parse env_id
    owner_name, env_name = _parse_environment_slug(env_id)

    try:
        client = APIClient(require_auth=False)

        result = client.get(
            f"/environmentshub/{owner_name}/{env_name}/status",
        )

        data = result.get("data", result)

        if output == "json":
            output_data_as_json(data, console)
        else:
            # Header
            env_display_name = data.get("name", env_name)
            console.print(f"\n[bold cyan]Environment:[/bold cyan] {owner_name}/{env_display_name}")
            if data.get("description"):
                console.print(f"[dim]Description:[/dim] {data['description']}")
            console.print(f"[dim]Visibility:[/dim] {data.get('visibility', 'UNKNOWN')}")

            # Latest Version section
            console.print("\n[bold]Latest Version:[/bold]")
            latest_version = data.get("latest_version")
            if latest_version:
                content_hash = latest_version.get("content_hash") or ""
                version_str = latest_version.get("semantic_version") or content_hash[:8]
                console.print(f"  Version: {version_str}")
                console.print(f"  Hash: {(latest_version.get('content_hash') or '-')[:12]}")
                created_at = latest_version.get("created_at")
                console.print(f"  Created: {format_time_ago(created_at)}")
            else:
                console.print("  [dim]No versions found[/dim]")

            # Action status section
            action_data = data.get("action")
            if action_data:
                console.print("\n[bold]Action Status:[/bold]")
                action_status_value = action_data.get("status")
                action_text = _format_action_status(action_status_value)
                console.print("  Status: ", end="")
                console.print(action_text)
                if action_data.get("job_id"):
                    console.print(f"  Job ID: [dim]{action_data.get('job_id')}[/dim]")

            console.print()

    except APIError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        raise typer.Exit(1)


@app.command(rich_help_panel="Manage")
def push(
    path: str = typer.Option(".", "--path", "-p", help="Path to environment directory"),
    name: Optional[str] = typer.Option(
        None, "--name", "-n", help="Override environment name (defaults to pyproject.toml name)"
    ),
    owner: Optional[str] = typer.Option(
        None,
        "--owner",
        "-o",
        help="Owner slug (user or team) to push to (for collaborators with write access)",
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
    rc: bool = typer.Option(False, "--rc", help="Bump or create a .rc pre-release (rc0 -> rc1)"),
    post: bool = typer.Option(
        False,
        "--post",
        help="Bump or create a .post release (post0 -> post1)",
    ),
) -> None:
    """Push environment to registry"""

    try:
        env_path = Path(path).resolve()

        # Display upstream environment info if metadata exists
        display_upstream_environment_info(env_path)

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
            if auto_bump or rc or post:
                flags_set = sum(bool(x) for x in (auto_bump, rc, post))
                if flags_set > 1:
                    console.print(
                        "[red]Error: --auto-bump, --rc, and --post are mutually exclusive[/red]"
                    )
                    raise typer.Exit(1)
                current_version = project_info.get("version")
                if not current_version:
                    console.print(
                        "[red]Error: No version found in pyproject.toml for auto-bump[/red]"
                    )
                    raise typer.Exit(1)

                if auto_bump:
                    new_version = bump_version(current_version)
                elif rc:
                    new_version = bump_rc_version(current_version)
                else:
                    new_version = bump_post_version(current_version)

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
            if owner:
                # Push to a specific owner (user or team) - for collaborators with write access
                resolve_data["owner_slug"] = owner
            elif team:
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

            # Extract Requires-Dist from wheel METADATA (includes URL dependencies)
            requires_dist = extract_requires_dist_from_wheel(wheel_path)

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
                    "requires_dist": requires_dist,  # Include full dependency specs from wheel
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
                    console.print(
                        "[dim]Alternatively, use the --auto-bump flag to push "
                        "a new version without content changes[/dim]"
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

                    # Check tarball size
                    tarball_size = Path(tmp.name).stat().st_size
                    tarball_size_formatted = format_file_size(tarball_size)
                    console.print(f"Source archive size: {tarball_size_formatted}")

                    if tarball_size > MAX_TARBALL_SIZE_LIMIT:
                        max_size_formatted = format_file_size(MAX_TARBALL_SIZE_LIMIT)
                        console.print(
                            f"\n[yellow]⚠ Warning: Your tarball size ({tarball_size_formatted}) "
                            f"exceeds the recommended limit of {max_size_formatted}.[/yellow]"
                        )
                        console.print(
                            "[yellow]Large environment uploads may cause issues. Consider:[/yellow]"
                        )
                        console.print(
                            "[yellow]  • Excluding large data files or model weights[/yellow]"
                        )
                        console.print(
                            "[yellow]  • Checking for accidentally included build "
                            "artifacts[/yellow]"
                        )
                        console.print(
                            "[yellow]  • Using .gitignore patterns to exclude unnecessary "
                            "files[/yellow]\n"
                        )

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
                            console.print(
                                "[dim]Alternatively, use the --auto-bump flag to push "
                                "a new version without content changes[/dim]"
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

                # Save or update environment hub metadata for future reference
                try:
                    prime_dir = env_path / ".prime"
                    prime_dir.mkdir(exist_ok=True)
                    metadata_path = prime_dir / ".env-metadata.json"

                    # Backwards compatibility: Migrate .env-metadata.json from root to .prime/
                    # This handles environments that were pulled/pushed before we moved
                    # to .prime/ subfolder
                    old_metadata_path = env_path / ".env-metadata.json"
                    migration_failed = False
                    if old_metadata_path.exists() and not metadata_path.exists():
                        try:
                            # Move the old file to the new location
                            old_metadata_path.rename(metadata_path)
                            console.print(
                                "[dim]Migrated environment metadata from root "
                                "to .prime/ subfolder[/dim]"
                            )
                        except (OSError, IOError) as e:
                            migration_failed = True
                            console.print(
                                f"[yellow]Warning: Could not migrate old .env-metadata.json "
                                f"file to .prime/ subfolder: {e}[/yellow]"
                            )
                    elif old_metadata_path.exists() and metadata_path.exists():
                        # Both exist - prefer the one in .prime/ and remove the old one
                        try:
                            old_metadata_path.unlink()
                        except (OSError, IOError):
                            console.print(
                                "[yellow]Warning: Could not remove old .env-metadata.json[/yellow]"
                            )

                    # Read existing metadata if it exists
                    existing_metadata = {}
                    if metadata_path.exists():
                        try:
                            with open(metadata_path, "r") as f:
                                existing_metadata = json.load(f)
                        except (json.JSONDecodeError, IOError) as e:
                            console.print(
                                f"[yellow]Warning: Could not read existing metadata: {e}[/yellow]"
                            )
                            existing_metadata = {}
                    elif migration_failed and old_metadata_path.exists():
                        # If migration failed, read from old location to preserve metadata
                        try:
                            with open(old_metadata_path, "r") as f:
                                existing_metadata = json.load(f)
                        except (json.JSONDecodeError, IOError) as e:
                            console.print(
                                f"[yellow]Warning: Could not read existing metadata from "
                                f"old location: {e}[/yellow]"
                            )
                            existing_metadata = {}

                    # Check if upstream (owner/name) changed
                    old_owner = existing_metadata.get("owner")
                    old_name = existing_metadata.get("name")
                    upstream_changed = False
                    if existing_metadata and (old_owner != owner_name or old_name != env_name):
                        upstream_changed = True

                    # Merge existing metadata with new push information
                    env_metadata = {
                        **existing_metadata,  # Preserve existing fields
                        "environment_id": env_id,
                        "owner": owner_name,
                        "name": env_name,
                        "pushed_at": datetime.now().isoformat(),
                        "wheel_sha256": wheel_sha256,
                    }

                    with open(metadata_path, "w") as f:
                        json.dump(env_metadata, f, indent=2)

                    if existing_metadata:
                        message = Text("Updated environment metadata in ", style="dim")
                        message.append(str(metadata_path), style="dim")
                        console.print(message)
                    else:
                        message = Text("Saved environment metadata to ", style="dim")
                        message.append(str(metadata_path), style="dim")
                        console.print(message)

                    # Report upstream change if it occurred
                    if upstream_changed:
                        upstream_message = Text("Upstream set to ", style="dim")
                        upstream_message.append(f"{owner_name}/{env_name}", style="dim")
                        console.print(upstream_message)
                except Exception as e:
                    console.print(
                        f"[yellow]Warning: Could not save environment metadata: {e}[/yellow]"
                    )

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


@app.command(no_args_is_help=True, rich_help_panel="Manage")
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


@app.command(no_args_is_help=True, rich_help_panel="Manage")
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
            # Check if the base directory exists and add index suffix if needed
            base_dir = Path.cwd() / name
            target_dir = base_dir
            if target_dir.exists():
                # Find the next available directory with index suffix
                index = 1
                while target_dir.exists():
                    target_dir = Path.cwd() / f"{name}-{index}"
                    index += 1
                console.print(
                    f"[yellow]Directory {base_dir} already exists. "
                    f"Using {target_dir} instead.[/yellow]"
                )

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

        # Create .env-metadata.json for proper resolution
        try:
            prime_dir = target_dir / ".prime"
            prime_dir.mkdir(exist_ok=True)
            metadata_path = prime_dir / ".env-metadata.json"
            env_metadata = {
                "environment_id": details.get("id"),
                "owner": owner,
                "name": name,
            }
            with open(metadata_path, "w") as f:
                json.dump(env_metadata, f, indent=2)
            message = Text("Created environment metadata at ", style="dim")
            message.append(str(metadata_path), style="dim")
            console.print(message)
        except Exception as e:
            console.print(f"[yellow]Warning: Could not create metadata file: {e}[/yellow]")

        try:
            all_files = list(target_dir.iterdir())
            # Filter out .prime directory and .env-metadata.json files
            # (created locally, not extracted)
            extracted_files = [
                f for f in all_files if f.name != ".prime" and f.name != ".env-metadata.json"
            ]
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


def extract_requires_dist_from_wheel(wheel_path: Path) -> List[str]:
    """Extract Requires-Dist entries from a wheel's METADATA file.

    A wheel is a zip file containing a .dist-info directory with a METADATA file.
    This function extracts all Requires-Dist entries which include dependencies.

    Args:
        wheel_path: Path to the wheel file

    Returns:
        List of Requires-Dist entries (e.g., ["requests>=2.0", "tau2@ git+https://..."])
    """
    requires_dist = []
    try:
        with zipfile.ZipFile(wheel_path, "r") as whl:
            # Find the METADATA file in the .dist-info directory
            metadata_files = [
                name for name in whl.namelist() if name.endswith(".dist-info/METADATA")
            ]
            if not metadata_files:
                return requires_dist

            metadata_content = whl.read(metadata_files[0]).decode("utf-8")

            # Parse the METADATA file (RFC 822 format)
            parser = EmailParser()
            metadata = parser.parsestr(metadata_content)

            # Get all Requires-Dist entries
            requires_dist = metadata.get_all("Requires-Dist") or []

    except (zipfile.BadZipFile, KeyError, UnicodeDecodeError) as e:
        console.print(f"[yellow]Warning: Could not extract metadata from wheel: {e}[/yellow]")

    return requires_dist


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


def bump_rc_version(version: str) -> str:
    """
    Bump or create an .post suffix.
    Examples:
      1.2.3 -> 1.2.3.post0
      1.2.3.post0 -> 1.2.3.post1
      1.2.3post2 -> 1.2.3post3
    """
    m = re.match(r"^(?P<base>.*?)(?:\.rc|rc)(?P<num>\d+)$", version)
    if m:
        base = m.group("base")
        num = int(m.group("num"))
        return f"{base}.rc{num + 1}"
    else:
        base = re.sub(r"([+-].*)$", "", version)
        return f"{base}.rc0"


def bump_post_version(version: str) -> str:
    """
    Bump or create an .post suffix.
    Examples:
      1.2.3 -> 1.2.3.post0
      1.2.3.post0 -> 1.2.3.post1
      1.2.3post2 -> 1.2.3post3
    """
    m = re.match(r"^(?P<base>.*?)(?:\.post|post)(?P<num>\d+)$", version)
    if m:
        base = m.group("base")
        num = int(m.group("num"))
        return f"{base}.post{num + 1}"
    else:
        base = re.sub(r"([+-].*)$", "", version)
        return f"{base}.post0"


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


def get_install_command(
    tool: str, wheel_url: str, package_name: str, no_upgrade: bool = False
) -> List[str]:
    """Generate install command for the specified tool.

    Args:
        tool: Package manager to use ('uv' or 'pip')
        wheel_url: URL to the wheel file
        package_name: Package name for targeted upgrade with -P flag (uv only)
        no_upgrade: If True, don't include upgrade flags (preserves locked dependencies)
    """
    if tool == "uv":
        cmd = ["uv", "pip", "install"]
        if not no_upgrade:
            # Use -P to only upgrade this package, not its dependencies
            cmd.extend(["-P", package_name])
        cmd.append(wheel_url)
        return cmd
    elif tool == "pip":
        cmd = ["pip", "install"]
        if not no_upgrade:
            cmd.append("--upgrade")
        cmd.append(wheel_url)
        return cmd
    else:
        raise ValueError(f"Unsupported package manager: {tool}. Use 'uv' or 'pip'.")


@app.command(no_args_is_help=True, rich_help_panel="Explore")
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
        Exception: If installation fails (caller should catch)
    """
    console.print(f"\n[cyan]Installing {env_id}@{version} with {tool}...[/cyan]")
    console.print(f"[dim]Command: {' '.join(cmd)}[/dim]")

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
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
        raise Exception(f"Installation failed with exit code {return_code}")

    console.print(f"\n[green]✓ Successfully installed {env_id}@{version}[/green]")


@app.command(no_args_is_help=True, rich_help_panel="Manage")
def install(
    env_ids: List[str] = typer.Argument(
        ..., help="Environment ID(s) to install (owner/name or local name)"
    ),
    with_tool: str = typer.Option(
        "uv",
        "--with",
        help="Package manager to use (uv or pip)",
    ),
    path: str = typer.Option(
        "./environments",
        "--path",
        "-p",
        help="Path to local environments directory (for local installs)",
    ),
    no_upgrade: bool = typer.Option(
        False,
        "--no-upgrade",
        help="Don't upgrade existing packages. Useful with locked dependencies (uv.lock).",
    ),
) -> None:
    """Install a verifiers environment.

    \b
    Examples:
        prime env install gsm8k                    # local install from ./environments
        prime env install gsm8k -p /path/to/envs   # local install from custom path
        prime env install owner/environment        # install from Prime Hub
        prime env install owner/environment@0.2.3  # specific version
        prime env install owner/environment --with pip
        prime env install env1 env2 env3           # install multiple
    """
    try:
        client = APIClient(require_auth=False)

        # Validate package manager
        if with_tool not in ["uv", "pip"]:
            console.print(
                f"[red]Error: Unsupported package manager '{with_tool}'. Use 'uv' or 'pip'.[/red]"
            )
            raise typer.Exit(1)

        # Check if tool is installed
        if not shutil.which(with_tool):
            console.print(f"[red]Error: {with_tool} is not installed.[/red]")
            raise typer.Exit(1)

        # De-dup environment IDs just in case
        env_ids = list(dict.fromkeys(env_ids))

        # Resolving and validating environments
        installable_envs = []
        failed_envs = []
        skipped_envs = []

        console.print(
            f"[bold]Resolving {len(env_ids)} "
            f"environment{'s' if len(env_ids) != 1 else ''}...[/bold]"
        )
        for env_id in env_ids:
            # Check if this is a local environment (no "/" in the name)
            local_name = env_id.split("@")[0]
            if "/" not in local_name:
                if not local_name or not local_name.strip():
                    skipped_envs.append((env_id, "Empty environment name"))
                    console.print("[yellow]⚠ Skipping: Empty environment name[/yellow]")
                    continue
                env_folder = local_name.replace("-", "_")
                env_path = Path(path) / env_folder
                if env_path.exists():
                    if with_tool == "uv":
                        cmd_parts = ["uv", "pip", "install", "-e", str(env_path)]
                    else:
                        cmd_parts = ["pip", "install", "-e", str(env_path)]
                    installable_envs.append((cmd_parts, local_name, "local", local_name))
                    console.print(f"[green]✓ Found local environment: {env_path}[/green]")
                else:
                    failed_envs.append((local_name, f"Local path not found: {env_path}"))
                    console.print(f"[red]✗ Local environment not found: {env_path}[/red]")
                    if "-" in local_name:
                        alt_path = Path(path) / local_name
                        if alt_path.exists():
                            console.print(
                                f"[yellow]  Hint: Found '{alt_path}' but expected "
                                f"'{env_path}'[/yellow]"
                            )
                            console.print(
                                "[yellow]  Python packages use underscores, not dashes. "
                                f"Rename folder to '{env_folder}'[/yellow]"
                            )
                continue

            # Validate environment ID format (owner/name)
            try:
                env_id, target_version = validate_env_id(env_id)
            except ValueError as e:
                skipped_envs.append((env_id, f"Invalid format: {e}"))
                console.print(f"[yellow]⚠ Skipping {env_id}: Invalid format[/yellow]")
                continue

            owner, name = env_id.split("/")

            # Fetch environment details
            try:
                details = fetch_environment_details(client, owner, name, target_version)
            except APIError as e:
                failed_envs.append((f"{env_id}@{target_version}", f"{e}"))
                console.print(f"[red]✗ Failed to resolve {env_id}@{target_version}: {e}[/red]")
                continue

            # Get both simple index URL and wheel URL
            simple_index_url = details.get("simple_index_url")
            wheel_url = process_wheel_url(details.get("wheel_url"))
            url_dependencies = details.get("url_dependencies", [])

            # Check if this is a private environment - pull, build, and install from cache
            if not simple_index_url and not wheel_url and details.get("visibility") == "PRIVATE":
                console.print("[dim]Private environment detected, pulling and building...[/dim]")
                try:
                    # Pull, build, and get actual version (resolves "latest" from pyproject.toml)
                    wheel_path, resolved_version = _pull_and_build_private_env(
                        client, owner, name, target_version, details
                    )
                    normalized_name = normalize_package_name(name)
                    if with_tool == "uv":
                        cmd_parts = ["uv", "pip", "install"]
                        if not no_upgrade:
                            # Use -P to only upgrade this package, not its dependencies
                            cmd_parts.extend(["-P", normalized_name])
                        cmd_parts.append(str(wheel_path))
                    else:
                        cmd_parts = ["pip", "install", str(wheel_path)]
                        if not no_upgrade:
                            cmd_parts.append("--upgrade")
                    installable_envs.append((cmd_parts, env_id, resolved_version, name))
                    console.print(f"[green]✓ Built {env_id}@{resolved_version}[/green]")
                except Exception as e:
                    failed_envs.append((f"{env_id}@{target_version}", f"Failed to build: {e}"))
                    console.print(
                        f"[red]✗ Failed to build private environment {env_id}@{target_version}: "
                        f"{e}[/red]"
                    )
                continue
            elif not simple_index_url and not wheel_url:
                skipped_envs.append((f"{env_id}@{target_version}", "No installation method"))
                console.print(
                    f"[yellow]⚠ Skipping {env_id}@{target_version}: "
                    f"No installation method available[/yellow]"
                )
                console.print(
                    "[dim]  Use 'prime env info' to see available options "
                    "or 'pull' to download source.[/dim]"
                )
                continue

            console.print(f"[green]✓ Found {env_id}@{target_version}[/green]")

            cmd_parts = _build_install_command(
                name,
                target_version,
                simple_index_url,
                wheel_url,
                with_tool,
                no_upgrade,
                url_dependencies,
            )
            if not cmd_parts:
                skipped_envs.append((f"{env_id}@{target_version}", "No installation method"))
                console.print(
                    f"[yellow]⚠ Skipping {env_id}@{target_version}: No installation method[/yellow]"
                )
                continue

            installable_envs.append((cmd_parts, env_id, target_version, name))

        if not installable_envs:
            console.print("[red]Error: Unable to resolve installable environments[/red]")
            raise typer.Exit(1)

        # Install resolved environments
        installed_envs = []
        install_failed_envs = []

        console.print(
            f"\n[bold]Installing {len(installable_envs)} "
            f"environment{'s' if len(installable_envs) != 1 else ''}...[/bold]"
        )
        for cmd_parts, env_id, target_version, name in installable_envs:
            try:
                execute_install_command(cmd_parts, env_id, target_version, with_tool)
                installed_envs.append((env_id, target_version))

                # Display usage instructions
                console.print("\n[dim]Use in Python:[/dim]")
                console.print("  from verifiers import load_environment")
                console.print(f"  env = load_environment('{name}')")
            except FileNotFoundError:
                error_msg = f"{cmd_parts[0]} command not found"
                install_failed_envs.append((f"{env_id}@{target_version}", error_msg))
                console.print(f"[red]✗ Installation failed: {error_msg}[/red]")
            except Exception as e:
                install_failed_envs.append((f"{env_id}@{target_version}", str(e)))
                console.print(f"[red]✗ Installation failed: {e}[/red]")

        # Display final summary of installed/failed environments
        if installed_envs:
            console.print(
                f"\n[bold]Installed {len(installed_envs)} "
                f"environment{'s' if len(installed_envs) != 1 else ''}:[/bold]"
            )
            for env_id, version in installed_envs:
                console.print(f"[green]✓ {env_id}@{version}[/green]")

        if install_failed_envs:
            console.print(
                f"\n[bold]Failed to install {len(install_failed_envs)} "
                f"environment{'s' if len(install_failed_envs) != 1 else ''}:[/bold]"
            )
            for env_id, reason in install_failed_envs:
                console.print(f"[red]✗ {env_id} - {reason}")

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


@app.command(no_args_is_help=True, rich_help_panel="Manage")
def uninstall(
    env_name: str = typer.Argument(..., help="Environment name to uninstall"),
    with_tool: str = typer.Option(
        "uv",
        "--with",
        help="Package manager to use (uv or pip)",
    ),
) -> None:
    """Uninstall a verifiers environment.

    \b
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
app.add_typer(version_app, name="version", rich_help_panel="Manage")


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
                        created_date = iso_timestamp(created_date)
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


@app.command(no_args_is_help=True, rich_help_panel="Manage")
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


def _safe_tar_extract(tar: tarfile.TarFile, dest_path: Path) -> None:
    """Safely extract tar archive, preventing path traversal and symlink attacks.

    Args:
        tar: Open tarfile object
        dest_path: Destination directory for extraction

    Raises:
        ValueError: If archive contains unsafe paths, symlinks, or hardlinks
    """
    dest_path = dest_path.resolve()

    for member in tar.getmembers():
        member_path = Path(member.name)

        # Block symlinks - they can be used to write outside destination
        # (e.g., symlink "evil" -> "/tmp", then file "evil/malicious.txt")
        if member.issym():
            raise ValueError(f"Refusing to extract symlink: {member.name}")

        # Block hardlinks - they can also be used for attacks
        if member.islnk():
            raise ValueError(f"Refusing to extract hardlink: {member.name}")

        # Block absolute paths
        if member_path.is_absolute():
            raise ValueError(f"Refusing to extract absolute path: {member.name}")

        # Block path traversal
        if ".." in member_path.parts:
            raise ValueError(f"Refusing to extract path with '..': {member.name}")

        # Verify resolved path is within destination
        target_path = (dest_path / member_path).resolve()
        if not target_path.is_relative_to(dest_path):
            raise ValueError(f"Path escapes destination directory: {member.name}")

    # All members validated, safe to extract
    tar.extractall(dest_path)


def _get_env_cache_dir() -> Path:
    """Get the cache directory for private environment wheels."""
    cache_dir = Path.home() / ".prime" / "wheel_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _validate_path_component(component: str, component_name: str) -> None:
    """Validate a path component doesn't contain traversal sequences.

    Args:
        component: The path component to validate (owner, name, or version)
        component_name: Name of the component for error messages

    Raises:
        ValueError: If component contains unsafe characters
    """
    if not component:
        raise ValueError(f"{component_name} cannot be empty")

    # Block path traversal sequences
    if ".." in component:
        raise ValueError(f"{component_name} cannot contain '..'")

    # Block path separators
    if "/" in component or "\\" in component:
        raise ValueError(f"{component_name} cannot contain path separators")

    # Block null bytes
    if "\x00" in component:
        raise ValueError(f"{component_name} cannot contain null bytes")


def _get_version_from_pyproject(env_path: Path) -> Optional[str]:
    """Extract version from pyproject.toml in the environment directory."""
    pyproject_path = env_path / "pyproject.toml"
    if not pyproject_path.exists():
        return None
    try:
        pyproject_data = toml.load(pyproject_path)
        return pyproject_data.get("project", {}).get("version")
    except Exception:
        return None


def _pull_and_build_private_env(
    client: APIClient,
    owner: str,
    name: str,
    version: str,
    details: Dict[str, Any],
) -> Tuple[Path, str]:
    """Pull a private environment, build it, and return the wheel path and resolved version.

    Args:
        client: API client with authentication
        owner: Environment owner
        name: Environment name
        version: Environment version (may be "latest")
        details: Environment details from API

    Returns:
        Tuple of (wheel_path, resolved_version)

    Raises:
        Exception: If download, extraction, or build fails
    """
    # Validate path components to prevent directory traversal
    _validate_path_component(owner, "owner")
    _validate_path_component(name, "name")
    _validate_path_component(version, "version")

    download_url = details.get("package_url")
    if not download_url:
        raise ValueError("No downloadable package found for private environment")

    cache_dir = _get_env_cache_dir()

    # If version is not "latest", check cache directly
    if version != "latest":
        env_cache_path = cache_dir / owner / name / version
        if not env_cache_path.resolve().is_relative_to(cache_dir.resolve()):
            raise ValueError("Cache path escapes cache directory")
        wheel_cache_path = env_cache_path / "dist"
        if wheel_cache_path.exists():
            existing_wheels = list(wheel_cache_path.glob("*.whl"))
            if existing_wheels:
                console.print(f"[dim]Using cached wheel at {existing_wheels[0]}[/dim]")
                return existing_wheels[0], version

    # Download to temp directory first to determine actual version
    temp_extract_dir = None
    temp_file_path = None
    try:
        temp_extract_dir = tempfile.mkdtemp(prefix="prime_env_")
        temp_extract_path = Path(temp_extract_dir)

        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
            temp_file_path = tmp.name
            headers = {}
            if client.api_key:
                headers["Authorization"] = f"Bearer {client.api_key}"

            with httpx.stream("GET", download_url, headers=headers, timeout=60.0) as resp:
                resp.raise_for_status()
                with open(tmp.name, "wb") as f:
                    for chunk in resp.iter_bytes(chunk_size=8192):
                        f.write(chunk)

            # Extract to temp path (with path traversal protection)
            with tarfile.open(tmp.name, "r:gz") as tar:
                _safe_tar_extract(tar, temp_extract_path)

        # Get actual version from pyproject.toml
        actual_version = _get_version_from_pyproject(temp_extract_path) or version
        _validate_path_component(actual_version, "version")

        # Now we know the real version - check if it's already cached
        env_cache_path = cache_dir / owner / name / actual_version
        if not env_cache_path.resolve().is_relative_to(cache_dir.resolve()):
            raise ValueError("Cache path escapes cache directory")
        wheel_cache_path = env_cache_path / "dist"

        if wheel_cache_path.exists():
            existing_wheels = list(wheel_cache_path.glob("*.whl"))
            if existing_wheels:
                console.print(f"[dim]Using cached wheel at {existing_wheels[0]}[/dim]")
                return existing_wheels[0], actual_version

        # Move extracted content to final cache location
        env_cache_path.mkdir(parents=True, exist_ok=True)
        for item in temp_extract_path.iterdir():
            shutil.move(str(item), str(env_cache_path / item.name))

    finally:
        if temp_file_path and Path(temp_file_path).exists():
            Path(temp_file_path).unlink()
        if temp_extract_dir and Path(temp_extract_dir).exists():
            shutil.rmtree(temp_extract_dir, ignore_errors=True)

    # Build the wheel
    console.print("[dim]Building wheel...[/dim]")
    try:
        if shutil.which("uv"):
            subprocess.run(
                ["uv", "build", "--wheel", "--out-dir", "dist"],
                cwd=env_cache_path,
                capture_output=True,
                text=True,
                check=True,
            )
        else:
            subprocess.run(
                [sys.executable, "-m", "build", "--wheel", str(env_cache_path)],
                capture_output=True,
                text=True,
                check=True,
            )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to build wheel: {e.stderr}") from e

    # Find the built wheel
    wheels = list(wheel_cache_path.glob("*.whl"))
    if not wheels:
        raise RuntimeError("No wheel file found after build")

    wheel_path = wheels[0]

    # Create metadata file for tracking
    try:
        prime_dir = env_cache_path / ".prime"
        prime_dir.mkdir(exist_ok=True)
        metadata_path = prime_dir / ".env-metadata.json"
        env_metadata = {
            "environment_id": details.get("id"),
            "owner": owner,
            "name": name,
            "version": actual_version,
            "cached_at": datetime.now().isoformat(),
            "wheel_path": str(wheel_path),
        }
        with open(metadata_path, "w") as f:
            json.dump(env_metadata, f, indent=2)
    except Exception:
        pass  # Non-critical if metadata save fails

    return wheel_path, actual_version


def _is_environment_installed(env_name: str, required_version: Optional[str] = None) -> bool:
    """Check if an environment package is installed."""
    try:
        pkg_name = normalize_package_name(env_name)
        result = subprocess.run(
            ["uv", "pip", "show", pkg_name],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            return False

        if required_version and required_version != "latest":
            for line in result.stdout.splitlines():
                if line.startswith("Version:"):
                    installed_version = line.split(":", 1)[1].strip()
                    return installed_version == required_version
            return False

        return True
    except Exception:
        return False


def _build_install_command(
    name: str,
    version: str,
    simple_index_url: Optional[str],
    wheel_url: Optional[str],
    tool: str = "uv",
    no_upgrade: bool = False,
    url_dependencies: Optional[List[str]] = None,
) -> Optional[List[str]]:
    """Build install command for an environment. Returns None if no install method available.

    Args:
        name: Package name
        version: Package version
        simple_index_url: Simple index URL for the package
        wheel_url: Direct wheel URL
        tool: Package manager to use ('uv' or 'pip')
        no_upgrade: If True, don't include upgrade flags (preserves locked dependencies)
        url_dependencies: List of URL dependencies to install as direct requirements
    """
    normalized_name = normalize_package_name(name)

    if simple_index_url:
        if tool == "uv":
            cmd = ["uv", "pip", "install"]
            if not no_upgrade:
                # Use -P to only upgrade this package, not its dependencies
                cmd.extend(["-P", normalized_name])
            if version and version != "latest":
                cmd.append(f"{normalized_name}=={version}")
            else:
                cmd.append(normalized_name)
            # Add URL dependencies as direct requirements (uv requires this)
            if url_dependencies:
                cmd.extend(url_dependencies)
            cmd.extend(["--extra-index-url", simple_index_url])
            return cmd
        else:  # pip
            cmd = ["pip", "install"]
            if not no_upgrade:
                cmd.append("--upgrade")
            if version and version != "latest":
                cmd.append(f"{normalized_name}=={version}")
            else:
                cmd.append(normalized_name)
            # Add URL dependencies for consistency
            if url_dependencies:
                cmd.extend(url_dependencies)
            cmd.extend(["--extra-index-url", simple_index_url])
            return cmd
    elif wheel_url:
        try:
            cmd = get_install_command(tool, wheel_url, normalized_name, no_upgrade)
            # Add URL dependencies for wheel-only installs too
            if url_dependencies:
                cmd.extend(url_dependencies)
            return cmd
        except ValueError:
            return None

    return None


def _install_single_environment(env_slug: str, tool: str = "uv") -> bool:
    """Install a single environment from the hub. Returns True on success."""
    try:
        env_id, version = validate_env_id(env_slug)
    except ValueError as e:
        console.print(f"[red]Invalid environment format: {e}[/red]")
        return False

    owner, name = env_id.split("/")

    try:
        client = APIClient(require_auth=False)
        details = fetch_environment_details(client, owner, name, version)
    except APIError as e:
        console.print(f"[red]Failed to find environment {env_slug}: {e}[/red]")
        return False

    simple_index_url = details.get("simple_index_url")
    wheel_url = process_wheel_url(details.get("wheel_url"))
    url_dependencies = details.get("url_dependencies", [])

    if not simple_index_url and not wheel_url:
        if details.get("visibility") == "PRIVATE":
            console.print(
                f"[red]Cannot install private environment {env_slug}.[/red]\n"
                "[yellow]Use 'prime env pull' to download and install locally.[/yellow]"
            )
        else:
            console.print(f"[red]No installation method available for {env_slug}[/red]")
        return False

    cmd_parts = _build_install_command(
        name, version, simple_index_url, wheel_url, tool, url_dependencies=url_dependencies
    )
    if not cmd_parts:
        console.print(f"[red]Failed to build install command for {env_slug}[/red]")
        return False

    try:
        execute_install_command(cmd_parts, env_id, version, tool)
        return True
    except Exception as e:
        console.print(f"[red]Installation failed: {e}[/red]")
        return False


def run_eval(
    environment: str,
    model: str,
    num_examples: Optional[int],
    rollouts_per_example: Optional[int],
    max_concurrent: Optional[int],
    max_concurrent_generation: Optional[int],
    max_concurrent_scoring: Optional[int],
    max_tokens: Optional[int],
    temperature: Optional[float],
    sampling_args: Optional[str],
    verbose: bool,
    no_interleave_scoring: bool,
    state_columns: Optional[str],
    save_results: bool,
    save_every: int,
    independent_scoring: bool,
    save_to_hf_hub: bool,
    hf_hub_dataset_name: Optional[str],
    env_args: Optional[str],
    extra_env_kwargs: Optional[str],
    env_dir_path: Optional[str],
    api_key_var: Optional[str],
    api_base_url: Optional[str],
    skip_upload: bool,
    env_path: Optional[str],
    endpoints_path: Optional[str] = None,
    headers: Optional[List[str]] = None,
) -> None:
    """
    Run verifiers' vf-eval with Prime Inference
    """
    is_slug = (
        "/" in environment and not environment.startswith("./") and not environment.startswith("/")
    )

    upstream_owner = None
    upstream_name = None
    env_name_for_vf_eval = environment

    if is_slug:
        env_slug = environment
        requested_version = "latest"
        if "@" in environment:
            env_slug, requested_version = environment.rsplit("@", 1)

        parts = env_slug.split("/")
        if len(parts) == 2 and parts[0] and parts[1]:
            upstream_owner, upstream_name = parts
            env_name_for_vf_eval = upstream_name
            console.print(
                f"[dim]Using upstream environment {upstream_owner}/{upstream_name}[/dim]\n"
            )

            if not _is_environment_installed(upstream_name, requested_version):
                console.print(f"[cyan]Installing {environment}...[/cyan]")
                if not _install_single_environment(environment):
                    raise typer.Exit(1)
                console.print()

            is_resolved = True
        else:
            console.print(f"[red]Invalid environment slug format: {environment}[/red]")
            raise typer.Exit(1)
    else:
        check_path = Path(env_path) if env_path else Path.cwd()
        is_resolved = display_upstream_environment_info(
            env_path=check_path, environment_name=environment
        )
        if not is_resolved and not skip_upload:
            console.print(
                "[yellow]Evaluation results will not be uploaded or viewable on the platform "
                "without a specified upstream environment. Use `prime env push` "
                "to set an upstream.[/yellow]"
            )

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

    cmd = ["uv", "run", "vf-eval", env_name_for_vf_eval]

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
    if extra_env_kwargs:
        cmd += ["-x", extra_env_kwargs]
    if env_dir_path:
        cmd += ["-p", env_dir_path]
    if num_examples is not None:
        cmd += ["-n", str(num_examples)]
    if rollouts_per_example is not None:
        cmd += ["-r", str(rollouts_per_example)]
    if max_concurrent is not None:
        cmd += ["-c", str(max_concurrent)]
    if max_concurrent_generation is not None:
        cmd += ["--max-concurrent-generation", str(max_concurrent_generation)]
    if max_concurrent_scoring is not None:
        cmd += ["--max-concurrent-scoring", str(max_concurrent_scoring)]
    if max_tokens is not None:
        cmd += ["-t", str(max_tokens)]
    if temperature is not None:
        cmd += ["-T", str(temperature)]
    if sampling_args:
        cmd += ["-S", sampling_args]
    if verbose:
        cmd += ["-v"]
    if no_interleave_scoring:
        cmd += ["-N"]
    if state_columns:
        cmd += ["-C", state_columns]
    if save_results or not skip_upload:
        cmd += ["-s"]
    if save_every is not None:
        cmd += ["-f", str(save_every)]
    if independent_scoring:
        cmd += ["-R"]
    if save_to_hf_hub:
        cmd += ["-H"]
    if hf_hub_dataset_name:
        cmd += ["-D", hf_hub_dataset_name]
    if endpoints_path:
        cmd += ["-e", endpoints_path]
    if headers:
        for header in headers:
            cmd += ["--header", header]

    # Generate job_id for end-to-end tracing of eval runs
    eval_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_uuid = str(uuid.uuid4())[:8]
    sanitized_env = env_name_for_vf_eval.replace("-", "_").replace("/", "_")
    sanitized_model = model.replace("/", "_").replace("-", "_")
    job_id = f"{sanitized_env}_{sanitized_model}_{eval_timestamp}_{job_uuid}"

    # Pass tracking header to vf-eval
    cmd += ["--header", f"X-PI-Job-Id: {job_id}"]

    # If a team is configured, pass it to vf-eval via header
    if config.team_id:
        cmd += ["--header", f"X-Prime-Team-ID: {config.team_id}"]

    console.print(f"[dim]Eval job_id: {job_id}[/dim]")

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

    # Automatically push to hub after successful eval (unless --skip-upload is used)
    if not skip_upload:
        if is_resolved:
            try:
                if is_slug and upstream_owner and upstream_name:
                    push_eval_results_to_hub(
                        env_name=env_name_for_vf_eval,
                        model=model,
                        job_id=job_id,
                        env_path=Path(env_path) if env_path else None,
                        upstream_slug=f"{upstream_owner}/{upstream_name}",
                    )
                else:
                    check_path = Path(env_path) if env_path else Path.cwd()
                    push_eval_results_to_hub(
                        env_name=env_name_for_vf_eval,
                        model=model,
                        job_id=job_id,
                        env_path=check_path,
                    )
            except Exception as e:
                console.print(f"[red]Failed to push results to hub:[/red] {e}")
                console.print("[yellow]Evaluation completed but results were not pushed.[/yellow]")
                raise typer.Exit(1)
        else:
            console.print(
                "[dim]No upstream environment found. Skipped uploading evaluation "
                "results to platform.\nUse `prime env push` to set an "
                "upstream, or use `--env-path` to specify the correct path to the "
                "environment if it's not the current directory.[/dim]"
            )
    else:
        console.print("[dim]Skipped uploading evaluation results[/dim]")


@app.command(
    "eval",
    no_args_is_help=True,
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    deprecated=True,
    rich_help_panel="Deprecated",
)
def eval_env(
    ctx: typer.Context,
    environment: str = typer.Argument(
        ...,
        help="Environment name (e.g. 'wordle') or slug (e.g. 'primeintellect/wordle')",
    ),
    model: str = typer.Option(
        "openai/gpt-4.1-mini",
        "--model",
        "-m",
        help=(
            "Model to use (e.g. 'openai/gpt-4.1-mini', 'prime-intellect/intellect-3', "
            "see 'prime inference models' for available models)"
        ),
    ),
    num_examples: Optional[int] = typer.Option(
        None, "--num-examples", "-n", help="Number of examples"
    ),
    rollouts_per_example: Optional[int] = typer.Option(
        None, "--rollouts-per-example", "-r", help="Rollouts per example"
    ),
    max_concurrent: Optional[int] = typer.Option(
        32, "--max-concurrent", "-c", help="Max concurrent requests"
    ),
    max_concurrent_generation: Optional[int] = typer.Option(
        None, "--max-concurrent-generation", help="Max concurrent generation requests"
    ),
    max_concurrent_scoring: Optional[int] = typer.Option(
        None, "--max-concurrent-scoring", help="Max concurrent scoring requests"
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
    no_interleave_scoring: bool = typer.Option(
        False, "--no-interleave-scoring", "-N", help="Disable interleaving of scoring"
    ),
    state_columns: Optional[str] = typer.Option(
        None,
        "--state-columns",
        "-C",
        help="Comma-separated list of state columns to save (e.g., 'turn,timing')",
    ),
    save_results: bool = typer.Option(False, "--save-results", "-s", help="Save results to disk"),
    save_every: int = typer.Option(-1, "--save-every", "-f", help="Save dataset every n rollouts"),
    independent_scoring: bool = typer.Option(
        False,
        "--independent-scoring",
        "-R",
        help="Score each rollout individually instead of scoring by group",
    ),
    save_to_hf_hub: bool = typer.Option(False, "--save-to-hf-hub", "-H", help="Save to HF Hub"),
    hf_hub_dataset_name: Optional[str] = typer.Option(
        None, "--hf-hub-dataset-name", "-D", help="HF Hub dataset name"
    ),
    env_args: Optional[str] = typer.Option(
        None, "--env-args", "-a", help='Environment args as JSON, e.g. \'{"key":"value"}\''
    ),
    extra_env_kwargs: Optional[str] = typer.Option(
        None,
        "--extra-env-kwargs",
        "-x",
        help='Extra environment kwargs as JSON, e.g. \'{"key":"value"}\'',
    ),
    env_dir_path: Optional[str] = typer.Option(
        None, "--env-dir-path", "-p", help="Path to environments directory"
    ),
    api_key_var: Optional[str] = typer.Option(
        None, "--api-key-var", "-k", help="Override api key variable instead of using PRIME_API_KEY"
    ),
    api_base_url: Optional[str] = typer.Option(
        None,
        "--api-base-url",
        "-b",
        help=(
            "Override api base url variable instead of using prime inference url, "
            "should end in '/v1'"
        ),
    ),
    skip_upload: bool = typer.Option(
        False,
        "--skip-upload",
        help="Skip uploading results to Prime Evals Hub (results are uploaded by default)",
    ),
    env_path: Optional[str] = typer.Option(
        None,
        "--env-path",
        help=(
            "Path to the environment directory "
            "(used to locate .prime/.env-metadata.json for upstream resolution)"
        ),
    ),
) -> None:
    """Use 'prime eval' instead."""

    run_eval(
        environment=environment,
        model=model,
        num_examples=num_examples,
        rollouts_per_example=rollouts_per_example,
        max_concurrent=max_concurrent,
        max_concurrent_generation=max_concurrent_generation,
        max_concurrent_scoring=max_concurrent_scoring,
        max_tokens=max_tokens,
        temperature=temperature,
        sampling_args=sampling_args,
        verbose=verbose,
        no_interleave_scoring=no_interleave_scoring,
        state_columns=state_columns,
        save_results=save_results,
        save_every=save_every,
        independent_scoring=independent_scoring,
        save_to_hf_hub=save_to_hf_hub,
        hf_hub_dataset_name=hf_hub_dataset_name,
        env_args=env_args,
        extra_env_kwargs=extra_env_kwargs,
        env_dir_path=env_dir_path,
        api_key_var=api_key_var,
        api_base_url=api_base_url,
        skip_upload=skip_upload,
        env_path=env_path,
        endpoints_path=None,
        headers=None,
    )
