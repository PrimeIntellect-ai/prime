import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from datetime import datetime

# Wheel METADATA files use RFC 822 format (PEP 566), same as email headers
from email.parser import Parser as EmailParser
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import httpx
import toml
import typer
from gitignore_parser import parse_gitignore
from rich.table import Table
from rich.text import Text

from ..client import APIClient, APIError
from ..utils import (
    PlainAwareTyperGroup,
    PlainTyper,
    get_console,
    json_output_help,
    output_data_as_json,
)
from ..utils.env_metadata import find_environment_metadata
from ..utils.environment_runtime import (
    VERIFIERS_V1,
    classify_runtime_from_metadata,
    parse_runtime_option,
)
from ..utils.formatters import format_file_size
from ..utils.prompt import (
    any_provided,
    prompt_for_value,
    require_selection,
    validate_env_var_name,
)
from ..utils.time_utils import format_time_ago, iso_timestamp
from .config import TEAM_ID_PATTERN

ENV_COMMAND_ORDER = ("list", "info", "pull", "push", "delete", "version", "secret", "var")


class _EnvGroup(PlainAwareTyperGroup):
    def list_commands(self, ctx):
        return sorted(super().list_commands(ctx), key=ENV_COMMAND_ORDER.index)


app = PlainTyper(
    cls=_EnvGroup,
    help="Manage environments (list, info, pull, push, delete, version, secret, var)",
    no_args_is_help=True,
)
console = get_console()

# Constants
MAX_FILES_TO_SHOW = 10
DEFAULT_HASH_LENGTH = 8
DEFAULT_LIST_LIMIT = 20
MAX_TARBALL_SIZE_LIMIT = 250 * 1024 * 1024  # 250MB

# Secret subcommand app
secret_app = PlainTyper(help="Manage environment secrets", no_args_is_help=True)
app.add_typer(secret_app, name="secret")

# Variable subcommand app
var_app = PlainTyper(help="Manage environment variables", no_args_is_help=True)
app.add_typer(var_app, name="var")

ENV_LIST_JSON_HELP = json_output_help(
    ".environments[] = {environment, description, visibility, version, stars, updated_at, tags[]?}",
    ".total = number",
    ".page = number",
    ".per_page = number",
)

ENV_INFO_JSON_HELP = json_output_help(
    ". = environment version object from the Environments Hub",
    ".latest_version? = {semantic_version?, content_hash?, created_at?}",
)

ENV_SECRET_LIST_JSON_HELP = json_output_help(
    ".secrets[] = {id, name, source, description?, createdAt, updatedAt?}",
)

ENV_SECRET_DETAIL_JSON_HELP = json_output_help(
    ". = {id, name, source, description?, value?, createdAt, updatedAt?}",
)

ENV_SECRET_LINK_JSON_HELP = json_output_help(
    ". = {id, secretId, secretName, environmentId, createdAt}",
)

ENV_VAR_LIST_JSON_HELP = json_output_help(
    ".variables[] = {id, name, value, description?, createdAt, updatedAt?}",
)

ENV_VAR_DETAIL_JSON_HELP = json_output_help(
    ". = {id, name, value, description?, createdAt, updatedAt?}",
)


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


def _resolve_environment(environment: Optional[str]) -> Tuple[str, str]:
    """Resolve environment slug from argument or auto-detect from current directory."""
    if environment:
        return _parse_environment_slug(environment)

    metadata = find_environment_metadata()
    if metadata:
        owner = metadata.get("owner")
        name = metadata.get("name")
        if owner and name:
            console.print(f"[dim]Using environment: {owner}/{name}[/dim]")
            return owner, name

    console.print(
        "[red]Error: No environment specified and none detected in current directory[/red]"
    )
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


def _environment_ref(
    owner: Any,
    name: Any,
    *,
    environment_id: Any = None,
    version: Any = None,
) -> Dict[str, str]:
    if not owner or not name:
        return {}
    ref = {"owner": str(owner), "name": str(name)}
    if environment_id is not None:
        ref["environment_id"] = str(environment_id)
    if version is not None:
        ref["version"] = str(version)
    return ref


def _environment_fork_chain(
    metadata: Dict[str, Any],
    upstream: Dict[str, str] | None = None,
) -> List[Dict[str, str]]:
    chain: List[Dict[str, str]] = []
    for value in metadata.get("fork_chain") or ():
        if isinstance(value, dict):
            ref = _environment_ref(
                value.get("owner"),
                value.get("name"),
                environment_id=value.get("environment_id"),
                version=value.get("version"),
            )
            if ref:
                chain.append(ref)
    if isinstance(metadata.get("origin"), dict):
        origin = metadata["origin"]
        ref = _environment_ref(
            origin.get("owner"),
            origin.get("name"),
            environment_id=origin.get("environment_id"),
            version=origin.get("version"),
        )
        if ref:
            chain.insert(0, ref)
    if upstream:
        chain.append(upstream)

    deduped: List[Dict[str, str]] = []
    seen: set[Tuple[str, str, str, str]] = set()
    for ref in chain:
        key = (
            ref.get("owner", ""),
            ref.get("name", ""),
            ref.get("environment_id", ""),
            ref.get("version", ""),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(ref)
    return deduped


def _environment_push_metadata(
    existing_metadata: Dict[str, Any],
    *,
    environment_id: str,
    owner: str,
    name: str,
    version: Any,
    pushed_at: str,
    wheel_sha256: str,
) -> Dict[str, Any]:
    old_owner = existing_metadata.get("owner")
    old_name = existing_metadata.get("name")
    upstream_changed = bool(existing_metadata and (old_owner != owner or old_name != name))
    old_upstream = _environment_ref(
        old_owner,
        old_name,
        environment_id=existing_metadata.get("environment_id"),
        version=existing_metadata.get("version"),
    )
    fork_chain = _environment_fork_chain(
        existing_metadata,
        old_upstream if upstream_changed else None,
    )
    existing_forked_from: Dict[str, str] = {}
    if isinstance(existing_metadata.get("forked_from"), dict):
        forked_from = existing_metadata["forked_from"]
        existing_forked_from = _environment_ref(
            forked_from.get("owner"),
            forked_from.get("name"),
            environment_id=forked_from.get("environment_id"),
            version=forked_from.get("version"),
        )
    stale_fork_keys = {"forked_from", "origin", "fork_chain"}
    metadata = {
        key: value for key, value in existing_metadata.items() if key not in stale_fork_keys
    } | {
        "environment_id": environment_id,
        "owner": owner,
        "name": name,
        "pushed_at": pushed_at,
        "wheel_sha256": wheel_sha256,
    }
    if version is not None:
        metadata["version"] = version
    if fork_chain:
        metadata["origin"] = fork_chain[0]
        metadata["fork_chain"] = fork_chain
    if upstream_changed and old_upstream:
        metadata["forked_from"] = old_upstream
    elif existing_forked_from:
        metadata["forked_from"] = existing_forked_from
    return metadata


def should_include_file_in_archive(file_path: Path, base_path: Path) -> bool:
    """Determine if a file should be included in the archive based on filtering rules."""
    if not file_path.is_file():
        return False

    # Skip symlinks - they cause extraction failures in _safe_tar_extract
    if file_path.is_symlink():
        return False

    rel_path = file_path.relative_to(base_path)

    # Skip hidden files
    if file_path.name.startswith("."):
        return False

    # Skip files in __pycache__ directories
    if "__pycache__" in rel_path.parts:
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


def _build_gitignore_matcher(env_path: Path) -> Optional[Callable[[str], bool]]:
    """Build a matcher for the root .gitignore file, if present."""
    gitignore_path = env_path / ".gitignore"
    if not gitignore_path.exists():
        return None
    return parse_gitignore(str(gitignore_path), base_dir=str(env_path))


def _collect_archive_files(env_path: Path) -> List[Path]:
    """Collect archive file paths in deterministic order, honoring .gitignore."""
    ignore_matcher = _build_gitignore_matcher(env_path)
    files_by_rel_path: Dict[str, Path] = {}

    def maybe_add_file(file_path: Path) -> None:
        if not should_include_file_in_archive(file_path, env_path):
            return
        if ignore_matcher is not None and ignore_matcher(str(file_path)):
            return

        rel_path = str(file_path.relative_to(env_path)).replace("\\", "/")
        files_by_rel_path[rel_path] = file_path

    for pattern in ["README.md", "pyproject.toml", "*.py"]:
        for file_path in sorted(env_path.glob(pattern), key=lambda p: p.name):
            maybe_add_file(file_path)

    def is_nested_dir_ignored(dir_path: Path) -> bool:
        """Check if a nested directory should be pruned from traversal."""
        if not should_include_directory_in_archive(dir_path):
            return True
        if ignore_matcher is not None and ignore_matcher(str(dir_path)):
            return True
        return False

    for subdir in sorted(env_path.iterdir(), key=lambda path: path.name):
        if not should_include_directory_in_archive(subdir):
            continue
        if is_nested_dir_ignored(subdir):
            continue

        for root, dirnames, filenames in os.walk(subdir):
            root_path = Path(root)
            dirnames[:] = sorted(
                dirname for dirname in dirnames if not is_nested_dir_ignored(root_path / dirname)
            )
            for filename in sorted(filenames):
                maybe_add_file(root_path / filename)

    return [files_by_rel_path[rel_path] for rel_path in sorted(files_by_rel_path)]


def compute_content_hash(env_path: Path) -> str:
    """Compute deterministic, cross-platform content hash for environment files.

    Args:
        env_path: Path to the environment directory

    Returns:
        SHA256 hexdigest of the environment content
    """
    content_hasher = hashlib.sha256()

    for file_path in _collect_archive_files(env_path):
        normalized_path = str(file_path.relative_to(env_path)).replace("\\", "/")
        content_hasher.update(f"file:{normalized_path}".encode("utf-8"))
        try:
            with open(file_path, "rb") as f:
                content_hasher.update(f.read())
        except IOError:
            # Skip files that can't be read
            pass

    return content_hasher.hexdigest()


@app.command("list", epilog=ENV_LIST_JSON_HELP)
def list_cmd(
    num: int = typer.Option(DEFAULT_LIST_LIMIT, "--num", "-n", help="Items per page"),
    page: int = typer.Option(1, "--page", "-p", help="Page number"),
    owner: Optional[str] = typer.Option(None, "--owner", "-o", help="Filter by owner name"),
    visibility: Optional[str] = typer.Option(
        None, "--visibility", "-v", help="Filter by visibility (PUBLIC/PRIVATE)"
    ),
    search: Optional[str] = typer.Option(
        None, "--search", "-s", help="Filter by name or description"
    ),
    tag: Optional[List[str]] = typer.Option(None, "--tag", "-t", help="Filter by tag (repeatable)"),
    sort: str = typer.Option(
        "created_at", "--sort", help="Sort by: name, created_at, updated_at, stars"
    ),
    order: str = typer.Option("desc", "--order", help="Sort order: asc, desc"),
    starred: bool = typer.Option(
        False, "--starred", help="Filter to only environments you have starred"
    ),
    mine: bool = typer.Option(
        False, "--mine", help="Filter to only your own environments (personal + team)"
    ),
    as_json: bool = typer.Option(False, "--json", help="Output JSON instead of a table"),
) -> None:
    """List environments from the Environments Hub.

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

    if num < 1 or page < 1:
        console.print("[red]Error:[/red] --num and --page must be at least 1")
        raise typer.Exit(1)

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

        offset = (page - 1) * num
        params: Dict[str, Any] = {
            "include_teams": True,
            "limit": num,
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
        if starred:
            params["starred_only"] = True
        if mine:
            params["mine_only"] = True

        result = client.get("/environmentshub/", params=params)

        environments = result.get("data", result.get("environments", []))
        total = result.get("total_count", result.get("total", 0))

        if not environments:
            if as_json:
                output_data_as_json(
                    {"environments": [], "total": 0, "page": page, "per_page": num}, console
                )
            elif page > 1:
                console.print("[yellow]No more results.[/yellow]")
            else:
                console.print("No environments found.", style="yellow")
            return

        if as_json:
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
                if env.get("tags"):
                    env_entry["tags"] = env.get("tags")
                env_data.append(env_entry)

            output_data = {
                "environments": env_data,
                "total": total,
                "page": page,
                "per_page": num,
            }
            output_data_as_json(output_data, console)
        else:
            # Table output
            # One line per environment: the description absorbs the width and is cut
            # with an ellipsis so the table always fits the terminal.
            table = Table(expand=True)
            table.add_column(
                "Environment", style="cyan", no_wrap=True, overflow="ellipsis", max_width=40
            )
            table.add_column(
                "Description", style="green", no_wrap=True, overflow="ellipsis", ratio=1
            )
            table.add_column("Version", style="blue", no_wrap=True)
            table.add_column("Stars", style="yellow", justify="right", no_wrap=True)
            table.add_column("Updated", style="dim", no_wrap=True)

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

                table.add_row(env_id, description, version, stars, updated_at)

            console.print(table)

            pages = max(1, -(-total // num))
            footer = f"Page {page}/{pages} - {total} environment(s)"
            if page < pages:
                footer += f" - use --page {page + 1} for the next"
            console.print(f"\n[dim]{footer}[/dim]")

    except APIError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        raise typer.Exit(1)


def _resolve_push_environment_path(path: Optional[str], env_id: Optional[str]) -> Path:
    """Resolve the local environment directory for `prime env push`."""
    if env_id:
        env_folder = env_id.split("/")[-1].replace("-", "_")
        parent = Path(path) if path else Path("./environments")
        return (parent / env_folder).resolve()

    return Path(path or ".").resolve()


def _environment_resolve_data(
    env_name: str,
    *,
    visibility: Optional[str],
    owner: Optional[str],
    team: Optional[str],
    configured_team: Optional[str],
) -> Dict[str, str]:
    """Build the /environmentshub/resolve payload for `prime env push`."""
    resolve_data = {"name": env_name}
    if visibility:
        resolve_data["visibility"] = visibility
    if owner:
        resolve_data["owner_slug"] = owner
    elif team:
        resolve_data["team_slug"] = team
    elif configured_team:
        configured_team = configured_team.strip()
        if TEAM_ID_PATTERN.match(configured_team):
            resolve_data["team_id"] = configured_team
        else:
            resolve_data["team_slug"] = configured_team
    return resolve_data


def _resolve_pull_environment_path(target: Optional[str], env_name: str) -> Path:
    """Resolve the local target directory for `prime env pull`."""
    if target:
        return Path(target)

    env_folder = env_name.replace("-", "_")
    cwd = Path.cwd()
    parent = cwd / "environments" if (cwd / "environments").is_dir() else cwd
    return parent / env_folder


@app.command()
def push(
    env_id: Optional[str] = typer.Argument(
        None,
        help="Optional environment ID used as the local folder name (hyphens map to underscores)",
    ),
    path: Optional[str] = typer.Option(
        None,
        "--path",
        "-p",
        help=(
            "Path to environment directory. Defaults to '.' without env_id, "
            "or './environments' as the parent directory with env_id."
        ),
    ),
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
    visibility: Optional[str] = typer.Option(
        None, "--visibility", "-v", help="Environment visibility (PUBLIC/PRIVATE)"
    ),
    runtime: Optional[str] = typer.Option(
        None,
        "--runtime",
        help=(
            "Verifiers API the package targets: v0 or v1. Defaults to the package's "
            "verifiers requirement (a lower bound of 0.2.0 or newer means v1)."
        ),
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
        declared_runtime = parse_runtime_option(runtime)
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    try:
        env_path = _resolve_push_environment_path(path, env_id)

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

        console.print("\nUploading to the Environments Hub...")

        try:
            client = APIClient()

            console.print("Resolving environment...")
            resolve_data = _environment_resolve_data(
                env_name,
                visibility=visibility,
                owner=owner,
                team=team,
                configured_team=client.config.team_id,
            )

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

            runtime_hint = declared_runtime or classify_runtime_from_metadata(
                requires_dist or project_metadata.get("dependencies", [])
            )
            if runtime_hint is None:
                console.print(
                    "[yellow]No verifiers requirement found, so the Environments Hub will list "
                    "this package as Unclassified; pass --runtime v0|v1 to declare it.[/yellow]"
                )
            else:
                label = "verifiers v1" if runtime_hint == VERIFIERS_V1 else "legacy verifiers v0"
                source = "--runtime" if declared_runtime else "the verifiers requirement"
                console.print(f"Publishing as {label} (from {source})")

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
            if runtime_hint is not None:
                wheel_data["runtime_hint"] = runtime_hint

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
                            timeout=300.0,
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
                        for file_path in _collect_archive_files(env_path):
                            arcname = file_path.relative_to(env_path)
                            tar.add(file_path, arcname=str(arcname))

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
                    if runtime_hint is not None:
                        source_data["runtime_hint"] = runtime_hint

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

                # Save or update Environments Hub metadata for future reference
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

                    env_metadata = _environment_push_metadata(
                        existing_metadata,
                        environment_id=env_id,
                        owner=owner_name,
                        name=env_name,
                        version=project_metadata.get("version"),
                        pushed_at=datetime.now().isoformat(),
                        wheel_sha256=wheel_sha256,
                    )

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
                    if env_metadata.get("forked_from"):
                        upstream_message = Text("Upstream set to ", style="dim")
                        upstream_message.append(f"{owner_name}/{env_name}", style="dim")
                        console.print(upstream_message)
                except Exception as e:
                    console.print(
                        f"[yellow]Warning: Could not save environment metadata: {e}[/yellow]"
                    )

                # Show Environments Hub page link for the environment
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

        download_url = _environment_package_download_url(details)
        if not download_url:
            console.print("[red]Error: No downloadable package found[/red]")
            raise typer.Exit(1)

        base_dir = _resolve_pull_environment_path(target, name)
        target_dir = base_dir
        if not target and target_dir.exists():
            # Find the next available directory with index suffix
            index = 1
            while target_dir.exists():
                target_dir = base_dir.parent / f"{base_dir.name}-{index}"
                index += 1
            console.print(
                f"[yellow]Directory {base_dir} already exists. Using {target_dir} instead.[/yellow]"
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
                            "GET",
                            download_url,
                            headers=headers,
                            timeout=60.0,
                            follow_redirects=True,
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
            version_value = (
                details.get("semantic_version")
                or details.get("semanticVersion")
                or details.get("version")
                or version
            )
            source_metadata = details.get("metadata")
            if not isinstance(source_metadata, dict):
                source_metadata = {}
            origin = _environment_ref(
                owner,
                name,
                environment_id=details.get("id"),
                version=version_value,
            )
            fork_chain = _environment_fork_chain(source_metadata, origin)
            env_metadata = {
                "environment_id": details.get("id"),
                "owner": owner,
                "name": name,
                "version": version_value,
                "origin": origin,
                "fork_chain": fork_chain,
                "pulled_at": datetime.now().isoformat(),
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


def _environment_package_download_url(details: Dict[str, Any]) -> Optional[str]:
    """Return the platform package URL, preferring tracked public downloads."""
    for key in ("tracked_package_url", "package_url"):
        url = details.get(key)
        if isinstance(url, str) and url:
            return url
    return None


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


@app.command(no_args_is_help=True, epilog=ENV_INFO_JSON_HELP)
def info(
    env_id: str = typer.Argument(..., help="Environment ID (owner/name)"),
    version: str = typer.Option("latest", "--version", "-v", help="Version to show"),
    as_json: bool = typer.Option(False, "--json", help="Output JSON instead of a table"),
) -> None:
    """Show environment details, visibility, latest version and installation commands"""
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

        try:
            response = client.get(f"/environmentshub/{owner}/{name}/@{target_version}")
            details = response.get("data", response)
            status_response = client.get(f"/environmentshub/{owner}/{name}/status")
            status = status_response.get("data", status_response)
        except APIError as e:
            console.print(f"[red]Failed to get environment details: {e}[/red]")
            raise typer.Exit(1)

        if as_json:
            details["latest_version"] = status.get("latest_version")
            output_data_as_json(details, console)
            return

        wheel_url = process_wheel_url(details.get("wheel_url"))

        console.print()
        console.print(f"[bold cyan]{owner}/{name}[/bold cyan][dim]@{target_version}[/dim]")
        description = (details.get("metadata") or {}).get("description") or status.get(
            "description"
        )
        if description:
            console.print(f"[dim]{description}[/dim]")
        console.print(f"[dim]Visibility:[/dim] {status.get('visibility', 'UNKNOWN')}")

        latest_version = status.get("latest_version")
        if latest_version:
            content_hash = latest_version.get("content_hash") or ""
            version_str = latest_version.get("semantic_version") or content_hash[:8]
            created = format_time_ago(latest_version.get("created_at"))
            console.print(
                f"[dim]Latest:[/dim] {version_str} ({content_hash[:12] or '-'}, created {created})"
            )

        console.print()

        # Display key installation commands based on availability
        simple_index_url = details.get("install_index_url") or details.get("simple_index_url")

        if wheel_url or simple_index_url:
            normalized_name = normalize_package_name(name)

            console.print("[bold yellow]Install (choose one)[/bold yellow]")

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
    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        raise typer.Exit(1)


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


version_app = PlainTyper(help="Manage environment versions", no_args_is_help=True)
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

        table = Table()
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
    """Delete a specific environment version from the Environments Hub using its content hash"""
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
                    f"hash '{content_hash}' from '{env_id}' on the Environments Hub?"
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
    """Delete an entire environment from the Environments Hub"""
    try:
        if not force:
            try:
                delete_msg = (
                    f"Are you sure you want to permanently delete entire environment "
                    f"'{env_id}' and ALL its versions from the Environments Hub?"
                )
                confirm = typer.confirm(delete_msg)
                if not confirm:
                    console.print("Deletion cancelled.")
                    raise typer.Exit()
            except typer.Abort:
                console.print("Deletion cancelled.")
                raise typer.Exit()

        client = APIClient()
        console.print(f"Deleting {env_id} from the Environments Hub...")

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


def _get_environment_id(client: APIClient, owner: str, env_name: str) -> str:
    """Resolve environment slug to environment ID using the detail endpoint."""
    response = client.get(f"/environmentshub/{owner}/{env_name}/@latest")
    data = response.get("data", {})
    env_id = data.get("id")
    if not env_id:
        raise APIError(f"Environment {owner}/{env_name} not found")
    return env_id


def _fetch_env_secrets(client: APIClient, env_id: str) -> List[Dict[str, Any]]:
    """Fetch secrets for an environment."""
    response = client.get(f"/environmentshub/{env_id}/secrets")
    return response.get("data", [])


@secret_app.command("list", epilog=ENV_SECRET_LIST_JSON_HELP)
def env_secret_list(
    environment: Optional[str] = typer.Argument(
        None,
        help="Environment slug (e.g., 'owner/environment-name'). Auto-detected if not provided.",
    ),
    as_json: bool = typer.Option(False, "--json", help="Output JSON instead of a table"),
) -> None:
    """List all secrets for an environment."""
    owner, env_name = _resolve_environment(environment)

    try:
        client = APIClient()
        env_id = _get_environment_id(client, owner, env_name)
        secrets = _fetch_env_secrets(client, env_id)

        if as_json:
            output_data_as_json({"secrets": secrets}, console)
            return

        if not secrets:
            console.print("[yellow]No secrets found for this environment.[/yellow]")
            return

        table = Table()
        table.add_column("ID", style="dim", no_wrap=True)
        table.add_column("Name", style="cyan")
        table.add_column("Source", style="blue")
        table.add_column("Description", style="dim")
        table.add_column("Created", style="dim")

        for secret in secrets:
            secret_id = secret.get("id", "")
            name = secret.get("name", "")
            source = secret.get("source", "")
            description = secret.get("description") or ""
            created = secret.get("createdAt", "")
            if created:
                created = format_time_ago(created)
            table.add_row(secret_id, name, source, description, created)

        console.print(table)

    except APIError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@secret_app.command("create", epilog=ENV_SECRET_DETAIL_JSON_HELP)
def env_secret_create(
    environment: Optional[str] = typer.Argument(
        None,
        help="Environment slug (e.g., 'owner/environment-name'). Auto-detected if not provided.",
    ),
    name: Optional[str] = typer.Option(
        None,
        "--name",
        "-n",
        help="Secret name (must be uppercase with underscores, e.g., MY_SECRET)",
    ),
    value: Optional[str] = typer.Option(
        None,
        "--value",
        "-v",
        help="Secret value",
    ),
    description: Optional[str] = typer.Option(
        None,
        "--description",
        "-d",
        help="Secret description",
    ),
    as_json: bool = typer.Option(False, "--json", help="Output JSON instead of a table"),
) -> None:
    """Create an environment-specific secret."""
    owner, env_name = _resolve_environment(environment)

    try:
        if not name:
            name = prompt_for_value("Secret name")
            if not name:
                console.print("\n[dim]Cancelled.[/dim]")
                raise typer.Exit()

        if not validate_env_var_name(name, "secret"):
            raise typer.Exit(1)

        if not value:
            value = prompt_for_value("Secret value", hide_input=True)
            if not value:
                console.print("\n[dim]Cancelled.[/dim]")
                raise typer.Exit()

        with console.status("[bold blue]Creating secret...", spinner="dots"):
            client = APIClient()
            env_id = _get_environment_id(client, owner, env_name)

            payload: Dict[str, Any] = {"name": name, "value": value}
            if description:
                payload["description"] = description

            response = client.post(f"/environmentshub/{env_id}/secrets", json=payload)
            secret = response.get("data", {})

        if as_json:
            output_data_as_json(secret, console)
            return

        console.print(f"[green]✓ Created secret '{name}' for {owner}/{env_name}[/green]")
        console.print(f"[dim]ID: {secret.get('id')}[/dim]")

    except KeyboardInterrupt:
        console.print("\n[dim]Cancelled.[/dim]")
        raise typer.Exit()
    except APIError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@secret_app.command("update", epilog=ENV_SECRET_DETAIL_JSON_HELP)
def env_secret_update(
    environment: Optional[str] = typer.Argument(
        None,
        help="Environment slug (e.g., 'owner/environment-name'). Auto-detected if not provided.",
    ),
    secret_id: Optional[str] = typer.Option(
        None,
        "--id",
        help="Secret ID to update (interactive selection if not provided)",
    ),
    name: Optional[str] = typer.Option(
        None,
        "--name",
        "-n",
        help="New secret name",
    ),
    value: Optional[str] = typer.Option(
        None,
        "--value",
        "-v",
        help="New secret value",
    ),
    description: Optional[str] = typer.Option(
        None,
        "--description",
        "-d",
        help="New secret description",
    ),
    as_json: bool = typer.Option(False, "--json", help="Output JSON instead of a table"),
) -> None:
    """Update an environment-specific secret."""
    owner, env_name = _resolve_environment(environment)

    try:
        client = APIClient()
        env_id = _get_environment_id(client, owner, env_name)

        if not secret_id:
            secrets = _fetch_env_secrets(client, env_id)
            selected = require_selection(
                secrets, "update", f"No secrets to update for {owner}/{env_name}."
            )
            secret_id = selected.get("id")

        if not any_provided(name, value, description):
            console.print("\n[bold]What would you like to update?[/bold]")
            new_value = prompt_for_value("New value", required=False, hide_input=True)
            if new_value:
                value = new_value

            if not value:
                console.print("\n[dim]No changes made.[/dim]")
                raise typer.Exit()

        if name is not None and not validate_env_var_name(name, "secret"):
            raise typer.Exit(1)

        payload: Dict[str, Any] = {}
        if name is not None:
            payload["name"] = name
        if value is not None:
            payload["value"] = value
        if description is not None:
            payload["description"] = description

        response = client.patch(f"/environmentshub/{env_id}/secrets/{secret_id}", json=payload)
        secret = response.get("data", {})

        if as_json:
            output_data_as_json(secret, console)
            return

        console.print(
            f"[green]✓ Updated secret '{secret.get('name')}' for {owner}/{env_name}[/green]"
        )

    except KeyboardInterrupt:
        console.print("\n[dim]Cancelled.[/dim]")
        raise typer.Exit()
    except APIError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@secret_app.command("delete")
def env_secret_delete(
    environment: Optional[str] = typer.Argument(
        None,
        help="Environment slug (e.g., 'owner/environment-name'). Auto-detected if not provided.",
    ),
    secret_id: Optional[str] = typer.Option(
        None,
        "--id",
        help="Secret ID to delete (interactive selection if not provided)",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip confirmation prompt",
    ),
) -> None:
    """Delete an environment-specific secret."""
    owner, env_name = _resolve_environment(environment)

    try:
        client = APIClient()
        env_id = _get_environment_id(client, owner, env_name)

        if not secret_id:
            secrets = _fetch_env_secrets(client, env_id)
            selected = require_selection(
                secrets, "delete", f"No secrets to delete for {owner}/{env_name}."
            )
            secret_id = selected.get("id")
            secret_name = selected.get("name")
        else:
            secrets = _fetch_env_secrets(client, env_id)
            secret_data = next((s for s in secrets if s.get("id") == secret_id), None)
            secret_name = secret_data.get("name") if secret_data else secret_id

        if not yes:
            confirm = typer.confirm(f"Delete secret '{secret_name}' from {owner}/{env_name}?")
            if not confirm:
                console.print("\n[dim]Cancelled.[/dim]")
                raise typer.Exit()

        client.delete(f"/environmentshub/{env_id}/secrets/{secret_id}")
        console.print(f"[green]✓ Deleted secret '{secret_name}' from {owner}/{env_name}[/green]")

    except KeyboardInterrupt:
        console.print("\n[dim]Cancelled.[/dim]")
        raise typer.Exit()
    except APIError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@secret_app.command("link", epilog=ENV_SECRET_LINK_JSON_HELP)
def env_secret_link(
    global_secret_id: str = typer.Argument(
        ...,
        help="Global secret ID to link",
    ),
    environment: Optional[str] = typer.Argument(
        None,
        help="Environment slug (e.g., 'owner/environment-name'). Auto-detected if not provided.",
    ),
    as_json: bool = typer.Option(False, "--json", help="Output JSON instead of a table"),
) -> None:
    """Link a global secret to an environment."""
    owner, env_name = _resolve_environment(environment)

    try:
        client = APIClient()
        env_id = _get_environment_id(client, owner, env_name)

        response = client.post(
            f"/environmentshub/{env_id}/secrets/link/{global_secret_id}",
            json={},
        )
        linked = response.get("data", {})

        if as_json:
            output_data_as_json(linked, console)
            return

        secret_name = linked.get("secretName", global_secret_id)
        console.print(
            f"[green]✓ Linked global secret '{secret_name}' to {owner}/{env_name}[/green]"
        )

    except APIError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@secret_app.command("unlink")
def env_secret_unlink(
    global_secret_id: str = typer.Argument(
        ...,
        help="Global secret ID to unlink",
    ),
    environment: Optional[str] = typer.Argument(
        None,
        help="Environment slug (e.g., 'owner/environment-name'). Auto-detected if not provided.",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip confirmation prompt",
    ),
) -> None:
    """Unlink a global secret from an environment."""
    owner, env_name = _resolve_environment(environment)

    try:
        if not yes:
            confirm = typer.confirm(
                f"Unlink global secret {global_secret_id} from {owner}/{env_name}?"
            )
            if not confirm:
                console.print("\n[dim]Cancelled.[/dim]")
                raise typer.Exit()

        client = APIClient()
        env_id = _get_environment_id(client, owner, env_name)
        client.delete(f"/environmentshub/{env_id}/secrets/link/{global_secret_id}")
        console.print(f"[green]✓ Unlinked global secret from {owner}/{env_name}[/green]")

    except KeyboardInterrupt:
        console.print("\n[dim]Cancelled.[/dim]")
        raise typer.Exit()
    except APIError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@var_app.command("list", epilog=ENV_VAR_LIST_JSON_HELP)
def var_list(
    environment: Optional[str] = typer.Argument(
        None,
        help="Environment slug (e.g., 'owner/environment-name'). Auto-detected if not provided.",
    ),
    as_json: bool = typer.Option(False, "--json", help="Output JSON instead of a table"),
) -> None:
    """List all variables for an environment."""
    owner, env_name = _resolve_environment(environment)

    try:
        client = APIClient()
        env_id = _get_environment_id(client, owner, env_name)
        response = client.get(f"/environmentshub/{env_id}/variables")
        variables = response.get("data", [])

        if as_json:
            output_data_as_json({"variables": variables}, console)
            return

        if not variables:
            console.print("[yellow]No variables found for this environment.[/yellow]")
            return

        table = Table()
        table.add_column("ID", style="dim", no_wrap=True)
        table.add_column("Name", style="cyan")
        table.add_column("Value", style="green")
        table.add_column("Description", style="dim")
        table.add_column("Created", style="dim")

        for var in variables:
            var_id = var.get("id", "")
            name = var.get("name", "")
            value = var.get("value", "")
            if len(value) > 30:
                value = value[:27] + "..."
            description = var.get("description") or ""
            created = var.get("createdAt", "")
            if created:
                created = format_time_ago(created)
            table.add_row(var_id, name, value, description, created)

        console.print(table)

    except APIError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@var_app.command("create", epilog=ENV_VAR_DETAIL_JSON_HELP)
def var_create(
    environment: Optional[str] = typer.Argument(
        None,
        help="Environment slug (e.g., 'owner/environment-name'). Auto-detected if not provided.",
    ),
    name: Optional[str] = typer.Option(
        None,
        "--name",
        "-n",
        help="Variable name (must be uppercase with underscores, e.g., MY_VAR)",
    ),
    value: Optional[str] = typer.Option(
        None,
        "--value",
        "-v",
        help="Variable value",
    ),
    description: Optional[str] = typer.Option(
        None,
        "--description",
        "-d",
        help="Variable description",
    ),
    as_json: bool = typer.Option(False, "--json", help="Output JSON instead of a table"),
) -> None:
    """Create an environment variable."""
    owner, env_name = _resolve_environment(environment)

    try:
        if not name:
            name = prompt_for_value("Variable name")
            if not name:
                console.print("\n[dim]Cancelled.[/dim]")
                raise typer.Exit()

        if not validate_env_var_name(name, "variable"):
            raise typer.Exit(1)

        if not value:
            value = prompt_for_value("Variable value")
            if not value:
                console.print("\n[dim]Cancelled.[/dim]")
                raise typer.Exit()

        with console.status("[bold blue]Creating variable...", spinner="dots"):
            client = APIClient()
            env_id = _get_environment_id(client, owner, env_name)

            payload: Dict[str, Any] = {"name": name, "value": value}
            if description:
                payload["description"] = description

            response = client.post(f"/environmentshub/{env_id}/variables", json=payload)
            var = response.get("data", {})

        if as_json:
            output_data_as_json(var, console)
            return

        console.print(f"[green]✓ Created variable '{name}' for {owner}/{env_name}[/green]")
        console.print(f"[dim]ID: {var.get('id')}[/dim]")

    except KeyboardInterrupt:
        console.print("\n[dim]Cancelled.[/dim]")
        raise typer.Exit()
    except APIError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@var_app.command("update", epilog=ENV_VAR_DETAIL_JSON_HELP)
def var_update(
    var_id: str = typer.Argument(
        ...,
        help="Variable ID to update",
    ),
    environment: Optional[str] = typer.Argument(
        None,
        help="Environment slug (e.g., 'owner/environment-name'). Auto-detected if not provided.",
    ),
    name: Optional[str] = typer.Option(
        None,
        "--name",
        "-n",
        help="New variable name",
    ),
    value: Optional[str] = typer.Option(
        None,
        "--value",
        "-v",
        help="New variable value",
    ),
    description: Optional[str] = typer.Option(
        None,
        "--description",
        "-d",
        help="New variable description",
    ),
    as_json: bool = typer.Option(False, "--json", help="Output JSON instead of a table"),
) -> None:
    """Update an environment variable."""

    if not any_provided(name, value, description):
        console.print(
            "[red]Error: At least one of --name, --value, or --description is required[/red]"
        )
        raise typer.Exit(1)

    owner, env_name = _resolve_environment(environment)

    try:
        client = APIClient()
        env_id = _get_environment_id(client, owner, env_name)

        if name is not None and not validate_env_var_name(name, "variable"):
            raise typer.Exit(1)

        payload: Dict[str, Any] = {}
        if name is not None:
            payload["name"] = name
        if value is not None:
            payload["value"] = value
        if description is not None:
            payload["description"] = description

        response = client.patch(
            f"/environmentshub/{env_id}/variables/{var_id}",
            json=payload,
        )
        var = response.get("data", {})

        if as_json:
            output_data_as_json(var, console)
            return

        console.print(f"[green]✓ Updated variable '{var.get('name')}'[/green]")

    except APIError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@var_app.command("delete")
def var_delete(
    var_id: str = typer.Argument(
        ...,
        help="Variable ID to delete",
    ),
    environment: Optional[str] = typer.Argument(
        None,
        help="Environment slug (e.g., 'owner/environment-name'). Auto-detected if not provided.",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip confirmation prompt",
    ),
) -> None:
    """Delete an environment variable."""
    owner, env_name = _resolve_environment(environment)

    try:
        if not yes:
            confirm = typer.confirm(f"Delete variable {var_id} from {owner}/{env_name}?")
            if not confirm:
                console.print("\n[dim]Cancelled.[/dim]")
                raise typer.Exit()

        client = APIClient()
        env_id = _get_environment_id(client, owner, env_name)
        client.delete(f"/environmentshub/{env_id}/variables/{var_id}")
        console.print("[green]✓ Variable deleted[/green]")

    except KeyboardInterrupt:
        console.print("\n[dim]Cancelled.[/dim]")
        raise typer.Exit()
    except APIError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
