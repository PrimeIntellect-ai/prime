import ast
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import zipfile
from datetime import datetime

# Wheel METADATA files use RFC 822 format (PEP 566), same as email headers
from email.parser import Parser as EmailParser
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import httpx
import toml
import yaml
from prime_sandboxes import APIClient as SandboxAPIClient
from prime_sandboxes import Config as SandboxConfig
from rich.table import Table
from rich.text import Text

from ..client import APIClient, APIError
from ..lab_hygiene import LabHygieneOptions, find_lab_workspace, run_lab_hygiene_preflight
from ..utils import (
    get_console,
    is_plain_mode,
    json_output_help,
    output_data_as_json,
    validate_output_format,
)
from ..utils.env_metadata import (
    collect_archive_files,
    compute_content_hash,
    download_environment_source,
    find_environment_metadata,
    install_environment_from_hub,
    normalize_package_name,
    parse_env_id,
)
from ..utils.formatters import format_file_size
from ..utils.formatters import strip_ansi as _strip_ansi
from ..utils.prompt import (
    any_provided,
    confirm,
    prompt,
    prompt_for_value,
    require_selection,
    validate_env_var_name,
)
from ..utils.time_utils import format_time_ago, iso_timestamp
from ..verifiers_bridge import (
    build_verifiers_command,
    exec_verifiers_process,
    resolve_workspace_python,
    verifiers_environment,
)
from .config import TEAM_ID_PATTERN
from .env_configs import (
    EnvActionListConfig,
    EnvActionLogsConfig,
    EnvActionRetryConfig,
    EnvBuildConfig,
    EnvDeleteConfig,
    EnvInfoConfig,
    EnvInspectConfig,
    EnvInstallConfig,
    EnvListConfig,
    EnvPullConfig,
    EnvPushConfig,
    EnvSecretCreateConfig,
    EnvSecretDeleteConfig,
    EnvSecretLinkConfig,
    EnvSecretListConfig,
    EnvSecretUnlinkConfig,
    EnvSecretUpdateConfig,
    EnvStatusConfig,
    EnvUninstallConfig,
    EnvVarCreateConfig,
    EnvVarDeleteConfig,
    EnvVarListConfig,
    EnvVarUpdateConfig,
    EnvVersionDeleteConfig,
    EnvVersionListConfig,
)

console = get_console()

# Constants
MAX_FILES_TO_SHOW = 10
DEFAULT_HASH_LENGTH = 8
DEFAULT_LIST_LIMIT = 20
MAX_TARBALL_SIZE_LIMIT = 250 * 1024 * 1024  # 250MB
DEFAULT_BUILD_WAIT_TIMEOUT_S = 1200
DEFAULT_BUILD_WAIT_INTERVAL_S = 5.0

ACTION_LIST_JSON_HELP = json_output_help(
    ".actions[] = {id, name|job_type, status, version, trigger, created_at}",
    ".total = number",
)

ACTION_RETRY_JSON_HELP = json_output_help(
    ". = {success, job_id?, version_id?, message?}",
)

ENV_LIST_JSON_HELP = json_output_help(
    ".environments[] = {environment, description, visibility, version, stars, "
    "updated_at, action_status?, tags[]?}",
    ".total = number",
    ".page = number",
    ".per_page = number",
)

ENV_STATUS_JSON_HELP = json_output_help(
    ". = {name, description?, visibility, latest_version?, action?}",
    ".latest_version? = {semantic_version?, content_hash?, created_at?}",
    ".action? = {status, job_id?}",
)

ENV_INSPECT_JSON_HELP = json_output_help(
    ". = {kind, path, version_id, entry?, entries[]?, content?, truncated, total_bytes?}",
    ".entry? = {name, path, is_directory, size?, modified_at?, content_hash?}",
    ".entries[] = {name, path, is_directory, size?, modified_at?, content_hash?}",
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


def _uv_pip_command(subcommand: str, *args: str) -> List[str]:
    """Run uv pip against the workspace interpreter."""
    return ["uv", "pip", subcommand, "--python", resolve_workspace_python(), *args]


def _parse_environment_slug(environment: str) -> Tuple[str, str]:
    """Parse an unversioned owner/name Hub slug."""
    try:
        owner, name, version = parse_env_id(environment)
    except ValueError:
        console.print(f"[red]Invalid environment format: {environment}[/red]")
        console.print("[dim]Use format: owner/environment-name[/dim]")
        raise SystemExit(1)
    if version is not None:
        console.print("[red]Environment versions are not accepted by this command[/red]")
        raise SystemExit(1)
    return owner, name


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
    raise SystemExit(1)


def actions_list(config: EnvActionListConfig) -> None:
    """List actions (CI jobs) for an environment."""
    environment = config.environment
    version_id = config.version_id
    num = config.num
    page = config.page
    output = config.output

    validate_output_format(output, console)

    if num < 1 or page < 1:
        console.print("[red]Error:[/red] --num and --page must be at least 1")
        raise SystemExit(1)

    owner, env_name = _parse_environment_slug(environment)

    try:
        client = APIClient()
        offset = (page - 1) * num
        params: dict[str, int | str] = {
            "limit": num,
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
            if page > 1:
                console.print("[yellow]No more results.[/yellow]")
            else:
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
        if total > page * num:
            console.print(
                f"\n[yellow]Showing page {page} of results. "
                f"Use --page {page + 1} to see more.[/yellow]"
            )
        else:
            console.print(f"\n[dim]Total: {total} action(s)[/dim]")

    except APIError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise SystemExit(1)


def actions_logs(config: EnvActionLogsConfig) -> None:
    """Get logs for a specific action."""
    environment = config.environment
    action_id = config.action_id
    tail = config.tail
    follow = config.follow

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
        raise SystemExit(1)


def actions_retry(config: EnvActionRetryConfig) -> None:
    """Retry an action (integration test) for an environment.

    If no action ID is provided, retries the latest action.
    """
    environment = config.environment
    action_id = config.action_id
    output = config.output

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
            job_id = data.get("job_id")
            console.print(
                f"\n[dim]Use 'prime env action logs {environment} {job_id}' to view logs[/dim]"
            )
        else:
            console.print(f"[red]Retry failed:[/red] {data.get('message', 'Unknown error')}")
            raise SystemExit(1)

    except APIError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise SystemExit(1)


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


def _print_env_inspect_examples(owner: str, name: str, version: str) -> None:
    """Print inspect commands for an environment version."""
    console.print("[bold yellow]Inspect[/bold yellow]")
    console.print(f"  [green]$[/green] prime env inspect {owner}/{name}@{version}")
    console.print(f"  [green]$[/green] prime env inspect {owner}/{name}@{version} README.md")


def list_cmd(config: EnvListConfig) -> None:
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
    num = config.num
    page = config.page
    owner = config.owner
    visibility = config.visibility
    output = config.output
    search = config.search
    tag = config.tag
    action_status = config.action_status
    sort = config.sort
    order = config.order
    show_actions = config.show_actions
    starred = config.starred
    mine = config.mine

    validate_output_format(output, console)

    if num < 1 or page < 1:
        console.print("[red]Error:[/red] --num and --page must be at least 1")
        raise SystemExit(1)

    # Validate sort and order
    if sort not in ("name", "created_at", "updated_at", "stars"):
        console.print(
            "[red]Error: --sort must be one of: name, created_at, updated_at, stars[/red]"
        )
        raise SystemExit(1)
    if order.lower() not in ("asc", "desc"):
        console.print("[red]Error: --order must be one of: asc, desc[/red]")
        raise SystemExit(1)

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
                    {"environments": [], "total": 0, "page": page, "per_page": num}, console
                )
            elif page > 1:
                console.print("[yellow]No more results.[/yellow]")
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
                "page": page,
                "per_page": num,
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

            if total > page * num:
                console.print(
                    f"\n[yellow]Showing page {page} of results. "
                    f"Use --page {page + 1} to see more.[/yellow]"
                )
            else:
                console.print(f"\n[dim]Total: {total} environment(s)[/dim]")

    except APIError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise SystemExit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        raise SystemExit(1)


def status_cmd(config: EnvStatusConfig) -> None:
    """Show action status for an environment.

    \b
    Examples:
        prime env status owner/my-env
        prime env status owner/my-env --output json
    """
    env_id = config.env_id
    output = config.output

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
        raise SystemExit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        raise SystemExit(1)


def _resolve_push_environment_path(path: Optional[str], env_id: Optional[str]) -> Path:
    """Resolve the local environment directory for `prime env push`."""
    if env_id:
        env_folder = env_id.split("/")[-1].replace("-", "_")
        parent = Path(path) if path else Path("./environments")
        return (parent / env_folder).resolve()

    return Path(path or ".").resolve()


def _emit_lab_hygiene_message(message: str) -> None:
    console.print(message, markup=False)


def _run_env_init_lab_hygiene_preflight() -> None:
    workspace = find_lab_workspace(Path.cwd())
    if workspace is None:
        return
    run_lab_hygiene_preflight(
        LabHygieneOptions(fix=True),
        workspace=workspace,
        emit=_emit_lab_hygiene_message,
    )


def _run_env_push_lab_hygiene_preflight(env_path: Path) -> None:
    workspace = find_lab_workspace(env_path)
    if workspace is None:
        return
    result = run_lab_hygiene_preflight(
        LabHygieneOptions(fix=False, fail_on_tracked=True),
        workspace=workspace,
        emit=_emit_lab_hygiene_message,
    )
    if result.exit_code != 0:
        raise SystemExit(result.exit_code)


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


def push(config: EnvPushConfig) -> None:
    """Push environment to registry"""
    env_id = config.env_id
    path = config.path
    name = config.name
    owner = config.owner
    team = config.team
    visibility = config.visibility
    auto_bump = config.auto_bump
    rc = config.rc
    post = config.post

    try:
        env_path = _resolve_push_environment_path(path, env_id)
        _run_env_push_lab_hygiene_preflight(env_path)

        # Display upstream environment info if metadata exists
        display_upstream_environment_info(env_path)

        # Validate basic structure
        pyproject_path = env_path / "pyproject.toml"
        if not pyproject_path.exists():
            console.print("[red]Error: pyproject.toml not found[/red]")
            raise SystemExit(1)

        try:
            pyproject_data = toml.load(pyproject_path)
            project_info = pyproject_data.get("project", {})

            env_name = name or project_info.get("name")
            if not env_name:
                console.print(
                    "[red]Error: No name found in pyproject.toml and no --name provided[/red]"
                )
                raise SystemExit(1)

            # Auto-bump version if requested
            if auto_bump or rc or post:
                flags_set = sum(bool(x) for x in (auto_bump, rc, post))
                if flags_set > 1:
                    console.print(
                        "[red]Error: --auto-bump, --rc, and --post are mutually exclusive[/red]"
                    )
                    raise SystemExit(1)
                current_version = project_info.get("version")
                if not current_version:
                    console.print(
                        "[red]Error: No version found in pyproject.toml for auto-bump[/red]"
                    )
                    raise SystemExit(1)

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
                    raise SystemExit(1)

            console.print(f"Environment name: {env_name}")

        except Exception as e:
            console.print(f"[red]Failed to parse pyproject.toml: {e}[/red]")
            raise SystemExit(1)

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
            raise SystemExit(1)

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
            raise SystemExit(1)
        except FileNotFoundError:
            console.print("[red]Build tool not found. Please install 'uv' or 'build'.[/red]")
            raise SystemExit(1)

        dist_dir = env_path / "dist"
        wheels = list(dist_dir.glob("*.whl"))
        if not wheels:
            console.print("[red]Error: No wheel file found after build[/red]")
            raise SystemExit(1)

        wheel_path = wheels[0]
        wheel_size = wheel_path.stat().st_size
        console.print(f"[green]✓ Built {wheel_path.name} ({wheel_size:,} bytes)[/green]")

        console.print("\nUploading to Prime Intellect Hub...")

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
                                prompt(
                                    "Enter your desired username",
                                )
                                .strip()
                                .lower()
                            )
                        except KeyboardInterrupt:
                            console.print("[red]Cancelled by user[/red]")
                            raise SystemExit(1)

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
                                raise SystemExit(1)

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
                        raise SystemExit(1)
                else:
                    console.print(f"[red]Failed to resolve environment: {e}[/red]")
                    raise SystemExit(1)

            console.print("Uploading wheel ...")

            try:
                with open(wheel_path, "rb") as f:
                    wheel_sha256 = hashlib.sha256(f.read()).hexdigest()
            except IOError as e:
                console.print(f"[red]Failed to read wheel file: {e}[/red]")
                raise SystemExit(1)

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
                raise SystemExit(1)

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
                    raise SystemExit(1)
                except IOError as e:
                    console.print(f"[red]Failed to read wheel file for upload: {e}[/red]")
                    raise SystemExit(1)

                try:
                    client.post(f"/environmentshub/{env_id}/wheels/{wheel_id}/finalize")
                except APIError as e:
                    console.print(f"[red]Failed to finalize wheel upload: {e}[/red]")
                    raise SystemExit(1)

            console.print("Creating source archive...")
            temp_file_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
                    temp_file_path = tmp.name
                    with tarfile.open(tmp.name, "w:gz") as tar:
                        for file_path in collect_archive_files(env_path):
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
                        raise SystemExit(1)

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
                        raise SystemExit(1)
                    except IOError as e:
                        console.print(f"[red]Failed to read source archive for upload: {e}[/red]")
                        raise SystemExit(1)

                    # Finalize
                    try:
                        response = client.post(
                            f"/environmentshub/{env_id}/versions/{version_id}/finalize"
                        )

                        finalize_response = response["data"]

                    except APIError as e:
                        console.print(f"[red]Failed to finalize source upload: {e}[/red]")
                        raise SystemExit(1)

            except (tarfile.TarError, OSError) as e:
                console.print(f"[red]Failed to create source archive: {e}[/red]")
                raise SystemExit(1)
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
                raise SystemExit(1)

        except APIError as e:
            console.print(f"[red]API Error: {e}[/red]")
            raise SystemExit(1)
        except Exception as e:
            console.print(f"[red]Upload failed: {e}[/red]")
            raise SystemExit(1)

    except subprocess.CalledProcessError as e:
        console.print(f"[red]Build error: {e}[/red]")
        raise SystemExit(1)
    except FileNotFoundError as e:
        console.print(f"[red]File not found: {e}[/red]")
        raise SystemExit(1)
    except PermissionError as e:
        console.print(f"[red]Permission error: {e}[/red]")
        raise SystemExit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        raise SystemExit(1)


def init(argv: list[str]) -> None:
    """Initialize a V1 or V0 environment with Verifiers."""
    result = subprocess.run(
        build_verifiers_command("init", argv),
        env=verifiers_environment(plain=is_plain_mode()),
    )
    if result.returncode != 0:
        raise SystemExit(result.returncode)
    if argv and not any(arg in ("-h", "--help") for arg in argv):
        _run_env_init_lab_hygiene_preflight()


def _resolve_build_project_dir(environments_root: Path, env_id_underscore: str) -> Path:
    env_path = environments_root / env_id_underscore
    if not env_path.is_dir():
        raise FileNotFoundError(
            f"Environment not found: {env_path}. Expected directory "
            f"'{env_id_underscore}' under {environments_root}."
        )

    project_dir = env_path / "proj"
    if not project_dir.is_dir():
        raise FileNotFoundError(
            f"Embedded project directory not found: {project_dir}. "
            "Required structure: environments/<env_id_underscore>/proj/."
        )

    for required in (project_dir / "openenv.yaml", project_dir / "pyproject.toml"):
        if not required.exists():
            raise FileNotFoundError(
                f"Required file missing: {required}. Expected project files under proj/."
            )
    return project_dir


def _resolve_build_app_module(project_dir: Path, app_name: str) -> Path:
    module = app_name.split(":", 1)[0].strip()
    if not module:
        raise RuntimeError(f"Invalid app entrypoint in openenv.yaml: {app_name}")
    app_module = project_dir / Path(*module.split("."))
    for candidate in (app_module.with_suffix(".py"), app_module / "__init__.py"):
        if candidate.exists():
            return candidate
    raise RuntimeError(
        f"Could not resolve app module from openenv.yaml app='{app_name}'. "
        f"Expected {app_module.with_suffix('.py')} or {app_module / '__init__.py'}."
    )


def _build_ast_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return ""


def _detect_build_contract(project_dir: Path, app_name: str) -> str:
    """Detect the OpenEnv MCP or Gym contract from its create_app call."""
    app_file = _resolve_build_app_module(project_dir, app_name)
    tree = ast.parse(app_file.read_text(), filename=str(app_file))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or _build_ast_name(node.func) != "create_app":
            continue
        if len(node.args) < 3:
            continue
        action_name = _build_ast_name(node.args[1])
        observation_name = _build_ast_name(node.args[2])
        if action_name == "CallToolAction" and observation_name == "CallToolObservation":
            return "mcp"
        return "gym"
    raise RuntimeError(
        f"Could not detect OpenEnv contract: no supported create_app(...) call found in {app_file}."
    )


def _write_build_manifest(
    project_dir: Path,
    image: str,
    port: int,
    env_id: str,
    status: str,
    start_command: str,
    app_name: str,
    contract: str,
) -> Path:
    manifest = {
        "schema_version": 1,
        "environment_id": env_id,
        "image": image,
        "port": port,
        "app": app_name,
        "contract": contract,
        "start_command": start_command,
        "image_status": status,
    }
    manifest_path = project_dir / ".build.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest_path


def _extract_build_image_ref(item: dict[str, Any]) -> Optional[str]:
    for key in ("displayRef", "fullImagePath"):
        value = item.get(key)
        if isinstance(value, str) and value:
            return value
    image_name = item.get("imageName")
    image_tag = item.get("imageTag")
    if isinstance(image_name, str) and image_name:
        return f"{image_name}:{image_tag}" if image_tag else image_name
    for key in ("image", "image_reference", "image_ref", "name", "ref"):
        value = item.get(key)
        if isinstance(value, str) and value:
            return value
        if isinstance(value, dict) and isinstance(value.get("name"), str):
            return f"{value['name']}:{value['tag']}" if value.get("tag") else value["name"]
    return None


def _wait_for_build(
    image_ref: str,
    timeout_s: int = DEFAULT_BUILD_WAIT_TIMEOUT_S,
    interval_s: float = DEFAULT_BUILD_WAIT_INTERVAL_S,
) -> Optional[str]:
    start = time.time()
    last_status = None
    params = {"limit": "250", "offset": "0"}
    team_id = SandboxConfig().team_id
    if team_id:
        params["teamId"] = team_id
    client = SandboxAPIClient()
    while time.time() - start < timeout_s:
        response = client.request("GET", "/images", params=params)
        items = response.get("data") or response.get("items") or response.get("images") or []
        for item in items:
            if not isinstance(item, dict) or _extract_build_image_ref(item) != image_ref:
                continue
            status = item.get("status") or item.get("state")
            last_status = str(status) if status is not None else None
            break
        if last_status and last_status.lower() in {
            "ready",
            "succeeded",
            "completed",
            "failed",
            "error",
        }:
            return last_status
        time.sleep(interval_s)
    return last_status


def _resolve_build_target(raw_env_id: Optional[str], raw_path: str) -> tuple[str, Path]:
    base_path = Path(raw_path).expanduser().resolve()
    if raw_env_id is None:
        env_dir = base_path
        env_id_underscore = env_dir.name
        if not env_id_underscore:
            raise ValueError("Could not infer environment id from --path.")
    else:
        env_id_dash = raw_env_id.strip()
        if not env_id_dash:
            raise ValueError("Environment id cannot be empty.")
        if "_" in env_id_dash:
            raise ValueError(
                "Environment id must use hyphens (e.g. openenv-echo), not underscores."
            )
        env_id_underscore = env_id_dash.replace("-", "_")
        env_dir = base_path / env_id_underscore
    return env_id_underscore.replace("_", "-"), env_dir


def _build_environment(env_id: Optional[str], path: str) -> int:
    try:
        env_id_dash, env_path = _resolve_build_target(env_id, path)
        project_dir = _resolve_build_project_dir(env_path.parent, env_path.name)
        dockerfile = project_dir / "server" / "Dockerfile"
        if not dockerfile.exists():
            raise FileNotFoundError(
                f"No Dockerfile found at {dockerfile}. "
                "Expected enforced layout with proj/server/Dockerfile."
            )
    except (ValueError, FileNotFoundError) as exc:
        console.print(str(exc), markup=False)
        return 2

    image = f"{env_id_dash}:latest"
    openenv_config = yaml.safe_load((project_dir / "openenv.yaml").read_text())
    if not isinstance(openenv_config, dict):
        openenv_config = {}
    port = int(openenv_config.get("port", 8000))
    raw_app_name = openenv_config.get("app")
    app_name = (
        raw_app_name.strip()
        if isinstance(raw_app_name, str) and raw_app_name.strip()
        else "server.app:app"
    )
    contract = _detect_build_contract(project_dir, app_name)
    start_command = (
        f'sh -lc "cd /app/env && /app/.venv/bin/uvicorn {app_name} --host 0.0.0.0 --port {port}"'
    )

    from .images import push_image
    from .images_configs import ImagesPushConfig

    resolved_image = push_image(
        ImagesPushConfig(
            image_reference=image,
            context=str(project_dir),
            dockerfile=str(dockerfile),
            platform="linux/amd64",
        )
    )
    status = _wait_for_build(resolved_image)
    if status is None:
        console.print(
            "Timed out waiting for image status. Run `prime images list` to check progress."
        )
        return 1
    if status.lower() not in {"ready", "succeeded", "completed"}:
        console.print(f"Image build did not complete successfully (status={status}).")
        return 1

    manifest_path = _write_build_manifest(
        project_dir,
        resolved_image,
        port,
        env_id_dash,
        status,
        start_command,
        app_name,
        contract,
    )
    console.print(
        f"Wrote {manifest_path} with image='{resolved_image}' port={port} app='{app_name}' "
        f"contract='{contract}' start_command='{start_command}' status={status}",
        markup=False,
    )
    return 0


def build(config: EnvBuildConfig) -> None:
    """Build an OpenEnv-backed environment image."""
    env_id = config.env_id
    path = config.path

    code = _build_environment(env_id, path)
    if code != 0:
        raise SystemExit(code)


def validate(argv: list[str]) -> None:
    """Run a taskset's model-free validation with Verifiers."""
    exec_verifiers_process("validate", argv, plain=is_plain_mode())


def serve(argv: list[str]) -> None:
    """Serve a V1 or V0 environment with Verifiers."""
    exec_verifiers_process("serve", argv, plain=is_plain_mode())


def pull(config: EnvPullConfig) -> None:
    """Pull environment for local inspection"""
    env_id = config.env_id
    target = config.target
    version = config.version

    try:
        client = APIClient(require_auth=False)

        try:
            owner, name, parsed_version = parse_env_id(env_id)
        except ValueError as exc:
            console.print(f"[red]Error: {exc}[/red]")
            raise SystemExit(1)
        version = parsed_version or version
        env_id = f"{owner}/{name}"

        console.print(f"Pulling {env_id}@{version}...")

        try:
            response = client.get(f"/environmentshub/{owner}/{name}/@{version}")
            details = response["data"]
        except APIError as e:
            console.print(f"[red]Failed to get environment details: {e}[/red]")
            raise SystemExit(1)

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

        console.print(f"Downloading to {target_dir}...")
        try:
            download_environment_source(
                details,
                target_dir,
                api_key=client.api_key,
                base_url=client.base_url,
            )
        except (OSError, ValueError, httpx.HTTPError) as exc:
            console.print(f"[red]Download failed: {exc}[/red]")
            raise SystemExit(1) from exc

        console.print(f"[green]✓ Environment pulled to {target_dir}[/green]")

        # Record the published upstream for hosted workflows.
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
            extracted_files = [f for f in all_files if f.name != ".prime"]
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
        raise SystemExit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        raise SystemExit(1)


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


def info(config: EnvInfoConfig) -> None:
    """Show environment details and installation commands"""
    env_id = config.env_id
    version = config.version

    try:
        client = APIClient(require_auth=False)

        try:
            owner, name, parsed_version = parse_env_id(env_id)
        except ValueError as e:
            console.print(f"[red]Error: {e}[/red]")
            raise SystemExit(1)
        env_id = f"{owner}/{name}"
        target_version = parsed_version or version

        console.print(f"Fetching {env_id}@{target_version}...")

        # Fetch environment details
        try:
            response = client.get(f"/environmentshub/{owner}/{name}/@{target_version}")
            details = response.get("data", response)
        except APIError as e:
            console.print(f"[red]Failed to get environment details: {e}[/red]")
            raise SystemExit(1)

        # Process wheel URL
        raw_wheel_url = details.get("wheel_url")
        parsed_wheel_url = urlparse(raw_wheel_url) if isinstance(raw_wheel_url, str) else None
        wheel_url = (
            raw_wheel_url
            if parsed_wheel_url
            and parsed_wheel_url.scheme in ("http", "https")
            and parsed_wheel_url.netloc
            else None
        )

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
        _print_env_inspect_examples(owner, name, target_version)
        console.print()

        package_url = details.get("tracked_package_url") or details.get("package_url")
        if wheel_url or simple_index_url or package_url:
            reference = f"{owner}/{name}@{target_version}"
            module = normalize_package_name(name)
            console.print("[bold yellow]Install[/bold yellow]")
            console.print(f"  [green]$[/green] prime env install {reference}")
            console.print()
            console.print("[bold yellow]Run[/bold yellow]")
            console.print(f"  [green]$[/green] prime eval run {module}       # V1")
            console.print(f"  [green]$[/green] prime eval run --id {module}  # V0")
        else:
            console.print("[yellow]No installable artifact is available for this version[/yellow]")

        console.print()

    except APIError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise SystemExit(1)
    except SystemExit:
        raise
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        raise SystemExit(1)


def inspect_cmd(config: EnvInspectConfig) -> None:
    """Inspect environment source without downloading the archive locally."""
    env_id = config.env_id
    source_path = config.source_path
    version = config.version
    output = config.output
    max_bytes = config.max_bytes

    validate_output_format(output, console)

    try:
        try:
            owner, name, parsed_version = parse_env_id(env_id)
        except ValueError as e:
            console.print(f"[red]Error: {e}[/red]")
            raise SystemExit(1)
        target_version = parsed_version or version
        client = APIClient(require_auth=False)
        params: Dict[str, Any] = {"max_bytes": max_bytes}
        if source_path:
            params["path"] = source_path

        response = client.get(
            f"/environmentshub/{owner}/{name}/@{target_version}/inspect",
            params=params,
        )
        data = response.get("data", response)

        if output == "json":
            output_data_as_json(data, console)
            return

        if data.get("kind") == "file":
            inspected_path = data.get("path") or source_path or "/"
            console.print()
            console.print(f"[bold cyan]{owner}/{name}[/bold cyan][dim]@{target_version}[/dim]")
            console.print(f"[dim]{inspected_path}[/dim]")
            console.print()

            content = data.get("content") or ""
            if content:
                console.print(content, markup=False, highlight=False)
                if not content.endswith("\n"):
                    console.print()
            else:
                console.print("[dim](empty file)[/dim]")

            if data.get("truncated"):
                total_bytes = data.get("total_bytes") or max_bytes
                console.print(
                    f"[yellow]Output truncated from {format_file_size(total_bytes)}. "
                    f"Re-run with --max-bytes > {max_bytes} to view more.[/yellow]"
                )
            return

        entries = data.get("entries", [])
        title_path = data.get("path") or "/"
        table = Table(title=f"Source: {owner}/{name}@{target_version} (path: {title_path})")
        table.add_column("Type", style="blue", no_wrap=True)
        table.add_column("Path", style="cyan")
        table.add_column("Size", style="dim", justify="right")

        for entry in entries:
            is_directory = bool(entry.get("is_directory"))
            entry_type = "dir" if is_directory else "file"
            size_value = entry.get("size")
            size_display = (
                "-" if is_directory or size_value is None else format_file_size(size_value)
            )
            table.add_row(entry_type, entry.get("path", ""), size_display)

        console.print(table)
        if not entries:
            console.print("[dim]No files found in this directory.[/dim]")
            return

        example_path = next(
            (entry.get("path") for entry in entries if not entry.get("is_directory")),
            entries[0].get("path"),
        )
        if example_path:
            inspect_example = f"prime env inspect {owner}/{name}@{target_version} {example_path}"
            console.print(f"\n[dim]Inspect a file with: {inspect_example}[/dim]")

    except APIError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise SystemExit(1)
    except SystemExit:
        raise
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        raise SystemExit(1)


def install(config: EnvInstallConfig) -> None:
    """Install a verifiers environment.

    \b
    Examples:
        prime env install gsm8k                    # local install from ./environments
        prime env install gsm8k -p /path/to/envs   # local install from custom path
        prime env install owner/environment        # install from Prime Hub
        prime env install owner/environment@0.2.3  # specific version
        prime env install env1 env2 env3           # install multiple
    """
    env_ids = config.env_ids
    path = config.path
    prerelease = config.prerelease

    if not shutil.which("uv"):
        console.print("[red]Error: uv is not installed.[/red]")
        raise SystemExit(1)

    failed: list[tuple[str, str]] = []
    for env_id in dict.fromkeys(env_ids):
        try:
            if "/" in env_id:
                owner, name, version = parse_env_id(env_id)
                client = APIClient(require_auth=False)
                response = client.get(f"/environmentshub/{owner}/{name}/@{version or 'latest'}")
                details = response.get("data", response)
                if not isinstance(details, dict):
                    raise ValueError(f"Invalid Environments Hub response for {env_id!r}")
                module = install_environment_from_hub(
                    env_id,
                    details,
                    api_key=client.api_key,
                    base_url=client.base_url,
                    python_executable=resolve_workspace_python(),
                    prerelease=prerelease,
                )
            else:
                env_path = Path(path) / env_id.replace("-", "_")
                if not env_path.is_dir():
                    raise FileNotFoundError(f"Local environment not found: {env_path}")
                subprocess.run(_uv_pip_command("install", "-e", str(env_path)), check=True)
                module = normalize_package_name(env_id)
            console.print(f"[green]✓ Installed {env_id}[/green]")
            console.print(f"[dim]V1: prime eval run {module}[/dim]")
            console.print(f"[dim]V0: prime eval run --id {module}[/dim]")
        except (APIError, OSError, ValueError, subprocess.CalledProcessError) as exc:
            failed.append((env_id, str(exc)))
            console.print(f"[red]✗ {env_id}: {exc}[/red]")

    if failed:
        raise SystemExit(1)


def uninstall(config: EnvUninstallConfig) -> None:
    """Uninstall an environment distribution with uv."""
    env_name = config.env_name

    package = normalize_package_name(env_name.rsplit("/", 1)[-1].split("@", 1)[0])
    try:
        subprocess.run(_uv_pip_command("uninstall", package), check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        console.print(f"[red]Uninstall failed: {exc}[/red]")
        raise SystemExit(1) from exc
    console.print(f"[green]✓ Uninstalled {package}[/green]")


def list_versions(config: EnvVersionListConfig) -> None:
    """List all versions of an environment"""
    env_id = config.env_id
    full_hashes = config.full_hashes

    try:
        client = APIClient(require_auth=False)

        parts = env_id.split("/")
        if len(parts) != 2:
            console.print("[red]Error: Invalid environment ID format. Expected: owner/name[/red]")
            raise SystemExit(1)

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
            raise SystemExit(1)

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
        raise SystemExit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        raise SystemExit(1)


def delete_version(config: EnvVersionDeleteConfig) -> None:
    """Delete a specific environment version from the environments hub using its content hash"""
    env_id = config.env_id
    content_hash = config.content_hash
    force = config.force

    try:
        # Validate that we have a proper content hash (basic validation)
        if len(content_hash) < 8:
            console.print(
                "[red]Error: Please provide a valid content hash (at least 8 characters)[/red]"
            )
            console.print(
                "[yellow]Use 'prime env version list' to see available content hashes[/yellow]"
            )
            raise SystemExit(1)

        if not force:
            try:
                confirm_msg = (
                    f"Are you sure you want to permanently delete version with content "
                    f"hash '{content_hash}' from '{env_id}' on the environments hub?"
                )
                confirmed = confirm(confirm_msg)
                if not confirmed:
                    console.print("Deletion cancelled.")
                    raise SystemExit(0)
            except KeyboardInterrupt:
                console.print("Deletion cancelled.")
                raise SystemExit(0)

        client = APIClient()

        parts = env_id.split("/")
        if len(parts) != 2:
            console.print("[red]Error: Invalid environment ID format. Expected: owner/name[/red]")
            raise SystemExit(1)

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
            raise SystemExit(1)

    except APIError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise SystemExit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        raise SystemExit(1)


def delete(config: EnvDeleteConfig) -> None:
    """Delete an entire environment from the environments hub"""
    env_id = config.env_id
    force = config.force

    try:
        if not force:
            try:
                delete_msg = (
                    f"Are you sure you want to permanently delete entire environment "
                    f"'{env_id}' and ALL its versions from the environments hub?"
                )
                confirmed = confirm(delete_msg)
                if not confirmed:
                    console.print("Deletion cancelled.")
                    raise SystemExit(0)
            except KeyboardInterrupt:
                console.print("Deletion cancelled.")
                raise SystemExit(0)

        client = APIClient()
        console.print(f"Deleting {env_id} from remote hub...")

        try:
            client.delete(f"/environmentshub/{env_id}")
            console.print(f"[green]✓ Environment {env_id} deleted successfully[/green]")
        except APIError as e:
            console.print(f"[red]Failed to delete environment: {e}[/red]")
            raise SystemExit(1)

    except APIError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise SystemExit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        raise SystemExit(1)


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


def env_secret_list(config: EnvSecretListConfig) -> None:
    """List all secrets for an environment."""
    environment = config.environment
    output = config.output

    validate_output_format(output, console)
    owner, env_name = _resolve_environment(environment)

    try:
        client = APIClient()
        env_id = _get_environment_id(client, owner, env_name)
        secrets = _fetch_env_secrets(client, env_id)

        if output == "json":
            output_data_as_json({"secrets": secrets}, console)
            return

        if not secrets:
            console.print("[yellow]No secrets found for this environment.[/yellow]")
            return

        table = Table(title=f"Secrets for {owner}/{env_name}")
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
        raise SystemExit(1)


def env_secret_create(config: EnvSecretCreateConfig) -> None:
    """Create an environment-specific secret."""
    environment = config.environment
    name = config.name
    value = config.value
    description = config.description
    output = config.output

    validate_output_format(output, console)
    owner, env_name = _resolve_environment(environment)

    try:
        if not name:
            name = prompt_for_value("Secret name")
            if not name:
                console.print("\n[dim]Cancelled.[/dim]")
                raise SystemExit(0)

        if not validate_env_var_name(name, "secret"):
            raise SystemExit(1)

        if not value:
            value = prompt_for_value("Secret value", hide_input=True)
            if not value:
                console.print("\n[dim]Cancelled.[/dim]")
                raise SystemExit(0)

        with console.status("[bold blue]Creating secret...", spinner="dots"):
            client = APIClient()
            env_id = _get_environment_id(client, owner, env_name)

            payload: Dict[str, Any] = {"name": name, "value": value}
            if description:
                payload["description"] = description

            response = client.post(f"/environmentshub/{env_id}/secrets", json=payload)
            secret = response.get("data", {})

        if output == "json":
            output_data_as_json(secret, console)
            return

        console.print(f"[green]✓ Created secret '{name}' for {owner}/{env_name}[/green]")
        console.print(f"[dim]ID: {secret.get('id')}[/dim]")

    except KeyboardInterrupt:
        console.print("\n[dim]Cancelled.[/dim]")
        raise SystemExit(0)
    except APIError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise SystemExit(1)


def env_secret_update(config: EnvSecretUpdateConfig) -> None:
    """Update an environment-specific secret."""
    environment = config.environment
    secret_id = config.secret_id
    name = config.name
    value = config.value
    description = config.description
    output = config.output

    validate_output_format(output, console)
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
                raise SystemExit(0)

        if name is not None and not validate_env_var_name(name, "secret"):
            raise SystemExit(1)

        payload: Dict[str, Any] = {}
        if name is not None:
            payload["name"] = name
        if value is not None:
            payload["value"] = value
        if description is not None:
            payload["description"] = description

        response = client.patch(f"/environmentshub/{env_id}/secrets/{secret_id}", json=payload)
        secret = response.get("data", {})

        if output == "json":
            output_data_as_json(secret, console)
            return

        console.print(
            f"[green]✓ Updated secret '{secret.get('name')}' for {owner}/{env_name}[/green]"
        )

    except KeyboardInterrupt:
        console.print("\n[dim]Cancelled.[/dim]")
        raise SystemExit(0)
    except APIError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise SystemExit(1)


def env_secret_delete(config: EnvSecretDeleteConfig) -> None:
    """Delete an environment-specific secret."""
    environment = config.environment
    secret_id = config.secret_id
    yes = config.yes

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
            confirmed = confirm(f"Delete secret '{secret_name}' from {owner}/{env_name}?")
            if not confirmed:
                console.print("\n[dim]Cancelled.[/dim]")
                raise SystemExit(0)

        client.delete(f"/environmentshub/{env_id}/secrets/{secret_id}")
        console.print(f"[green]✓ Deleted secret '{secret_name}' from {owner}/{env_name}[/green]")

    except KeyboardInterrupt:
        console.print("\n[dim]Cancelled.[/dim]")
        raise SystemExit(0)
    except APIError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise SystemExit(1)


def env_secret_link(config: EnvSecretLinkConfig) -> None:
    """Link a global secret to an environment."""
    global_secret_id = config.global_secret_id
    environment = config.environment
    output = config.output

    validate_output_format(output, console)
    owner, env_name = _resolve_environment(environment)

    try:
        client = APIClient()
        env_id = _get_environment_id(client, owner, env_name)

        response = client.post(
            f"/environmentshub/{env_id}/secrets/link/{global_secret_id}",
            json={},
        )
        linked = response.get("data", {})

        if output == "json":
            output_data_as_json(linked, console)
            return

        secret_name = linked.get("secretName", global_secret_id)
        console.print(
            f"[green]✓ Linked global secret '{secret_name}' to {owner}/{env_name}[/green]"
        )

    except APIError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise SystemExit(1)


def env_secret_unlink(config: EnvSecretUnlinkConfig) -> None:
    """Unlink a global secret from an environment."""
    global_secret_id = config.global_secret_id
    environment = config.environment
    yes = config.yes

    owner, env_name = _resolve_environment(environment)

    try:
        if not yes:
            confirmed = confirm(f"Unlink global secret {global_secret_id} from {owner}/{env_name}?")
            if not confirmed:
                console.print("\n[dim]Cancelled.[/dim]")
                raise SystemExit(0)

        client = APIClient()
        env_id = _get_environment_id(client, owner, env_name)
        client.delete(f"/environmentshub/{env_id}/secrets/link/{global_secret_id}")
        console.print(f"[green]✓ Unlinked global secret from {owner}/{env_name}[/green]")

    except KeyboardInterrupt:
        console.print("\n[dim]Cancelled.[/dim]")
        raise SystemExit(0)
    except APIError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise SystemExit(1)


def var_list(config: EnvVarListConfig) -> None:
    """List all variables for an environment."""
    environment = config.environment
    output = config.output

    validate_output_format(output, console)
    owner, env_name = _resolve_environment(environment)

    try:
        client = APIClient()
        env_id = _get_environment_id(client, owner, env_name)
        response = client.get(f"/environmentshub/{env_id}/variables")
        variables = response.get("data", [])

        if output == "json":
            output_data_as_json({"variables": variables}, console)
            return

        if not variables:
            console.print("[yellow]No variables found for this environment.[/yellow]")
            return

        table = Table(title=f"Variables for {owner}/{env_name}")
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
        raise SystemExit(1)


def var_create(config: EnvVarCreateConfig) -> None:
    """Create an environment variable."""
    environment = config.environment
    name = config.name
    value = config.value
    description = config.description
    output = config.output

    validate_output_format(output, console)
    owner, env_name = _resolve_environment(environment)

    try:
        if not name:
            name = prompt_for_value("Variable name")
            if not name:
                console.print("\n[dim]Cancelled.[/dim]")
                raise SystemExit(0)

        if not validate_env_var_name(name, "variable"):
            raise SystemExit(1)

        if not value:
            value = prompt_for_value("Variable value")
            if not value:
                console.print("\n[dim]Cancelled.[/dim]")
                raise SystemExit(0)

        with console.status("[bold blue]Creating variable...", spinner="dots"):
            client = APIClient()
            env_id = _get_environment_id(client, owner, env_name)

            payload: Dict[str, Any] = {"name": name, "value": value}
            if description:
                payload["description"] = description

            response = client.post(f"/environmentshub/{env_id}/variables", json=payload)
            var = response.get("data", {})

        if output == "json":
            output_data_as_json(var, console)
            return

        console.print(f"[green]✓ Created variable '{name}' for {owner}/{env_name}[/green]")
        console.print(f"[dim]ID: {var.get('id')}[/dim]")

    except KeyboardInterrupt:
        console.print("\n[dim]Cancelled.[/dim]")
        raise SystemExit(0)
    except APIError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise SystemExit(1)


def var_update(config: EnvVarUpdateConfig) -> None:
    """Update an environment variable."""
    var_id = config.var_id
    environment = config.environment
    name = config.name
    value = config.value
    description = config.description
    output = config.output

    validate_output_format(output, console)

    if not any_provided(name, value, description):
        console.print(
            "[red]Error: At least one of --name, --value, or --description is required[/red]"
        )
        raise SystemExit(1)

    owner, env_name = _resolve_environment(environment)

    try:
        client = APIClient()
        env_id = _get_environment_id(client, owner, env_name)

        if name is not None and not validate_env_var_name(name, "variable"):
            raise SystemExit(1)

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

        if output == "json":
            output_data_as_json(var, console)
            return

        console.print(f"[green]✓ Updated variable '{var.get('name')}'[/green]")

    except APIError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise SystemExit(1)


def var_delete(config: EnvVarDeleteConfig) -> None:
    """Delete an environment variable."""
    var_id = config.var_id
    environment = config.environment
    yes = config.yes

    owner, env_name = _resolve_environment(environment)

    try:
        if not yes:
            confirmed = confirm(f"Delete variable {var_id} from {owner}/{env_name}?")
            if not confirmed:
                console.print("\n[dim]Cancelled.[/dim]")
                raise SystemExit(0)

        client = APIClient()
        env_id = _get_environment_id(client, owner, env_name)
        client.delete(f"/environmentshub/{env_id}/variables/{var_id}")
        console.print("[green]✓ Variable deleted[/green]")

    except KeyboardInterrupt:
        console.print("\n[dim]Cancelled.[/dim]")
        raise SystemExit(0)
    except APIError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise SystemExit(1)
