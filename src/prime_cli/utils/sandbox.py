from typing import Optional, Tuple, Dict, Any
from rich.console import Console

from prime_cli.api.sandbox import Sandbox, SandboxClient
from prime_cli.utils.formatters import format_resources
from prime_cli.utils.time_utils import human_age, iso_timestamp
from prime_cli.utils.formatters import obfuscate_env_vars
from prime_cli.utils.debug import debug_log


def parse_cp_arg(arg: str) -> Tuple[Optional[str], str]:
    """Parse cp-style arg: either "<sandbox-id>:<path>" or local path.

    Returns (sandbox_id, path). sandbox_id is None if local.
    """
    if ":" in arg and not arg.startswith(":"):
        sandbox_id, path = arg.split(":", 1)
        if sandbox_id:
            return sandbox_id, path
    return None, arg


def expand_home_in_path(path: str) -> str:
    """
    Safely expand $HOME to /sandbox-workspace in paths.
    Only allows $HOME expansion for security - no other environment variables.
    """
    if "$HOME" in path:
        # Replace $HOME with the sandbox workspace path
        expanded_path = path.replace("$HOME", "/sandbox-workspace")
        return expanded_path
    return path

def handle_local_to_sandbox(
    sandbox_client: SandboxClient,
    source_path: str,
    sandbox_id: str,
    destination_path: str,
    working_dir: Optional[str],
    console: Console,
) -> None:
    """Handle copying from local to sandbox."""
    import logging

    logger = logging.getLogger(__name__)

    logger.debug("🔄 Local to Sandbox Copy Debug:")
    logger.debug(f"   Source path: {source_path}")
    logger.debug(f"   Sandbox ID: {sandbox_id}")
    logger.debug(f"   Destination path: {destination_path}")
    logger.debug(f"   Working dir: {working_dir}")

    console.print(
        f"[blue]Uploading {source_path} to sandbox {sandbox_id}:{destination_path}...[/blue]"
    )

    try:
        with console.status("[bold blue]Uploading...", spinner="dots"):
            logger.debug("📤 Calling sandbox_client.upload_path...")
            result = sandbox_client.upload_path(
                sandbox_id,
                source_path,
                destination_path,
                working_dir=working_dir,
            )
            logger.debug(f"✅ upload_path result: {result}")

        # Success output
        console.print(f"[green]Upload completed[/green] {result.message}")
        if result.files_uploaded:
            console.print(f"Files uploaded: {result.files_uploaded}")
        if result.bytes_uploaded:
            console.print(f"Bytes uploaded: {result.bytes_uploaded}")

    except Exception as e:
        logger.error(f"❌ Upload failed with exception: {e}")
        logger.error(f"   Exception type: {type(e)}")
        console.print(f"[red]Upload failed:[/red] {e}")
        raise


def handle_sandbox_to_local(
    sandbox_client: SandboxClient,
    sandbox_id: str,
    source_path: str,
    destination_path: str,
    working_dir: Optional[str],
    console: Console,
) -> None:
    """Handle copying from sandbox to local."""
    debug_log(
        f"_handle_sandbox_to_local called with sandbox_id={sandbox_id}, "
        f"source_path={source_path}, destination_path={destination_path}"
    )

    console.print(
        f"[blue]Downloading from sandbox {sandbox_id}:{source_path} to {destination_path}...[/blue]"
    )

    try:
        debug_log("About to call sandbox_client.download_path")
        with console.status("[bold blue]Downloading...", spinner="dots"):
            sandbox_client.download_path(
                sandbox_id,
                source_path,
                destination_path,
                working_dir=working_dir,
            )

        console.print(f"[green]Download completed to {destination_path}[/green]")

    except Exception as e:
        debug_log(f"Exception caught in _handle_sandbox_to_local: {e}")
        console.print(f"[red]Download failed:[/red] {e}")
        raise

def format_sandbox_for_list(sandbox: Sandbox) -> Dict[str, Any]:
    """Format sandbox data for list display (both table and JSON)"""
    return {
        "id": sandbox.id,
        "name": sandbox.name,
        "image": sandbox.docker_image,
        "status": sandbox.status,
        "resources": format_resources(sandbox.cpu_cores, sandbox.memory_gb, sandbox.gpu_count),
        "created_at": iso_timestamp(sandbox.created_at),  # For JSON output
        "age": human_age(sandbox.created_at),  # For table output
    }


def format_sandbox_for_details(sandbox: Sandbox) -> Dict[str, Any]:
    """Format sandbox data for details display (both table and JSON)"""
    data: Dict[str, Any] = {
        "id": sandbox.id,
        "name": sandbox.name,
        "docker_image": sandbox.docker_image,
        "start_command": sandbox.start_command,
        "status": sandbox.status,
        "cpu_cores": sandbox.cpu_cores,
        "memory_gb": sandbox.memory_gb,
        "disk_size_gb": sandbox.disk_size_gb,
        "disk_mount_path": sandbox.disk_mount_path,
        "gpu_count": sandbox.gpu_count,
        "timeout_minutes": sandbox.timeout_minutes,
        "created_at": iso_timestamp(sandbox.created_at),
        "user_id": sandbox.user_id,
        "team_id": sandbox.team_id,
    }

    if sandbox.started_at:
        data["started_at"] = iso_timestamp(sandbox.started_at)
    if sandbox.terminated_at:
        data["terminated_at"] = iso_timestamp(sandbox.terminated_at)
    if sandbox.environment_vars:
        data["environment_vars"] = obfuscate_env_vars(sandbox.environment_vars)
    if sandbox.advanced_configs:
        data["advanced_configs"] = sandbox.advanced_configs.model_dump()

    return data