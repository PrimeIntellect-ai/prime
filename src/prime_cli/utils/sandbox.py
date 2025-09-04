from typing import Optional, Tuple, Dict, Any

from prime_cli.utils.formatters import format_resources
from prime_cli.utils.time_utils import human_age, iso_timestamp
from prime_cli.utils.formatters import obfuscate_env_vars


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

def format_sandbox_for_list(sandbox: Any) -> Dict[str, Any]:
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


def format_sandbox_for_details(sandbox: Any) -> Dict[str, Any]:
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