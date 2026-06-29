"""Pydantic Config schemas for the ``prime sandbox`` surface.

These were previously defined as one-file-per-command leaves under
``prime_cli.leaves``; they now live next to their command callbacks and are
consumed by the router via ``Command.config_attr``.
"""

from __future__ import annotations

from pydantic import AliasChoices, Field
from pydantic_config import BaseConfig


class SandboxCreateConfig(BaseConfig):
    """Create a new sandbox"""

    docker_image: str | None = Field(
        None, description="Image to run. When using --vm, provide the VM image reference."
    )
    name: str | None = Field(
        None, description="Name for the sandbox (auto-generated if not provided)"
    )
    start_command: str | None = Field(
        "tail -f /dev/null", description="Command to run in the container"
    )
    cpu_cores: float = Field(1.0, description="Number of CPU cores")
    memory_gb: float = Field(2.0, description="Memory in GB")
    disk_size_gb: float = Field(10.0, description="Disk size in GB")
    gpu_count: int = Field(0, description="Number of GPUs")
    gpu_type: str | None = Field(
        None,
        description="GPU type/model (e.g. H100_80GB, A100_80GB). Required when --gpu-count > 0",
    )
    vm: bool = Field(
        False,
        description="Create a VM-backed sandbox. Required when requesting GPUs.",
    )
    network_access: bool = Field(
        True, description="Allow outbound internet access (enabled by default)"
    )
    timeout_minutes: int = Field(60, description="Timeout in minutes")
    idle_timeout_minutes: int | None = Field(
        None,
        description="Terminate after this many idle minutes. Disabled by default.",
    )
    team_id: str | None = Field(None, description="Team ID (uses config team_id if not specified)")
    region: str | None = Field(
        None,
        description="Sandbox cluster region. Uses the backend default when omitted.",
    )
    registry_credentials_id: str | None = Field(
        None, description="Registry credentials ID for pulling private images"
    )
    env: list[str] | None = Field(
        None,
        description="Environment variables in KEY=VALUE format. Can be specified multiple times.",
    )
    secret: list[str] | None = Field(
        None, description="Secrets in KEY=VALUE format. Can be specified multiple times."
    )
    labels: list[str] | None = Field(
        None, validation_alias=AliasChoices("labels", "l"), description="Labels for the sandbox."
    )
    guaranteed: bool = Field(
        False,
        description="Use Guaranteed QoS. Admin only; incompatible with --vm.",
    )
    yes: bool = Field(
        False, validation_alias=AliasChoices("yes", "y"), description="Skip confirmation prompt"
    )


class SandboxDeleteConfig(BaseConfig):
    """Delete one or more sandboxes by ID, by label, or all sandboxes with --all"""

    sandbox_ids: list[str] | None = Field(
        None, description="Sandbox ID(s) to delete (space or comma-separated)"
    )
    all: bool = Field(
        False, validation_alias=AliasChoices("all", "a"), description="Delete all sandboxes"
    )
    labels: list[str] | None = Field(
        None,
        validation_alias=AliasChoices("labels", "l"),
        description="Delete sandboxes having all provided labels.",
    )
    yes: bool = Field(
        False, validation_alias=AliasChoices("yes", "y"), description="Skip confirmation prompt"
    )
    all_users: bool = Field(
        False,
        validation_alias=AliasChoices("all_users", "A"),
        description="Delete across every user in the team. Requires team admin role.",
    )
    target_user_id: str | None = Field(
        None,
        validation_alias=AliasChoices("target_user_id", "user", "u"),
        description="Target one teammate. Requires team admin; conflicts with --all-users.",
    )


class SandboxDownloadConfig(BaseConfig):
    """Download a file from a sandbox"""

    sandbox_id: str = Field(..., description="Sandbox ID to download file from")
    remote_path: str = Field(..., description="Path to file in sandbox")
    local_file: str = Field(..., description="Path where file should be saved locally")


class SandboxExposeConfig(BaseConfig):
    """Expose a port from a sandbox."""

    sandbox_id: str = Field(..., description="Sandbox ID to expose port from")
    port: int = Field(..., description="Port number to expose")
    name: str | None = Field(None, description="Optional name for the exposed port")
    protocol: str = Field(
        "HTTP", validation_alias=AliasChoices("protocol", "p"), description="Protocol: HTTP or TCP"
    )
    output: str = Field(
        "table",
        validation_alias=AliasChoices("output", "o"),
        description="Output format: table or json",
    )


class SandboxGetConfig(BaseConfig):
    """Get detailed information about a specific sandbox"""

    sandbox_id: str = Field(...)
    output: str = Field(
        "table",
        validation_alias=AliasChoices("output", "o"),
        description="Output format: table or json",
    )


class SandboxListConfig(BaseConfig):
    """List your sandboxes."""

    team_id: str | None = Field(
        None, description="Filter by team ID (uses config team_id if not specified)"
    )
    status: str | None = Field(None, description="Filter by status")
    labels: list[str] | None = Field(
        None,
        validation_alias=AliasChoices("labels", "l"),
        description="Filter by labels; sandboxes must have all provided labels.",
    )
    page: int = Field(1, validation_alias=AliasChoices("page", "p"), description="Page number")
    num: int = Field(50, validation_alias=AliasChoices("num", "n"), description="Items per page")
    all: bool = Field(False, description="Show all sandboxes including terminated ones")
    output: str = Field(
        "table",
        validation_alias=AliasChoices("output", "o"),
        description="Output format: table or json",
    )


class SandboxListPortsConfig(BaseConfig):
    """List exposed ports for a sandbox, or all sandboxes if no ID is provided"""

    sandbox_id: str | None = Field(
        None, description="Sandbox ID (omit to list all exposed ports across all sandboxes)"
    )
    output: str = Field(
        "table",
        validation_alias=AliasChoices("output", "o"),
        description="Output format: table or json",
    )


class SandboxLogsConfig(BaseConfig):
    """Get logs from a sandbox"""

    sandbox_id: str = Field(...)


class SandboxResetCacheConfig(BaseConfig):
    """Reset sandbox authentication cache"""

    yes: bool = Field(
        False, validation_alias=AliasChoices("yes", "y"), description="Skip confirmation prompt"
    )


class SandboxRunConfig(BaseConfig):
    """Execute a command in a sandbox."""

    sandbox_id: str = Field(...)
    command: list[str] = Field(
        ...,
        description="Command to execute. Use -- before command options.",
    )
    working_dir: str | None = Field(
        None, validation_alias=AliasChoices("working_dir", "w"), description="Working directory"
    )
    env: list[str] | None = Field(
        None,
        validation_alias=AliasChoices("env", "e"),
        description="Environment variables in KEY=VALUE format. Can be specified multiple times.",
    )
    timeout: int | None = Field(None, description="Timeout for the command in seconds")
    user: str | None = Field(
        None,
        validation_alias=AliasChoices("user", "u"),
        description="Container username or UID, optionally USER:GROUP.",
    )


class SandboxSshConfig(BaseConfig):
    """Connect to a sandbox via SSH."""

    sandbox_id: str | None = Field(
        None, description="Sandbox ID to SSH into (interactive selection if not provided)"
    )
    ssh_args: list[str] | None = Field(
        None, description="Additional SSH arguments (e.g., -- -v for verbose)"
    )
    shell: str | None = Field(
        None,
        validation_alias=AliasChoices("shell", "s"),
        description="Shell to use (e.g., bash, zsh, sh). Auto-detected if not specified.",
    )


class SandboxUnexposeConfig(BaseConfig):
    """Unexpose a port from a sandbox"""

    sandbox_id: str = Field(..., description="Sandbox ID")
    exposure_id: str = Field(..., description="Exposure ID to remove")
    yes: bool = Field(
        False, validation_alias=AliasChoices("yes", "y"), description="Skip confirmation prompt"
    )


class SandboxUploadConfig(BaseConfig):
    """Upload a file to a sandbox"""

    sandbox_id: str = Field(..., description="Sandbox ID to upload file to")
    local_file: str = Field(..., description="Path to local file to upload")
    remote_path: str = Field(..., description="Path where file should be stored in sandbox")
