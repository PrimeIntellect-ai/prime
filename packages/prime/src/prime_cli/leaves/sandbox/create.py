"""Pydantic leaf for ``prime sandbox create``."""

from __future__ import annotations

from pydantic import AliasChoices, Field
from pydantic_config import BaseConfig


class Config(BaseConfig):
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


POSITIONALS = ("docker_image",)


def run(config: Config):
    from prime_cli.commands.sandbox import create as callback

    return callback(config)
