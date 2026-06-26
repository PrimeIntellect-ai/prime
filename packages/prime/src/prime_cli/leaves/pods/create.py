"""Pydantic leaf for ``prime pods create``."""

from __future__ import annotations

from pydantic import AliasChoices, Field
from pydantic_config import BaseConfig


class Config(BaseConfig):
    """Create a new pod with an interactive setup process"""

    id: str | None = Field(None, description="Short ID from availability list")
    cloud_id: str | None = Field(None, description="Cloud ID from cloud provider")
    gpu_type: str | None = Field(None, description="GPU type (e.g. A100, V100)")
    gpu_count: int | None = Field(None, description="Number of GPUs")
    name: str | None = Field(None, description="Name for the pod")
    disk_size: int | None = Field(None, description="Disk size in GB")
    vcpus: int | None = Field(None, description="Number of vCPUs")
    memory: int | None = Field(None, description="Memory in GB")
    image: str | None = Field(
        None, description="Image name or 'custom_template' when using custom template ID"
    )
    custom_template_id: str | None = Field(None, description="Custom template ID")
    team_id: str | None = Field(
        None, description="Team ID to use for the pod (uses config team_id if not specified)"
    )
    disks: list[str] | None = Field(
        None, description="Attach existing disk IDs to the pod. Repeat option for multiple disks."
    )
    env: list[str] | None = Field(
        None,
        description="Environment variables to set in the pod.",
    )
    share_with_team: bool = Field(False, description="Share the pod with all team members")
    add_members: bool = Field(
        False, description="Interactively select team members to share the pod with"
    )
    yes: bool = Field(
        False, validation_alias=AliasChoices("yes", "y"), description="Skip confirmation prompt"
    )


POSITIONALS = ()


def run(config: Config):
    from prime_cli.commands.pods import create as callback

    return callback(config)
