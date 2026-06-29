"""Pydantic Config schemas for the ``prime pods`` surface.

These were previously defined as one-file-per-command leaves under
``prime_cli.leaves``; they now live next to their command callbacks and are
consumed by the router via ``Command.config_attr``.
"""

from __future__ import annotations

from pydantic import AliasChoices, Field
from pydantic_config import BaseConfig


class PodsCreateConfig(BaseConfig):
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


class PodsHistoryConfig(BaseConfig):
    """List your pods history (terminated pods)"""

    limit: int = Field(100, description="Maximum number of history items to list")
    offset: int = Field(0, description="Number of history items to skip")
    output: str = Field(
        "table",
        validation_alias=AliasChoices("output", "o"),
        description="Output format: table or json",
    )


class PodsListConfig(BaseConfig):
    """List your running pods"""

    limit: int = Field(100, description="Maximum number of pods to list")
    offset: int = Field(0, description="Number of pods to skip")
    watch: bool = Field(
        False,
        validation_alias=AliasChoices("watch", "w"),
        description="Watch pods list in real-time",
    )
    output: str = Field(
        "table",
        validation_alias=AliasChoices("output", "o"),
        description="Output format: table or json",
    )


class PodsSshConfig(BaseConfig):
    """SSH / connect to a pod using configured SSH key"""

    pod_id: str = Field(...)


class PodsStatusConfig(BaseConfig):
    """Get detailed status of a specific pod"""

    pod_id: str = Field(...)
    output: str = Field(
        "table",
        validation_alias=AliasChoices("output", "o"),
        description="Output format: table or json",
    )


class PodsTerminateConfig(BaseConfig):
    """Terminate a pod"""

    pod_id: str = Field(...)
    yes: bool = Field(
        False, validation_alias=AliasChoices("yes", "y"), description="Skip confirmation prompt"
    )
