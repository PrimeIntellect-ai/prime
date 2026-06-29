"""Pydantic Config schemas for the ``prime disks`` surface.

These were previously defined as one-file-per-command leaves under
``prime_cli.leaves``; they now live next to their command callbacks and are
consumed by the router via ``Command.config_attr``.
"""

from __future__ import annotations

from pydantic import AliasChoices, Field
from pydantic_config import BaseConfig


class DisksCreateConfig(BaseConfig):
    """Create a new storage disk"""

    id: str | None = Field(None, description="Short ID from availability list")
    size: int = Field(..., description="Size of the disk in GB")
    name: str | None = Field(None, description="Name for the disk")
    country: str | None = Field(None, description="Country location")
    cloud_id: str | None = Field(None, description="Cloud ID from availability")
    data_center_id: str | None = Field(None, description="Data center ID")
    team_id: str | None = Field(
        None, description="Team ID to use for the disk (uses config team_id if not specified)"
    )
    provider_type: str | None = Field(None, description="Provider type (e.g., lambda, runpod)")
    yes: bool = Field(
        False, validation_alias=AliasChoices("yes", "y"), description="Skip confirmation prompt"
    )


class DisksGetConfig(BaseConfig):
    """Get detailed information about a specific disk"""

    disk_id: str = Field(...)
    output: str = Field(
        "table",
        validation_alias=AliasChoices("output", "o"),
        description="Output format: table or json",
    )


class DisksListConfig(BaseConfig):
    """List your persistent disks"""

    limit: int = Field(100, description="Maximum number of disks to list")
    offset: int = Field(0, description="Number of disks to skip")
    watch: bool = Field(
        False,
        validation_alias=AliasChoices("watch", "w"),
        description="Watch disks list in real-time",
    )
    output: str = Field(
        "table",
        validation_alias=AliasChoices("output", "o"),
        description="Output format: table or json",
    )


class DisksTerminateConfig(BaseConfig):
    """Terminate a disk"""

    disk_id: str = Field(...)
    yes: bool = Field(
        False, validation_alias=AliasChoices("yes", "y"), description="Skip confirmation prompt"
    )


class DisksUpdateConfig(BaseConfig):
    """Update a disk's name"""

    disk_id: str = Field(...)
    name: str = Field(..., description="New name for the disk")
