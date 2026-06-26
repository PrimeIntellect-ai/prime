"""Pydantic leaf for ``prime disks create``."""

from __future__ import annotations

from pydantic import AliasChoices, Field
from pydantic_config import BaseConfig


class Config(BaseConfig):
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


POSITIONALS = ()


def run(config: Config):
    from prime_cli.commands.disks import create as callback

    return callback(config)
