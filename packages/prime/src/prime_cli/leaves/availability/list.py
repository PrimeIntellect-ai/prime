"""Pydantic leaf for ``prime availability list``."""

from __future__ import annotations

from pydantic import AliasChoices, Field
from pydantic_config import BaseConfig


class Config(BaseConfig):
    """List available GPU resources"""

    gpu_type: str | None = Field(None, description="GPU type (e.g., H100_80GB)")
    gpu_count: int | None = Field(None, description="Number of GPUs required")
    regions: list[str] | None = Field(None, description="Filter by regions (e.g., united_states)")
    socket: str | None = Field(None, description="Filter by socket type (e.g., PCIe, SXM5, SXM4)")
    provider: str | None = Field(None, description="Filter by provider (e.g., aws, azure, google)")
    disks: list[str] | None = Field(None, description="Filter by disk ids")
    group_similar: bool = Field(True, description="Group similar configurations from same provider")
    output: str = Field(
        "table",
        validation_alias=AliasChoices("output", "o"),
        description="Output format: table or json",
    )


POSITIONALS = ()


def run(config: Config):
    from prime_cli.commands.availability import list as callback

    return callback(config)
