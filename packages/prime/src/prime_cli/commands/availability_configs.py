"""Pydantic Config schemas for the ``prime availability`` surface.

These were previously defined as one-file-per-command leaves under
``prime_cli.leaves``; they now live next to their command callbacks and are
consumed by the router via ``Command.config_attr``.
"""

from __future__ import annotations

from pydantic import AliasChoices, Field
from pydantic_config import BaseConfig


class AvailabilityDisksConfig(BaseConfig):
    """List available disks"""

    regions: list[str] | None = Field(None, description="Filter by regions (e.g., united_states)")
    data_center_id: str | None = Field(None, description="Filter by data center ID (e.g., US-1)")
    output: str = Field(
        "table",
        validation_alias=AliasChoices("output", "o"),
        description="Output format: table or json",
    )


class AvailabilityGpuTypesConfig(BaseConfig):
    """List available GPU types"""

    output: str = Field(
        "table",
        validation_alias=AliasChoices("output", "o"),
        description="Output format: table or json",
    )


class AvailabilityListConfig(BaseConfig):
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
