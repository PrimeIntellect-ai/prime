"""Pydantic Config schemas for the ``prime registry`` surface.

These were previously defined as one-file-per-command leaves under
``prime_cli.leaves``; they now live next to their command callbacks and are
consumed by the router via ``Command.config_attr``.
"""

from __future__ import annotations

from pydantic import AliasChoices, Field
from pydantic_config import BaseConfig


class RegistryCheckImageConfig(BaseConfig):
    """Verify that an image is accessible (optionally using registry credentials)."""

    image: str = Field(..., description="Image reference, e.g. ghcr.io/org/repo:tag")
    registry_credentials_id: str | None = Field(
        None, description="Registry credentials ID for private images"
    )


class RegistryListConfig(BaseConfig):
    """List registry credentials available to the current user."""

    output: str = Field(
        "table",
        validation_alias=AliasChoices("output", "o"),
        description="Output format: table or json",
    )
