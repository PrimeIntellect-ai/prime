"""Pydantic Config schemas for the ``prime fork`` surface.

These were previously defined as one-file-per-command leaves under
``prime_cli.leaves``; they now live next to their command callbacks and are
consumed by the router via ``Command.config_attr``.
"""

from __future__ import annotations

from pydantic import AliasChoices, Field
from pydantic_config import BaseConfig


class ForkConfig(BaseConfig):
    """Fork a public environment into your Prime Intellect namespace."""

    environment: str = Field(..., description="Public environment to fork, in owner/name format")
    team: str | None = Field(
        None,
        validation_alias=AliasChoices("team", "t"),
        description="Team slug to fork into (uses configured team ID if omitted)",
    )
    output: str = Field(
        "table",
        validation_alias=AliasChoices("output", "o"),
        description="Output format: table or json",
    )
