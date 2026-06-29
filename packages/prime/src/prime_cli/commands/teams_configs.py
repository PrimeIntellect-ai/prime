"""Pydantic Config schemas for the ``prime teams`` surface.

These were previously defined as one-file-per-command leaves under
``prime_cli.leaves``; they now live next to their command callbacks and are
consumed by the router via ``Command.config_attr``.
"""

from __future__ import annotations

from pydantic import AliasChoices, Field
from pydantic_config import BaseConfig


class TeamsListConfig(BaseConfig):
    """List teams for the current user."""

    limit: int = Field(100, description="Maximum number of teams to list")
    offset: int = Field(0, description="Number of teams to skip")
    output: str = Field(
        "table",
        validation_alias=AliasChoices("output", "o"),
        description="Output format: table or json",
    )


class TeamsMembersConfig(BaseConfig):
    """List members of a team."""

    team_id: str | None = Field(None, description="Team ID (uses config team_id if omitted)")
    output: str = Field(
        "table",
        validation_alias=AliasChoices("output", "o"),
        description="Output format: table or json",
    )
