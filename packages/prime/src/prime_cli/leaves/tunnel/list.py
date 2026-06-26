"""Pydantic leaf for ``prime tunnel list``."""

from __future__ import annotations

from pydantic import AliasChoices, Field
from pydantic_config import BaseConfig


class Config(BaseConfig):
    """List active tunnels."""

    team_id: str | None = Field(
        None, description="Team ID to list team tunnels (uses config team_id if not specified)"
    )
    labels: list[str] | None = Field(
        None, validation_alias=AliasChoices("labels", "l"), description="Filter by labels."
    )
    status: str | None = Field(None, description="Filter by status")
    sort_by: str = Field(
        "createdAt", description="Sort field: createdAt, status, name, expiresAt, connectedAt"
    )
    sort_order: str = Field("desc", description="Sort order: asc or desc")
    page: int = Field(1, validation_alias=AliasChoices("page", "p"), description="Page number")
    num: int = Field(50, validation_alias=AliasChoices("num", "n"), description="Items per page")
    output: str = Field(
        "table",
        validation_alias=AliasChoices("output", "o"),
        description="Output format: table or json",
    )


POSITIONALS = ()


def run(config: Config):
    from prime_cli.commands.tunnel import list_tunnels as callback

    return callback(config)
