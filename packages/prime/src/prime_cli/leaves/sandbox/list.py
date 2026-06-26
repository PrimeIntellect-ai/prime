"""Pydantic leaf for ``prime sandbox list``."""

from __future__ import annotations

from pydantic import AliasChoices, Field
from pydantic_config import BaseConfig


class Config(BaseConfig):
    """List your sandboxes."""

    team_id: str | None = Field(
        None, description="Filter by team ID (uses config team_id if not specified)"
    )
    status: str | None = Field(None, description="Filter by status")
    labels: list[str] | None = Field(
        None,
        validation_alias=AliasChoices("labels", "l"),
        description="Filter by labels; sandboxes must have all provided labels.",
    )
    page: int = Field(1, validation_alias=AliasChoices("page", "p"), description="Page number")
    num: int = Field(50, validation_alias=AliasChoices("num", "n"), description="Items per page")
    all: bool = Field(False, description="Show all sandboxes including terminated ones")
    output: str = Field(
        "table",
        validation_alias=AliasChoices("output", "o"),
        description="Output format: table or json",
    )


POSITIONALS = ()


def run(config: Config):
    from prime_cli.commands.sandbox import list_sandboxes_cmd as callback

    return callback(config)
