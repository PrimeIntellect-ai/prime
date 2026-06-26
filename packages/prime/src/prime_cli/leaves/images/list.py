"""Pydantic leaf for ``prime images list``."""

from __future__ import annotations

from pydantic import AliasChoices, Field
from pydantic_config import BaseConfig


class Config(BaseConfig):
    """List all images you've pushed to Prime Intellect registry."""

    output: str = Field(
        "table",
        validation_alias=AliasChoices("output", "o"),
        description="Output format (table or json)",
    )
    search: str | None = Field(
        None,
        validation_alias=AliasChoices("search", "q"),
        description="Case-insensitive substring match on image name, tag, or reference",
    )
    page: int = Field(1, validation_alias=AliasChoices("page", "p"), description="Page number")
    num: int = Field(
        50, validation_alias=AliasChoices("num", "n"), description="Items per page (max 250)"
    )


POSITIONALS = ()


def run(config: Config):
    from prime_cli.commands.images import list_images as callback

    return callback(config)
