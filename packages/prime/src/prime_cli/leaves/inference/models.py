"""Pydantic leaf for ``prime inference models``."""

from __future__ import annotations

from pydantic import AliasChoices, Field
from pydantic_config import BaseConfig


class Config(BaseConfig):
    """List available models from Prime Inference (/v1/models)."""

    output: str = Field(
        "table", validation_alias=AliasChoices("output", "o"), description="table|json"
    )
    search: str | None = Field(
        None,
        validation_alias=AliasChoices("search", "q"),
        description="Case-insensitive substring match on model id",
    )
    sort: str = Field(
        "id", validation_alias=AliasChoices("sort", "s"), description="Sort by: id, input, output"
    )
    order: str = Field(
        "asc",
        validation_alias=AliasChoices("order", "d"),
        description="Sort order (direction): asc, desc",
    )


POSITIONALS = ()


def run(config: Config):
    from prime_cli.commands.inference import list_models as callback

    return callback(config)
