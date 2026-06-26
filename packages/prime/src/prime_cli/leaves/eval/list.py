"""Pydantic leaf for ``prime eval list``."""

from __future__ import annotations

from pydantic import AliasChoices, Field
from pydantic_config import BaseConfig


class Config(BaseConfig):
    """List evaluations."""

    output: str = Field(
        "table", validation_alias=AliasChoices("output", "o"), description="table|json"
    )
    num: int = Field(20, validation_alias=AliasChoices("num", "n"), description="Items per page")
    page: int = Field(1, validation_alias=AliasChoices("page", "p"), description="Page number")
    env: str | None = Field(
        None,
        validation_alias=AliasChoices("env", "env_name", "e"),
        description="Filter by environment (e.g., 'gsm8k' or 'owner/gsm8k')",
    )


POSITIONALS = ()


def run(config: Config):
    from prime_cli.commands.evals import list_evals as callback

    return callback(config)
