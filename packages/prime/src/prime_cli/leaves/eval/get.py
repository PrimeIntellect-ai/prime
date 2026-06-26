"""Pydantic leaf for ``prime eval get``."""

from __future__ import annotations

from pydantic import AliasChoices, Field
from pydantic_config import BaseConfig


class Config(BaseConfig):
    """Show evaluation details."""

    eval_id: str = Field(..., description="The ID of the evaluation to retrieve")
    output: str = Field(
        "json", validation_alias=AliasChoices("output", "o"), description="json|pretty"
    )


POSITIONALS = ("eval_id",)


def run(config: Config):
    from prime_cli.commands.evals import get_eval as callback

    return callback(config)
