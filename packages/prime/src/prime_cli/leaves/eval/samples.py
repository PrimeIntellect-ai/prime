"""Pydantic leaf for ``prime eval samples``."""

from __future__ import annotations

from pydantic import AliasChoices, Field
from pydantic_config import BaseConfig


class Config(BaseConfig):
    """"""

    eval_id: str = Field(..., description="The ID of the evaluation")
    page: int = Field(1, validation_alias=AliasChoices("page", "p"), description="Page number")
    num: int = Field(100, validation_alias=AliasChoices("num", "n"), description="Items per page")
    output: str = Field(
        "json", validation_alias=AliasChoices("output", "o"), description="json|pretty"
    )


POSITIONALS = ("eval_id",)


def run(config: Config):
    from prime_cli.commands.evals import get_samples as callback

    return callback(config)
