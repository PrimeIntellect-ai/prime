"""Pydantic leaf for ``prime eval stop``."""

from __future__ import annotations

from pydantic import Field
from pydantic_config import BaseConfig


class Config(BaseConfig):
    """Stop a running hosted evaluation."""

    eval_id: str = Field(..., description="Evaluation id to stop")


POSITIONALS = ("eval_id",)


def run(config: Config):
    from prime_cli.commands.evals import stop_cmd as callback

    return callback(config)
