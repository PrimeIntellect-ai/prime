"""Pydantic leaf for ``prime eval logs``."""

from __future__ import annotations

from pydantic import AliasChoices, Field
from pydantic_config import BaseConfig


class Config(BaseConfig):
    """Get logs for a hosted evaluation."""

    eval_id: str = Field(..., description="Evaluation id to get logs for")
    tail: int = Field(
        1000, validation_alias=AliasChoices("tail", "n"), description="Number of lines to show"
    )
    follow: bool = Field(
        False, validation_alias=AliasChoices("follow", "f"), description="Follow log output"
    )
    poll_interval: float = Field(5.0, description="Polling interval in seconds when following logs")


POSITIONALS = ("eval_id",)


def run(config: Config):
    from prime_cli.commands.evals import logs_cmd as callback

    return callback(config)
