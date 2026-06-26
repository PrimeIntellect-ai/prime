"""Pydantic leaf for ``prime train progress``."""

from __future__ import annotations

from pydantic import Field
from pydantic_config import BaseConfig


class Config(BaseConfig):
    """Get progress information, including which steps have samples and distributions."""

    run_id: str = Field(..., description="Run ID to get progress for")


POSITIONALS = ("run_id",)


def run(config: Config):
    from prime_cli.commands.rl import get_progress as callback

    return callback(config)
