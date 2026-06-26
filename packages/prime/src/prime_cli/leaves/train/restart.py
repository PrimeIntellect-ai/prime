"""Pydantic leaf for ``prime train restart``."""

from __future__ import annotations

from pydantic import AliasChoices, Field
from pydantic_config import BaseConfig


class Config(BaseConfig):
    """Restart a running run from its latest checkpoint."""

    run_id: str = Field(..., description="Run ID to restart")
    force: bool = Field(
        False, validation_alias=AliasChoices("force", "f"), description="Skip confirmation"
    )


POSITIONALS = ("run_id",)


def run(config: Config):
    from prime_cli.commands.rl import restart_run as callback

    return callback(config)
