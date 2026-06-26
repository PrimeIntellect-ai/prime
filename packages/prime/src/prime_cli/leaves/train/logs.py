"""Pydantic leaf for ``prime train logs``."""

from __future__ import annotations

from pydantic import AliasChoices, Field
from pydantic_config import BaseConfig


class Config(BaseConfig):
    """Get logs for a run."""

    run_id: str = Field(..., description="Run ID to get logs for")
    component: str | None = Field(
        None,
        validation_alias=AliasChoices("component", "c"),
        description="Pod component: orchestrator, trainer, inference, or env-server.",
    )
    env: str | None = Field(
        None,
        description="Env-server name or name/N. Implies --component=env-server.",
    )
    tail: int = Field(
        1000, validation_alias=AliasChoices("tail", "n"), description="Number of lines to show"
    )
    follow: bool = Field(
        False, validation_alias=AliasChoices("follow", "f"), description="Follow log output"
    )
    raw: bool = Field(
        False,
        validation_alias=AliasChoices("raw", "r"),
        description="Show raw logs without formatting",
    )
    search: str | None = Field(
        None,
        description="Filter lines by substring, or by regex with --regex.",
    )
    regex: bool = Field(False, description="Treat --search as a regex (RE2 syntax).")
    level: str | None = Field(
        None, description="Filter to one log level: ERROR | WARNING | SUCCESS | INFO | DEBUG."
    )
    since: str | None = Field(
        None,
        description="Filtered query window, such as 15m, 1h, 24h, or seconds.",
    )


POSITIONALS = ("run_id",)


def run(config: Config):
    from prime_cli.commands.rl import get_logs as callback

    return callback(config)
