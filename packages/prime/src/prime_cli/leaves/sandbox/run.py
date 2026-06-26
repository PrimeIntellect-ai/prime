"""Pydantic leaf for ``prime sandbox run``."""

from __future__ import annotations

from pydantic import AliasChoices, Field
from pydantic_config import BaseConfig


class Config(BaseConfig):
    """Execute a command in a sandbox."""

    sandbox_id: str = Field(...)
    command: list[str] = Field(
        ...,
        description="Command to execute. Use -- before command options.",
    )
    working_dir: str | None = Field(
        None, validation_alias=AliasChoices("working_dir", "w"), description="Working directory"
    )
    env: list[str] | None = Field(
        None,
        validation_alias=AliasChoices("env", "e"),
        description="Environment variables in KEY=VALUE format. Can be specified multiple times.",
    )
    timeout: int | None = Field(None, description="Timeout for the command in seconds")
    user: str | None = Field(
        None,
        validation_alias=AliasChoices("user", "u"),
        description="Container username or UID, optionally USER:GROUP.",
    )


POSITIONALS = ("sandbox_id", "command")


def run(config: Config):
    from prime_cli.commands.sandbox import run as callback

    return callback(config)
