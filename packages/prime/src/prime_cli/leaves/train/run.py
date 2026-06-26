"""Pydantic leaf for ``prime train run``."""

from __future__ import annotations

from pydantic import AliasChoices, Field
from pydantic_config import BaseConfig


class Config(BaseConfig):
    """Launch a Hosted Training run from a config file."""

    config_path: str = Field(
        ..., description="Path to a TOML config file to launch as a Hosted Training run."
    )
    env: list[str] | None = Field(
        None,
        validation_alias=AliasChoices("env", "e", "env_var"),
        description="Environment value: KEY=VALUE, KEY from the shell, or an env file.",
    )
    env_file: list[str] | None = Field(
        None,
        description=".env file containing secrets. Supports ${VAR} expansion.",
    )
    output: str = Field(
        "table",
        validation_alias=AliasChoices("output", "o"),
        description="Output format: table or json",
    )
    skip_action_check: bool = Field(
        False, description="Skip action status check and run even if environment action failed."
    )
    yes: bool = Field(
        False, validation_alias=AliasChoices("yes", "y"), description="Skip confirmation prompt"
    )
    image_tag: str | None = Field(
        None,
        description="prime-rl image tag for full-FT runs.",
    )


POSITIONALS = ("config_path",)


def run(config: Config):
    from prime_cli.commands.rl import create_run as callback

    return callback(config)
