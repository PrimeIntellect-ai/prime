"""Pydantic leaf for ``prime sandbox ssh``."""

from __future__ import annotations

from pydantic import AliasChoices, Field
from pydantic_config import BaseConfig


class Config(BaseConfig):
    """Connect to a sandbox via SSH."""

    sandbox_id: str | None = Field(
        None, description="Sandbox ID to SSH into (interactive selection if not provided)"
    )
    ssh_args: list[str] | None = Field(
        None, description="Additional SSH arguments (e.g., -- -v for verbose)"
    )
    shell: str | None = Field(
        None,
        validation_alias=AliasChoices("shell", "s"),
        description="Shell to use (e.g., bash, zsh, sh). Auto-detected if not specified.",
    )


POSITIONALS = ("sandbox_id", "ssh_args")


def run(config: Config):
    from prime_cli.commands.sandbox import ssh_connect as callback

    return callback(config)
