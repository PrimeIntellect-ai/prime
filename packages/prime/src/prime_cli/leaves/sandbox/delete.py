"""Pydantic leaf for ``prime sandbox delete``."""

from __future__ import annotations

from pydantic import AliasChoices, Field
from pydantic_config import BaseConfig


class Config(BaseConfig):
    """Delete one or more sandboxes by ID, by label, or all sandboxes with --all"""

    sandbox_ids: list[str] | None = Field(
        None, description="Sandbox ID(s) to delete (space or comma-separated)"
    )
    all: bool = Field(
        False, validation_alias=AliasChoices("all", "a"), description="Delete all sandboxes"
    )
    labels: list[str] | None = Field(
        None,
        validation_alias=AliasChoices("labels", "l"),
        description="Delete sandboxes having all provided labels.",
    )
    yes: bool = Field(
        False, validation_alias=AliasChoices("yes", "y"), description="Skip confirmation prompt"
    )
    all_users: bool = Field(
        False,
        validation_alias=AliasChoices("all_users", "A"),
        description="Delete across every user in the team. Requires team admin role.",
    )
    target_user_id: str | None = Field(
        None,
        validation_alias=AliasChoices("target_user_id", "user", "u"),
        description="Target one teammate. Requires team admin; conflicts with --all-users.",
    )


POSITIONALS = ("sandbox_ids",)


def run(config: Config):
    from prime_cli.commands.sandbox import delete as callback

    return callback(config)
