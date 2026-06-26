"""Pydantic leaf for ``prime tunnel stop``."""

from __future__ import annotations

from pydantic import AliasChoices, Field
from pydantic_config import BaseConfig


class Config(BaseConfig):
    """Stop and delete one or more tunnels."""

    tunnel_ids: list[str] | None = Field(
        None,
        description="Tunnel IDs to stop. Cannot be combined with filters.",
    )
    all: bool = Field(
        False,
        validation_alias=AliasChoices("all", "a"),
        description="Stop every tunnel in scope. May be narrowed with --status.",
    )
    labels: list[str] | None = Field(
        None,
        validation_alias=AliasChoices("labels", "l"),
        description="Stop tunnels carrying all labels. Conflicts with IDs and --all.",
    )
    status: str | None = Field(
        None,
        description="Filter by pending, connected, or disconnected status.",
    )
    team_id: str | None = Field(
        None,
        description="Team for filtered operations. Required with --all-users.",
    )
    yes: bool = Field(
        False, validation_alias=AliasChoices("yes", "y"), description="Skip confirmation prompt"
    )
    all_users: bool = Field(
        False, description="Target every team member's tunnels. Requires a team."
    )


POSITIONALS = ("tunnel_ids",)


def run(config: Config):
    from prime_cli.commands.tunnel import stop_tunnel as callback

    return callback(config)
