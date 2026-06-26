"""Pydantic leaf for ``prime tunnel start``."""

from __future__ import annotations

from pydantic import AliasChoices, Field
from pydantic_config import BaseConfig


class Config(BaseConfig):
    """Start a tunnel to expose a local port."""

    port: int = Field(
        8765, validation_alias=AliasChoices("port", "p"), description="Local port to tunnel"
    )
    name: str | None = Field(
        None, validation_alias=AliasChoices("name", "n"), description="Friendly name for the tunnel"
    )
    labels: list[str] | None = Field(
        None, validation_alias=AliasChoices("labels", "l"), description="Labels for the tunnel."
    )
    team_id: str | None = Field(
        None, description="Team ID for team tunnels (uses config team_id if not specified)"
    )
    auth: str | None = Field(
        None,
        description="Basic-auth username. A generated password is shown once.",
    )


POSITIONALS = ()


def run(config: Config):
    from prime_cli.commands.tunnel import start_tunnel as callback

    return callback(config)
