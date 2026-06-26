"""Pydantic leaf for ``prime sandbox expose``."""

from __future__ import annotations

from pydantic import AliasChoices, Field
from pydantic_config import BaseConfig


class Config(BaseConfig):
    """Expose a port from a sandbox."""

    sandbox_id: str = Field(..., description="Sandbox ID to expose port from")
    port: int = Field(..., description="Port number to expose")
    name: str | None = Field(None, description="Optional name for the exposed port")
    protocol: str = Field(
        "HTTP", validation_alias=AliasChoices("protocol", "p"), description="Protocol: HTTP or TCP"
    )
    output: str = Field(
        "table",
        validation_alias=AliasChoices("output", "o"),
        description="Output format: table or json",
    )


POSITIONALS = ("sandbox_id", "port")


def run(config: Config):
    from prime_cli.commands.sandbox import expose_port as callback

    return callback(config)
