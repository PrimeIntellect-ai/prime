"""Pydantic leaf for ``prime lab mcp``."""

from __future__ import annotations

import pathlib

from pydantic import Field
from pydantic_config import BaseConfig


class Config(BaseConfig):
    """Run the Lab MCP server over stdio."""

    workspace: pathlib.Path | None = Field(
        None, description="Workspace whose running Lab TUI should receive MCP tool calls."
    )


POSITIONALS = ()


def run(config: Config):
    from prime_cli.commands.lab import mcp as callback

    return callback(config)
