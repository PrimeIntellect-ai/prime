"""Pydantic leaf for ``prime train components``."""

from __future__ import annotations

from pydantic import Field
from pydantic_config import BaseConfig


class Config(BaseConfig):
    """List pods (orchestrator + env-servers) for a run."""

    run_id: str = Field(..., description="Run ID to list components for")


POSITIONALS = ("run_id",)


def run(config: Config):
    from prime_cli.commands.rl import list_components as callback

    return callback(config)
