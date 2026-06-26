"""Pydantic leaf for ``prime deployments delete``."""

from __future__ import annotations

from pydantic import Field
from pydantic_config import BaseConfig


class Config(BaseConfig):
    """Unload a model from inference."""

    model_id: str = Field(description="Model ID to unload")


POSITIONALS = ("model_id",)


def run(config: Config):
    from prime_cli.commands.deployments import delete_deployment as callback

    return callback(config)
