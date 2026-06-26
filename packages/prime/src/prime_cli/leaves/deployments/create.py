"""Pydantic leaf for ``prime deployments create``."""

from __future__ import annotations

from pydantic import AliasChoices, Field
from pydantic_config import BaseConfig


class Config(BaseConfig):
    """Deploy a model for inference."""

    model_id: str = Field(description="Model ID to deploy")
    yes: bool = Field(
        False, validation_alias=AliasChoices("yes", "y"), description="Skip confirmation prompt"
    )


POSITIONALS = ("model_id",)


def run(config: Config):
    from prime_cli.commands.deployments import create_deployment as callback

    return callback(config)
