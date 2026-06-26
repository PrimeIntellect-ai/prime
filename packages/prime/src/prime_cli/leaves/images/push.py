"""Pydantic leaf for ``prime images push``."""

from __future__ import annotations

from pydantic import AliasChoices, Field
from pydantic_config import BaseConfig


class Config(BaseConfig):
    """Build and push a Docker image to Prime Intellect registry."""

    image_reference: str = Field(
        ..., description="Image reference (e.g., 'myapp:v1.0.0' or 'myapp:latest')"
    )
    context: str = Field(
        ".", validation_alias=AliasChoices("context", "c"), description="Build context directory"
    )
    dockerfile: str | None = Field(
        None, validation_alias=AliasChoices("dockerfile", "f"), description="Path to Dockerfile"
    )
    platform: str = Field(
        "linux/amd64",
        description="Target platform (defaults to linux/amd64 for Kubernetes compatibility)",
    )
    public: bool = Field(False, description="Make the image public when the build completes")
    private: bool = Field(False, description="Make the image private when the build completes")


POSITIONALS = ("image_reference",)


def run(config: Config):
    from prime_cli.commands.images import push_image as callback

    return callback(config)
