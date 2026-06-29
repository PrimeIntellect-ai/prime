"""Pydantic Config schemas for the ``prime images`` surface.

These were previously defined as one-file-per-command leaves under
``prime_cli.leaves``; they now live next to their command callbacks and are
consumed by the router via ``Command.config_attr``.
"""

from __future__ import annotations

from pydantic import AliasChoices, Field
from pydantic_config import BaseConfig


class ImagesDeleteConfig(BaseConfig):
    """Delete an image from your registry."""

    image_reference: str = Field(
        ...,
        description="Personal or team image reference to delete.",
    )
    yes: bool = Field(
        False, validation_alias=AliasChoices("yes", "y"), description="Skip confirmation prompt"
    )


class ImagesListConfig(BaseConfig):
    """List all images you've pushed to Prime Intellect registry."""

    output: str = Field(
        "table",
        validation_alias=AliasChoices("output", "o"),
        description="Output format (table or json)",
    )
    search: str | None = Field(
        None,
        validation_alias=AliasChoices("search", "q"),
        description="Case-insensitive substring match on image name, tag, or reference",
    )
    page: int = Field(1, validation_alias=AliasChoices("page", "p"), description="Page number")
    num: int = Field(
        50, validation_alias=AliasChoices("num", "n"), description="Items per page (max 250)"
    )


class ImagesPublishConfig(BaseConfig):
    """Make an image public so other Prime users can run it."""

    image_reference: str = Field(
        ...,
        description="Personal or team image reference to make public.",
    )


class ImagesPushConfig(BaseConfig):
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


class ImagesUnpublishConfig(BaseConfig):
    """Make a public image private again."""

    image_reference: str = Field(
        ...,
        description="Personal or team image reference to make private.",
    )
