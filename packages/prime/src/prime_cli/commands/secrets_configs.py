"""Pydantic Config schemas for the ``prime secrets`` surface.

These were previously defined as one-file-per-command leaves under
``prime_cli.leaves``; they now live next to their command callbacks and are
consumed by the router via ``Command.config_attr``.
"""

from __future__ import annotations

from pydantic import AliasChoices, Field
from pydantic_config import BaseConfig


class SecretCreateConfig(BaseConfig):
    """Create a new global secret."""

    name: str | None = Field(
        None,
        validation_alias=AliasChoices("name", "n"),
        description="Secret name (used as environment variable name)",
    )
    value: str | None = Field(
        None, validation_alias=AliasChoices("value", "v"), description="Secret value"
    )
    description: str | None = Field(
        None, validation_alias=AliasChoices("description", "d"), description="Secret description"
    )
    file: bool = Field(
        False,
        validation_alias=AliasChoices("file", "f"),
        description="Treat value as file content (base64 encoded)",
    )
    output: str = Field(
        "table",
        validation_alias=AliasChoices("output", "o"),
        description="Output format: table or json",
    )


class SecretDeleteConfig(BaseConfig):
    """Delete a global secret."""

    secret_id: str | None = Field(
        None, description="Secret ID to delete (interactive selection if not provided)"
    )
    yes: bool = Field(
        False, validation_alias=AliasChoices("yes", "y"), description="Skip confirmation prompt"
    )


class SecretGetConfig(BaseConfig):
    """Get details of a specific secret."""

    secret_id: str = Field(..., description="Secret ID to get")
    output: str = Field(
        "table",
        validation_alias=AliasChoices("output", "o"),
        description="Output format: table or json",
    )


class SecretListConfig(BaseConfig):
    """List your global secrets."""

    output: str = Field(
        "table",
        validation_alias=AliasChoices("output", "o"),
        description="Output format: table or json",
    )


class SecretUpdateConfig(BaseConfig):
    """Update an existing global secret."""

    secret_id: str | None = Field(
        None, description="Secret ID to update (interactive selection if not provided)"
    )
    name: str | None = Field(
        None, validation_alias=AliasChoices("name", "n"), description="New secret name"
    )
    value: str | None = Field(
        None, validation_alias=AliasChoices("value", "v"), description="New secret value"
    )
    description: str | None = Field(
        None,
        validation_alias=AliasChoices("description", "d"),
        description="New secret description",
    )
    output: str = Field(
        "table",
        validation_alias=AliasChoices("output", "o"),
        description="Output format: table or json",
    )
