"""Pydantic Config schemas for the ``prime config`` surface.

These were previously defined as one-file-per-command leaves under
``prime_cli.leaves``; they now live next to their command callbacks and are
consumed by the router via ``Command.config_attr``.
"""

from __future__ import annotations

from pydantic import AliasChoices, Field
from pydantic_config import BaseConfig


class ConfigDeleteConfig(BaseConfig):
    """Delete a saved environment"""

    name: str = Field(..., description="Name of the saved environment")


class ConfigEnvsConfig(BaseConfig):
    """List available environments"""

    pass


class ConfigRemoveTeamIdConfig(BaseConfig):
    """Remove team ID to use personal account"""

    pass


class ConfigResetConfig(BaseConfig):
    """Reset configuration to defaults"""

    yes: bool = Field(
        False, validation_alias=AliasChoices("yes", "y"), description="Skip confirmation prompt"
    )


class ConfigSaveConfig(BaseConfig):
    """Save current config as environment (including API key)"""

    name: str = Field(..., description="Name for the environment")


class ConfigSetApiKeyConfig(BaseConfig):
    """Set your API key (prompts securely if not provided)"""

    api_key: str | None = Field(
        None,
        description="Your Prime Intellect API key. If not provided, you'll be prompted securely.",
    )


class ConfigSetBaseUrlConfig(BaseConfig):
    """Set the API base URL (prompts if not provided)"""

    url: str | None = Field(
        None,
        description="Base URL for the Prime Intellect API. If not provided, you'll be prompted.",
    )


class ConfigSetFrontendUrlConfig(BaseConfig):
    """Set the frontend URL (prompts if not provided)"""

    url: str | None = Field(
        None,
        description="Prime Intellect web app URL. Prompts when omitted.",
    )


class ConfigSetInferenceUrlConfig(BaseConfig):
    """Set the inference URL (prompts if not provided)"""

    url: str | None = Field(
        None,
        description="Inference URL for Prime Inference API. If not provided, you'll be prompted.",
    )


class ConfigSetShareResourcesWithTeamConfig(BaseConfig):
    """Set whether to automatically share new resources with all team members"""

    enabled: str = Field(..., description="Enable or disable auto-sharing with team: true or false")


class ConfigSetSshKeyPathConfig(BaseConfig):
    """Set the SSH private key path"""

    path: str = Field(..., description="Path to your SSH private key file")


class ConfigSetTeamIdConfig(BaseConfig):
    """Set your team ID."""

    team_id: str = Field(..., description="Your Prime Intellect team ID.")


class ConfigUseConfig(BaseConfig):
    """Switch to a different environment"""

    env: str = Field(
        ..., description="Environment name: 'production' or a custom saved environment"
    )


class ConfigViewConfig(BaseConfig):
    """View current configuration"""

    pass
