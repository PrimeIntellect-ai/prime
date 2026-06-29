"""Pydantic Config schemas for the ``prime tunnel`` surface.

These were previously defined as one-file-per-command leaves under
``prime_cli.leaves``; they now live next to their command callbacks and are
consumed by the router via ``Command.config_attr``.
"""

from __future__ import annotations

from pydantic import AliasChoices, Field
from pydantic_config import BaseConfig


class TunnelListConfig(BaseConfig):
    """List active tunnels."""

    team_id: str | None = Field(
        None, description="Team ID to list team tunnels (uses config team_id if not specified)"
    )
    labels: list[str] | None = Field(
        None, validation_alias=AliasChoices("labels", "l"), description="Filter by labels."
    )
    status: str | None = Field(None, description="Filter by status")
    sort_by: str = Field(
        "createdAt", description="Sort field: createdAt, status, name, expiresAt, connectedAt"
    )
    sort_order: str = Field("desc", description="Sort order: asc or desc")
    page: int = Field(1, validation_alias=AliasChoices("page", "p"), description="Page number")
    num: int = Field(50, validation_alias=AliasChoices("num", "n"), description="Items per page")
    output: str = Field(
        "table",
        validation_alias=AliasChoices("output", "o"),
        description="Output format: table or json",
    )


class TunnelStartConfig(BaseConfig):
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


class TunnelStatusConfig(BaseConfig):
    """Get status of a specific tunnel."""

    tunnel_id: str = Field(..., description="Tunnel ID to check")


class TunnelStopConfig(BaseConfig):
    """Stop and delete one or more tunnels."""

    tunnel_ids: list[str] | None = Field(
        None,
        description="Tunnel IDs to stop. Cannot be combined with filters.",
    )
    all: bool = Field(
        False,
        validation_alias=AliasChoices("all", "a"),
        description="Stop every tunnel in scope. May be narrowed with --status.",
    )
    labels: list[str] | None = Field(
        None,
        validation_alias=AliasChoices("labels", "l"),
        description="Stop tunnels carrying all labels. Conflicts with IDs and --all.",
    )
    status: str | None = Field(
        None,
        description="Filter by pending, connected, or disconnected status.",
    )
    team_id: str | None = Field(
        None,
        description="Team for filtered operations. Required with --all-users.",
    )
    yes: bool = Field(
        False, validation_alias=AliasChoices("yes", "y"), description="Skip confirmation prompt"
    )
    all_users: bool = Field(
        False, description="Target every team member's tunnels. Requires a team."
    )
