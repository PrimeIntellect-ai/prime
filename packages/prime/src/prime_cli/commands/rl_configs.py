"""Pydantic Config schemas for the ``prime rl`` surface.

These were previously defined as one-file-per-command leaves under
``prime_cli.leaves``; they now live next to their command callbacks and are
consumed by the router via ``Command.config_attr``.
"""

from __future__ import annotations

from pydantic import AliasChoices, Field
from pydantic_config import BaseConfig


class TrainCheckpointsConfig(BaseConfig):
    """List checkpoints for a run."""

    run_id: str = Field(..., description="Run ID to list checkpoints for")
    status_filter: str | None = Field(
        None,
        validation_alias=AliasChoices("status_filter", "status", "s"),
        description="Filter by status (READY, PENDING, UPLOADING, FAILED)",
    )
    output: str = Field(
        "table",
        validation_alias=AliasChoices("output", "o"),
        description="Output format: table or json",
    )


class TrainComponentsConfig(BaseConfig):
    """List pods (orchestrator + env-servers) for a run."""

    run_id: str = Field(..., description="Run ID to list components for")


class TrainConfigsConfig(BaseConfig):
    """List available configuration options for Hosted Training."""

    output: str = Field(
        "table",
        validation_alias=AliasChoices("output", "o"),
        description="Output format: table or json",
    )


class TrainDeleteConfig(BaseConfig):
    """Delete a run."""

    run_id: str = Field(..., description="Run ID to delete")
    force: bool = Field(
        False, validation_alias=AliasChoices("force", "f"), description="Skip confirmation"
    )


class TrainDistributionsConfig(BaseConfig):
    """Get reward/advantage distribution histogram for a run."""

    run_id: str = Field(..., description="Run ID to get distributions for")
    distribution_type: str | None = Field(
        None,
        validation_alias=AliasChoices("distribution_type", "type", "t"),
        description="Distribution type (defaults to all)",
    )
    step: int | None = Field(
        None,
        validation_alias=AliasChoices("step", "s"),
        description="Step number (defaults to latest)",
    )


class TrainGetConfig(BaseConfig):
    """Get details of a specific run."""

    run_id: str = Field(..., description="Run ID to get details for")
    output: str = Field(
        "table",
        validation_alias=AliasChoices("output", "o"),
        description="Output format: table or json",
    )


class TrainInitConfig(BaseConfig):
    """Generate a template config file for a Hosted Training run."""

    output_path: str = Field("rl.toml", description="Output path for the config file")
    force: bool = Field(
        False, validation_alias=AliasChoices("force", "f"), description="Overwrite existing file"
    )


class TrainListConfig(BaseConfig):
    """List your runs."""

    team: str | None = Field(
        None, validation_alias=AliasChoices("team", "t"), description="Filter by team ID"
    )
    mine: bool = Field(
        False, description="Filter to only your own runs (useful for admin accounts)"
    )
    num: int = Field(20, validation_alias=AliasChoices("num", "n"), description="Items per page")
    page: int = Field(1, validation_alias=AliasChoices("page", "p"), description="Page number")
    output: str = Field(
        "table",
        validation_alias=AliasChoices("output", "o"),
        description="Output format: table or json",
    )


class TrainLogsConfig(BaseConfig):
    """Get logs for a run."""

    run_id: str = Field(..., description="Run ID to get logs for")
    component: str | None = Field(
        None,
        validation_alias=AliasChoices("component", "c"),
        description="Pod component: orchestrator, trainer, inference, or env-server.",
    )
    env: str | None = Field(
        None,
        description="Env-server name or name/N. Implies --component=env-server.",
    )
    tail: int = Field(
        1000, validation_alias=AliasChoices("tail", "n"), description="Number of lines to show"
    )
    follow: bool = Field(
        False, validation_alias=AliasChoices("follow", "f"), description="Follow log output"
    )
    raw: bool = Field(
        False,
        validation_alias=AliasChoices("raw", "r"),
        description="Show raw logs without formatting",
    )
    search: str | None = Field(
        None,
        description="Filter lines by substring, or by regex with --regex.",
    )
    regex: bool = Field(False, description="Treat --search as a regex (RE2 syntax).")
    level: str | None = Field(
        None, description="Filter to one log level: ERROR | WARNING | SUCCESS | INFO | DEBUG."
    )
    since: str | None = Field(
        None,
        description="Filtered query window, such as 15m, 1h, 24h, or seconds.",
    )


class TrainMetricsConfig(BaseConfig):
    """Get Hosted Training metrics for a run."""

    run_id: str = Field(..., description="Run ID to get metrics for")
    min_step: int | None = Field(None, description="Minimum step (inclusive)")
    max_step: int | None = Field(None, description="Maximum step (inclusive)")
    limit: int | None = Field(
        None, validation_alias=AliasChoices("limit", "n"), description="Maximum number of records"
    )


class TrainModelsConfig(BaseConfig):
    """List available models for Hosted Training."""

    output: str = Field(
        "table",
        validation_alias=AliasChoices("output", "o"),
        description="Output format: table or json",
    )


class TrainProgressConfig(BaseConfig):
    """Get progress information, including which steps have samples and distributions."""

    run_id: str = Field(..., description="Run ID to get progress for")


class TrainRequestConfig(BaseConfig):
    """Request models for Hosted Training."""

    pass


class TrainRestartConfig(BaseConfig):
    """Restart a running run from its latest checkpoint."""

    run_id: str = Field(..., description="Run ID to restart")
    force: bool = Field(
        False, validation_alias=AliasChoices("force", "f"), description="Skip confirmation"
    )


class TrainRolloutsConfig(BaseConfig):
    """Get rollout samples for a run."""

    run_id: str = Field(..., description="Run ID to get rollouts for")
    step: int = Field(
        ...,
        validation_alias=AliasChoices("step", "s"),
        description="Step number to get rollouts for",
    )
    page: int = Field(
        1, validation_alias=AliasChoices("page", "p"), description="Page number (1-indexed)"
    )
    num: int = Field(100, validation_alias=AliasChoices("num", "n"), description="Items per page")


class TrainRunConfig(BaseConfig):
    """Launch a Hosted Training run from a config file."""

    config_path: str = Field(
        ..., description="Path to a TOML config file to launch as a Hosted Training run."
    )
    env: list[str] | None = Field(
        None,
        validation_alias=AliasChoices("env", "e", "env_var"),
        description="Environment value: KEY=VALUE, KEY from the shell, or an env file.",
    )
    env_file: list[str] | None = Field(
        None,
        description=".env file containing secrets. Supports ${VAR} expansion.",
    )
    output: str = Field(
        "table",
        validation_alias=AliasChoices("output", "o"),
        description="Output format: table or json",
    )
    skip_action_check: bool = Field(
        False, description="Skip action status check and run even if environment action failed."
    )
    yes: bool = Field(
        False, validation_alias=AliasChoices("yes", "y"), description="Skip confirmation prompt"
    )
    image_tag: str | None = Field(
        None,
        description="prime-rl image tag for full-FT runs.",
    )


class TrainStopConfig(BaseConfig):
    """Stop a run."""

    run_id: str = Field(..., description="Run ID to stop")
    force: bool = Field(
        False, validation_alias=AliasChoices("force", "f"), description="Skip confirmation"
    )
