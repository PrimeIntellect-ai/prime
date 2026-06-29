"""Pydantic Config schemas for the ``prime evals`` surface.

These were previously defined as one-file-per-command leaves under
``prime_cli.leaves``; they now live next to their command callbacks and are
consumed by the router via ``Command.config_attr``.
"""

from __future__ import annotations

from pydantic import AliasChoices, Field
from pydantic_config import BaseConfig


class EvalGetConfig(BaseConfig):
    """Show evaluation details."""

    eval_id: str = Field(..., description="The ID of the evaluation to retrieve")
    output: str = Field(
        "json", validation_alias=AliasChoices("output", "o"), description="json|pretty"
    )


class EvalListConfig(BaseConfig):
    """List evaluations."""

    output: str = Field(
        "table", validation_alias=AliasChoices("output", "o"), description="table|json"
    )
    num: int = Field(20, validation_alias=AliasChoices("num", "n"), description="Items per page")
    page: int = Field(1, validation_alias=AliasChoices("page", "p"), description="Page number")
    env: str | None = Field(
        None,
        validation_alias=AliasChoices("env", "env_name", "e"),
        description="Filter by environment (e.g., 'gsm8k' or 'owner/gsm8k')",
    )


class EvalLogsConfig(BaseConfig):
    """Get logs for a hosted evaluation."""

    eval_id: str = Field(..., description="Evaluation id to get logs for")
    tail: int = Field(
        1000, validation_alias=AliasChoices("tail", "n"), description="Number of lines to show"
    )
    follow: bool = Field(
        False, validation_alias=AliasChoices("follow", "f"), description="Follow log output"
    )
    poll_interval: float = Field(5.0, description="Polling interval in seconds when following logs")


class EvalPushConfig(BaseConfig):
    """Push native or legacy evaluation data to Prime Evals."""

    config_path: str | None = Field(
        None,
        description="Native V1 or legacy evaluation run directory. Auto-discovers when omitted.",
    )
    env_id: str | None = Field(
        None,
        validation_alias=AliasChoices("env_id", "env", "e"),
        description="Published environment slug (owner/name).",
    )
    run_id: str | None = Field(
        None,
        validation_alias=AliasChoices("run_id", "r"),
        description="Link to existing training run id",
    )
    eval_id: str | None = Field(
        None,
        validation_alias=AliasChoices("eval_id", "eval"),
        description="Push to existing evaluation id",
    )
    name: str | None = Field(None, description="Explicit evaluation name override")
    output: str = Field(
        "pretty", validation_alias=AliasChoices("output", "o"), description="json|pretty"
    )
    is_public: bool = Field(
        False,
        validation_alias=AliasChoices("is_public", "public"),
        description="Make the pushed evaluation public. Evaluations are private by default.",
    )


class EvalSamplesConfig(BaseConfig):
    """"""

    eval_id: str = Field(..., description="The ID of the evaluation")
    page: int = Field(1, validation_alias=AliasChoices("page", "p"), description="Page number")
    num: int = Field(100, validation_alias=AliasChoices("num", "n"), description="Items per page")
    output: str = Field(
        "json", validation_alias=AliasChoices("output", "o"), description="json|pretty"
    )


class EvalStopConfig(BaseConfig):
    """Stop a running hosted evaluation."""

    eval_id: str = Field(..., description="Evaluation id to stop")


class EvalSubmitConfig(BaseConfig):
    """Submit a hosted V0 evaluation"""

    environment: str | None = Field(
        None, description="Environment name/slug or V0 TOML config path"
    )
    env_path: str | None = Field(
        None,
        description="Environment directory used for upstream resolution.",
    )
    poll_interval: float = Field(
        10.0, description="Polling interval in seconds for hosted evaluation status"
    )
    follow: bool = Field(
        False, description="Follow hosted evaluation status and stream logs until completion"
    )
    model: str | None = Field(
        None, validation_alias=AliasChoices("model", "m"), description="Inference model"
    )
    num_examples: int | None = Field(
        None,
        validation_alias=AliasChoices("num_examples", "n"),
        description="Examples per environment",
    )
    rollouts_per_example: int | None = Field(
        None,
        validation_alias=AliasChoices("rollouts_per_example", "r"),
        description="Rollouts per example",
    )
    env_args: str | None = Field(None, description="V0 load_environment arguments as JSON")
    extra_env_kwargs: str | None = Field(
        None, description="V0 post-load environment arguments as JSON"
    )
    timeout_minutes: int | None = Field(
        None, description="Timeout in minutes for hosted evaluation"
    )
    allow_sandbox_access: bool | None = Field(
        None, description="Allow sandbox read/write access for hosted evaluations"
    )
    allow_instances_access: bool | None = Field(
        None, description="Allow instance creation and management for hosted evaluations"
    )
    allow_tunnel_access: bool | None = Field(
        None, description="Allow tunnel creation and management for hosted evaluations"
    )
    custom_secrets: str | None = Field(
        None, description='Custom secrets for hosted eval as JSON (e.g. \'{"API_KEY":"xxx"}\')'
    )
    sampling_args: str | None = Field(
        None,
        description="Sampling arguments as JSON.",
    )
    eval_name: str | None = Field(None, description="Custom name for the hosted evaluation")
    max_concurrent: int | None = Field(None, description="Maximum concurrent rollouts")
    max_retries: int | None = Field(None, description="Retries per rollout")
    state_columns: list[str] | None = Field(
        None,
        description="State columns to retain.",
    )
    independent_scoring: bool | None = Field(None, description="Score rollouts independently")
    verbose: bool | None = Field(None, description="Enable verbose evaluator logs")
    header: list[str] | None = Field(
        None, description="Extra HTTP header as 'Name: Value'; repeat as needed"
    )
    api_client_type: str | None = Field(None, description="V0 model client type")
    api_base_url: str | None = Field(None, description="V0 model API base URL")
    api_key_var: str | None = Field(
        None, description="Environment variable containing the model API key"
    )


class EvalViewConfig(BaseConfig):
    """Launch the interactive evaluation viewer."""

    limit: int = Field(
        50, validation_alias=AliasChoices("limit", "n"), description="Max evaluation rows to load"
    )
    env_dir: str | None = Field(
        None,
        validation_alias=AliasChoices("env_dir", "e"),
        description="Path to environments directory",
    )
    outputs_dir: str | None = Field(
        None,
        validation_alias=AliasChoices("outputs_dir", "o"),
        description="Path to outputs directory",
    )
