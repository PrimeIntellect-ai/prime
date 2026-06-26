"""Pydantic leaf for ``prime eval submit``."""

from __future__ import annotations

from pydantic import AliasChoices, Field
from pydantic_config import BaseConfig


class Config(BaseConfig):
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


POSITIONALS = ("environment",)


def run(config: Config):
    from prime_cli.commands.evals import submit_eval_cmd as callback

    return callback(config)
