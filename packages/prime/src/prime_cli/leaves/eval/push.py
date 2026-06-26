"""Pydantic leaf for ``prime eval push``."""

from __future__ import annotations

from pydantic import AliasChoices, Field
from pydantic_config import BaseConfig


class Config(BaseConfig):
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


POSITIONALS = ("config_path",)


def run(config: Config):
    from prime_cli.commands.evals import push_eval as callback

    return callback(config)
