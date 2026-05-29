"""Shared Lab config templates and TOML rendering."""

from __future__ import annotations

from typing import Any

import toml

from .toml_format import format_toml_blocks

DEFAULT_ENV_ID = "primeintellect/gsm8k"


def format_lab_config(config: dict[str, Any]) -> str:
    """Render a Lab config with the user-facing TOML formatter."""

    return format_toml_blocks(toml.dumps(config)).rstrip()


def evaluation_config(
    *,
    env_id: str = DEFAULT_ENV_ID,
    model: str = "",
    version: str = "",
    num_examples: int | None = None,
    rollouts_per_example: int = 1,
    max_tokens: int | None = 512,
    max_concurrent: int | None = None,
    save_results: bool = True,
) -> dict[str, Any]:
    eval_config: dict[str, Any] = {
        "env_id": env_id,
        "rollouts_per_example": rollouts_per_example,
    }
    if version:
        eval_config["version"] = version
    if num_examples is not None:
        eval_config["num_examples"] = num_examples
    if max_tokens is not None:
        eval_config["sampling_args"] = {"max_tokens": max_tokens}
    config: dict[str, Any] = {
        "model": model,
        "save_results": save_results,
        "eval": [eval_config],
    }
    if max_concurrent is not None:
        config["max_concurrent"] = max_concurrent
    return filter_empty_config_values(config)


def rl_config(
    *,
    env_id: str = DEFAULT_ENV_ID,
    model: str = "",
    version: str = "",
    max_steps: int = 100,
    batch_size: int = 256,
    rollouts_per_example: int = 8,
    max_tokens: int | None = 512,
) -> dict[str, Any]:
    env: dict[str, Any] = {"id": env_id}
    if version:
        env["version"] = version
    config: dict[str, Any] = {
        "model": model,
        "max_steps": max_steps,
        "batch_size": batch_size,
        "rollouts_per_example": rollouts_per_example,
        "env": [env],
    }
    if max_tokens is not None:
        config["sampling"] = {"max_tokens": max_tokens}
    return filter_empty_config_values(config)


def filter_empty_config_values(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): child
            for key, child in (
                (key, filter_empty_config_values(child)) for key, child in value.items()
            )
            if child is not None and child != {} and child != []
        }
    if isinstance(value, list):
        return [
            child
            for child in (filter_empty_config_values(child) for child in value)
            if child is not None and child != {} and child != []
        ]
    return value
