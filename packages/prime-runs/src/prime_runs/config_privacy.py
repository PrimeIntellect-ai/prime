"""The config summary prime-runs may upload.

This is a summary, not a replayable config. Unknown fields are omitted, including
source files, URLs, paths, commands, task data, headers and harness environment
values. A credential-name denylist cannot make arbitrary configuration public.
Keep the evaluation wire policy aligned with platform's config_privacy module and the
dashboard's evaluationConfig projection; none trusts another to sanitize input.
"""

import math
import re
from datetime import datetime
from typing import Any

NUMBER_FIELDS = (
    "num_examples",
    "rollouts_per_example",
    "num_tasks",
    "num_rollouts",
    "max_concurrent",
    "max_retries",
    "timeout_minutes",
    "max_tokens",
    "temperature",
    "top_k",
    "top_p",
    "frequency_penalty",
    "presence_penalty",
    "repetition_penalty",
    "seed",
)
BOOL_FIELDS = (
    "auto_max_concurrent",
    "independent_scoring",
    "verbose",
    "shuffle",
    "push",
    "allow_sandbox_access",
    "allow_instances_access",
    "allow_tunnel_access",
    "use_platform_billing",
)
SAMPLING_FIELDS = (
    "max_tokens",
    "temperature",
    "top_k",
    "top_p",
    "frequency_penalty",
    "presence_penalty",
    "repetition_penalty",
    "seed",
    "min_p",
)


def _numbers(source: dict, fields: tuple[str, ...]) -> dict[str, Any]:
    return {
        key: value
        for key in fields
        if type(value := source.get(key)) in (int, float)
        and -(2**53) < value < 2**53
        and math.isfinite(value)
    }


def config_summary(value: Any) -> dict[str, Any]:
    """Copy only supported, correctly typed controls. Never parse raw files."""
    if not isinstance(value, dict):
        return {}
    result = _numbers(value, NUMBER_FIELDS)
    result.update({key: value[key] for key in BOOL_FIELDS if type(value.get(key)) is bool})
    for key in ("sampling", "sampling_args"):
        sampling = value.get(key)
        if isinstance(sampling, dict):
            projected = _numbers(sampling, SAMPLING_FIELDS)
            effort = sampling.get("reasoning_effort")
            if isinstance(effort, str) and effort in (
                "none",
                "minimal",
                "low",
                "medium",
                "high",
                "xhigh",
            ):
                projected["reasoning_effort"] = effort
            if projected:
                result[key] = projected
    client = value.get("client")
    if isinstance(client, dict) and client.get("type") in ("eval", "train"):
        result["client"] = {"type": client["type"]}
    serve = value.get("serve")
    if isinstance(serve, dict):
        projected = _numbers(serve, ("max_concurrent",))
        pool = serve.get("pool")
        if isinstance(pool, dict):
            projected_pool = _numbers(pool, ("num_workers", "max_workers", "multiplex"))
            if pool.get("type") in ("static", "elastic"):
                projected_pool["type"] = pool["type"]
            if projected_pool:
                projected["pool"] = projected_pool
        if projected:
            result["serve"] = projected
    return result


def metadata_summary(value: Any) -> dict[str, Any]:
    """Config controls plus the typed metadata used by statistics and columns.

    Run identity belongs in the API's name/model/environment fields, not arbitrary
    metadata strings. Column names identify published sample fields; their values
    are never copied here. Producer errors may contain credentials and are omitted.
    """
    result = config_summary(value)
    if not isinstance(value, dict):
        return result
    result.update(_numbers(value, ("avg_reward", "avg_score")))
    columns = value.get("state_columns")
    if isinstance(columns, list):
        result["state_columns"] = [
            column
            for column in columns[:128]
            if isinstance(column, str) and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,127}", column)
        ]
    terminal = value.get("prime_runs")
    if isinstance(terminal, dict):
        status = terminal.get("status")
        if status in ("running", "completed", "failed", "cancelled", "crashed"):
            result["prime_runs"] = {"status": status}
            finished_at = terminal.get("finished_at")
            if isinstance(finished_at, str):
                try:
                    result["prime_runs"]["finished_at"] = datetime.fromisoformat(
                        finished_at.replace("Z", "+00:00")
                    ).isoformat()
                except ValueError:
                    pass
    return result


def training_config_summary(value: Any) -> dict[str, Any]:
    """External training metadata is also a summary, never a full config file."""
    result = config_summary(value)
    if not isinstance(value, dict):
        return result
    fields = (
        "max_steps",
        "batch_size",
        "seq_len",
        "lr",
        "learning_rate",
        "weight_decay",
        "warmup_steps",
    )
    result.update(_numbers(value, fields))
    trainer = value.get("trainer")
    if isinstance(trainer, dict):
        projected = _numbers(trainer, fields)
        if projected:
            result["trainer"] = projected
    return result
