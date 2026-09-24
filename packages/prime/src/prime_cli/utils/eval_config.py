"""Command-line and TOML parsing for `prime eval run --hosted`.

The hosted-evaluation API takes the flags and TOML fields of the verifiers 0.2.0
eval CLI. This module keeps that exact surface without importing verifiers: the
parser declares only the flags a hosted evaluation accepts, and the TOML loader
follows verifiers 0.2.0's schema (global defaults, `[[eval]]`, `[[ablation]]`).
"""

from __future__ import annotations

import argparse
import itertools
import json
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "openai/gpt-4.1-mini"
DEFAULT_ENV_DIR_PATH = "./environments"
DEFAULT_ENDPOINTS_PATH = "./configs/endpoints.toml"
DEFAULT_MAX_CONCURRENT = 32
DEFAULT_MAX_RETRIES = 3

API_CLIENT_TYPES = (
    "openai_completions",
    "openai_chat_completions",
    "openai_chat_completions_token",
    "openai_responses",
    "renderer",
    "anthropic_messages",
    "nemorl_chat_completions",
)

CHAT_TEMPLATE_KWARG_FIELDS = ("reasoning_effort", "enable_thinking")
FREEFORM_ABLATION_SWEEP_FIELDS = {"args", "env_args"}

# TOML fields verifiers 0.2.0 accepts; the hosted CLI rejects the ones it cannot send.
TOML_VALID_FIELDS = {
    "env_id",
    "name",
    "args",
    "env_args",
    "taskset",
    "harness",
    "env_dir_path",
    "endpoints_path",
    "extra_env_kwargs",
    "provider",
    "endpoint_id",
    "model",
    "api_client_type",
    "api_key_var",
    "api_base_url",
    "header",
    "headers",
    "header_from_state",
    "headers_from_state",
    "sampling",
    "sampling_args",
    "max_tokens",
    "temperature",
    "num_examples",
    "rollouts_per_example",
    "shuffle",
    "shuffle_seed",
    "max_concurrent",
    "independent_scoring",
    "max_retries",
    "num_workers",
    "disable_env_server",
    "timeout",
    "verbose",
    "disable_tui",
    "output_dir",
    "state_columns",
    "save_results",
    "resume",
    "resume_path",
    "save_to_hf_hub",
    "hf_hub_dataset_name",
}


def build_hosted_eval_parser() -> argparse.ArgumentParser:
    """The verifiers 0.2.0 eval flags that a hosted evaluation supports."""
    parser = argparse.ArgumentParser(prog="prime eval run", allow_abbrev=False, add_help=False)
    parser.add_argument("env_id_or_config", type=str)
    parser.add_argument("--env-args", "-a", type=json.loads, default={})
    parser.add_argument("--env-dir-path", type=str, default=DEFAULT_ENV_DIR_PATH)
    parser.add_argument("--model", "-m", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--api-client-type", type=str, default=None, choices=API_CLIENT_TYPES)
    parser.add_argument("--api-key-var", "-k", type=str, default=None)
    parser.add_argument("--api-base-url", "-b", type=str, default=None)
    parser.add_argument("--header", action="append", default=None)
    parser.add_argument("--num-examples", "-n", type=int, default=None)
    parser.add_argument("--rollouts-per-example", "-r", type=int, default=None)
    parser.add_argument("--max-concurrent", "-c", type=int, default=DEFAULT_MAX_CONCURRENT)
    parser.add_argument("--max-tokens", "-t", type=int, default=None)
    parser.add_argument("--temperature", "-T", type=float, default=None)
    parser.add_argument("--sampling-args", "-S", type=json.loads, default=None)
    parser.add_argument("--verbose", "-v", default=False, action="store_true")
    parser.add_argument(
        "--state-columns",
        "-C",
        type=lambda raw: [column.strip() for column in raw.split(",")],
        default=[],
    )
    parser.add_argument("--independent-scoring", "-i", default=False, action="store_true")
    parser.add_argument("--extra-env-kwargs", "-x", type=json.loads, default={})
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    return parser


def merge_sampling_args(
    sampling_args: dict[str, Any] | None,
    *,
    max_tokens: int | None = None,
    temperature: float | None = None,
    prefer_existing_keys: bool = True,
) -> dict[str, Any]:
    merged = dict(sampling_args or {})
    if max_tokens is not None and (not prefer_existing_keys or "max_tokens" not in merged):
        merged["max_tokens"] = max_tokens
    if temperature is not None and (not prefer_existing_keys or "temperature" not in merged):
        merged["temperature"] = temperature
    return merged


def _validate_headers_table(value: object) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError("extra_headers must be a dict")
    headers: dict[str, str] = {}
    for key, header_value in value.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("extra_headers keys must be non-empty strings")
        if not isinstance(header_value, str):
            raise ValueError("extra_headers values must be strings")
        headers[key] = header_value
    return headers


def build_extra_headers(raw: Mapping[str, Any]) -> dict[str, str]:
    """Merge a `headers` table with repeated `header = "Name: Value"` entries."""
    table: dict[str, str] = {}
    if raw.get("headers") is not None:
        table = _validate_headers_table(raw["headers"])

    header_values = raw.get("header")
    if header_values is None:
        header_values = []
    if not isinstance(header_values, list):
        raise ValueError("'header' must be a list of 'Name: Value' strings")

    from_list: dict[str, str] = {}
    for header_value in header_values:
        if not isinstance(header_value, str):
            raise ValueError(
                f"Each 'header' entry must be a string 'Name: Value', got: {header_value!r}"
            )
        if ":" not in header_value:
            raise ValueError(f"--header must be 'Name: Value', got: {header_value!r}")
        name, value = (part.strip() for part in header_value.split(":", 1))
        if not name:
            raise ValueError("--header name cannot be empty")
        from_list[name] = value
    return {**table, **from_list}


def _config_table(value: object, field: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a table.")
    return dict(value)


def _normalize_env_config_sections(raw: Mapping[str, Any]) -> dict[str, Any]:
    config = dict(raw)
    env_args = _config_table(config.pop("env_args", {}), "env_args")
    args = _config_table(config.pop("args", {}), "args")
    overlap = set(env_args) & set(args)
    if overlap:
        raise ValueError(f"Environment arg key(s) {overlap} appear in both args and env_args.")
    env_args = {**env_args, **args}

    child_config: dict[str, Any] = {}
    for section in ("taskset", "harness"):
        value = config.pop(section, None)
        if value is not None:
            child_config[section] = _config_table(value, section)
    if child_config:
        existing = _config_table(env_args.get("config", {}), "env_args.config")
        overlap = set(existing) & set(child_config)
        if overlap:
            raise ValueError(
                f"Environment config section(s) {overlap} appear in both config and "
                "top-level sections."
            )
        env_args["config"] = {**existing, **child_config}

    if env_args:
        config["env_args"] = env_args
    return config


def _normalize_env_id_alias(config: Mapping[str, Any], section: str) -> dict[str, Any]:
    normalized = dict(config)
    if "id" in normalized and "env_id" in normalized:
        raise ValueError(f"{section} cannot contain both id and env_id.")
    if "id" in normalized:
        normalized["env_id"] = normalized.pop("id")
    return normalized


def _eval_env_id(config: Mapping[str, Any], section: str) -> str:
    env_id = config.get("env_id")
    if isinstance(env_id, str) and env_id:
        return env_id
    taskset = config.get("taskset")
    if not isinstance(taskset, Mapping):
        env_args = config.get("env_args")
        if isinstance(env_args, Mapping):
            env_config = env_args.get("config")
            if isinstance(env_config, Mapping):
                taskset = env_config.get("taskset")
    if isinstance(taskset, Mapping):
        taskset_id = taskset.get("id") or taskset.get("taskset_id")
        if isinstance(taskset_id, str) and taskset_id:
            return taskset_id
    raise ValueError(f"{section} must contain env_id or taskset.id.")


def _merge_sampling_tables(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = {**dict(base), **dict(override)}
    base_extra_body = base.get("extra_body")
    override_extra_body = override.get("extra_body")
    if isinstance(base_extra_body, Mapping) and isinstance(override_extra_body, Mapping):
        extra_body = {**dict(base_extra_body), **dict(override_extra_body)}
        base_kwargs = base_extra_body.get("chat_template_kwargs")
        override_kwargs = override_extra_body.get("chat_template_kwargs")
        if isinstance(base_kwargs, Mapping) and isinstance(override_kwargs, Mapping):
            extra_body["chat_template_kwargs"] = {**dict(base_kwargs), **dict(override_kwargs)}
        merged["extra_body"] = extra_body
    return merged


def _normalize_sampling_args(sampling_args: Mapping[str, Any], section: str) -> dict[str, Any]:
    """Move `reasoning_effort` / `enable_thinking` into `extra_body.chat_template_kwargs`."""
    normalized = dict(sampling_args)
    chat_template_kwargs = {
        key: normalized[key] for key in CHAT_TEMPLATE_KWARG_FIELDS if key in normalized
    }
    if not chat_template_kwargs:
        return normalized

    raw_extra_body = normalized.get("extra_body") or {}
    if not isinstance(raw_extra_body, Mapping):
        raise ValueError(f"{section}.extra_body must be a table.")
    extra_body = dict(raw_extra_body)
    raw_kwargs = extra_body.get("chat_template_kwargs") or {}
    if not isinstance(raw_kwargs, Mapping):
        raise ValueError(f"{section}.extra_body.chat_template_kwargs must be a table.")
    extra_body["chat_template_kwargs"] = {**dict(raw_kwargs), **chat_template_kwargs}
    normalized["extra_body"] = extra_body
    return normalized


def _normalize_sampling_config(
    config: Mapping[str, Any], section: str, *, merge_with_existing: bool = False
) -> dict[str, Any]:
    """Fold a `[sampling]` table into `sampling_args`."""
    normalized = dict(config)
    if "sampling_args" in normalized:
        if not isinstance(normalized["sampling_args"], Mapping):
            raise ValueError(f"{section}.sampling_args must be a table.")
        normalized["sampling_args"] = _normalize_sampling_args(
            normalized["sampling_args"], f"{section}.sampling_args"
        )

    if "sampling" not in normalized:
        return normalized

    if "sampling_args" in normalized:
        if not merge_with_existing:
            raise ValueError(f"{section} cannot contain both sampling and sampling_args.")
        existing: Mapping[str, Any] = normalized["sampling_args"]
    else:
        existing = {}

    sampling = normalized.pop("sampling")
    if not isinstance(sampling, Mapping):
        raise ValueError(f"{section}.sampling must be a table.")
    normalized["sampling_args"] = _merge_sampling_tables(
        existing, _normalize_sampling_args(sampling, f"{section}.sampling")
    )
    return normalized


def _expand_ablation(ablation: Mapping[str, Any], global_defaults: dict) -> list[dict]:
    """Expand an `[[ablation]]` block into one config per sweep combination."""
    ablation = dict(ablation)
    sweep = dict(ablation.pop("sweep", {}))
    args_sweep = sweep.pop("args", {})
    env_args_sweep = sweep.pop("env_args", {})

    dimensions: list[tuple[str, list]] = []
    for prefix, table, label in (
        ("", sweep, ""),
        ("env_args.", env_args_sweep, ".env_args"),
        ("args.", args_sweep, ".args"),
    ):
        for key, values in table.items():
            if not isinstance(values, list):
                raise ValueError(
                    f"Ablation sweep{label} values must be lists, got "
                    f"{type(values).__name__} for '{key}'"
                )
            dimensions.append((f"{prefix}{key}", values))
    if not dimensions:
        raise ValueError("[[ablation]] block must have a non-empty [ablation.sweep] section")

    fixed_env_args = {
        **_config_table(ablation.get("env_args", {}), "ablation.env_args"),
        **_config_table(ablation.get("args", {}), "ablation.args"),
    }
    overlap = set(fixed_env_args) & (set(env_args_sweep) | set(args_sweep))
    if overlap:
        raise ValueError(
            f"environment arg key(s) {overlap} appear in both fixed args and "
            "sweep args — use one or the other"
        )

    explicit_keys = set(ablation) | set(sweep)
    fixed = {**global_defaults, **ablation}
    if "endpoint_id" in explicit_keys and "model" not in explicit_keys:
        fixed.pop("model", None)
    if "model" in explicit_keys and "endpoint_id" not in explicit_keys:
        fixed.pop("endpoint_id", None)

    keys = [key for key, _ in dimensions]
    expanded = []
    for combo in itertools.product(*(values for _, values in dimensions)):
        config = {k: (dict(v) if isinstance(v, dict) else v) for k, v in fixed.items()}
        for key, value in zip(keys, combo):
            for prefix in ("env_args.", "args."):
                if key.startswith(prefix):
                    table = prefix[:-1]
                    config[table] = {**config.get(table, {}), key[len(prefix) :]: value}
                    break
            else:
                config[key] = value
        expanded.append(config)
    return expanded


def load_toml_config(path: Path, extra_valid_fields: set[str] | None = None) -> list[dict]:
    """Load an eval TOML into one merged config per evaluation."""
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "rb") as handle:
        raw_config = tomllib.load(handle)

    eval_list = raw_config.get("eval", [])
    ablation_list = raw_config.get("ablation", [])
    if not isinstance(eval_list, list):
        raise ValueError(
            "Config file uses [eval] but should use [[eval]] (double brackets) "
            f"for array of tables: {path}"
        )
    if not isinstance(ablation_list, list):
        raise ValueError(
            "Config file uses [ablation] but should use [[ablation]] (double brackets) "
            f"for array of tables: {path}"
        )
    if not eval_list and not ablation_list:
        raise ValueError(
            f"Config file must contain at least one [[eval]] or [[ablation]] section: {path}"
        )

    eval_list = [_normalize_env_id_alias(config, "[[eval]]") for config in eval_list]
    ablation_list = [_normalize_env_id_alias(config, "[[ablation]]") for config in ablation_list]
    valid_fields = TOML_VALID_FIELDS | (extra_valid_fields or set())

    global_defaults = {k: v for k, v in raw_config.items() if k not in ("eval", "ablation")}
    invalid_global = set(global_defaults) - (valid_fields - {"name"})
    if invalid_global:
        raise ValueError(
            f"Invalid global field(s) {invalid_global}. "
            f"Valid fields are: {sorted(valid_fields - {'name'})}"
        )
    global_defaults = _normalize_sampling_config(global_defaults, "global config")

    merged_configs: list[dict] = []
    for eval_config in eval_list:
        invalid = set(eval_config) - valid_fields
        if invalid:
            raise ValueError(
                f"Invalid field(s) {invalid} for {eval_config.get('env_id', 'unknown')}. "
                f"Valid fields are: {sorted(valid_fields)}"
            )
        eval_config["env_id"] = _eval_env_id(eval_config, "[[eval]]")
        eval_config = _normalize_sampling_config(eval_config, f"[[eval]] {eval_config['env_id']}")
        merged = {**global_defaults, **eval_config}
        if "endpoint_id" in eval_config and "model" not in eval_config:
            merged.pop("model", None)
        if "model" in eval_config and "endpoint_id" not in eval_config:
            merged.pop("endpoint_id", None)
        merged_configs.append(_normalize_env_config_sections(merged))

    for ablation in ablation_list:
        if isinstance(ablation.get("sweep"), Mapping):
            ablation = {
                **ablation,
                "sweep": _normalize_env_id_alias(ablation["sweep"], "[ablation.sweep]"),
            }
        invalid = (set(ablation) - {"sweep"}) - valid_fields
        if invalid:
            raise ValueError(
                f"Invalid field(s) {invalid} in [[ablation]] block. "
                f"Valid fields are: {sorted(valid_fields)}"
            )
        ablation = _normalize_sampling_config(ablation, "[[ablation]] block")
        invalid_sweep = set(ablation.get("sweep", {})) - valid_fields
        invalid_sweep -= FREEFORM_ABLATION_SWEEP_FIELDS
        if invalid_sweep:
            raise ValueError(
                f"Invalid sweep field(s) {invalid_sweep} in [[ablation]] block. "
                f"Valid fields are: {sorted(valid_fields)}"
            )
        merged_configs.extend(
            _normalize_env_config_sections(
                _normalize_sampling_config(
                    config, "expanded [[ablation]] config", merge_with_existing=True
                )
            )
            for config in _expand_ablation(ablation, global_defaults)
        )

    for config in merged_configs:
        config["env_id"] = _eval_env_id(config, "eval config")
        endpoints_path = config.get("endpoints_path")
        if isinstance(endpoints_path, str) and not Path(endpoints_path).is_absolute():
            config["endpoints_path"] = str((path.parent / endpoints_path).resolve())
    return merged_configs


def resolve_endpoints_file(endpoints_path: str) -> Path | None:
    """The endpoints.toml an `endpoints_path` points at (a file or a directory)."""
    path = Path(endpoints_path)
    if path.is_dir():
        toml_file = path / "endpoints.toml"
        return toml_file if toml_file.exists() else None
    return path


def load_endpoint_models(endpoints_path: str) -> dict[str, list[str]]:
    """Map each `endpoint_id` in an endpoints.toml registry to its models.

    Raises ValueError when the registry is missing or malformed."""
    endpoints_file = resolve_endpoints_file(endpoints_path)
    if endpoints_file is None or not endpoints_file.exists():
        raise ValueError(f"endpoints.toml not found at {endpoints_path}")
    with open(endpoints_file, "rb") as handle:
        raw = tomllib.load(handle)

    entries = raw.get("endpoint", [])
    if not isinstance(entries, list):
        raise ValueError(f"Expected [[endpoint]] array-of-tables in {endpoints_file}")
    models: dict[str, list[str]] = {}
    for index, entry in enumerate(entries):
        source = f"{endpoints_file} ([[endpoint]] index {index})"
        if not isinstance(entry, dict):
            raise ValueError(f"Each [[endpoint]] entry must be a table in {source}")
        endpoint_id = entry.get("endpoint_id")
        if not isinstance(endpoint_id, str) or not endpoint_id:
            raise ValueError(
                f"Each [[endpoint]] entry must include non-empty string 'endpoint_id' in {source}"
            )
        model = entry.get("model")
        if not isinstance(model, str) or not model:
            raise ValueError(f"Field 'model' must be a non-empty string in {source}")
        models.setdefault(endpoint_id, []).append(model)
    return models
