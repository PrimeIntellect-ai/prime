"""Training config display and rerun helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import toml

from .models import LabItem
from .palette import PRIMARY
from .toml_format import format_toml_blocks


def training_config_toml(raw: dict[str, Any]) -> str:
    config: dict[str, Any] = {}
    if model := raw.get("base_model") or raw.get("baseModel"):
        config["model"] = model

    for key in (
        "max_steps",
        "rollouts_per_example",
        "batch_size",
        "learning_rate",
        "lora_alpha",
        "oversampling_factor",
        "max_inflight_rollouts",
        "checkpoint_id",
        "cluster_name",
    ):
        if key in raw:
            config[key] = raw[key]

    sampling = _dict_value(raw.get("sampling"))
    if "max_tokens" in raw:
        sampling.setdefault("max_tokens", raw["max_tokens"])
    if "temperature" in raw:
        sampling.setdefault("temperature", raw["temperature"])
    if sampling:
        config["sampling"] = sampling

    environments = raw.get("environments")
    if isinstance(environments, list) and environments:
        config["env"] = _rl_env_configs(environments)

    if isinstance(raw.get("eval_config"), dict):
        config["eval"] = _rl_eval_config(raw["eval_config"])
    if isinstance(raw.get("buffer_config"), dict):
        config["buffer"] = raw["buffer_config"]

    filtered = _filter_empty_values(normalize_rl_config(config))
    return format_toml_blocks(toml.dumps(filtered)).rstrip() if filtered else ""


def normalize_rl_config(config: dict[str, Any]) -> dict[str, Any]:
    updated = dict(config)
    updated.pop("seq_len", None)
    max_tokens = updated.pop("max_tokens", None)
    if max_tokens is not None:
        sampling = _dict_value(updated.get("sampling"))
        sampling.setdefault("max_tokens", max_tokens)
        updated["sampling"] = sampling

    environments = updated.pop("environments", None)
    if "env" in updated:
        updated["env"] = _rl_env_configs(_list_value(updated["env"]))
    elif isinstance(environments, list) and environments:
        updated["env"] = _rl_env_configs(environments)

    if "eval_config" in updated and "eval" not in updated:
        eval_config = updated.pop("eval_config")
        if isinstance(eval_config, dict):
            updated["eval"] = _rl_eval_config(eval_config)
    else:
        updated.pop("eval_config", None)
    updated.pop("val_config", None)
    if "buffer_config" in updated and "buffer" not in updated:
        buffer_config = updated.pop("buffer_config")
        if isinstance(buffer_config, dict):
            updated["buffer"] = buffer_config
    else:
        updated.pop("buffer_config", None)
    updated.pop("run_config", None)
    return updated


def training_config_item(
    item: LabItem,
    raw: dict[str, Any],
    *,
    workspace: Path | None,
) -> LabItem | None:
    toml_text = training_config_toml(raw)
    if not toml_text:
        return None
    run_id = training_run_id(raw, fallback=item.title)
    stem = safe_config_stem(
        str(raw.get("name") or _metadata_value(item, "Name") or f"training-{run_id}")
    )
    model = str(raw.get("base_model") or raw.get("baseModel") or "")
    envs = training_environment_names(_list_value(raw.get("environments")))
    metadata: list[tuple[str, str]] = [("Source run", run_id)]
    if model:
        metadata.append(("Model", model))
    if envs:
        metadata.append(("Environments", ", ".join(envs)))
    if raw.get("seq_len"):
        metadata.append(("Seq len", str(raw["seq_len"])))
    return LabItem(
        key=f"{item.key}:config",
        section="home",
        title="training-config.toml",
        subtitle="Training config copy",
        status="rl",
        status_style=PRIMARY,
        metadata=tuple(metadata),
        raw={
            "type": "config_file",
            "config_kind": "rl",
            "toml": toml_text,
            "workspace": str((workspace or Path.cwd()).expanduser().resolve()),
            "path": f"{stem}.toml",
        },
    )


def training_platform_url(
    frontend_url: str,
    raw: dict[str, Any],
    *,
    fallback_id: str = "",
) -> str | None:
    run_id = training_run_id(raw, fallback=fallback_id)
    if not frontend_url or not run_id:
        return None
    return f"{frontend_url.rstrip('/')}/dashboard/training/{run_id}"


def training_run_id(raw: dict[str, Any], *, fallback: str = "") -> str:
    value = raw.get("id") or raw.get("run_id") or raw.get("runId") or fallback
    return str(value or "").strip()


def safe_config_stem(value: str) -> str:
    chars: list[str] = []
    last_dash = False
    for char in value.lower():
        if "a" <= char <= "z" or "0" <= char <= "9":
            chars.append(char)
            last_dash = False
        elif not last_dash:
            chars.append("-")
            last_dash = True
    stem = "".join(chars).strip("-")
    return stem or "training-config"


def training_environment_names(environments: list[Any]) -> list[str]:
    names: list[str] = []
    for env in environments:
        if isinstance(env, dict):
            env_id = env.get("id") or env.get("name")
            if env_id:
                names.append(str(env_id))
        elif env:
            names.append(str(env))
    return names


def _rl_env_configs(environments: list[Any]) -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    for env in environments:
        if isinstance(env, dict):
            env_id = env.get("id") or env.get("slug") or env.get("name")
            if not env_id:
                continue
            config: dict[str, Any] = {"id": str(env_id)}
            if env.get("name") and env.get("name") != env_id:
                config["name"] = env["name"]
            if isinstance(env.get("args"), dict):
                config["args"] = env["args"]
            if env.get("version"):
                config["version"] = env["version"]
            configs.append(config)
        elif env:
            env_id, version = _split_environment_token(str(env))
            config = {"id": env_id}
            if version:
                config["version"] = version
            configs.append(config)
    return configs


def _rl_eval_config(config: dict[str, Any]) -> dict[str, Any]:
    updated = dict(config)
    environments = updated.pop("environments", None)
    if "env" in updated:
        updated["env"] = _rl_env_configs(_list_value(updated["env"]))
    elif isinstance(environments, list) and environments:
        updated["env"] = _rl_env_configs(environments)
    return updated


def _dict_value(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _split_environment_token(token: str) -> tuple[str, str | None]:
    if "@" in token:
        env_id, version = token.rsplit("@", 1)
        return env_id.strip(), version.strip() or None
    if ":" in token and "/" in token:
        env_id, version = token.rsplit(":", 1)
        return env_id.strip(), version.strip() or None
    return token, None


def _filter_empty_values(value: Any) -> Any:
    if isinstance(value, dict):
        filtered = {
            str(key): _filter_empty_values(child)
            for key, child in value.items()
            if child is not None
        }
        return {
            key: child
            for key, child in filtered.items()
            if child is not None and child != {} and child != []
        }
    if isinstance(value, list):
        return [
            child
            for child in (_filter_empty_values(child) for child in value)
            if child is not None and child != {} and child != []
        ]
    return value


def _list_value(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _metadata_value(item: LabItem, key: str) -> str | None:
    for item_key, value in item.metadata:
        if item_key == key:
            return value
    return None
