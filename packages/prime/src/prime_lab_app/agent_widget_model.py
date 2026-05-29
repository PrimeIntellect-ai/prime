"""Logical agent control models and config helpers for Lab chat."""

from __future__ import annotations

import shlex
from collections.abc import Iterable
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import toml
from prime_cli.api.rl import RLClient
from prime_cli.client import APIClient, APIError
from prime_cli.core import Config

from .agent_widget_titles import clean_widget_title
from .config_factory import evaluation_config, filter_empty_config_values, rl_config
from .config_screen import initial_config_field_values
from .models import LabItem
from .toml_format import format_toml_blocks

_TRAINING_MODEL_OPTIONS_CACHE: tuple[tuple[str, str], ...] | None = None
_TRAINING_MODEL_NAMES_CACHE: tuple[str, ...] | None = None
_TRAINING_MODEL_OPTIONS_CACHE_KEY: tuple[str, str, str] | None = None
_TRAINING_MODEL_NAMES_CACHE_KEY: tuple[str, str, str] | None = None
_TRAINING_MODEL_OPTION_METADATA: dict[str, tuple[str, dict[str, str]]] = {}


@dataclass(frozen=True)
class AgentWidgetFieldSpec:
    """One editable field in a logical agent widget."""

    name: str
    label: str
    value: str
    input_type: str = "text"
    disabled: bool = False
    widget: str = "input"
    options: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class AgentWidgetActionSpec:
    """One action exposed by a logical agent widget."""

    name: str
    label: str
    variant: str = "default"


@dataclass(frozen=True)
class AgentWidgetModel:
    """Logical Lab control request independent of its Textual rendering skin."""

    action: dict[str, Any]
    payload: dict[str, Any]
    config_context: dict[str, Any] | None
    title: str
    fields: tuple[AgentWidgetFieldSpec, ...]
    actions: tuple[AgentWidgetActionSpec, ...]


def build_agent_widget_model(message: Any, workspace: Path) -> AgentWidgetModel:
    """Normalize one agent-emitted widget request into a renderable logical model."""

    action = _widget_action(message)
    payload = _widget_payload(action)
    context = _widget_config_context(action, workspace)
    return AgentWidgetModel(
        action=action,
        payload=payload,
        config_context=context,
        title=_widget_display_title(action, context),
        fields=_widget_field_specs(context),
        actions=_widget_actions(action),
    )


def widget_payload(action: dict[str, Any]) -> dict[str, Any]:
    """Return the canonical payload object for a widget action."""

    return _widget_payload(action)


def widget_display_title(
    action: dict[str, Any],
    context: dict[str, Any] | None,
) -> str:
    """Return the user-facing title for a widget action."""

    return _widget_display_title(action, context)


def widget_config_item(action: dict[str, Any], workspace: Path) -> LabItem | None:
    """Return a config item represented by a widget action, when available."""

    return _widget_config_item(action, workspace)


def widget_command_text(payload: dict[str, Any], workspace: Path) -> str:
    """Return a runnable command represented by a widget payload."""

    return _widget_command_text(payload, workspace)


def widget_config_field_name(field_id: str) -> str:
    """Map a visible field id to the shared config field name."""

    return _widget_config_field_name(field_id)


def widget_training_model_option_parts(value: str) -> tuple[str, dict[str, str]]:
    """Return the model name and sampling controls encoded by a training model option."""

    return _training_model_option_parts(value)


def training_model_names() -> tuple[str, ...]:
    """Return Hosted Training model ids available to the active account."""

    return _training_model_names()


def widget_config_path(payload: dict[str, Any], workspace: Path) -> Path:
    """Return the resolved config path represented by a widget payload."""

    return _widget_config_path(payload, workspace)


def widget_generated_config_path(
    action: dict[str, Any],
    workspace: Path,
    config_kind: str,
    source_path: Path,
) -> Path:
    """Return the generated config copy path for a widget launch."""

    return _widget_generated_config_path(action, workspace, config_kind, source_path)


def widget_single_eval_config(config: dict[str, Any]) -> dict[str, Any] | None:
    """Return the single eval block from an eval config, when it has exactly one."""

    return _widget_single_eval_config(config)


def dedupe_single_eval_globals(config: dict[str, Any]) -> None:
    """Drop duplicate global/single-eval values from an eval config."""

    _dedupe_single_eval_globals(config)


def filter_widget_empty_values(value: Any) -> Any:
    """Remove empty optional values from a widget config object."""

    return _filter_widget_empty_values(value)


def candidate_by_id(candidates: Any, choice_id: str) -> dict[str, Any]:
    """Return the choice candidate with the given id."""

    return _candidate_by_id(candidates, choice_id)


def candidate_label(candidate: dict[str, Any], fallback: str) -> str:
    """Return a readable choice label."""

    return _candidate_label(candidate, fallback)


def coerce_optional_int(value: Any) -> int | None:
    """Parse an optional integer value."""

    return _coerce_optional_int(value)


def parse_optional_int(label: str, value: str, errors: list[str]) -> int | None:
    """Parse a user-entered optional integer and append validation errors."""

    return _parse_optional_int(label, value, errors)


def relative_path(path: Path, workspace: Path) -> str:
    """Return a workspace-relative path when possible."""

    return _relative_path(path, workspace)


def _widget_action(message: Any) -> dict[str, Any]:
    return message.metadata if isinstance(message.metadata, dict) else {}


def _widget_payload(action: dict[str, Any]) -> dict[str, Any]:
    payload = action.get("payload")
    return payload if isinstance(payload, dict) else action


def _widget_context_payload(action: dict[str, Any]) -> dict[str, Any]:
    payload = _widget_payload(action)
    if payload is action:
        return payload
    top_level = {key: value for key, value in action.items() if key != "payload"}
    return {**top_level, **payload}


def _widget_config_context(action: dict[str, Any], workspace: Path) -> dict[str, Any] | None:
    payload = _widget_context_payload(action)
    kind = str(action.get("kind") or payload.get("kind") or "")
    config_kind = str(payload.get("config_kind") or action.get("config_kind") or "").strip()
    if kind not in {"run_launcher", "config_editor"} or config_kind not in {"eval", "rl", "gepa"}:
        return None
    env_hint = _widget_environment_from_payload(payload, Path(""), workspace)
    config_path = _widget_config_path(payload, workspace)
    if not str(payload.get("config_path") or "").strip():
        config_path = _widget_existing_config_path(workspace, config_kind, env_hint) or config_path
    toml_text = _widget_toml(payload, config_path)
    config = _safe_toml_loads(toml_text) if toml_text else {}
    if not config:
        config = _widget_default_config(payload, config_kind, config_path, workspace)
        toml_text = format_toml_blocks(toml.dumps(config)).rstrip() + "\n"
    fallback_name = config_path.stem or clean_widget_title(
        str(action.get("title") or payload.get("title") or "config")
    )
    values = initial_config_field_values(config, config_kind, fallback_name=fallback_name)
    values.update(_widget_extra_field_values(config, config_kind))
    if not values.get("envs"):
        values["envs"] = env_hint
    model_options = _widget_model_options(workspace, config, config_path, config_kind)
    if config_kind == "rl" and model_options:
        values["model"] = _selected_training_model_option_value(
            str(values.get("model") or ""),
            config,
            model_options,
        )
    elif not values.get("model") and model_options:
        values["model"] = str(model_options[0][1])
    if config_kind == "eval":
        if not values.get("num_examples"):
            values["num_examples"] = "50"
        if not values.get("rollouts_per_example"):
            values["rollouts_per_example"] = "3"
        if not values.get("max_tokens"):
            values["max_tokens"] = "1024"
        if not values.get("max_concurrent"):
            values["max_concurrent"] = "auto"
    elif config_kind == "rl":
        if not values.get("max_steps"):
            values["max_steps"] = "100"
        if not values.get("rollouts_per_example"):
            values["rollouts_per_example"] = "8"
        if not values.get("batch_size"):
            values["batch_size"] = "256"
        if not values.get("max_tokens"):
            values["max_tokens"] = "8192"
    return {
        "config_kind": config_kind,
        "config_path": config_path,
        "workspace": workspace,
        "toml": toml_text,
        "config": config,
        "values": values,
        "model_options": model_options,
    }


def _widget_display_title(action: dict[str, Any], context: dict[str, Any] | None) -> str:
    payload = _widget_payload(action)
    raw_title = clean_widget_title(str(action.get("title") or payload.get("title") or "Action"))
    if context is None:
        return raw_title
    config_kind = str(context.get("config_kind") or "")
    env = str(context.get("values", {}).get("envs") or raw_title).split(",", 1)[0].strip()
    env_name = env.rsplit("/", 1)[-1] if env else raw_title
    if config_kind == "eval":
        return f"Evaluate {env_name}"
    if config_kind == "rl":
        return f"Train {env_name}"
    if config_kind == "gepa":
        return f"Optimize {env_name}"
    return raw_title


def _widget_field_specs(context: dict[str, Any] | None) -> tuple[AgentWidgetFieldSpec, ...]:
    if context is None:
        return ()
    values = context["values"]
    config_kind = str(context["config_kind"])
    model_options = tuple(context.get("model_options") or ())
    names = (
        ("envs", "Environment", "text", False),
        ("model", "Model", "text", False),
        ("num_examples", "Examples", "integer", False),
        ("rollouts_per_example", "Rollouts per example", "integer", False),
        ("max_tokens", "Max tokens", "integer", False),
        ("max_concurrent", "Max concurrent", "text", False),
    )
    if config_kind == "rl":
        names = (
            ("envs", "Environment", "text", False),
            ("model", "Model", "text", False),
            ("max_steps", "Steps", "integer", False),
            ("rollouts_per_example", "Rollouts per example", "integer", False),
            ("batch_size", "Rollouts per batch", "integer", False),
            ("max_tokens", "Max tokens", "integer", False),
            ("seq_len", "Seq len", "integer", True),
        )
    elif config_kind == "gepa":
        names = (
            ("envs", "Environment", "text", False),
            ("model", "Model", "text", False),
        )
    fields: list[AgentWidgetFieldSpec] = []
    for name, label, input_type, disabled in names:
        value = str(values.get(name) or "")
        if not value and name in {"seq_len"}:
            continue
        widget = "input"
        options: tuple[tuple[str, str], ...] = ()
        if name == "model" and model_options:
            option_values = {str(option_value) for _label, option_value in model_options}
            if config_kind == "rl" and value not in option_values:
                value = str(model_options[0][1])
            elif value and value not in option_values:
                model_options = ((value, value), *model_options)
            elif not value:
                value = str(model_options[0][1])
            widget = "select"
            options = model_options
        fields.append(
            AgentWidgetFieldSpec(
                name=name,
                label=label,
                value=value,
                input_type=input_type,
                disabled=disabled,
                widget=widget,
                options=options,
            )
        )
    return tuple(fields)


def _widget_actions(action: dict[str, Any]) -> tuple[AgentWidgetActionSpec, ...]:
    payload = _widget_payload(action)
    kind = str(action.get("kind") or payload.get("kind") or "")
    actions: list[AgentWidgetActionSpec] = []
    if kind == "choice_picker":
        for index, candidate in enumerate(_candidate_list(payload.get("candidates"))[:6]):
            choice_id = _candidate_id(candidate, index)
            actions.append(
                AgentWidgetActionSpec(
                    name=f"choice:{choice_id}",
                    label=_candidate_label(candidate, choice_id),
                    variant="primary"
                    if choice_id == str(payload.get("default_id") or "")
                    else "default",
                )
            )
    if kind in {"run_launcher", "config_editor"}:
        actions.append(AgentWidgetActionSpec("launch", "Launch", "primary"))
        actions.append(AgentWidgetActionSpec("stop", "Stop", "default"))
    return tuple(actions)


def _candidate_list(candidates: Any) -> list[dict[str, Any]]:
    if not isinstance(candidates, list):
        return []
    return [candidate for candidate in candidates if isinstance(candidate, dict)]


def _candidate_id(candidate: dict[str, Any], index: int) -> str:
    raw_id = str(candidate.get("id") or candidate.get("value") or candidate.get("key") or "")
    return raw_id.strip() or f"choice-{index + 1}"


def _candidate_by_id(candidates: Any, choice_id: str) -> dict[str, Any]:
    for index, candidate in enumerate(_candidate_list(candidates)):
        if _candidate_id(candidate, index) == choice_id:
            return candidate
    return {}


def _candidate_label(candidate: dict[str, Any], fallback: str) -> str:
    label = str(
        candidate.get("label")
        or candidate.get("name")
        or candidate.get("title")
        or candidate.get("id")
        or fallback
    )
    return label.strip() or fallback


def _widget_default_config(
    payload: dict[str, Any],
    config_kind: str,
    config_path: Path,
    workspace: Path,
) -> dict[str, Any]:
    defaults = payload.get("defaults")
    defaults = defaults if isinstance(defaults, dict) else {}
    env_id = _widget_environment_from_payload(payload, config_path, workspace)
    model = str(
        defaults.get("model")
        or payload.get("model")
        or _widget_default_model(workspace, {}, config_path, config_kind)
    )
    if config_kind == "eval":
        sampling = defaults.get("sampling_args")
        if not isinstance(sampling, dict):
            sampling = defaults.get("sampling")
        max_tokens = _coerce_int(defaults.get("max_tokens"), 1024)
        config = evaluation_config(
            env_id=env_id,
            model=model,
            num_examples=_coerce_int(defaults.get("num_examples"), 50),
            rollouts_per_example=_coerce_int(defaults.get("rollouts_per_example"), 3),
            max_tokens=None if isinstance(sampling, dict) and sampling else max_tokens,
            max_concurrent=_coerce_optional_int(defaults.get("max_concurrent")),
        )
        if isinstance(sampling, dict) and sampling:
            config["eval"][0]["sampling_args"] = dict(sampling)
        return filter_empty_config_values(config)
    if config_kind == "rl":
        return rl_config(
            env_id=env_id,
            model=model,
            max_steps=_coerce_int(defaults.get("max_steps"), 100),
            rollouts_per_example=_coerce_int(defaults.get("rollouts_per_example"), 8),
            batch_size=_coerce_int(defaults.get("batch_size"), 256),
            max_tokens=_coerce_int(defaults.get("max_tokens"), 8192),
        )
    return _filter_widget_empty_values(
        {
            "model": model,
            "env": [{"id": env_id}],
        }
    )


def _widget_extra_field_values(config: dict[str, Any], config_kind: str) -> dict[str, str]:
    if config_kind != "eval":
        return {}
    eval_config = _widget_single_eval_config(config)
    num_examples = _first_present(
        config.get("num_examples"),
        eval_config.get("num_examples") if eval_config else None,
    )
    max_concurrent = _first_present(
        config.get("max_concurrent"),
        eval_config.get("max_concurrent") if eval_config else None,
    )
    return {
        "num_examples": "" if num_examples is None else str(num_examples),
        "max_concurrent": "auto" if max_concurrent is None else str(max_concurrent),
    }


def _widget_config_field_name(field_id: str) -> str:
    names = {
        "name": "name",
        "model": "model",
        "envs": "envs",
        "max_steps": "max_steps",
        "rollouts_per_example": "rollouts_per_example",
        "batch_size": "batch_size",
        "max_tokens": "max_tokens",
        "seq_len": "seq_len",
        "enable_thinking": "enable_thinking",
        "reasoning_effort": "reasoning_effort",
        "config-name": "name",
        "config-model": "model",
        "config-envs": "envs",
        "config-max-steps": "max_steps",
        "config-rollouts": "rollouts_per_example",
        "config-batch-size": "batch_size",
        "config-max-tokens": "max_tokens",
        "config-seq-len": "seq_len",
        "config-enable-thinking": "enable_thinking",
        "config-reasoning-effort": "reasoning_effort",
    }
    return names.get(field_id, "")


def _widget_existing_config_path(
    workspace: Path,
    config_kind: str,
    env_hint: str,
) -> Path | None:
    names = _widget_candidate_stems(env_hint)
    roots = (
        workspace / "configs" / config_kind,
        workspace / ".prime" / "lab" / "configs" / config_kind,
    )
    for root in roots:
        for name in names:
            path = root / f"{name}.toml"
            if path.is_file():
                return path.resolve()
    return None


def _widget_candidate_stems(value: str) -> tuple[str, ...]:
    candidates: list[str] = []
    for raw in (value, value.rsplit("/", 1)[-1]):
        raw = raw.strip()
        if not raw:
            continue
        for candidate in (raw, raw.replace("_", "-"), raw.replace("-", "_"), _slug(raw)):
            if candidate and candidate not in candidates:
                candidates.append(candidate)
    return tuple(candidates)


def _widget_generated_config_path(
    action: dict[str, Any],
    workspace: Path,
    config_kind: str,
    source_path: Path,
) -> Path:
    payload = _widget_payload(action)
    if config_kind == "rl":
        env_id = _widget_environment_from_payload(payload, source_path, workspace)
        if env_id:
            stem = _slug(env_id.rsplit("/", 1)[-1])
            return workspace / ".prime" / "lab" / "configs" / config_kind / f"{stem}.toml"
    title = clean_widget_title(str(action.get("title") or payload.get("title") or source_path.stem))
    stem = _slug(title) or source_path.stem or "run"
    return workspace / ".prime" / "lab" / "configs" / config_kind / f"{stem}.toml"


def _widget_model_options(
    workspace: Path,
    config: dict[str, Any],
    config_path: Path,
    config_kind: str,
) -> tuple[tuple[str, str], ...]:
    options: list[tuple[str, str]] = []
    if config_kind == "rl":
        options.extend(_training_model_options())
    else:
        for endpoint_path in _widget_endpoint_paths(workspace, config, config_path):
            for option in _read_endpoint_model_options(endpoint_path):
                if option not in options:
                    options.append(option)
    current = str(config.get("model") or "").strip()
    if not current:
        endpoint_id = str(config.get("endpoint_id") or "").strip()
        current = _model_for_endpoint_id(endpoint_id, workspace, config, config_path)
    if current and config_kind != "rl" and current not in {value for _label, value in options}:
        options.insert(0, (current, current))
    return tuple(options)


def _widget_default_model(
    workspace: Path,
    config: dict[str, Any],
    config_path: Path,
    config_kind: str,
) -> str:
    options = _widget_model_options(workspace, config, config_path, config_kind)
    if options:
        value = str(options[0][1])
        if config_kind == "rl":
            model, _controls = _training_model_option_parts(value)
            return model
        return value
    return ""


def _training_model_options() -> tuple[tuple[str, str], ...]:
    global _TRAINING_MODEL_OPTIONS_CACHE, _TRAINING_MODEL_OPTIONS_CACHE_KEY
    names = _training_model_names()
    if not names:
        return ()
    cache_key = _TRAINING_MODEL_NAMES_CACHE_KEY
    if _TRAINING_MODEL_OPTIONS_CACHE is not None and _TRAINING_MODEL_OPTIONS_CACHE_KEY == cache_key:
        return _TRAINING_MODEL_OPTIONS_CACHE
    _TRAINING_MODEL_OPTION_METADATA.clear()
    _TRAINING_MODEL_OPTIONS_CACHE = _training_model_options_from_names(names)
    _TRAINING_MODEL_OPTIONS_CACHE_KEY = _TRAINING_MODEL_NAMES_CACHE_KEY
    return _TRAINING_MODEL_OPTIONS_CACHE


def _training_model_names() -> tuple[str, ...]:
    global _TRAINING_MODEL_NAMES_CACHE, _TRAINING_MODEL_NAMES_CACHE_KEY
    global _TRAINING_MODEL_OPTIONS_CACHE, _TRAINING_MODEL_OPTIONS_CACHE_KEY
    config = Config()
    if not config.api_key:
        return ()
    cache_key = _training_model_cache_key(config)
    if _TRAINING_MODEL_NAMES_CACHE is not None and _TRAINING_MODEL_NAMES_CACHE_KEY == cache_key:
        return _TRAINING_MODEL_NAMES_CACHE
    try:
        models = RLClient(APIClient()).list_models(team_id=config.team_id)
    except APIError:
        return ()
    if _TRAINING_MODEL_NAMES_CACHE_KEY != cache_key:
        _TRAINING_MODEL_OPTIONS_CACHE = None
        _TRAINING_MODEL_OPTIONS_CACHE_KEY = None
    names: list[str] = []
    for model in sorted(models, key=lambda model: model.name):
        name = str(model.name).strip()
        if not name:
            continue
        names.append(name)
    _TRAINING_MODEL_NAMES_CACHE = tuple(names)
    _TRAINING_MODEL_NAMES_CACHE_KEY = cache_key
    return _TRAINING_MODEL_NAMES_CACHE


def _training_model_cache_key(config: Any) -> tuple[str, str, str]:
    return (
        str(getattr(config, "base_url", "")),
        str(getattr(config, "current_environment", "")),
        str(getattr(config, "team_id", None) or getattr(config, "team_name", "") or ""),
    )


def _training_model_options_from_names(names: Iterable[str]) -> tuple[tuple[str, str], ...]:
    options: list[tuple[str, str]] = []
    for name in names:
        for option in _training_model_options_for_name(name):
            if option not in options:
                options.append(option)
    return tuple(options)


def _training_model_options_for_name(name: str) -> tuple[tuple[str, str], ...]:
    normalized = name.strip()
    if not normalized:
        return ()
    lowered = normalized.lower()
    if "gpt-oss" in lowered:
        return tuple(
            (
                f"{normalized} ({effort})",
                _training_model_option_value(normalized, reasoning_effort=effort),
            )
            for effort in ("low", "medium", "high")
        )
    if "qwen3.5" in lowered or "qwen3.6" in lowered or "nemotron" in lowered:
        return (
            (
                f"{normalized} (thinking)",
                _training_model_option_value(normalized, enable_thinking="true"),
            ),
            (
                f"{normalized} (instruct)",
                _training_model_option_value(normalized, enable_thinking="false"),
            ),
        )
    return ((normalized, normalized),)


def _training_model_option_value(
    model: str,
    *,
    enable_thinking: str = "",
    reasoning_effort: str = "",
) -> str:
    controls: dict[str, str] = {}
    if enable_thinking:
        controls["enable_thinking"] = enable_thinking
    if reasoning_effort:
        controls["reasoning_effort"] = reasoning_effort
    control_key = ",".join(f"{key}={value}" for key, value in sorted(controls.items()))
    digest = sha256(f"{model}\0{control_key}".encode("utf-8")).hexdigest()[:16]
    value = f"training-model-{digest}"
    _TRAINING_MODEL_OPTION_METADATA[value] = (model, controls)
    return value


def _selected_training_model_option_value(
    model: str,
    config: dict[str, Any],
    options: tuple[tuple[str, str], ...],
) -> str:
    option_values = {str(value) for _label, value in options}
    sampling = config.get("sampling")
    sampling = sampling if isinstance(sampling, dict) else {}
    if sampling.get("enable_thinking") is not None:
        candidate = _training_model_option_value(
            model,
            enable_thinking=str(bool(sampling["enable_thinking"])).lower(),
        )
        if candidate in option_values:
            return candidate
    reasoning_effort = str(sampling.get("reasoning_effort") or "").strip()
    if reasoning_effort:
        candidate = _training_model_option_value(model, reasoning_effort=reasoning_effort)
        if candidate in option_values:
            return candidate
    medium_reasoning = _training_model_option_value(model, reasoning_effort="medium")
    if medium_reasoning in option_values:
        return medium_reasoning
    for _label, value in _training_model_options_for_name(model):
        if value in option_values:
            return value
    return _preferred_training_model_option_value(options)


def _preferred_training_model_option_value(options: tuple[tuple[str, str], ...]) -> str:
    first_value = str(options[0][1])
    first_model, _controls = _training_model_option_parts(first_value)
    medium_reasoning = _training_model_option_value(first_model, reasoning_effort="medium")
    if medium_reasoning in {str(value) for _label, value in options}:
        return medium_reasoning
    return first_value


def _training_model_option_parts(value: str) -> tuple[str, dict[str, str]]:
    metadata = _TRAINING_MODEL_OPTION_METADATA.get(value)
    if metadata is None:
        return value, {}
    model, controls = metadata
    return model, dict(controls)


def _widget_endpoint_paths(
    workspace: Path,
    config: dict[str, Any],
    config_path: Path,
) -> tuple[Path, ...]:
    candidates: list[Path] = []
    raw_path = config.get("endpoints_path")
    if isinstance(raw_path, str) and raw_path.strip():
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = config_path.parent / path
        candidates.append(path)
    candidates.extend((workspace / "configs" / "endpoints.toml", workspace / "endpoints.toml"))
    paths: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file() and resolved not in paths:
            paths.append(resolved)
    return tuple(paths)


def _read_endpoint_model_options(path: Path) -> tuple[tuple[str, str], ...]:
    try:
        parsed = toml.loads(path.read_text(encoding="utf-8"))
    except (OSError, toml.TomlDecodeError):
        return ()
    endpoints = parsed.get("endpoint")
    if not isinstance(endpoints, list):
        endpoints = parsed.get("endpoints")
    if not isinstance(endpoints, list):
        return ()
    options: list[tuple[str, str]] = []
    for endpoint in endpoints:
        if not isinstance(endpoint, dict):
            continue
        model = str(endpoint.get("model") or "").strip()
        if model:
            option = (model, model)
            if option not in options:
                options.append(option)
    return tuple(options)


def _model_for_endpoint_id(
    endpoint_id: str,
    workspace: Path,
    config: dict[str, Any],
    config_path: Path,
) -> str:
    if not endpoint_id:
        return ""
    for endpoint_path in _widget_endpoint_paths(workspace, config, config_path):
        try:
            parsed = toml.loads(endpoint_path.read_text(encoding="utf-8"))
        except (OSError, toml.TomlDecodeError):
            continue
        endpoints = parsed.get("endpoint")
        if not isinstance(endpoints, list):
            continue
        for endpoint in endpoints:
            if not isinstance(endpoint, dict):
                continue
            if str(endpoint.get("endpoint_id") or "").strip() == endpoint_id:
                return str(endpoint.get("model") or "").strip()
    return ""


def _resolve_widget_environment(
    workspace: Path,
    value: str,
    *,
    allow_inference: bool = True,
) -> str:
    candidate = value.strip()
    if not candidate:
        return ""
    if not allow_inference:
        return candidate
    if "/" in candidate:
        return candidate
    local_names = _widget_local_environment_names(workspace)
    normalized = _normalize_widget_env_name(candidate)
    for local_name in local_names:
        if _normalize_widget_env_name(local_name) == normalized:
            return local_name
    for local_name in local_names:
        short_name = local_name.rsplit("/", 1)[-1]
        if _normalize_widget_env_name(short_name) == normalized:
            return local_name
    return f"primeintellect/{candidate.replace('_', '-')}"


def _widget_local_environment_names(workspace: Path) -> tuple[str, ...]:
    env_root = workspace / "environments"
    if not env_root.is_dir():
        return ()
    names: list[str] = []
    for path in sorted(env_root.iterdir(), key=lambda item: item.name):
        if not path.is_dir() or path.name.startswith("."):
            continue
        name = _widget_environment_project_name(path) or path.name.replace("_", "-")
        if name and name not in names:
            names.append(name)
    return tuple(names)


def _widget_environment_project_name(path: Path) -> str:
    pyproject = path / "pyproject.toml"
    if not pyproject.is_file():
        return ""
    try:
        parsed = toml.loads(pyproject.read_text(encoding="utf-8"))
    except (OSError, toml.TomlDecodeError):
        return ""
    project = parsed.get("project")
    if not isinstance(project, dict):
        return ""
    return str(project.get("name") or "").strip()


def _normalize_widget_env_name(value: str) -> str:
    return value.strip().lower().replace("_", "-")


def _widget_environment_from_payload(
    payload: dict[str, Any],
    config_path: Path,
    workspace: Path,
) -> str:
    config_kind = str(payload.get("config_kind") or "eval")
    config = payload.get("config")
    if isinstance(config, dict):
        envs = initial_config_field_values(
            config,
            config_kind,
            fallback_name="",
        )
        if envs.get("envs"):
            return _resolve_widget_environment(
                workspace,
                _widget_environment_token(envs["envs"]),
                allow_inference=config_kind != "rl",
            )
        if env_value := _widget_environment_from_mapping(config, config_kind):
            return _resolve_widget_environment(
                workspace,
                _widget_environment_token(env_value),
                allow_inference=config_kind != "rl",
            )
    defaults = payload.get("defaults")
    if isinstance(defaults, dict):
        envs = initial_config_field_values(defaults, config_kind, fallback_name="")
        if envs.get("envs"):
            return _resolve_widget_environment(
                workspace,
                _widget_environment_token(envs["envs"]),
                allow_inference=config_kind != "rl",
            )
        if env_value := _widget_environment_from_mapping(defaults, config_kind):
            return _resolve_widget_environment(
                workspace,
                _widget_environment_token(env_value),
                allow_inference=config_kind != "rl",
            )
    if env_value := _widget_environment_from_mapping(payload, config_kind):
        return _resolve_widget_environment(
            workspace,
            _widget_environment_token(env_value),
            allow_inference=config_kind != "rl",
        )
    if config_kind == "rl":
        return ""
    command_env = _environment_from_command(payload.get("command"))
    if command_env:
        return _resolve_widget_environment(workspace, _widget_environment_token(command_env))
    title = str(payload.get("title") or "")
    if ":" in title:
        candidate = title.split(":", 1)[1].strip()
        if candidate:
            return _resolve_widget_environment(workspace, candidate)
    if config_path.stem and config_path.stem != "agent-config":
        return _resolve_widget_environment(workspace, config_path.stem)
    return _resolve_widget_environment(workspace, "")


def _widget_environment_from_mapping(value: dict[str, Any], config_kind: str) -> str:
    if config_kind == "rl":
        return _widget_rl_environment_value(value.get("env"))
    keys = (
        "env_id",
        "environment",
        "environment_id",
        "env",
        "envs",
        "environments",
    )
    for key in keys:
        env_value = _widget_environment_value(value.get(key))
        if env_value:
            return env_value
    return ""


def _widget_rl_environment_value(value: Any) -> str:
    if not isinstance(value, list):
        return ""
    for item in value:
        if not isinstance(item, dict):
            continue
        env_id = item.get("id")
        if isinstance(env_id, str) and env_id.strip():
            return env_id.strip()
    return ""


def _widget_environment_value(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        for key in ("env_id", "id", "environment", "environment_id", "name", "slug"):
            env_value = value.get(key)
            if isinstance(env_value, str) and env_value.strip():
                return env_value.strip()
        return ""
    if isinstance(value, list):
        for item in value:
            env_value = _widget_environment_value(item)
            if env_value:
                return env_value
    return ""


def _widget_environment_token(value: str) -> str:
    return value.split(",", 1)[0].split("@", 1)[0].strip()


def _environment_from_command(command: Any) -> str:
    if isinstance(command, list):
        parts = [str(part) for part in command]
    elif isinstance(command, str) and command.strip():
        try:
            parts = shlex.split(command)
        except ValueError:
            parts = command.split()
    else:
        return ""
    for index, part in enumerate(parts):
        if part == "run" and index + 1 < len(parts):
            candidate = parts[index + 1]
            if not candidate.endswith((".toml", ".yaml", ".yml")):
                return candidate
    return ""


def _widget_single_eval_config(config: dict[str, Any]) -> dict[str, Any] | None:
    evals = config.get("eval")
    if isinstance(evals, list) and len(evals) == 1 and isinstance(evals[0], dict):
        return evals[0]
    return None


def _dedupe_single_eval_globals(config: dict[str, Any]) -> None:
    eval_config = _widget_single_eval_config(config)
    if eval_config is None:
        return
    for key in ("num_examples", "rollouts_per_example"):
        if key in config and eval_config.get(key) == config.get(key):
            eval_config.pop(key, None)


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _coerce_int(value: Any, fallback: int) -> int:
    parsed = _coerce_optional_int(value)
    return fallback if parsed is None else parsed


def _coerce_optional_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_bool(value: Any, fallback: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1", "on"}:
            return True
        if normalized in {"false", "no", "0", "off"}:
            return False
    return fallback


def _parse_optional_int(label: str, value: str, errors: list[str]) -> int | None:
    if not value or value.strip().lower() == "auto":
        return None
    try:
        return int(value)
    except ValueError:
        errors.append(f"{label} must be an integer")
        return None


def _parse_optional_bool(label: str, value: str, errors: list[str]) -> bool | None:
    if not value:
        return None
    normalized = value.strip().lower()
    if normalized in {"true", "yes", "1", "on"}:
        return True
    if normalized in {"false", "no", "0", "off"}:
        return False
    errors.append(f"{label} must be true or false")
    return None


def _bool_text(value: Any) -> str:
    return "true" if _coerce_bool(value, False) else "false"


def _filter_widget_empty_values(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): child
            for key, child in (
                (key, _filter_widget_empty_values(child)) for key, child in value.items()
            )
            if child is not None and child != {} and child != [] and child != ""
        }
    if isinstance(value, list):
        return [
            child
            for child in (_filter_widget_empty_values(child) for child in value)
            if child is not None and child != {} and child != [] and child != ""
        ]
    return value


def _slug(value: str) -> str:
    chars = [char.lower() if char.isalnum() else "-" for char in value.strip()]
    slug = "-".join(part for part in "".join(chars).split("-") if part)
    return slug[:80]


def _widget_config_item(action: dict[str, Any], workspace: Path) -> LabItem | None:
    payload = _widget_payload(action)
    config_kind = str(payload.get("config_kind") or "eval")
    config_path = _widget_config_path(payload, workspace)
    toml_text = _widget_toml(payload, config_path)
    if not toml_text:
        return None
    title = str(payload.get("title") or action.get("title") or config_path.stem)
    return LabItem(
        key=f"agent-widget:config:{config_kind}:{config_path}",
        section="workspace",
        title=title,
        subtitle=str(config_path),
        status=config_kind,
        metadata=(("Kind", config_kind), ("Path", str(config_path))),
        raw={
            "type": "config_file",
            "config_kind": config_kind,
            "workspace": str(workspace),
            "path": str(config_path),
            "relative_path": _relative_path(config_path, workspace),
            "toml": toml_text,
            "parsed": _safe_toml_loads(toml_text),
        },
    )


def _widget_toml(payload: dict[str, Any], config_path: Path) -> str:
    if isinstance(payload.get("toml"), str):
        return str(payload["toml"])
    config = payload.get("config")
    if isinstance(config, dict) and config:
        return toml.dumps(config).rstrip() + "\n"
    if config_path.is_file():
        try:
            return config_path.read_text(encoding="utf-8")
        except OSError:
            return ""
    return ""


def _safe_toml_loads(value: str) -> dict[str, Any]:
    try:
        parsed = toml.loads(value)
    except toml.TomlDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _widget_command_text(payload: dict[str, Any], workspace: Path) -> str:
    command = payload.get("command")
    if isinstance(command, list) and command:
        return " ".join(str(part) for part in command)
    if isinstance(command, str) and command.strip():
        return command.strip()
    config_path = _widget_config_path(payload, workspace)
    if not str(payload.get("config_path") or "").strip():
        return ""
    rel_path = _relative_path(config_path, workspace)
    config_kind = str(payload.get("config_kind") or "")
    if config_kind == "rl":
        return f"prime train run {rel_path} --yes"
    if config_kind == "eval":
        return f"prime eval run {rel_path} --hosted"
    if config_kind == "gepa":
        return f"prime gepa run {rel_path}"
    return ""


def _widget_config_path(payload: dict[str, Any], workspace: Path) -> Path:
    raw_path = str(payload.get("config_path") or "").strip()
    if not raw_path:
        config_kind = str(payload.get("config_kind") or "eval")
        if config_kind not in {"eval", "rl", "gepa"}:
            config_kind = "eval"
        raw_path = f".prime/lab/configs/{config_kind}/agent-config.toml"
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = workspace / path
    return path.resolve()


def _relative_path(path: Path, workspace: Path) -> str:
    try:
        return str(path.resolve().relative_to(workspace.resolve()))
    except ValueError:
        return str(path)


def _short_path(value: str) -> str:
    path = Path(value).expanduser()
    try:
        return f"~/{path.resolve().relative_to(Path.home())}"
    except ValueError:
        return value
