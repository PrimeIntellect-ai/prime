"""Shared action preparation for agent-requested Lab controls."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import toml

from .agent_widget_model import (
    AgentWidgetModel,
    candidate_by_id,
    candidate_label,
    coerce_optional_int,
    dedupe_single_eval_globals,
    filter_widget_empty_values,
    parse_optional_int,
    relative_path,
    widget_command_text,
    widget_config_field_name,
    widget_config_path,
    widget_display_title,
    widget_generated_config_path,
    widget_single_eval_config,
    widget_training_model_option_parts,
)
from .config_screen import (
    ConfigBuildResult,
    build_config_from_fields,
    launch_command_for_config,
)
from .toml_format import format_toml_blocks


@dataclass(frozen=True)
class AgentWidgetLaunchPlan:
    """Concrete command and metadata for an inline Lab control launch."""

    command: str
    config_kind: str
    config_path: Path | None
    follow_training_logs: bool
    errors: tuple[str, ...] = ()


def build_agent_widget_config(
    model: AgentWidgetModel,
    field_values: dict[str, str],
) -> ConfigBuildResult:
    """Build the current config represented by an agent widget model."""

    context = model.config_context
    if context is None:
        return ConfigBuildResult({}, "", ())
    model_value = field_values.get("model", str(context["values"].get("model", ""))).strip()
    model_name, model_controls = widget_training_model_option_parts(model_value)

    def field_value(field_id: str) -> str:
        field_name = widget_config_field_name(field_id)
        if not field_name:
            return ""
        if str(context["config_kind"]) == "rl":
            if field_name == "model":
                return model_name
            if field_name in {"enable_thinking", "reasoning_effort"}:
                return model_controls.get(field_name, "")
        if field_name in field_values:
            return field_values[field_name].strip()
        return str(context["values"].get(field_name, "")).strip()

    build = build_config_from_fields(
        dict(context["config"]),
        context["config_kind"],
        field_value,
    )
    build = remove_agent_training_run_name(build, str(context["config_kind"]))
    return apply_agent_widget_extra_fields(build, str(context["config_kind"]), field_values)


def prepare_agent_widget_launch(
    model: AgentWidgetModel,
    *,
    workspace: Path,
    field_values: dict[str, str],
) -> AgentWidgetLaunchPlan:
    """Return the runnable command for the current agent widget state."""

    if model.config_context is None:
        command = widget_command_text(model.payload, workspace)
        return AgentWidgetLaunchPlan(
            command=command,
            config_kind=str(model.payload.get("config_kind") or ""),
            config_path=None,
            follow_training_logs=False,
            errors=(),
        )

    build = build_agent_widget_config(model, field_values)
    if build.errors:
        return AgentWidgetLaunchPlan(
            command="",
            config_kind=str(model.config_context["config_kind"]),
            config_path=None,
            follow_training_logs=False,
            errors=build.errors,
        )

    config_kind = str(model.config_context["config_kind"])
    config_path = widget_generated_config_path(
        model.action,
        workspace,
        config_kind,
        model.config_context["config_path"],
    )
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(build.toml_text.rstrip() + "\n", encoding="utf-8")
    rel_path = relative_path(config_path, workspace)
    command = launch_command_for_config(config_kind, rel_path)
    if config_kind == "eval" and widget_single_eval_config(build.parsed) is not None:
        command += " --follow"
    return AgentWidgetLaunchPlan(
        command=command,
        config_kind=config_kind,
        config_path=config_path,
        follow_training_logs=config_kind == "rl",
        errors=(),
    )


def agent_widget_launch_action(
    model: AgentWidgetModel,
    *,
    command: str,
    workspace: Path,
    status: str,
    returncode: int | None = None,
    run_id: str = "",
) -> dict[str, Any]:
    """Return a durable action log record for a widget launch state."""

    config_path = ""
    config_kind = str(model.payload.get("config_kind") or "")
    if model.config_context is not None:
        config_kind = str(model.config_context["config_kind"])
        config_path = relative_path(
            widget_generated_config_path(
                model.action,
                workspace,
                config_kind,
                model.config_context["config_path"],
            ),
            workspace,
        )
    elif model.payload.get("config_path"):
        config_path = relative_path(widget_config_path(model.payload, workspace), workspace)
    action: dict[str, Any] = {
        "type": "agent_inline_launch",
        "status": status,
        "kind": str(model.action.get("kind") or model.payload.get("kind") or ""),
        "title": widget_display_title(model.action, model.config_context),
        "config_kind": config_kind,
        "config_path": config_path,
        "command": command,
        "workspace": str(workspace),
    }
    if returncode is not None:
        action["returncode"] = returncode
    if run_id:
        action["run_id"] = run_id
    return action


def agent_widget_choice_action(
    model: AgentWidgetModel,
    *,
    choice_id: str,
    workspace: Path,
) -> tuple[str, dict[str, Any]]:
    """Return the selected label and durable action for a choice widget."""

    choice = candidate_by_id(model.payload.get("candidates"), choice_id)
    label = candidate_label(choice, choice_id)
    return label, {
        "type": "agent_widget_choice_selected",
        "widget_id": str(model.action.get("widget_id") or ""),
        "kind": "choice_picker",
        "title": widget_display_title(model.action, model.config_context),
        "choice_id": choice_id,
        "choice_label": label,
        "workspace": str(workspace),
    }


def apply_agent_widget_extra_fields(
    build: ConfigBuildResult,
    config_kind: str,
    field_values: dict[str, str],
) -> ConfigBuildResult:
    """Apply compact widget-only fields on top of the shared config form builder."""

    if config_kind != "eval":
        return build
    config = dict(build.parsed)
    errors = list(build.errors)
    if "rollouts_per_example" in field_values:
        parsed_rollouts = coerce_optional_int(field_values["rollouts_per_example"])
        if parsed_rollouts is not None:
            config["rollouts_per_example"] = parsed_rollouts
        elif not field_values["rollouts_per_example"]:
            config.pop("rollouts_per_example", None)
    if "num_examples" in field_values:
        parsed_examples = parse_optional_int("examples", field_values["num_examples"], errors)
        if parsed_examples is None:
            config.pop("num_examples", None)
        else:
            config["num_examples"] = parsed_examples
    if "max_concurrent" in field_values:
        parsed_concurrent = parse_optional_int(
            "max concurrent",
            field_values["max_concurrent"],
            errors,
        )
        if parsed_concurrent is None:
            config.pop("max_concurrent", None)
        else:
            config["max_concurrent"] = parsed_concurrent
    config["save_results"] = True
    if config.get("model"):
        config.pop("endpoint_id", None)
    dedupe_single_eval_globals(config)
    toml_text = format_toml_blocks(toml.dumps(filter_widget_empty_values(config))).rstrip()
    return ConfigBuildResult(config, toml_text, tuple(errors))


def remove_agent_training_run_name(
    build: ConfigBuildResult,
    config_kind: str,
) -> ConfigBuildResult:
    """Keep agent-generated Hosted Training configs on the platform-generated run name path."""

    if config_kind != "rl" or "name" not in build.parsed:
        return build
    config = dict(build.parsed)
    config.pop("name", None)
    toml_text = format_toml_blocks(toml.dumps(filter_widget_empty_values(config))).rstrip()
    return ConfigBuildResult(config, toml_text, build.errors)
