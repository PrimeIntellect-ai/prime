"""Declarative Lab control tools exposed to coding agents."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .agent_widget_titles import clean_widget_title

LAB_WIDGET_NAMESPACE = "lab"

LAB_WIDGET_KINDS = (
    "choice_picker",
    "config_editor",
    "action_preview",
    "file_patch_summary",
    "run_launcher",
    "rollout_insight",
)
LAB_WIDGET_DIAGNOSTIC_TITLE = "Lab tool diagnostic"
EnvironmentSearch = Callable[[str, int], list[dict[str, str]]]


@dataclass(frozen=True)
class LabWidgetTool:
    """One named Lab control intent exposed to coding agents."""

    name: str
    kind: str
    description: str
    required: tuple[str, ...]
    properties: dict[str, Any]
    common: bool = True


LAB_WIDGET_TOOLS = (
    LabWidgetTool(
        name="choose",
        kind="choice_picker",
        description=(
            "Ask the user to choose from Lab objects or next actions. Use for ambiguous "
            "environments, configs, runs, evals, models, workspaces, or launch choices."
        ),
        required=("title", "candidates"),
        properties={
            "prompt": {"type": "string"},
            "candidates": {"type": "array", "items": {"type": "object"}},
            "allow_multiple": {"type": "boolean"},
            "default_id": {"type": "string"},
        },
    ),
    LabWidgetTool(
        name="search_environments",
        kind="environment_search",
        description=(
            "Search platform environments through the Prime CLI. Use before creating eval or "
            "training configs when the user names an environment."
        ),
        required=(),
        properties={
            "query": {"type": "string"},
            "limit": {"type": "integer"},
        },
        common=False,
    ),
    LabWidgetTool(
        name="train_model",
        kind="run_launcher",
        description=(
            "Open a native hosted-training config editor from explicit training fields. "
            "Use this for RL training requests after resolving an environment with "
            "`search_environments`."
        ),
        required=(
            "env",
            "model",
            "max_steps",
            "batch_size",
            "rollouts_per_example",
            "max_tokens",
        ),
        properties={
            "env": {
                "type": "string",
                "description": "Platform environment in owner/name form.",
                "pattern": "^[^/]+/[^/]+$",
            },
            "model": {"type": "string"},
            "max_steps": {"type": "integer", "minimum": 1},
            "batch_size": {"type": "integer", "minimum": 1},
            "rollouts_per_example": {"type": "integer", "minimum": 1},
            "max_tokens": {"type": "integer", "minimum": 1},
        },
        common=False,
    ),
    LabWidgetTool(
        name="edit_config",
        kind="config_editor",
        description=(
            "Open a native Lab config editor. Use when the user asks to create, tweak, "
            "clone, rerun, evaluate, train, or modify a config."
        ),
        required=("title", "config_kind"),
        properties={
            "config_kind": {"type": "string", "enum": ["eval", "rl", "gepa"]},
            "config": {"type": "object"},
            "source": {"type": "object"},
            "editable_fields": {"type": "array", "items": {"type": "string"}},
            "read_only_fields": {"type": "array", "items": {"type": "string"}},
            "defaults": {"type": "object"},
            "validation": {"type": "object"},
        },
    ),
    LabWidgetTool(
        name="preview_action",
        kind="action_preview",
        description=(
            "Show a side-effect preview with validation and confirm/cancel controls. Use before "
            "writes, syncs, installs, launches, pushes, deletes, or agent-applied remediations."
        ),
        required=("title", "actions"),
        properties={
            "actions": {"type": "array", "items": {"type": "object"}},
            "side_effects": {"type": "array", "items": {"type": "string"}},
            "validation": {"type": "object"},
            "requires_confirmation": {"type": "boolean"},
        },
    ),
    LabWidgetTool(
        name="launch_run",
        kind="run_launcher",
        description=(
            "Launch or rerun an evaluation/training job from a selected config. Use when the "
            "user is ready to run inside Lab and should see a native launch preview plus "
            "live-log handoff."
        ),
        required=("title", "config_kind"),
        properties={
            "config_kind": {"type": "string", "enum": ["eval", "rl", "gepa"]},
            "config_path": {"type": "string"},
            "config": {"type": "object"},
            "command": {"type": "array", "items": {"type": "string"}},
            "source_run_id": {"type": "string"},
        },
    ),
    LabWidgetTool(
        name="show_patch",
        kind="file_patch_summary",
        description=(
            "Show file changes or proposed source edits. Use after creating/editing environment "
            "code, configs, README content, skills, docs, or setup assets."
        ),
        required=("title", "files"),
        properties={
            "files": {"type": "array", "items": {"type": "object"}},
            "summary": {"type": "string"},
            "risk_notes": {"type": "array", "items": {"type": "string"}},
            "next_actions": {"type": "array", "items": {"type": "object"}},
        },
    ),
    LabWidgetTool(
        name="inspect_rollouts",
        kind="rollout_insight",
        description=(
            "Open rollout/eval sample inspection with selected failures, metrics, or examples. "
            "Use when diagnosing training/eval behavior or summarizing qualitative issues."
        ),
        required=("title",),
        properties={
            "run_id": {"type": "string"},
            "eval_id": {"type": "string"},
            "sample_ids": {"type": "array", "items": {"type": "string"}},
            "filters": {"type": "object"},
            "failure_categories": {"type": "array", "items": {"type": "object"}},
            "proposed_next_action": {"type": "object"},
        },
    ),
)
_TOOL_BY_NAME = {tool.name: tool for tool in LAB_WIDGET_TOOLS}


def lab_dynamic_tools() -> list[dict[str, Any]]:
    """Dynamic tool specs passed to Codex app-server threads."""

    training_models = _current_training_model_names()
    return [_tool_spec(tool, training_models=training_models) for tool in LAB_WIDGET_TOOLS]


def lab_widget_developer_instructions() -> str:
    """Instructions that teach the agent how Lab control requests work."""

    training_model_guidance = _training_model_guidance(_current_training_model_names())
    return (
        "You are running inside Prime Intellect Lab. Apply the Prime-managed Lab controls "
        "guidance for this session. Lab is an interactive terminal research app, so your "
        "default visible action surface is native Lab tools whenever the request involves "
        "choosing, editing, "
        "launching, syncing, inspecting, confirming, or comparing Lab objects. The user is using "
        "the Lab app by default, not the shell. Do not suggest CLI commands or tell the user to "
        "run commands unless they explicitly ask for CLI instructions. Do not describe a "
        "config or command when the user should be able to click, edit, or confirm it in Lab. "
        "Do not narrate repository searches, file formats, docs, folders, resolver order, TOML "
        "shape, or other implementation details unless the user explicitly asks. Your prose "
        "should stay product-facing: what is ready, what needs a decision, or what changed. "
        "Do not name internal implementation surfaces, tool plumbing, or rendering mechanics. "
        "Do not narrate that you are checking, reading, searching, or inspecting workspace "
        "files, configs, folders, docs, or resolver state; either call the native Lab tool or "
        "state the user-facing decision needed. Avoid implementation-facing words in visible "
        "chat such as workflow, skill, template, resolver, workspace file, or config shape. "
        "Use the smallest specific native Lab tool that matches the next decision: "
        "`choose` for "
        "ambiguity, `search_environments` to resolve platform environments, `train_model` for "
        "hosted-training config creation, `edit_config` for eval/GEPA config creation or tweaks, "
        "`preview_action` for side effects, `launch_run` for launch handoff, `show_patch` for "
        "code/config file changes, and `inspect_rollouts` for sample or metric diagnosis. "
        "For requests like make, create, set up, or run an eval, do not stop after searching "
        "for the environment: call `edit_config` with `config_kind` `eval` as soon as the "
        "environment and model are known, or call `choose` if either is ambiguous. For training "
        "requests, call `train_model` once environment, model, and core run limits are known, "
        "or `choose` for missing/ambiguous choices. "
        f"{training_model_guidance}"
        "Before calling a widget tool, decide: the Lab object kind, "
        "candidate IDs/paths, the default selection, editable versus read-only fields, "
        "validation blockers, and the next confirmed action. Lab owns rendering, validation, "
        "confirmation, execution, and result logging. After calling a native Lab tool, do not "
        "narrate that a control appeared, do not restate the full config, and do not ask the "
        "user to open another page; let the embedded control carry the action details. "
        "Keep payloads small and JSON-compatible."
    )


def lab_widget_diagnostic_prompt() -> str:
    """Prompt the active agent to exercise the native Lab tool contract once."""

    return (
        "Run the Lab native tool diagnostic now. Call the native Lab `choose` tool exactly once "
        f'with title "{LAB_WIDGET_DIAGNOSTIC_TITLE}", prompt "Lab tools ready.", '
        'candidates [{"id":"ok","label":"Lab tools connected"}], and default_id "ok". '
        "Do not provide additional text."
    )


def handle_lab_widget_tool_call(
    params: dict[str, Any],
    *,
    environment_search: EnvironmentSearch | None = None,
) -> tuple[str, str, dict[str, Any]]:
    """Validate and convert a Codex dynamic tool call into chat text and tool output."""

    namespace = params.get("namespace")
    tool = params.get("tool")
    if namespace != LAB_WIDGET_NAMESPACE:
        return _tool_error("Unsupported Lab dynamic tool.")

    arguments = _coerce_arguments(params.get("arguments"))
    if not isinstance(arguments, dict):
        return _tool_error("Action arguments must be an object.")
    if tool == "search_environments":
        return _handle_search_environments(arguments, environment_search)
    if tool == "train_model":
        model_error = _validate_train_model_model(arguments)
        if model_error:
            return _tool_error(model_error)

    normalized = normalize_widget_arguments(str(tool or ""), arguments)
    if normalized is None:
        return _tool_error(f"Unsupported Lab dynamic tool: {tool or '<empty>'}.")

    kind = str(normalized.get("kind") or "").strip()
    title = str(normalized.get("title") or "").strip()
    if kind not in LAB_WIDGET_KINDS:
        return _tool_error(f"Unsupported action kind: {kind or '<empty>'}.")
    if not title:
        return _tool_error("Widget title is required.")

    widget_id = str(params.get("callId") or "")
    description = str(arguments.get("description") or "").strip()
    summary = _widget_summary(widget_id, kind, title, description, normalized)
    output = {
        "ok": True,
        "widgetId": widget_id,
        "kind": kind,
        "title": title,
        "tool": str(tool or ""),
        "message": "displayed",
        "nextInstruction": (
            "The Lab app rendered the native control. Do not summarize the control, "
            "describe CLI steps, or ask the user to open another page."
        ),
    }
    return (
        "widget",
        summary,
        {
            "success": True,
            "contentItems": [{"type": "inputText", "text": json.dumps(output, sort_keys=True)}],
        },
    )


def normalize_widget_arguments(tool: str, arguments: dict[str, Any]) -> dict[str, Any] | None:
    """Normalize a named Lab tool call to the canonical widget payload."""

    tool = tool.strip()
    spec = _TOOL_BY_NAME.get(tool)
    if spec is None:
        return None
    normalized = dict(arguments)
    if tool == "choose":
        if "candidates" not in normalized and isinstance(normalized.get("options"), list):
            normalized["candidates"] = normalized["options"]
        if not str(normalized.get("title") or "").strip():
            normalized["title"] = str(normalized.get("prompt") or "Choose")
    if tool == "train_model":
        normalized = _normalize_train_model_arguments(normalized)
    return {**normalized, "kind": spec.kind, "tool": tool}


def _normalize_train_model_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    env_id = str(arguments.get("env") or "").strip()
    config: dict[str, Any] = {
        "model": str(arguments.get("model") or "").strip(),
        "max_steps": arguments.get("max_steps"),
        "batch_size": arguments.get("batch_size"),
        "rollouts_per_example": arguments.get("rollouts_per_example"),
        "env": [{"id": env_id}],
        "sampling": {"max_tokens": arguments.get("max_tokens")},
    }
    return {
        "title": f"Train {env_id.rsplit('/', 1)[-1] or 'model'}",
        "config_kind": "rl",
        "config": config,
    }


def _handle_search_environments(
    arguments: dict[str, Any],
    environment_search: EnvironmentSearch | None,
) -> tuple[str, str, dict[str, Any]]:
    query = str(arguments.get("query") or "").strip()
    limit = _positive_limit(arguments.get("limit"), default=12, maximum=30)
    results = environment_search(query, limit) if environment_search is not None else []
    output = {"ok": True, "query": query, "environments": results}
    return (
        "tool",
        f"Environment search\n{len(results)} result(s)",
        {
            "success": True,
            "contentItems": [{"type": "inputText", "text": json.dumps(output, sort_keys=True)}],
        },
    )


def _positive_limit(value: Any, *, default: int, maximum: int) -> int:
    if isinstance(value, bool):
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(1, min(maximum, parsed))


def lab_widget_action_from_tool_call(params: dict[str, Any]) -> dict[str, Any]:
    """Build a durable action event from a widget tool call."""

    arguments = _coerce_arguments(params.get("arguments"))
    if not isinstance(arguments, dict):
        arguments = {}
    normalized = normalize_widget_arguments(str(params.get("tool") or ""), arguments) or arguments
    return {
        "type": "widget_requested",
        "source": "native_tool",
        "tool": str(params.get("tool") or ""),
        "widget_id": str(params.get("callId") or ""),
        "kind": str(normalized.get("kind") or ""),
        "title": str(normalized.get("title") or ""),
        "payload": normalized,
    }


def _coerce_arguments(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None
    return value


def _tool_spec(
    tool: LabWidgetTool,
    *,
    training_models: tuple[str, ...] = (),
) -> dict[str, Any]:
    properties = dict(tool.properties)
    if tool.name == "train_model" and training_models:
        model_property = dict(properties.get("model") or {})
        model_property["enum"] = list(training_models)
        model_property["description"] = (
            "Hosted Training model id. Use exactly one currently available id from this enum."
        )
        properties["model"] = model_property
    if tool.common:
        properties = {
            "title": {"type": "string"},
            "description": {"type": "string"},
            **properties,
        }
    return {
        "namespace": LAB_WIDGET_NAMESPACE,
        "name": tool.name,
        "description": tool.description,
        "inputSchema": {
            "type": "object",
            "required": list(tool.required),
            "properties": properties,
            "additionalProperties": False,
        },
    }


def _validate_train_model_model(arguments: dict[str, Any]) -> str:
    model = str(arguments.get("model") or "").strip()
    training_models = _current_training_model_names()
    if not model or not training_models or model in set(training_models):
        return ""
    return (
        f"Hosted Training model '{model}' is not available. Available model ids: "
        f"{_format_training_models(training_models)}"
    )


def _training_model_guidance(training_models: tuple[str, ...]) -> str:
    if not training_models:
        return ""
    return (
        "Available Hosted Training model ids for `train_model.model`: "
        f"{_format_training_models(training_models)}. Use one exactly as written; do not invent "
        "short aliases or future model names. "
    )


def _format_training_models(training_models: tuple[str, ...]) -> str:
    return ", ".join(training_models)


def _current_training_model_names() -> tuple[str, ...]:
    try:
        from .agent_widget_model import training_model_names

        return training_model_names()
    except Exception:
        return ()


def _tool_error(message: str) -> tuple[str, str, dict[str, Any]]:
    return (
        "error",
        f"Action request failed\n{message}",
        {
            "success": False,
            "contentItems": [
                {
                    "type": "inputText",
                    "text": json.dumps({"ok": False, "error": message}, sort_keys=True),
                }
            ],
        },
    )


def _widget_summary(
    widget_id: str,
    kind: str,
    title: str,
    description: str,
    arguments: dict[str, Any],
) -> str:
    visible_title = clean_widget_title(title)
    lines = [visible_title]
    if widget_id:
        lines.append(f"ID {widget_id}")
    if description:
        lines.extend(("", description))
    for key in ("candidates", "fields", "actions"):
        value = arguments.get(key)
        count = len(value) if isinstance(value, (list, dict)) else 0
        if count:
            lines.append(f"{key.capitalize()}  {count}")
    return "\n".join(lines)
