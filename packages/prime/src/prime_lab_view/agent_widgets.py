"""Declarative Lab widget tools exposed to coding agents."""

from __future__ import annotations

import json
from typing import Any

LAB_WIDGET_NAMESPACE = "lab"
LAB_WIDGET_TOOL = "render_widget"

LAB_WIDGET_KINDS = (
    "choice_picker",
    "config_editor",
    "action_preview",
    "file_patch_summary",
    "run_launcher",
    "rollout_insight",
)


def lab_dynamic_tools() -> list[dict[str, Any]]:
    """Dynamic tool specs passed to Codex app-server threads."""

    return [
        {
            "namespace": LAB_WIDGET_NAMESPACE,
            "name": LAB_WIDGET_TOOL,
            "description": (
                "Request a native Lab UI widget. Use this when the user needs to choose, edit, "
                "confirm, or inspect a Lab action instead of replying with plain text."
            ),
            "inputSchema": {
                "type": "object",
                "required": ["kind", "title"],
                "properties": {
                    "kind": {
                        "type": "string",
                        "enum": list(LAB_WIDGET_KINDS),
                    },
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "candidates": {
                        "type": "array",
                        "items": {"type": "object"},
                    },
                    "fields": {"type": "object"},
                    "actions": {
                        "type": "array",
                        "items": {"type": "object"},
                    },
                    "metadata": {"type": "object"},
                },
                "additionalProperties": True,
            },
        }
    ]


def lab_widget_developer_instructions() -> str:
    """Instructions that teach the agent how Lab widget requests work."""

    return (
        "You are running inside Prime Intellect Lab. When a user asks to choose an environment, "
        "edit or launch a config, inspect runs, compare options, or confirm a side effect, call "
        "`lab.render_widget` with a declarative widget request instead of inventing terminal UI. "
        "Lab owns rendering, validation, confirmation, and execution. Keep widget payloads small "
        "and use plain JSON-compatible values."
    )


def handle_lab_widget_tool_call(params: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    """Validate and convert a Codex dynamic tool call into chat text and tool output."""

    namespace = params.get("namespace")
    tool = params.get("tool")
    if namespace != LAB_WIDGET_NAMESPACE or tool != LAB_WIDGET_TOOL:
        return _tool_error("Unsupported Lab dynamic tool.")

    arguments = _coerce_arguments(params.get("arguments"))
    if not isinstance(arguments, dict):
        return _tool_error("Widget arguments must be an object.")

    kind = str(arguments.get("kind") or "").strip()
    title = str(arguments.get("title") or "").strip()
    if kind not in LAB_WIDGET_KINDS:
        return _tool_error(f"Unsupported widget kind: {kind or '<empty>'}.")
    if not title:
        return _tool_error("Widget title is required.")

    widget_id = str(params.get("callId") or "")
    description = str(arguments.get("description") or "").strip()
    summary = _widget_summary(widget_id, kind, title, description, arguments)
    output = {
        "ok": True,
        "widgetId": widget_id,
        "kind": kind,
        "title": title,
        "message": "Lab accepted the widget request and displayed it to the user.",
    }
    return (
        "widget",
        summary,
        {
            "success": True,
            "contentItems": [{"type": "inputText", "text": json.dumps(output, sort_keys=True)}],
        },
    )


def _coerce_arguments(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None
    return value


def _tool_error(message: str) -> tuple[str, str, dict[str, Any]]:
    return (
        "error",
        f"Widget request failed\n{message}",
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
    lines = ["Widget requested", f"Kind   {kind}", f"Title  {title}"]
    if widget_id:
        lines.append(f"ID     {widget_id}")
    if description:
        lines.extend(("", description))
    for key in ("candidates", "fields", "actions"):
        value = arguments.get(key)
        count = len(value) if isinstance(value, (list, dict)) else 0
        if count:
            lines.append(f"{key.capitalize()}  {count}")
    return "\n".join(lines)
