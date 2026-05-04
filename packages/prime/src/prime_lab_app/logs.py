"""Training/system log parsing and rendering helpers."""

from __future__ import annotations

import json
from typing import Any

from rich.console import Group
from rich.table import Table
from rich.text import Text

from .palette import STATUS_ERROR, STATUS_INFO, STATUS_SUCCESS, STATUS_WARNING

LOG_RENDER_ROW_LIMIT = 5000


def visible_system_renderable(raw: dict[str, Any], *, visible_log_lines: int | None) -> Group:
    logs = raw.get("logs_tail")
    current_tail = _int_value(raw.get("log_tail_lines"))
    visible_logs = _visible_log_text(
        logs if isinstance(logs, str) else "",
        visible_log_lines,
    )
    return system_renderable(visible_logs, tail_lines=current_tail)


def system_renderable(value: str, *, tail_lines: int | None = None) -> Group:
    log_tail_lines = tail_lines or 1000
    logs = _tail_text(
        value,
        max_chars=min(max(16000, log_tail_lines * 320), 2_000_000),
    )
    records = parse_log_records(logs)
    if not records:
        return Group(Text("No logs available", style="dim"))
    title = Text("Logs", style="bold")
    title.append(
        f" · {log_tail_lines:,} line tail · "
        f"showing {min(len(records), LOG_RENDER_ROW_LIMIT):,}/{len(records):,}",
        style="dim",
    )
    if not any(record.get("structured") for record in records):
        return Group(title, Text(""), Text(logs))

    table = Table(show_header=True, header_style="bold dim", expand=True)
    table.add_column("Time", no_wrap=True)
    table.add_column("Level", no_wrap=True)
    table.add_column("Step", no_wrap=True)
    table.add_column("Log", ratio=1)

    for record in records[-LOG_RENDER_ROW_LIMIT:]:
        level = str(record.get("level") or "")
        table.add_row(
            str(record.get("time") or ""),
            Text(level, style=_log_level_style(level)),
            str(record.get("step") or ""),
            str(record.get("log") or ""),
        )
    return Group(title, Text(""), table)


def parse_log_records(value: str) -> list[dict[str, Any]]:
    stripped = value.strip()
    if not stripped:
        return []

    parsed = _parse_json_value(stripped)
    if isinstance(parsed, list):
        return [_log_record(record) for record in parsed]
    if isinstance(parsed, dict):
        return [_log_record(parsed)]

    records = []
    for line in stripped.splitlines():
        line = line.strip()
        if not line:
            continue
        line_value = _parse_json_value(line)
        records.append(_log_record(line_value if isinstance(line_value, dict) else line))
    return records


def _visible_log_text(value: str, visible_log_lines: int | None) -> str:
    if visible_log_lines is None:
        return value
    lines = value.splitlines()
    if visible_log_lines >= len(lines):
        return value
    if visible_log_lines <= 0:
        return ""
    return "\n".join(lines[-visible_log_lines:])


def _parse_json_value(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def _log_record(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {"message": str(value), "structured": False}

    consumed = {
        "time",
        "timestamp",
        "created_at",
        "createdAt",
        "level",
        "levelname",
        "severity",
        "message",
        "msg",
        "event",
        "text",
        "desc",
        "step",
        "global_step",
        "iteration",
        "type",
        "current",
        "total",
        "percent",
    }
    if value.get("type") == "progress":
        message = _progress_log_message(value)
    else:
        message = _first_text_field(value, "message", "msg", "event", "text", "desc")
    if message is None:
        message = json.dumps(value, sort_keys=True, default=str)
    step = _first_text_field(value, "step", "global_step", "iteration")
    fields = {
        key: child
        for key, child in value.items()
        if key not in consumed and child is not None and child != {} and child != []
    }
    return {
        "time": _first_text_field(value, "time", "timestamp", "created_at", "createdAt"),
        "level": _first_text_field(value, "level", "levelname", "severity"),
        "step": step,
        "message": message,
        "fields": _compact_fields(fields),
        "log": _join_log_parts(message, _compact_fields(fields)),
        "structured": True,
    }


def _join_log_parts(message: str, fields: str) -> str:
    if fields:
        return f"{message}  {fields}"
    return message


def _progress_log_message(value: dict[str, Any]) -> str:
    desc = str(value.get("desc") or "Progress")
    current = value.get("current")
    total = value.get("total")
    percent = value.get("percent")
    if current is not None and total is not None and percent is not None:
        return f"{desc}  {current}/{total} ({percent}%)"
    if current is not None and total is not None:
        return f"{desc}  {current}/{total}"
    return desc


def _first_text_field(value: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        field = value.get(key)
        if field is not None:
            return str(field)
    return None


def _compact_fields(value: dict[str, Any]) -> str:
    if not value:
        return ""
    parts = []
    for key, child in value.items():
        rendered = (
            json.dumps(child, sort_keys=True, default=str)
            if isinstance(child, dict | list)
            else str(child)
        )
        parts.append(f"{key}={rendered}")
    text = " ".join(parts)
    return text if len(text) <= 240 else text[:237] + "..."


def _log_level_style(level: str) -> str:
    normalized = level.upper()
    if normalized in {"ERROR", "ERR", "CRITICAL", "FATAL"}:
        return STATUS_ERROR
    if normalized in {"WARNING", "WARN"}:
        return STATUS_WARNING
    if normalized in {"INFO", "NOTICE"}:
        return STATUS_SUCCESS
    if normalized in {"DEBUG", "TRACE"}:
        return "dim"
    return STATUS_INFO


def _tail_text(value: str, *, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value or "No logs available"
    return value[-max_chars:]


def _int_value(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
