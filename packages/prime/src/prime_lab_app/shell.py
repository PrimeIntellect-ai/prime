"""Shell/status helpers shared by Lab screens."""

from __future__ import annotations

import json
from pathlib import Path

from rich.table import Table
from rich.text import Text

from .agent_runtime import AgentConnectionState
from .models import LabSnapshot
from .palette import NEUTRAL, PRIMARY, STATUS_ERROR, STATUS_SUCCESS, STATUS_WARNING


def lab_mark_text() -> Text:
    text = Text()
    text.append("L A B", style=f"bold {PRIMARY}")
    return text


def lab_logo_text() -> Text:
    text = Text()
    text.append("PRIME", style="bold white")
    text.append(" Intellect", style="italic white")
    return text


def lab_header(title: Text | str | None = None) -> Table:
    left = Text()
    left.append_text(lab_mark_text())
    if title is not None:
        left.append("  ", style="dim")
        if isinstance(title, Text):
            left.append_text(title)
        else:
            left.append(str(title), style="bold")
    header = Table.grid(expand=True)
    header.add_column(ratio=1)
    header.add_column(justify="right", no_wrap=True)
    header.add_row(left, lab_logo_text())
    return header


def statusbar_text(
    snapshot: LabSnapshot | None,
    agent_state: AgentConnectionState,
) -> Text:
    text = Text()
    if snapshot is None:
        text.append("?", style="dim")
        text.append(" -", style=NEUTRAL)
    else:
        indicator = "✓" if snapshot.authenticated else "×"
        indicator_style = STATUS_SUCCESS if snapshot.authenticated else STATUS_ERROR
        text.append(indicator, style=indicator_style)
        text.append(f" {status_identity(snapshot)}", style=NEUTRAL)
        text.append(" · ", style="dim")
        text.append(compact_path(snapshot.workspace), style="dim")
        if snapshot.warnings:
            text.append(" · ", style="dim")
            text.append(_count_label(len(snapshot.warnings), "warning"), style=STATUS_WARNING)

    agent_text = agent_status_text(agent_state)
    if agent_text.plain != "none":
        text.append("  |  ", style="dim")
        text.append_text(agent_text)
        if agent_state.endpoint:
            text.append(f" {agent_state.endpoint}", style="dim")
    return text


def action_hint_text(*pairs: tuple[str, str]) -> Text:
    """Render footer action hints as colored key labels plus muted action names."""
    text = Text()
    for index, (key, label) in enumerate(pairs):
        if index:
            text.append("  ·  ", style="dim")
        text.append(key, style=f"bold {PRIMARY}")
        if label:
            text.append(f" {label}", style="dim")
    return text


def warning_popover_text(
    warnings: tuple[str, ...],
    *,
    include_doctor_hint: bool = False,
) -> Text:
    text = Text()
    text.append("Warnings", style=STATUS_WARNING)
    if include_doctor_hint:
        text.append("  ", style="dim")
        text.append("Open workspace settings and run Doctor for deterministic fixes.", style="dim")
    for index, warning in enumerate(warnings, start=1):
        lines = _warning_lines(warning)
        text.append("\n")
        text.append(f"{index}. ", style=STATUS_WARNING)
        text.append(lines[0])
        for line in lines[1:]:
            text.append("\n   ", style="dim")
            text.append(line)
    return text


def _count_label(count: int, singular: str) -> str:
    suffix = "" if count == 1 else "s"
    return f"{count} {singular}{suffix}"


def _warning_lines(warning: str) -> tuple[str, ...]:
    lines = tuple(line.strip() for line in str(warning).splitlines() if line.strip())
    return lines or ("Unknown warning",)


def agent_status_label(state: AgentConnectionState) -> str:
    if state.status == "none":
        return "none"
    label = state.label or state.agent or "agent"
    if state.status == "connected":
        return f"✓ {label}"
    if state.status == "starting":
        return f"… {label}"
    if state.status == "unsupported":
        return f"! {label}"
    if state.status == "stopped":
        return f"× {label}"
    if state.message:
        return f"× {label}: {state.message}"
    return f"× {label}"


def agent_status_text(state: AgentConnectionState) -> Text:
    """Render compact agent status as an indicator plus agent label."""
    if state.status == "none":
        return Text("none", style="dim")
    label = state.label or state.agent or "agent"
    text = Text()
    if state.status == "connected":
        text.append("✓", style=STATUS_SUCCESS)
    elif state.status == "starting":
        text.append("…", style=STATUS_WARNING)
    elif state.status == "unsupported":
        text.append("!", style=STATUS_WARNING)
    else:
        text.append("×", style=STATUS_ERROR if state.status == "error" else STATUS_WARNING)
    text.append(f" {label}", style=NEUTRAL)
    if state.status in {"error", "unsupported"} and state.message:
        text.append(f": {state.message}", style="dim")
    return text


def agent_status_style(status: str) -> str:
    if status == "connected":
        return STATUS_SUCCESS
    if status == "error":
        return STATUS_ERROR
    if status in {"starting", "stopped", "unsupported"}:
        return STATUS_WARNING
    return "dim"


def status_identity(snapshot: LabSnapshot) -> str:
    context = _active_workspace_context(snapshot)
    team = context.get("team") or snapshot.team
    profile = context.get("user_name") or context.get("username") or context.get("profile")
    if team:
        return str(team)
    if profile:
        return str(profile)
    for key in ("user_id",):
        value = context.get(key)
        if value:
            return str(value)
    return "personal"


def _active_workspace_context(snapshot: LabSnapshot) -> dict[str, object]:
    section = snapshot.section("workspace")
    if section is None:
        return {}
    for item in section.items:
        if item.raw.get("type") == "workspace_context" and item.raw.get("active") is True:
            return item.raw
    return {}


def compact_path(path: Path) -> str:
    resolved = path.expanduser().resolve()
    home = Path.home().resolve()
    try:
        relative = resolved.relative_to(home)
    except ValueError:
        parts = resolved.parts
        if len(parts) <= 3:
            return str(resolved)
        return f".../{'/'.join(parts[-2:])}"
    home_path = f"~/{relative}" if str(relative) != "." else "~"
    if len(home_path) <= 44:
        return home_path
    parts = relative.parts
    if len(parts) <= 2:
        return f"~/{relative}"
    return f"~/.../{'/'.join(parts[-2:])}"


def configured_workspace_agent(snapshot: LabSnapshot) -> str:
    section = snapshot.section("workspace")
    if section is None:
        return ""
    for item in section.items:
        if item.raw.get("type") == "workspace_context" and item.raw.get("active") is True:
            choices = item.raw.get("choices")
            if isinstance(choices, dict):
                agent = choices.get("primary_agent")
                return str(agent) if agent else ""
    return ""


def write_workspace_agent_choice(workspace: Path, agent: str) -> None:
    workspace = workspace.expanduser().resolve()
    metadata_path = workspace / ".prime" / "lab.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        data = {}
    if not isinstance(data, dict):
        data = {}
    data.setdefault("setup_source", "prime lab")
    choices = data.get("choices")
    if not isinstance(choices, dict):
        choices = {}
        data["choices"] = choices
    choices["primary_agent"] = agent
    agents = choices.get("agents")
    if not isinstance(agents, list):
        agents = []
    if agent not in agents:
        agents.insert(0, agent)
    choices["agents"] = agents
    metadata_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
