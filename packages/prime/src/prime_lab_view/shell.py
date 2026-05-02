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
        text.append("auth ?", style="dim")
        text.append(" · ", style="dim")
        text.append("team -", style="dim")
        text.append(" · ", style="dim")
        text.append("~", style="dim")
    else:
        auth_label = "auth check" if snapshot.authenticated else "auth x"
        text.append(auth_label, style=STATUS_SUCCESS if snapshot.authenticated else STATUS_ERROR)
        text.append(" · ", style="dim")
        text.append(snapshot.team or "personal", style=NEUTRAL)
        text.append(" · ", style="dim")
        text.append(compact_path(snapshot.workspace), style="dim")
        if snapshot.warnings:
            text.append(" · ", style="dim")
            text.append(f"{len(snapshot.warnings)} warnings", style=STATUS_WARNING)
    text.append("  |  ", style="dim")
    text.append("agent ", style="dim")
    text.append(agent_status_label(agent_state), style=agent_status_style(agent_state.status))
    if agent_state.endpoint:
        text.append(f" {agent_state.endpoint}", style="dim")
    return text


def agent_status_label(state: AgentConnectionState) -> str:
    if state.status == "none":
        return "none"
    label = state.label or state.agent or "agent"
    if state.status == "connected":
        return f"{label} connected"
    if state.status == "starting":
        return f"{label} starting"
    if state.status == "stopped":
        return f"{label} stopped"
    if state.message:
        return f"{label} {state.status}: {state.message}"
    return f"{label} {state.status}"


def agent_status_style(status: str) -> str:
    if status == "connected":
        return STATUS_SUCCESS
    if status == "error":
        return STATUS_ERROR
    if status in {"starting", "stopped"}:
        return STATUS_WARNING
    return "dim"


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
