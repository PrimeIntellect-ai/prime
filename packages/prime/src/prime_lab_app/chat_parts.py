"""Lab-native chat message parts and renderers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

from rich.align import Align
from rich.console import Group, RenderableType
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .agent_runtime import AgentChatMessage, AgentConnectionState
from .agent_widget_titles import clean_widget_title
from .palette import PRIMARY, STATUS_ERROR, STATUS_SUCCESS, STATUS_WARNING, SUCCESS

ReferenceKind = Literal["environment", "config", "run", "eval", "file"]

_REFERENCE_RE = re.compile(r"@(?P<kind>env|environment|config|run|eval|file):(?P<id>[^\s,;]+)")


@dataclass(frozen=True)
class ChatPart:
    """Base class for a Lab chat render part."""


@dataclass(frozen=True)
class TextPart(ChatPart):
    """Plain text content."""

    text: str


@dataclass(frozen=True)
class MarkdownPart(ChatPart):
    """Markdown content."""

    markdown: str
    complete: bool = True


@dataclass(frozen=True)
class ReferencePart(ChatPart):
    """Structured Lab object reference embedded in a chat turn."""

    ref_type: ReferenceKind
    ref_id: str
    label: str


@dataclass(frozen=True)
class ActionPart(ChatPart):
    """Structured Lab-native action emitted in a chat turn."""

    action: str
    payload: dict[str, Any]


def message_parts(message: AgentChatMessage) -> tuple[ChatPart, ...]:
    """Parse an agent runtime message into Lab chat parts."""

    content = message.content or ""
    if message.role == "assistant":
        return (MarkdownPart(content or " ", complete=message.status != "streaming"),)
    parts: list[ChatPart] = []
    cursor = 0
    for match in _REFERENCE_RE.finditer(content):
        if match.start() > cursor:
            parts.append(TextPart(content[cursor : match.start()]))
        ref_type = _reference_kind(match.group("kind"))
        ref_id = match.group("id")
        parts.append(ReferencePart(ref_type=ref_type, ref_id=ref_id, label=f"@{ref_id}"))
        cursor = match.end()
    if cursor < len(content) or not parts:
        parts.append(TextPart(content[cursor:] if content else ""))
    return tuple(parts)


def chat_transcript(
    messages: tuple[AgentChatMessage, ...],
    state: AgentConnectionState,
) -> RenderableType:
    """Render a complete Lab chat transcript."""

    if not messages:
        text = Text(justify="center")
        if state.status == "none":
            text.append("No coding agent configured.", style="dim")
        elif state.status == "connected":
            text.append("Ask the agent to design, edit, evaluate, or launch.", style="dim")
        elif state.status == "unsupported":
            text.append(state.message or _connection_label(state), style=STATUS_WARNING)
        else:
            text.append(_connection_label(state), style=_connection_style(state.status))
        return Align.center(text, vertical="middle")

    renderables: list[RenderableType] = []
    for index, message in enumerate(messages):
        if index:
            renderables.append(_turn_gap(index))
        renderables.append(render_chat_turn(message))
    return Group(*renderables)


def render_chat_turn(message: AgentChatMessage) -> RenderableType:
    """Render one Lab chat turn from parsed parts."""

    if message.status == "widget":
        return _render_widget_turn(message)
    parts = message_parts(message)
    if message.role == "user":
        return _render_user_turn(parts)
    if message.role == "assistant":
        return _render_agent_turn(parts, streaming=message.status == "streaming")
    return _render_system_turn(parts, status=message.status)


def _render_user_turn(parts: tuple[ChatPart, ...]) -> Table:
    table = Table.grid(expand=True)
    table.add_column(width=2, no_wrap=True)
    table.add_column(ratio=1)
    body = Text("› ", style="#d8d7cf")
    body.append_text(_parts_text(parts))
    table.add_row(Text("┃", style=PRIMARY), body)
    return table


def _render_agent_turn(parts: tuple[ChatPart, ...], *, streaming: bool) -> Table:
    table = Table.grid(expand=True)
    table.add_column(width=2, no_wrap=True)
    table.add_column(ratio=1)
    style = SUCCESS if streaming else STATUS_SUCCESS
    table.add_row(Text("│", style=style), _parts_renderable(parts, streaming=streaming))
    return table


def _render_system_turn(parts: tuple[ChatPart, ...], *, status: str) -> Table:
    table = Table.grid(expand=True)
    table.add_column(width=2, no_wrap=True)
    table.add_column(ratio=1)
    if status == "error":
        style = STATUS_ERROR
        marker = "!"
    elif status == "tool":
        style = "dim"
        marker = "·"
    else:
        style = STATUS_WARNING
        marker = "·"
    heading = "tool" if status == "tool" else status or "notice"
    body = Text.assemble((heading, style), "\n", _parts_text(parts))
    table.add_row(Text(marker, style=style), body)
    return table


def _render_widget_turn(message: AgentChatMessage) -> Panel:
    payload = message.metadata.get("payload")
    action = payload if isinstance(payload, dict) else message.metadata
    title = clean_widget_title(str(action.get("title") or "Action"))
    description = str(action.get("description") or "").strip()

    body = Table.grid(padding=(0, 2))
    body.add_column(style="bold dim", no_wrap=True)
    body.add_column()
    if config_path := str(action.get("config_path") or "").strip():
        body.add_row("Path", config_path)
    if description:
        body.add_row("Summary", description)
    if not body.row_count:
        body.add_row("Status", "Ready")

    heading = Text.assemble((title, "bold"))
    return Panel(
        Group(heading, Text(""), body),
        border_style=SUCCESS,
        padding=(1, 2),
    )


def _parts_renderable(parts: tuple[ChatPart, ...], *, streaming: bool) -> RenderableType:
    if len(parts) == 1 and isinstance(parts[0], MarkdownPart):
        part = parts[0]
        if streaming and _markdown_is_unbalanced(part.markdown):
            return Text(part.markdown)
        return Markdown(part.markdown or " ", code_theme="nord-darker", hyperlinks=True)
    return _parts_text(parts)


def _parts_text(parts: tuple[ChatPart, ...]) -> Text:
    text = Text()
    for part in parts:
        if isinstance(part, TextPart):
            text.append(part.text)
        elif isinstance(part, ReferencePart):
            text.append(part.label, style=PRIMARY)
        elif isinstance(part, ActionPart):
            text.append(part.action, style=SUCCESS)
        elif isinstance(part, MarkdownPart):
            text.append(part.markdown)
    return text


def _turn_gap(index: int) -> Text:
    pattern = "· · ·" if index % 2 else "· ·"
    return Text(f"\n{pattern}\n", style="dim")


def _reference_kind(value: str) -> ReferenceKind:
    if value == "env":
        return "environment"
    if value in {"environment", "config", "run", "eval", "file"}:
        return value
    return "file"


def _markdown_is_unbalanced(value: str) -> bool:
    return value.count("```") % 2 == 1


def _connection_label(state: AgentConnectionState) -> str:
    if state.status == "none":
        return "no agent"
    if state.label:
        return f"{state.label} {state.status}"
    if state.agent:
        return f"{state.agent} {state.status}"
    return state.status


def _connection_style(status: str) -> str:
    if status == "connected":
        return STATUS_SUCCESS
    if status in {"starting", "stopped", "unsupported"}:
        return STATUS_WARNING
    if status == "error":
        return STATUS_ERROR
    return "dim"
