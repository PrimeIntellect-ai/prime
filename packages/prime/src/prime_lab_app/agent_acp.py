"""Agent Client Protocol helpers for Lab agent runtimes."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

AcpEventKind = Literal["assistant_delta", "tool_call", "tool_update", "ignored"]


@dataclass(frozen=True)
class AcpSessionSupport:
    """ACP session lifecycle capabilities advertised by an agent."""

    resume: bool = False
    load: bool = False
    close: bool = False


@dataclass(frozen=True)
class AcpUpdateEvent:
    """Normalized ACP session/update event."""

    kind: AcpEventKind
    text: str = ""
    tool_call_id: str = ""
    title: str = ""
    tool_kind: str = ""
    status: str = ""


def acp_lab_mcp_servers(workspace: Path) -> list[dict[str, Any]]:
    """Return the Prime Lab MCP server config in ACP session/new shape."""

    return [
        {
            "name": "prime_lab",
            "command": sys.executable,
            "args": [
                "-c",
                "from prime_cli.main import run; run()",
                "lab",
                "mcp",
                "--workspace",
                str(workspace.expanduser().resolve()),
            ],
            "env": [],
        }
    ]


def acp_session_params(workspace: Path, *, session_id: str = "") -> dict[str, Any]:
    """Build common ACP session lifecycle params."""

    params: dict[str, Any] = {
        "cwd": str(workspace.expanduser().resolve()),
        "mcpServers": acp_lab_mcp_servers(workspace),
    }
    if session_id:
        params["sessionId"] = session_id
    return params


def acp_session_support(initialize_result: dict[str, Any]) -> AcpSessionSupport:
    """Extract ACP resume/load/close support from initialize result."""

    capabilities = initialize_result.get("agentCapabilities")
    if not isinstance(capabilities, dict):
        capabilities = {}
    session_capabilities = capabilities.get("sessionCapabilities")
    if not isinstance(session_capabilities, dict):
        session_capabilities = {}
    return AcpSessionSupport(
        resume=_capability_enabled(session_capabilities.get("resume")),
        load=capabilities.get("loadSession") is True,
        close=_capability_enabled(session_capabilities.get("close")),
    )


def acp_update_event(update: dict[str, Any]) -> AcpUpdateEvent:
    """Normalize one ACP session/update payload for Lab transcript handling."""

    update_type = update.get("sessionUpdate")
    if update_type == "agent_message_chunk":
        return AcpUpdateEvent("assistant_delta", text=_content_text(update.get("content")))
    if update_type == "tool_call":
        return AcpUpdateEvent(
            "tool_call",
            tool_call_id=str(update.get("toolCallId") or ""),
            title=str(update.get("title") or ""),
            tool_kind=str(update.get("kind") or "other"),
            status=str(update.get("status") or "pending"),
        )
    if update_type == "tool_call_update":
        return AcpUpdateEvent(
            "tool_update",
            tool_call_id=str(update.get("toolCallId") or ""),
            title=str(update.get("title") or ""),
            tool_kind=str(update.get("kind") or "other"),
            status=str(update.get("status") or ""),
            text=_tool_update_content_text(update.get("content")),
        )
    return AcpUpdateEvent("ignored")


def _tool_update_content_text(value: Any) -> str:
    if not isinstance(value, list | tuple):
        return _content_text(value)
    chunks: list[str] = []
    for item in value:
        if isinstance(item, dict) and item.get("type") == "content":
            text = _content_text(item.get("content"))
        else:
            text = _content_text(item)
        if text:
            chunks.append(text)
    return "\n".join(chunks)


def _capability_enabled(value: Any) -> bool:
    return value is True or isinstance(value, dict)


def _content_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list | tuple):
        return "".join(_content_text(item) for item in value)
    if isinstance(value, dict):
        text = value.get("text")
        if isinstance(text, str):
            return text
        content = value.get("content")
        if content is not value:
            nested = _content_text(content)
            if nested:
                return nested
    text = getattr(value, "text", None)
    if isinstance(text, str):
        return text
    content = getattr(value, "content", None)
    if content is not None and content is not value:
        return _content_text(content)
    return ""
