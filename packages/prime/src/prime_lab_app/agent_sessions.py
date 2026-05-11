"""Durable Lab agent chat session storage."""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

from .agent_runtime import AgentChatMessage, AgentConnectionState, AgentRole

_PROMPT_HISTORY_LIMIT = 500


@dataclass(frozen=True)
class AgentSessionRecord:
    """A Lab-managed chat session for one workspace and agent."""

    session_id: str
    path: Path
    workspace: Path
    agent: str
    native_session_id: str = ""
    messages: tuple[AgentChatMessage, ...] = ()


def agent_sessions_root() -> Path:
    root = Path.home() / ".prime" / "lab" / "sessions"
    root.mkdir(parents=True, exist_ok=True)
    return root


def agent_prompt_history_path() -> Path:
    root = Path.home() / ".prime" / "lab"
    root.mkdir(parents=True, exist_ok=True)
    return root / "prompt-history.jsonl"


def workspace_session_key(workspace: Path) -> str:
    return hashlib.sha1(str(workspace.expanduser().resolve()).encode("utf-8")).hexdigest()


def latest_agent_session(workspace: Path, agent: str) -> AgentSessionRecord | None:
    root = _agent_root(workspace, agent)
    if not root.is_dir():
        return None
    candidates: list[AgentSessionRecord] = []
    for path in root.iterdir():
        if not path.is_dir():
            continue
        record = _load_agent_session(path, workspace, agent)
        if record is not None:
            candidates.append(record)
    if not candidates:
        return None
    return max(candidates, key=lambda record: _session_sort_key(record.path))


def create_agent_session(workspace: Path, agent: str) -> AgentSessionRecord:
    workspace = workspace.expanduser().resolve()
    agent = agent.strip() or "agent"
    created = datetime.now(timezone.utc)
    session_id = f"{created.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"
    path = _agent_root(workspace, agent) / session_id
    (path / "native").mkdir(parents=True, exist_ok=True)
    (path / "actions.jsonl").touch(exist_ok=True)
    record = AgentSessionRecord(
        session_id=session_id,
        path=path,
        workspace=workspace,
        agent=agent,
    )
    write_agent_session(record, AgentConnectionState(agent=agent, workspace=workspace), ())
    return record


def ensure_agent_session(workspace: Path, agent: str) -> AgentSessionRecord:
    return latest_agent_session(workspace, agent) or create_agent_session(workspace, agent)


def write_agent_session(
    record: AgentSessionRecord,
    state: AgentConnectionState,
    messages: tuple[AgentChatMessage, ...],
    *,
    base_url: str = "",
    team: str | None = None,
    authenticated: bool | None = None,
) -> AgentSessionRecord:
    record.path.mkdir(parents=True, exist_ok=True)
    (record.path / "native").mkdir(exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    native_session_id = state.session_id or record.native_session_id
    metadata = {
        "version": 1,
        "session_id": record.session_id,
        "workspace": str(record.workspace),
        "workspace_key": workspace_session_key(record.workspace),
        "agent": state.agent or record.agent,
        "agent_label": state.label,
        "transport": state.transport,
        "native_session_id": native_session_id,
        "endpoint": state.endpoint,
        "status": state.status,
        "base_url": base_url,
        "team": team or "",
        "authenticated": authenticated,
        "updated_at": now,
    }
    created_at = _read_json(record.path / "session.json").get("created_at")
    metadata["created_at"] = created_at or now
    (record.path / "session.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_transcript(record.path / "transcript.jsonl", messages)
    return AgentSessionRecord(
        session_id=record.session_id,
        path=record.path,
        workspace=record.workspace,
        agent=state.agent or record.agent,
        native_session_id=native_session_id,
        messages=messages,
    )


def append_agent_session_action(record: AgentSessionRecord, action: dict[str, Any]) -> None:
    record.path.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **action,
    }
    with (record.path / "actions.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")


def append_agent_prompt_history(workspace: Path, agent: str, prompt: str) -> None:
    prompt = prompt.rstrip()
    if not prompt:
        return
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "workspace": str(workspace.expanduser().resolve()),
        "workspace_key": workspace_session_key(workspace),
        "agent": agent.strip() or "agent",
        "prompt": prompt,
    }
    path = agent_prompt_history_path()
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")


def load_agent_prompt_history(*, limit: int = 50) -> tuple[str, ...]:
    path = agent_prompt_history_path()
    if not path.is_file():
        return ()
    prompts: list[str] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return ()
    for line in lines[-_PROMPT_HISTORY_LIMIT:]:
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        prompt = str(payload.get("prompt") or "").strip()
        if prompt:
            prompts.append(prompt)
    return tuple(_latest_unique(prompts)[-limit:])


def _agent_root(workspace: Path, agent: str) -> Path:
    safe_agent = _safe_component(agent.strip() or "agent")
    return agent_sessions_root() / workspace_session_key(workspace) / safe_agent


def _load_agent_session(path: Path, workspace: Path, agent: str) -> AgentSessionRecord | None:
    metadata = _read_json(path / "session.json")
    session_id = str(metadata.get("session_id") or path.name)
    if not session_id:
        return None
    messages = tuple(_read_transcript(path / "transcript.jsonl"))
    return AgentSessionRecord(
        session_id=session_id,
        path=path,
        workspace=workspace.expanduser().resolve(),
        agent=str(metadata.get("agent") or agent),
        native_session_id=str(metadata.get("native_session_id") or ""),
        messages=messages,
    )


def _read_transcript(path: Path) -> list[AgentChatMessage]:
    messages: list[AgentChatMessage] = []
    if not path.is_file():
        return messages
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return messages
    for line in lines:
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        role = payload.get("role")
        if role not in {"user", "assistant", "system"}:
            continue
        messages.append(
            AgentChatMessage(
                cast(AgentRole, role),
                str(payload.get("content") or ""),
                str(payload.get("status") or ""),
                _metadata(payload.get("metadata")),
            )
        )
    return messages


def _write_transcript(path: Path, messages: tuple[AgentChatMessage, ...]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for message in messages:
            handle.write(
                json.dumps(
                    {
                        "role": message.role,
                        "content": message.content,
                        "status": message.status,
                        "metadata": message.metadata,
                    },
                    sort_keys=True,
                    default=str,
                )
                + "\n"
            )


def _session_sort_key(path: Path) -> tuple[str, float]:
    metadata = _read_json(path / "session.json")
    updated_at = str(metadata.get("updated_at") or metadata.get("created_at") or "")
    try:
        mtime = (path / "session.json").stat().st_mtime
    except OSError:
        mtime = 0.0
    return updated_at, mtime


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _metadata(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _latest_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in reversed(values):
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    unique.reverse()
    return unique


def _safe_component(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"-", "_", "."} else "-" for char in value)
    return safe.strip(".-") or "agent"
