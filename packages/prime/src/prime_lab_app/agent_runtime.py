"""Long-lived coding-agent server runtime for Lab."""

from __future__ import annotations

import json
import re
import subprocess
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from .agent_acp import acp_session_params, acp_session_support, acp_update_event
from .agent_adapters import AgentAdapter, AgentServerSpec, write_agent_native_surface
from .agent_capabilities import AgentCapability, agent_capability
from .agent_widgets import (
    handle_lab_widget_tool_call,
    lab_dynamic_tools,
    lab_widget_action_from_tool_call,
    lab_widget_developer_instructions,
)

AgentStatus = Literal["none", "starting", "connected", "error", "stopped", "unsupported"]
AgentRole = Literal["user", "assistant", "system"]
_ASSISTANT_STREAM_KEY = "__assistant_stream__"
_ASSISTANT_CHUNK_SEEN_KEY = "__assistant_chunk_seen__"
_ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


@dataclass(frozen=True)
class AgentConnectionState:
    """Display state for the active coding-agent server."""

    agent: str = ""
    label: str = ""
    status: AgentStatus = "none"
    transport: str = ""
    workspace: Path | None = None
    endpoint: str = ""
    message: str = ""
    session_id: str = ""


@dataclass(frozen=True)
class AgentChatMessage:
    """One message in the current Lab agent session."""

    role: AgentRole
    content: str
    status: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


StateCallback = Callable[[AgentConnectionState], None]
MessagesCallback = Callable[[tuple[AgentChatMessage, ...]], None]
ActionCallback = Callable[[dict[str, Any]], None]
LabToolHandler = Callable[[dict[str, Any]], tuple[str, str, dict[str, Any]]]
PopenFactory = Callable[..., subprocess.Popen[str]]


class AgentRuntime:
    """Owns one workspace-scoped agent server process and chat session."""

    def __init__(
        self,
        *,
        on_state: StateCallback | None = None,
        on_messages: MessagesCallback | None = None,
        on_action: ActionCallback | None = None,
        lab_tool_handler: LabToolHandler | None = None,
        popen_factory: PopenFactory = subprocess.Popen,
    ) -> None:
        self._on_state = on_state
        self._on_messages = on_messages
        self._on_action = on_action
        self._lab_tool_handler = lab_tool_handler
        self._popen_factory = popen_factory
        self._lock = threading.RLock()
        self._pending: dict[int, tuple[threading.Event, dict[str, Any]]] = {}
        self._send_lock = threading.Lock()
        self._next_request_id = 1
        self._process: subprocess.Popen[str] | None = None
        self._spec: AgentServerSpec | None = None
        self._adapter: AgentAdapter | None = None
        self._workspace: Path | None = None
        self._agent = ""
        self._label = ""
        self._state = AgentConnectionState()
        self._messages: list[AgentChatMessage] = []
        self._reader_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._initializing_thread: threading.Thread | None = None
        self._intentional_stop = False
        self._active_turn_id = ""

    @property
    def state(self) -> AgentConnectionState:
        with self._lock:
            return self._state

    def messages(self) -> tuple[AgentChatMessage, ...]:
        with self._lock:
            return tuple(self._messages)

    def start(
        self,
        workspace: Path,
        agent_name: str,
        *,
        session_id: str = "",
        initial_messages: tuple[AgentChatMessage, ...] = (),
    ) -> None:
        agent_name = agent_name.strip()
        if not agent_name:
            self.stop()
            self._set_state(AgentConnectionState(status="none", message="No agent configured"))
            with self._lock:
                self._messages = []
                self._emit_messages_locked()
            return

        capability = agent_capability(agent_name)
        adapter = capability.adapter
        workspace = workspace.expanduser().resolve()
        with self._lock:
            if (
                self._process is not None
                and self._process.poll() is None
                and self._workspace == workspace
                and self._agent == adapter.name
            ):
                return

        self.stop()
        spec = adapter.server_spec(workspace)
        self._workspace = workspace
        self._agent = adapter.name
        self._label = adapter.label
        self._spec = spec
        self._adapter = adapter
        self._intentional_stop = False
        with self._lock:
            self._messages = list(initial_messages)
            self._emit_messages_locked()
        if capability.status == "not_supported" or adapter.lab_widget_contract == "not-supported":
            message = _unsupported_agent_message(capability)
            self._set_state(
                AgentConnectionState(
                    agent=adapter.name,
                    label=adapter.label,
                    status="unsupported",
                    transport=spec.transport,
                    workspace=workspace,
                    message=message,
                    session_id=session_id,
                )
            )
            with self._lock:
                self._append_message_locked(AgentChatMessage("system", message, "warning"))
            return
        self._set_state(
            AgentConnectionState(
                agent=adapter.name,
                label=adapter.label,
                status="starting",
                transport=spec.transport,
                workspace=workspace,
                message="Starting server",
                session_id=session_id,
            )
        )

        if spec.transport == "acp-stdio":
            try:
                write_agent_native_surface(workspace, adapter.name)
            except OSError as exc:
                self._set_state(
                    AgentConnectionState(
                        agent=adapter.name,
                        label=adapter.label,
                        status="error",
                        transport=spec.transport,
                        workspace=workspace,
                        message=f"Could not prepare {adapter.label} Lab tools: {exc}",
                        session_id=session_id,
                    )
                )
                return
        if spec.transport == "resumable-cli":
            try:
                write_agent_native_surface(workspace, adapter.name)
            except OSError as exc:
                self._set_state(
                    AgentConnectionState(
                        agent=adapter.name,
                        label=adapter.label,
                        status="error",
                        transport=spec.transport,
                        workspace=workspace,
                        message=f"Could not prepare {adapter.label} Lab tools: {exc}",
                        session_id=session_id,
                    )
                )
                return
            self._set_connected(message="CLI ready", session_id=session_id)
            return

        if adapter.session_dir_flag is not None:
            (workspace / ".prime" / "lab" / "agent-sessions" / adapter.name).mkdir(
                parents=True,
                exist_ok=True,
            )

        try:
            process = self._popen_factory(
                list(spec.command),
                cwd=workspace,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            self._set_state(
                AgentConnectionState(
                    agent=adapter.name,
                    label=adapter.label,
                    status="error",
                    transport=spec.transport,
                    workspace=workspace,
                    message=str(exc),
                )
            )
            return

        with self._lock:
            self._process = process

        self._reader_thread = threading.Thread(
            target=self._read_stdout,
            name=f"lab-agent-{adapter.name}-stdout",
            daemon=True,
        )
        self._reader_thread.start()
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr,
            name=f"lab-agent-{adapter.name}-stderr",
            daemon=True,
        )
        self._stderr_thread.start()

        if spec.transport == "acp-stdio":
            self._initializing_thread = threading.Thread(
                target=self._initialize_acp_stdio,
                name=f"lab-agent-{adapter.name}-init",
                daemon=True,
            )
            self._initializing_thread.start()
        elif spec.transport == "codex-app-stdio":
            self._initializing_thread = threading.Thread(
                target=self._initialize_codex_app_stdio,
                name=f"lab-agent-{adapter.name}-init",
                daemon=True,
            )
            self._initializing_thread.start()
        elif spec.transport == "stdio-jsonrpc":
            self._set_connected(message="RPC server ready")
        elif spec.transport in {"websocket", "acp-http", "http"}:
            self._initializing_thread = threading.Thread(
                target=self._wait_for_endpoint_or_alive,
                name=f"lab-agent-{adapter.name}-ready",
                daemon=True,
            )
            self._initializing_thread.start()
        else:
            self._set_connected(message="Server ready")

    def stop(self) -> None:
        with self._lock:
            process = self._process
            self._process = None
            self._spec = None
            self._adapter = None
            self._pending.clear()
            self._intentional_stop = True
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()

    def send_prompt(self, prompt: str) -> None:
        prompt = prompt.rstrip("\n")
        if not prompt:
            return
        with self._lock:
            state = self._state
            if state.status != "connected":
                message = state.message if state.status == "unsupported" and state.message else ""
                self._append_message_locked(
                    AgentChatMessage("system", message or "No connected coding agent.", "error")
                )
                return
            if self._spec is None:
                self._append_message_locked(
                    AgentChatMessage("system", "No connected coding agent.", "error")
                )
                return
            if self._spec.transport == "codex-app-stdio":
                self._append_message_locked(
                    AgentChatMessage("user", prompt),
                    AgentChatMessage("assistant", "", "streaming"),
                )
                transport = "codex-app-stdio"
                session_id = state.session_id
            elif self._spec.transport == "acp-stdio":
                self._append_message_locked(
                    AgentChatMessage("user", prompt),
                    AgentChatMessage("assistant", "", "streaming"),
                )
                transport = "acp-stdio"
                session_id = state.session_id
            elif self._spec.transport == "resumable-cli":
                self._append_message_locked(
                    AgentChatMessage("user", prompt),
                    AgentChatMessage("assistant", "", "streaming"),
                )
                transport = self._spec.transport
                session_id = state.session_id
            else:
                self._append_message_locked(
                    AgentChatMessage("user", prompt),
                    AgentChatMessage("assistant", "", "streaming"),
                )
                transport = ""
                session_id = ""

        try:
            if transport == "codex-app-stdio":
                result = self._request(
                    "turn/start",
                    {
                        "threadId": session_id,
                        "cwd": str(self._workspace) if self._workspace is not None else None,
                        "input": [{"type": "text", "text": prompt}],
                    },
                    timeout=3600,
                )
                self._record_codex_turn(result)
            else:
                if transport == "resumable-cli":
                    threading.Thread(
                        target=self._run_resumable_cli_prompt,
                        args=(prompt, session_id),
                        name=f"lab-agent-{self._agent}-cli",
                        daemon=True,
                    ).start()
                    return
                prompt_text = (
                    _agent_prompt_with_lab_context(prompt) if transport == "acp-stdio" else prompt
                )
                self._request(
                    "session/prompt",
                    {
                        "sessionId": session_id,
                        "messageId": str(uuid.uuid4()),
                        "prompt": [{"type": "text", "text": prompt_text}],
                    },
                    timeout=3600,
                )
        except Exception as exc:
            with self._lock:
                self._replace_last_streaming_locked(
                    AgentChatMessage("system", f"Agent request failed: {exc}", "error")
                )
                self._emit_messages_locked()
            return

        if transport == "acp-stdio":
            with self._lock:
                if self._messages and self._messages[-1].status == "streaming":
                    last = self._messages[-1]
                    content = _dedupe_streamed_text(last.content) or (
                        "(completed without response text)"
                    )
                    self._messages[-1] = AgentChatMessage("assistant", content)
                    self._emit_messages_locked()

    def handle_external_lab_tool(self, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Handle a native MCP Lab tool call forwarded by `prime lab mcp`."""

        params = {
            "namespace": "lab",
            "tool": tool,
            "callId": str(uuid.uuid4()),
            "arguments": arguments,
        }
        status, content, response = self._handle_lab_tool_call(params)
        action = lab_widget_action_from_tool_call(params) if status == "widget" else None
        self._record_dynamic_tool_call(status, content, action)
        if status == "widget":
            self._emit_action(action or {})
        return response

    def _run_resumable_cli_prompt(self, prompt: str, session_id: str) -> None:
        workspace = self._workspace
        adapter = self._adapter
        if workspace is None or adapter is None:
            with self._lock:
                self._replace_last_streaming_locked(
                    AgentChatMessage("system", "No agent workspace is configured.", "error")
                )
                self._emit_messages_locked()
            return

        command = adapter.stream_command(
            _agent_prompt_with_lab_context(prompt),
            session_id,
            workspace=workspace,
        )
        try:
            process = self._popen_factory(
                command,
                cwd=workspace,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            with self._lock:
                self._replace_last_streaming_locked(AgentChatMessage("system", str(exc), "error"))
                self._emit_messages_locked()
            return

        seen_messages: dict[str, str] = {}
        emitted_text = False
        stderr_lines: list[str] = []
        stdout_error_lines: list[str] = []
        if process.stderr is not None:
            threading.Thread(
                target=_collect_stream_lines,
                args=(process.stderr, stderr_lines),
                name=f"lab-agent-{adapter.name}-cli-stderr",
                daemon=True,
            ).start()
        if process.stdout is not None:
            for raw_line in process.stdout:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    self._append_streaming_assistant_text(raw_line)
                    continue
                if _field(event, "type") == "error":
                    error_message = _field(event, "message")
                    if isinstance(error_message, str) and error_message:
                        stdout_error_lines.append(error_message)
                next_session_id = _extract_agent_session_id(event)
                if next_session_id and next_session_id != session_id:
                    session_id = next_session_id
                    self._set_connected(message="Connected", session_id=session_id)
                text = _extract_stream_delta(event, seen_messages)
                if not text and not emitted_text:
                    text = _extract_result_text(event)
                if text:
                    emitted_text = True
                    self._append_streaming_assistant_text(text)
        code = process.wait()
        if code != 0:
            failure_lines = stderr_lines or stdout_error_lines
            if failure_lines:
                self._append_streaming_assistant_text("\n".join(failure_lines).strip())
        self._finish_streaming_process(code, failure_label=f"{adapter.label} request failed")

    def _initialize_acp_stdio(self) -> None:
        workspace = self._workspace
        if workspace is None:
            return
        try:
            init = self._request(
                "initialize",
                {
                    "protocolVersion": 1,
                    "clientCapabilities": {"terminal": False},
                    "clientInfo": {
                        "name": "prime-lab",
                        "title": "Lab",
                        "version": "0.0.0",
                    },
                },
                timeout=12,
            )
            session_support = acp_session_support(init)
            auth_errors: list[str] = []
            for method in init.get("authMethods") or []:
                if self._agent == "opencode":
                    continue
                method_id = method.get("id") if isinstance(method, dict) else None
                if method_id:
                    try:
                        self._request("authenticate", {"methodId": method_id}, timeout=12)
                    except Exception as exc:
                        auth_errors.append(str(exc))
            current_session_id = self._state.session_id
            mcp_servers = [] if self._agent == "opencode" else None
            if current_session_id and session_support.resume:
                session = self._request(
                    "session/resume",
                    acp_session_params(
                        workspace,
                        session_id=current_session_id,
                        mcp_servers=mcp_servers,
                    ),
                    timeout=30,
                )
            else:
                session = self._request(
                    "session/new",
                    acp_session_params(workspace, mcp_servers=mcp_servers),
                    timeout=30,
                )
        except Exception as exc:
            message = str(exc)
            if "auth_errors" in locals() and auth_errors:
                message = f"{message}; auth: {'; '.join(auth_errors)}"
            self._set_state(
                AgentConnectionState(
                    agent=self._agent,
                    label=self._label,
                    status="error",
                    transport=self._spec.transport if self._spec else "",
                    workspace=workspace,
                    message=message,
                )
            )
            return
        session_id = str(session.get("sessionId") or "")
        self._set_connected(message="Connected", session_id=session_id)

    def _initialize_codex_app_stdio(self) -> None:
        workspace = self._workspace
        if workspace is None:
            return
        try:
            self._request(
                "initialize",
                {
                    "clientInfo": {
                        "name": "prime-lab",
                        "title": "Lab",
                        "version": "0.0.0",
                    },
                    "capabilities": {"experimentalApi": True},
                },
                timeout=12,
            )
            thread = self._request(
                "thread/start",
                {
                    "cwd": str(workspace),
                    "developerInstructions": lab_widget_developer_instructions(),
                    "dynamicTools": lab_dynamic_tools(),
                    "sessionStartSource": "startup",
                },
                timeout=30,
            )
        except Exception as exc:
            self._set_state(
                AgentConnectionState(
                    agent=self._agent,
                    label=self._label,
                    status="error",
                    transport=self._spec.transport if self._spec else "",
                    workspace=workspace,
                    message=str(exc),
                )
            )
            return
        thread_payload = thread.get("thread")
        thread_id = str(thread_payload.get("id") or "") if isinstance(thread_payload, dict) else ""
        self._set_connected(message="Connected", session_id=thread_id)

    def _wait_for_endpoint_or_alive(self) -> None:
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if self._process is None or self._process.poll() is not None:
                return
            if self._state.endpoint:
                return
            time.sleep(0.05)
        if self._process is not None and self._process.poll() is None:
            self._set_connected(message="Server process ready")

    def _read_stdout(self) -> None:
        process = self._process
        if process is None or process.stdout is None:
            return
        try:
            for raw_line in process.stdout:
                line = raw_line.strip()
                if not line:
                    continue
                if line.startswith(("ws://", "http://", "https://")):
                    self._set_connected(endpoint=line, message="Connected")
                    continue
                try:
                    message = json.loads(line)
                except json.JSONDecodeError:
                    continue
                self._handle_jsonrpc_message(message)
        finally:
            code = process.poll()
            if code is None:
                code = process.wait()
            with self._lock:
                intentional = self._intentional_stop
                state = self._state
            if not intentional and state.status != "error":
                self._set_state(
                    AgentConnectionState(
                        agent=self._agent,
                        label=self._label,
                        status="stopped",
                        transport=self._spec.transport if self._spec else "",
                        workspace=self._workspace,
                        message=f"Server exited with {code}",
                    )
                )

    def _drain_stderr(self) -> None:
        process = self._process
        if process is None or process.stderr is None:
            return
        for _line in process.stderr:
            pass

    def _handle_jsonrpc_message(self, message: Any) -> None:
        if not isinstance(message, dict):
            return
        request_id = message.get("id")
        if request_id is not None and "method" not in message:
            pending_id = _jsonrpc_response_id(request_id)
            if pending_id is None:
                message_text = (
                    "Agent returned response with unsupported JSON-RPC id: "
                    f"{_short_repr(request_id)}"
                )
                self._set_state(
                    AgentConnectionState(
                        agent=self._agent,
                        label=self._label,
                        status="error",
                        transport=self._spec.transport if self._spec else "",
                        workspace=self._workspace,
                        message=message_text,
                    )
                )
                return
            self._complete_pending(pending_id, message)
            return
        method = message.get("method")
        if not isinstance(method, str):
            return
        params = message.get("params") if isinstance(message.get("params"), dict) else {}
        if method == "session/update":
            self._handle_session_update(params)
            if request_id is not None:
                self._respond(request_id, None)
            return
        if method == "item/agentMessage/delta":
            self._handle_codex_agent_delta(params)
            if request_id is not None:
                self._respond(request_id, None)
            return
        if method == "item/completed":
            self._handle_codex_item_completed(params)
            if request_id is not None:
                self._respond(request_id, None)
            return
        if method == "turn/completed":
            self._handle_codex_turn_completed(params)
            if request_id is not None:
                self._respond(request_id, None)
            return
        if method == "item/tool/call":
            status, content, response = self._handle_lab_tool_call(params)
            action = lab_widget_action_from_tool_call(params) if status == "widget" else None
            self._record_dynamic_tool_call(status, content, action)
            if status == "widget":
                self._emit_action(action or {})
            if request_id is not None:
                self._respond(request_id, response)
            return
        if method == "session/request_permission":
            if request_id is not None:
                self._respond(request_id, {"outcome": {"outcome": "cancelled"}})
            return
        if method.endswith("/approval/request") or method.endswith("/requestApproval"):
            if request_id is not None:
                self._respond(request_id, {"decision": "denied"})
            return
        if request_id is not None:
            self._respond_error(request_id, -32601, f"Unsupported client method: {method}")

    def _handle_lab_tool_call(self, params: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
        if self._lab_tool_handler is not None:
            return self._lab_tool_handler(params)
        return handle_lab_widget_tool_call(params)

    def _handle_session_update(self, params: dict[str, Any]) -> None:
        update = params.get("update")
        if not isinstance(update, dict):
            return
        event = acp_update_event(update)
        if event.kind == "assistant_delta":
            self._append_streaming_assistant_text(event.text)
            return
        if event.kind in {"tool_call", "tool_update"}:
            self._record_acp_tool_event(event.title, event.status, event.text)
            return

    def _record_acp_tool_event(self, title: str, status: str, text: str) -> None:
        title = title.strip()
        status = status.strip()
        text = text.strip()
        if not title and not text:
            return
        if text and _is_lab_widget_tool_result_text(text):
            return
        label = title or "Tool"
        content = label if not text else f"{label}\n{text}"
        with self._lock:
            if (
                not text
                and _is_lab_widget_tool_event_title(label)
                and self._messages
                and self._messages[-1].status == "widget"
            ):
                return
            if (
                self._messages
                and self._messages[-1].role == "system"
                and self._messages[-1].status == "tool"
                and self._messages[-1].metadata.get("title") == label
            ):
                self._messages[-1] = AgentChatMessage(
                    "system",
                    content,
                    "tool",
                    {"title": label, "tool_status": status},
                )
            else:
                self._messages.append(
                    AgentChatMessage(
                        "system",
                        content,
                        "tool",
                        {"title": label, "tool_status": status},
                    )
                )
            self._emit_messages_locked()

    def _handle_codex_agent_delta(self, params: dict[str, Any]) -> None:
        delta = params.get("delta")
        if not isinstance(delta, str) or not delta:
            return
        turn_id = params.get("turnId")
        if isinstance(turn_id, str):
            self._active_turn_id = turn_id
        self._append_streaming_assistant_text(delta)

    def _handle_codex_item_completed(self, params: dict[str, Any]) -> None:
        item = params.get("item")
        if not isinstance(item, dict) or item.get("type") != "agentMessage":
            return
        text = item.get("text")
        if not isinstance(text, str) or not text:
            return
        with self._lock:
            if not self._messages or self._messages[-1].role != "assistant":
                self._messages.append(AgentChatMessage("assistant", text, "streaming"))
            elif not self._messages[-1].content:
                self._messages[-1] = AgentChatMessage("assistant", text, "streaming")
            self._emit_messages_locked()

    def _handle_codex_turn_completed(self, params: dict[str, Any]) -> None:
        turn = params.get("turn")
        if not isinstance(turn, dict):
            return
        error = turn.get("error")
        with self._lock:
            if isinstance(error, dict):
                message = str(error.get("message") or "Codex turn failed")
                self._replace_last_streaming_locked(AgentChatMessage("system", message, "error"))
            else:
                self._finish_latest_streaming_assistant_locked()
            self._active_turn_id = ""
            self._emit_messages_locked()

    def _record_dynamic_tool_call(
        self,
        status: str,
        content: str,
        action: dict[str, Any] | None = None,
    ) -> None:
        with self._lock:
            self._finish_latest_streaming_assistant_locked(fallback="")
            self._messages.append(AgentChatMessage("system", content, status, action or {}))
            self._emit_messages_locked()

    def _record_codex_turn(self, result: dict[str, Any]) -> None:
        turn = result.get("turn")
        if not isinstance(turn, dict):
            return
        turn_id = turn.get("id")
        if isinstance(turn_id, str):
            self._active_turn_id = turn_id
        if turn.get("status") in {"completed", "failed", "interrupted"}:
            self._handle_codex_turn_completed({"turn": turn})

    def _append_streaming_assistant_text(self, text: str) -> None:
        text = _clean_agent_output_text(text)
        if not text:
            return
        with self._lock:
            if (
                not self._messages
                or self._messages[-1].role != "assistant"
                or self._messages[-1].status != "streaming"
            ):
                self._messages.append(AgentChatMessage("assistant", text, "streaming"))
            else:
                last = self._messages[-1]
                merged = _merge_stream_text(last.content, text)
                self._messages[-1] = AgentChatMessage(
                    "assistant",
                    _dedupe_streamed_text(merged),
                    "streaming",
                )
            self._emit_messages_locked()

    def _finish_streaming_process(self, code: int, *, failure_label: str) -> None:
        with self._lock:
            if code != 0:
                last_text = (
                    latest.content
                    if (latest := self._latest_streaming_assistant_locked()) is not None
                    else ""
                )
                message = f"{failure_label} with {code}"
                if last_text.strip():
                    message = f"{message}\n\n{last_text.rstrip()}"
                self._replace_last_streaming_locked(AgentChatMessage("system", message, "error"))
            else:
                self._finish_latest_streaming_assistant_locked()
            self._emit_messages_locked()

    def _request(
        self,
        method: str,
        params: dict[str, Any],
        *,
        timeout: float,
    ) -> dict[str, Any]:
        request_id = self._next_id()
        event = threading.Event()
        holder: dict[str, Any] = {}
        with self._lock:
            self._pending[request_id] = (event, holder)
        self._write_json({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
        if not event.wait(timeout):
            with self._lock:
                self._pending.pop(request_id, None)
            raise TimeoutError(f"{method} timed out")
        if "error" in holder:
            error = holder["error"] or {}
            message = error.get("message") if isinstance(error, dict) else str(error)
            raise RuntimeError(message or f"{method} failed")
        result = holder.get("result")
        return result if isinstance(result, dict) else {}

    def _respond(self, request_id: Any, result: Any) -> None:
        self._write_json({"jsonrpc": "2.0", "id": request_id, "result": result})

    def _respond_error(self, request_id: Any, code: int, message: str) -> None:
        self._write_json(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": code, "message": message},
            }
        )

    def _write_json(self, payload: dict[str, Any]) -> None:
        process = self._process
        if process is None or process.stdin is None:
            raise RuntimeError("agent process is not running")
        data = json.dumps(payload, separators=(",", ":"))
        with self._send_lock:
            process.stdin.write(data + "\n")
            process.stdin.flush()

    def _next_id(self) -> int:
        with self._lock:
            request_id = self._next_request_id
            self._next_request_id += 1
            return request_id

    def _complete_pending(self, request_id: int, message: dict[str, Any]) -> None:
        with self._lock:
            pending = self._pending.pop(request_id, None)
        if pending is None:
            return
        event, holder = pending
        holder.update(message)
        event.set()

    def _set_connected(
        self,
        *,
        endpoint: str = "",
        message: str = "Connected",
        session_id: str = "",
    ) -> None:
        with self._lock:
            current = self._state
        self._set_state(
            AgentConnectionState(
                agent=self._agent,
                label=self._label,
                status="connected",
                transport=self._spec.transport if self._spec else current.transport,
                workspace=self._workspace,
                endpoint=endpoint or current.endpoint,
                message=message,
                session_id=session_id or current.session_id,
            )
        )

    def _set_state(self, state: AgentConnectionState) -> None:
        with self._lock:
            self._state = state
        if self._on_state is not None:
            self._on_state(state)

    def _append_message_locked(self, *messages: AgentChatMessage) -> None:
        self._messages.extend(messages)
        self._emit_messages_locked()

    def _replace_last_streaming_locked(self, message: AgentChatMessage) -> None:
        for index in range(len(self._messages) - 1, -1, -1):
            if (
                self._messages[index].role == "assistant"
                and self._messages[index].status == "streaming"
            ):
                self._messages[index] = message
                return
        self._messages.append(message)

    def _latest_streaming_assistant_locked(self) -> AgentChatMessage | None:
        for message in reversed(self._messages):
            if message.role == "assistant" and message.status == "streaming":
                return message
        return None

    def _finish_latest_streaming_assistant_locked(
        self,
        *,
        fallback: str = "(completed without response text)",
    ) -> None:
        for index in range(len(self._messages) - 1, -1, -1):
            message = self._messages[index]
            if message.role != "assistant" or message.status != "streaming":
                continue
            content = _dedupe_streamed_text(message.content)
            if not content and not fallback:
                self._messages.pop(index)
            else:
                self._messages[index] = AgentChatMessage(
                    "assistant",
                    content or fallback,
                    metadata=message.metadata,
                )
            return

    def _emit_messages_locked(self) -> None:
        if self._on_messages is not None:
            self._on_messages(tuple(self._messages))

    def _emit_action(self, action: dict[str, Any]) -> None:
        if self._on_action is not None:
            self._on_action(action)


def _collect_stream_lines(stream: Any, lines: list[str]) -> None:
    for line in stream:
        text = str(line).rstrip()
        if text:
            lines.append(text)


def _unsupported_agent_message(capability: AgentCapability) -> str:
    supported = "Amp, Codex, Claude, Claude Code, Cursor, Factory Droid, OpenCode, Hermes, or Pi"
    reason = f" {capability.unsupported_reason}" if capability.unsupported_reason else ""
    return (
        f"{capability.label} is not yet supported for Lab-native chat actions.{reason} "
        f"Switch to {supported} for native Lab actions."
    )


def _extract_agent_session_id(value: Any) -> str:
    for key in (
        "session_id",
        "sessionId",
        "sessionID",
        "chat_id",
        "chatId",
        "chatID",
        "thread_id",
        "threadId",
        "threadID",
    ):
        found = _field(value, key)
        if isinstance(found, str) and found:
            return found
    session = _field(value, "session")
    if session is not None:
        for key in ("id", "session_id", "sessionId"):
            found = _field(session, key)
            if isinstance(found, str) and found:
                return found
    message = _field(value, "message")
    if message is not None:
        return _extract_agent_session_id(message)
    return ""


def _extract_result_text(value: Any) -> str:
    payload_type = _field(value, "type")
    if payload_type == "agent_end":
        messages = _field(value, "messages")
        if isinstance(messages, list):
            for message in reversed(messages):
                role = _field(message, "role")
                if isinstance(role, str) and role.lower() == "assistant":
                    text = _content_text(_field(message, "content"))
                    if text:
                        return text
        return ""
    if payload_type not in {"result", "completion", "done"}:
        return ""
    for key in ("result", "text", "content", "message"):
        found = _field(value, key)
        text = _content_text(found)
        if text:
            return text
    return ""


def _extract_stream_delta(value: Any, seen_messages: dict[str, str]) -> str:
    if _is_non_transcript_stream_event(value):
        return ""

    explicit_delta = _explicit_stream_delta(value)
    if explicit_delta:
        seen_messages[_ASSISTANT_CHUNK_SEEN_KEY] = "1"
        return _record_agent_stream_delta(seen_messages, explicit_delta)

    message = _field(value, "message")
    if message is None:
        part = _field(value, "part")
        if part is not None:
            message = part
    if message is None:
        message = value
    role = _field(message, "role") or _field(value, "role") or _field(value, "type")
    if isinstance(role, str) and role.lower() in {"user", "human"}:
        return ""
    text = _content_text(_field(message, "content"))
    if not text:
        text = _content_text(_field(message, "text"))
    if not text:
        text = _content_text(_field(value, "content"))
    if not text:
        return ""
    text = _clean_agent_output_text(text)
    if not text:
        return ""
    if _is_lab_widget_tool_result_text(text):
        return ""

    message_id = _message_id(value, message)
    stream_text = seen_messages.get(_ASSISTANT_STREAM_KEY, "")
    if stream_text:
        if seen_messages.get(_ASSISTANT_CHUNK_SEEN_KEY) == "1" and not _is_chunk_message(value):
            if message_id:
                seen_messages[message_id] = text
            return ""
        if text == stream_text:
            if message_id:
                seen_messages[message_id] = text
            return ""
        if text.startswith(stream_text):
            delta = text[len(stream_text) :]
            if message_id:
                seen_messages[message_id] = text
            return _record_agent_stream_delta(seen_messages, delta)

    if _is_chunk_message(value):
        seen_messages[_ASSISTANT_CHUNK_SEEN_KEY] = "1"
        return _record_agent_stream_delta(seen_messages, text)

    if message_id:
        previous = seen_messages.get(message_id, "")
        seen_messages[message_id] = text
        if previous and text.startswith(previous):
            return _record_agent_stream_delta(seen_messages, text[len(previous) :])
        if previous == text:
            return ""

    if not message_id:
        seen_messages[_ASSISTANT_STREAM_KEY] = text
    return text


def _is_non_transcript_stream_event(value: Any) -> bool:
    payload_type = _field(value, "type")
    if isinstance(payload_type, str) and payload_type.lower() in {
        "step_start",
        "step_finish",
        "tool",
        "tool_use",
    }:
        return True
    if isinstance(payload_type, str) and payload_type.lower() in {
        "thinking",
        "thought",
        "reasoning",
    }:
        return True
    message = _field(value, "message")
    role = _field(message, "role") or _field(value, "role")
    return isinstance(role, str) and role.lower() in {"thinking", "thought", "reasoning"}


def _explicit_stream_delta(value: Any) -> str:
    for key in ("delta", "text_delta", "content_delta"):
        found = _field(value, key)
        if isinstance(found, str) and found:
            return _clean_agent_output_text(found)
        text = _content_text(found)
        if text:
            return _clean_agent_output_text(text)

    event = _field(value, "event")
    if event is not None:
        event_type = _field(event, "type")
        if event_type in {"content_block_delta", "text_delta", "message_delta"}:
            return _explicit_stream_delta(event)

    return ""


def _record_agent_stream_delta(seen_messages: dict[str, str], delta: str) -> str:
    delta = _clean_agent_output_text(delta)
    if not delta:
        return ""
    seen_messages[_ASSISTANT_STREAM_KEY] = f"{seen_messages.get(_ASSISTANT_STREAM_KEY, '')}{delta}"
    return delta


def _is_chunk_message(value: Any) -> bool:
    if _field(value, "timestamp_ms") is not None:
        return True
    if _field(value, "timestampMs") is not None:
        return True
    subtype = _field(value, "subtype")
    return subtype in {"partial", "chunk", "delta"}


def _clean_agent_output_text(text: str) -> str:
    if not text:
        return ""
    cleaned = _ANSI_RE.sub("", text)
    if cleaned != text and not cleaned.strip():
        return ""
    return cleaned


def _is_lab_widget_tool_result_text(text: str) -> bool:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return False
    return _contains_lab_widget_tool_result(payload)


def _contains_lab_widget_tool_result(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    if value.get("ok") is True and isinstance(value.get("tool"), str):
        return True
    result = value.get("result")
    if isinstance(result, str):
        try:
            if _contains_lab_widget_tool_result(json.loads(result)):
                return True
        except json.JSONDecodeError:
            pass
    content_items = value.get("contentItems")
    if isinstance(content_items, list):
        for item in content_items:
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if not isinstance(text, str):
                continue
            try:
                if _contains_lab_widget_tool_result(json.loads(text)):
                    return True
            except json.JSONDecodeError:
                continue
    return False


def _is_lab_widget_tool_event_title(value: str) -> bool:
    normalized = value.strip().lower().replace("-", "_")
    return normalized.startswith(("prime_lab_", "mcp_prime_lab_"))


def _merge_stream_text(existing: str, delta: str) -> str:
    """Append streamed text while ignoring duplicate final snapshots."""

    if not existing:
        return delta
    if not delta:
        return existing
    if not delta.strip():
        return f"{existing}{delta}"
    if delta == existing or existing.endswith(delta):
        return existing
    if existing.startswith(delta):
        return existing
    stripped_delta = delta.strip()
    if (
        len(stripped_delta) >= 24
        and ("\n" in stripped_delta or "." in stripped_delta)
        and stripped_delta in existing
    ):
        return existing
    if delta.startswith(existing):
        return delta
    overlap = _suffix_prefix_overlap(existing, delta)
    if overlap:
        return f"{existing}{delta[overlap:]}"
    return f"{existing}{delta}"


def _dedupe_streamed_text(text: str) -> str:
    """Collapse duplicate adjacent blocks commonly emitted by headless agents."""

    if not text:
        return ""
    stripped = text.strip()
    if not stripped:
        return text
    midpoint = len(stripped) // 2
    if len(stripped) % 2 == 0 and stripped[:midpoint] == stripped[midpoint:]:
        return stripped[:midpoint]

    blocks = re.split(r"(\n{2,})", text)
    output: list[str] = []
    previous_normalized = ""
    for block in blocks:
        if not block:
            continue
        if block.startswith("\n"):
            if output and not str(output[-1]).startswith("\n"):
                output.append(block)
            continue
        normalized = " ".join(block.split())
        if normalized and normalized == previous_normalized:
            if output and str(output[-1]).startswith("\n"):
                output.pop()
            continue
        output.append(block)
        if normalized:
            previous_normalized = normalized
    return "".join(output)


def _suffix_prefix_overlap(existing: str, delta: str) -> int:
    limit = min(len(existing), len(delta))
    for size in range(limit, 0, -1):
        if existing.endswith(delta[:size]):
            return size
    return 0


def _jsonrpc_response_id(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _short_repr(value: Any) -> str:
    rendered = repr(value)
    return rendered if len(rendered) <= 80 else rendered[:77] + "..."


def _agent_prompt_with_lab_context(prompt: str) -> str:
    return (
        "<prime_lab_context>\n"
        "Apply the Prime-managed Lab controls guidance for this session.\n"
        "You are assisting from inside Prime Intellect Lab. Lab is an interactive terminal "
        "research app, not a plain chat window or shell session. The user is using the Lab app "
        "by default, not the CLI. When the next useful step is choosing a Lab "
        "object, editing a config, previewing side effects, launching a run, reviewing a patch, "
        "or inspecting rollouts, use the native Lab tools exposed by this agent surface "
        "instead of narrating manual clicks or CLI steps. Do not suggest CLI commands unless "
        "the user explicitly asks for CLI instructions. Native Lab tools are available "
        "for this session; use them directly for confirmable Lab actions. Do not narrate "
        "repository searches, file formats, docs, folders, resolver order, TOML shape, or other "
        "implementation details unless the user explicitly asks. Do not say that you are "
        "checking, reading, searching, or inspecting workspace files or configs; use a native "
        "Lab control or state the user-facing decision needed. Avoid implementation-facing "
        "words in visible chat such as workflow, skill, template, resolver, workspace file, "
        "or config shape. Do not name internal "
        "implementation surfaces, tool plumbing, or rendering mechanics. For eval creation, "
        "resolve the environment and then open a native eval config editor; do not stop at "
        "environment search prose. For training creation, open a native training launcher once "
        "the required fields are known, or ask the user to choose the missing field.\n"
        "</prime_lab_context>\n\n"
        f"{prompt}"
    )


def _message_id(value: Any, message: Any) -> str:
    for candidate in (message, value):
        for key in ("id", "message_id", "messageId", "uuid"):
            found = _field(candidate, key)
            if isinstance(found, str) and found:
                return found
    return ""


def _content_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list | tuple):
        return "".join(_content_text(item) for item in value)
    text = _field(value, "text")
    if isinstance(text, str):
        return text
    content = _field(value, "content")
    if content is not value:
        nested = _content_text(content)
        if nested:
            return nested
    return ""


def _field(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)
