"""Long-lived coding-agent server runtime for Lab."""

from __future__ import annotations

import json
import subprocess
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .agent_adapters import AgentAdapter, AgentServerSpec, agent_adapter

AgentStatus = Literal["none", "starting", "connected", "error", "stopped"]
AgentRole = Literal["user", "assistant", "system"]


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


StateCallback = Callable[[AgentConnectionState], None]
MessagesCallback = Callable[[tuple[AgentChatMessage, ...]], None]
PopenFactory = Callable[..., subprocess.Popen[str]]


class AgentRuntime:
    """Owns one workspace-scoped agent server process and chat session."""

    def __init__(
        self,
        *,
        on_state: StateCallback | None = None,
        on_messages: MessagesCallback | None = None,
        popen_factory: PopenFactory = subprocess.Popen,
    ) -> None:
        self._on_state = on_state
        self._on_messages = on_messages
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

    def start(self, workspace: Path, agent_name: str) -> None:
        agent_name = agent_name.strip()
        if not agent_name:
            self.stop()
            self._set_state(AgentConnectionState(status="none", message="No agent configured"))
            return

        adapter = agent_adapter(agent_name)
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
        self._set_state(
            AgentConnectionState(
                agent=adapter.name,
                label=adapter.label,
                status="starting",
                transport=spec.transport,
                workspace=workspace,
                message="Starting server",
            )
        )

        if spec.transport == "one-shot":
            self._set_connected(message="One-shot exec ready")
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
                self._append_message_locked(
                    AgentChatMessage("system", "No connected coding agent.", "error")
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
            elif self._spec.transport == "one-shot":
                self._append_message_locked(
                    AgentChatMessage("user", prompt),
                    AgentChatMessage("assistant", "", "streaming"),
                )
                transport = "one-shot"
                session_id = ""
            else:
                self._append_message_locked(
                    AgentChatMessage("user", prompt),
                    AgentChatMessage("assistant", "", "streaming"),
                )
                transport = "one-shot"
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
                if transport == "one-shot":
                    threading.Thread(
                        target=self._run_one_shot_prompt,
                        args=(prompt,),
                        name=f"lab-agent-{self._agent}-one-shot",
                        daemon=True,
                    ).start()
                    return
                self._request(
                    "session/prompt",
                    {
                        "sessionId": session_id,
                        "messageId": str(uuid.uuid4()),
                        "prompt": [{"type": "text", "text": prompt}],
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
                    content = last.content or "(completed without response text)"
                    self._messages[-1] = AgentChatMessage("assistant", content)
                    self._emit_messages_locked()

    def _run_one_shot_prompt(self, prompt: str) -> None:
        with self._lock:
            adapter = self._adapter
            workspace = self._workspace
        if adapter is None or workspace is None:
            with self._lock:
                self._replace_last_streaming_locked(
                    AgentChatMessage("system", "No one-shot agent is configured.", "error")
                )
                self._emit_messages_locked()
            return
        try:
            process = self._popen_factory(
                adapter.prompt_command(prompt),
                cwd=workspace,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            with self._lock:
                self._replace_last_streaming_locked(
                    AgentChatMessage("system", f"Agent request failed: {exc}", "error")
                )
                self._emit_messages_locked()
            return

        if process.stdout is not None:
            for line in process.stdout:
                self._append_streaming_assistant_text(line)
        code = process.wait()
        with self._lock:
            if code != 0:
                last_text = (
                    self._messages[-1].content
                    if self._messages and self._messages[-1].status == "streaming"
                    else ""
                )
                message = f"Agent command exited with {code}"
                if last_text.strip():
                    message = f"{message}\n\n{last_text.rstrip()}"
                self._replace_last_streaming_locked(AgentChatMessage("system", message, "error"))
            elif self._messages and self._messages[-1].status == "streaming":
                last = self._messages[-1]
                self._messages[-1] = AgentChatMessage(
                    "assistant",
                    last.content or "(completed without response text)",
                )
            self._emit_messages_locked()

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
            for method in init.get("authMethods") or []:
                method_id = method.get("id") if isinstance(method, dict) else None
                if method_id:
                    self._request("authenticate", {"methodId": method_id}, timeout=12)
            session = self._request(
                "session/new",
                {"cwd": str(workspace), "mcpServers": []},
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
            self._complete_pending(int(request_id), message)
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

    def _handle_session_update(self, params: dict[str, Any]) -> None:
        update = params.get("update")
        if not isinstance(update, dict):
            return
        update_type = update.get("sessionUpdate")
        content = update.get("content")
        if update_type not in {"agent_message_chunk", "agent_thought_chunk"}:
            return
        if not isinstance(content, dict) or content.get("type") != "text":
            return
        text = content.get("text")
        if not isinstance(text, str) or not text:
            return
        with self._lock:
            if not self._messages or self._messages[-1].role != "assistant":
                self._messages.append(AgentChatMessage("assistant", text, "streaming"))
            else:
                last = self._messages[-1]
                self._messages[-1] = AgentChatMessage(
                    "assistant",
                    f"{last.content}{text}",
                    "streaming",
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
            elif self._messages and self._messages[-1].status == "streaming":
                last = self._messages[-1]
                self._messages[-1] = AgentChatMessage(
                    "assistant",
                    last.content or "(completed without response text)",
                )
            self._active_turn_id = ""
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
        with self._lock:
            if not self._messages or self._messages[-1].role != "assistant":
                self._messages.append(AgentChatMessage("assistant", text, "streaming"))
            else:
                last = self._messages[-1]
                self._messages[-1] = AgentChatMessage(
                    "assistant",
                    f"{last.content}{text}",
                    "streaming",
                )
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
        if self._messages and self._messages[-1].status == "streaming":
            self._messages[-1] = message
        else:
            self._messages.append(message)

    def _emit_messages_locked(self) -> None:
        if self._on_messages is not None:
            self._on_messages(tuple(self._messages))
