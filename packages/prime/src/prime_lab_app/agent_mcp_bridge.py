"""Local IPC bridge between MCP servers and the running Lab app."""

from __future__ import annotations

import hashlib
import json
import os
import socket
import sys
import threading
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

LabMcpIpcHandler = Callable[[str, dict[str, Any]], dict[str, Any]]


class LabMcpIpcError(RuntimeError):
    """Raised when a Lab MCP IPC request cannot be completed."""


def lab_mcp_runtime_dir(workspace: Path) -> Path:
    """Runtime directory for a workspace-scoped Lab MCP socket."""

    digest = hashlib.sha256(str(workspace.expanduser().resolve()).encode("utf-8")).hexdigest()[:24]
    root = Path(os.environ.get("PRIME_LAB_RUNTIME_DIR", "/tmp"))
    return root / f"prime-lab-{os.getuid()}" / digest


def lab_mcp_socket_path(workspace: Path) -> Path:
    """Unix socket path used by `prime lab mcp` for this workspace."""

    return lab_mcp_runtime_dir(workspace) / "lab.sock"


def lab_mcp_server_config(workspace: Path) -> dict[str, Any]:
    """MCP server config object for agents that accept JSON MCP definitions."""

    return {
        "command": sys.executable,
        "args": [
            "-c",
            "from prime_cli.main import run; run()",
            "lab",
            "mcp",
            "--workspace",
            str(workspace.expanduser().resolve()),
        ],
    }


def write_lab_mcp_config(workspace: Path, path: Path) -> Path:
    """Write a Prime Lab MCP config file and return its path."""

    payload: dict[str, Any] = {}
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            loaded = {}
        if isinstance(loaded, dict):
            payload.update(loaded)
    servers = payload.get("mcpServers")
    if not isinstance(servers, dict):
        servers = {}
    servers["prime_lab"] = lab_mcp_server_config(workspace)
    payload["mcpServers"] = servers
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def write_amp_mcp_config(workspace: Path, path: Path | None = None) -> Path:
    """Write the Amp config fragment that exposes Prime Lab MCP tools."""

    path = path or workspace / ".prime" / "lab" / "agent-mcp" / "amp.json"
    payload = _read_json_object(path)
    payload["prime_lab"] = lab_mcp_server_config(workspace)
    payload.pop("mcpServers", None)
    payload.pop("amp.mcpServers", None)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def write_opencode_mcp_config(workspace: Path, path: Path | None = None) -> Path:
    """Write the workspace OpenCode config that exposes Prime Lab MCP tools."""

    path = path or workspace / "opencode.json"
    payload = _read_json_object(path)
    mcp = payload.get("mcp")
    if not isinstance(mcp, dict):
        mcp = {}
    server = lab_mcp_server_config(workspace)
    mcp["prime_lab"] = {
        "type": "local",
        "command": [server["command"], *server["args"]],
        "enabled": True,
        "timeout": 60000,
    }
    payload["mcp"] = mcp
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def write_hermes_mcp_config(workspace: Path, path: Path | None = None) -> Path:
    """Write the Hermes user config that exposes Prime Lab MCP tools."""

    path = path or Path.home() / ".hermes" / "config.yaml"
    payload = _read_yaml_object(path)
    servers = payload.get("mcp_servers")
    if not isinstance(servers, dict):
        servers = {}
    server = lab_mcp_server_config(workspace)
    servers["prime_lab"] = {
        "command": server["command"],
        "args": server["args"],
        "enabled": True,
    }
    payload["mcp_servers"] = servers
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_dump_yaml_object(payload), encoding="utf-8")
    return path


class LabMcpIpcServer:
    """Small JSON-lines Unix socket server owned by the running Lab TUI."""

    def __init__(self, workspace: Path, handler: LabMcpIpcHandler) -> None:
        self.workspace = workspace.expanduser().resolve()
        self.path = lab_mcp_socket_path(self.workspace)
        self._handler = handler
        self._socket: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        """Start serving tool calls for the workspace."""

        self.stop()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(str(self.path))
        server.listen()
        self._socket = server
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._serve,
            name=f"lab-mcp-ipc-{self.path.parent.name}",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop serving tool calls and remove the socket path."""

        self._stop_event.set()
        server = self._socket
        self._socket = None
        if server is not None:
            try:
                server.close()
            except OSError:
                pass
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass

    def _serve(self) -> None:
        while not self._stop_event.is_set():
            server = self._socket
            if server is None:
                return
            try:
                conn, _addr = server.accept()
            except OSError:
                return
            threading.Thread(
                target=self._handle_connection,
                args=(conn,),
                name=f"lab-mcp-ipc-client-{self.path.parent.name}",
                daemon=True,
            ).start()

    def _handle_connection(self, conn: socket.socket) -> None:
        with conn:
            try:
                payload = _recv_json_line(conn)
                response = self._handle_payload(payload)
            except Exception as exc:
                response = {"ok": False, "error": str(exc)}
            _send_json_line(conn, response)

    def _handle_payload(self, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise LabMcpIpcError("Invalid Lab MCP payload.")
        tool = payload.get("tool")
        if not isinstance(tool, str) or not tool:
            raise LabMcpIpcError("Lab MCP payload is missing a tool name.")
        arguments = payload.get("arguments")
        if not isinstance(arguments, dict):
            arguments = {}
        result = self._handler(tool, arguments)
        return {"ok": True, "result": result, "request_id": payload.get("request_id")}


def call_lab_mcp_tool(
    workspace: Path,
    tool: str,
    arguments: dict[str, Any] | None = None,
    *,
    timeout: float = 5.0,
) -> dict[str, Any]:
    """Forward one MCP tool call into the running Lab app for a workspace."""

    path = lab_mcp_socket_path(workspace)
    if not path.exists():
        raise LabMcpIpcError(
            "Lab is not running for this workspace. Open `prime lab` before using Lab MCP tools."
        )
    payload = {
        "request_id": str(uuid.uuid4()),
        "tool": tool,
        "arguments": arguments or {},
    }
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.settimeout(timeout)
    try:
        client.connect(str(path))
        _send_json_line(client, payload)
        response = _recv_json_line(client)
    except OSError as exc:
        raise LabMcpIpcError(f"Could not reach the running Lab app: {exc}") from exc
    finally:
        client.close()

    if not isinstance(response, dict):
        raise LabMcpIpcError("Invalid response from the running Lab app.")
    if not response.get("ok"):
        error = response.get("error")
        raise LabMcpIpcError(str(error or "Lab MCP tool call failed."))
    result = response.get("result")
    if not isinstance(result, dict):
        raise LabMcpIpcError("Invalid tool result from the running Lab app.")
    return result


def _recv_json_line(conn: socket.socket) -> Any:
    chunks: list[bytes] = []
    while True:
        chunk = conn.recv(65536)
        if not chunk:
            break
        if b"\n" in chunk:
            before, _sep, _after = chunk.partition(b"\n")
            chunks.append(before)
            break
        chunks.append(chunk)
    data = b"".join(chunks)
    if not data:
        raise LabMcpIpcError("Empty Lab MCP IPC message.")
    return json.loads(data.decode("utf-8"))


def _send_json_line(conn: socket.socket, payload: dict[str, Any]) -> None:
    conn.sendall(json.dumps(payload, separators=(",", ":")).encode("utf-8") + b"\n")


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    except (OSError, json.JSONDecodeError):
        loaded = {}
    return loaded if isinstance(loaded, dict) else {}


def _read_yaml_object(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ModuleNotFoundError:
        return {}
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) if path.is_file() else {}
    except (OSError, yaml.YAMLError):
        loaded = {}
    return loaded if isinstance(loaded, dict) else {}


def _dump_yaml_object(payload: dict[str, Any]) -> str:
    try:
        import yaml
    except ModuleNotFoundError:
        return _dump_simple_yaml(payload)
    return str(yaml.safe_dump(payload, sort_keys=False))


def _dump_simple_yaml(payload: dict[str, Any]) -> str:
    lines: list[str] = []

    def emit_value(key: str, value: Any, indent: int) -> None:
        prefix = " " * indent
        if isinstance(value, dict):
            lines.append(f"{prefix}{key}:")
            for child_key, child_value in value.items():
                emit_value(str(child_key), child_value, indent + 2)
            return
        if isinstance(value, list):
            lines.append(f"{prefix}{key}:")
            for item in value:
                lines.append(f"{prefix}  - {item}")
            return
        if isinstance(value, bool):
            lines.append(f"{prefix}{key}: {'true' if value else 'false'}")
            return
        lines.append(f"{prefix}{key}: {value}")

    for key, value in payload.items():
        emit_value(str(key), value, 0)
    return "\n".join(lines).rstrip() + "\n"
