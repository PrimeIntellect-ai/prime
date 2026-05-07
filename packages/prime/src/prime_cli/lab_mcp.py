"""MCP server for Lab-native widget tools."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, TextIO

from prime_lab_app.agent_mcp_bridge import (
    LabMcpIpcError,
    call_lab_mcp_tool,
)
from prime_lab_app.agent_widgets import lab_dynamic_tools

MCP_PROTOCOL_VERSION = "2024-11-05"


def run_lab_mcp_server(workspace: Path) -> None:
    """Run a minimal stdio MCP server that forwards tools into the Lab TUI."""

    _serve_lab_mcp_stdio(workspace.expanduser().resolve(), sys.stdin, sys.stdout)


def lab_mcp_tool_definitions() -> list[dict[str, Any]]:
    """MCP tool definitions mirrored from Lab's native widget contracts."""

    tools: list[dict[str, Any]] = []
    for spec in lab_dynamic_tools():
        name = str(spec.get("name") or "")
        if not name:
            continue
        input_schema = spec.get("inputSchema")
        if not isinstance(input_schema, dict):
            input_schema = {"type": "object", "additionalProperties": True}
        tools.append(
            {
                "name": name,
                "description": str(spec.get("description") or ""),
                "inputSchema": input_schema,
            }
        )
    return tools


__all__ = [
    "lab_mcp_tool_definitions",
    "run_lab_mcp_server",
]


def _serve_lab_mcp_stdio(workspace: Path, stdin: TextIO, stdout: TextIO) -> None:
    for raw_line in stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            _write_jsonrpc(stdout, _error_response(None, -32700, "Parse error"))
            continue
        if not isinstance(request, dict):
            _write_jsonrpc(stdout, _error_response(None, -32600, "Invalid request"))
            continue
        response = _handle_mcp_request(workspace, request)
        if response is not None:
            _write_jsonrpc(stdout, response)


def _handle_mcp_request(workspace: Path, request: dict[str, Any]) -> dict[str, Any] | None:
    request_id = request.get("id")
    method = request.get("method")
    if not isinstance(method, str):
        return _error_response(request_id, -32600, "Invalid request")

    if method.startswith("notifications/"):
        return None
    if method == "initialize":
        return _result_response(
            request_id,
            {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "prime-lab", "version": "0.1.0"},
                "instructions": (
                    "Prime Lab tools display native interactive controls inside the running "
                    "Lab TUI. Use them when the next step asks the user to choose, edit, "
                    "preview, launch, review a patch, or inspect rollouts."
                ),
            },
        )
    if method == "ping":
        return _result_response(request_id, {})
    if method == "tools/list":
        return _result_response(request_id, {"tools": lab_mcp_tool_definitions()})
    if method == "tools/call":
        params = request.get("params")
        if not isinstance(params, dict):
            return _error_response(request_id, -32602, "Invalid tool call params")
        name = params.get("name")
        if not isinstance(name, str) or not name:
            return _error_response(request_id, -32602, "Missing tool name")
        arguments = params.get("arguments")
        if not isinstance(arguments, dict):
            arguments = {}
        result = _call_lab_tool(workspace, name, arguments)
        return _result_response(
            request_id,
            {
                "content": [{"type": "text", "text": json.dumps(result, sort_keys=True)}],
                "structuredContent": result,
                "isError": _tool_result_is_error(result),
            },
        )
    return _error_response(request_id, -32601, f"Method not found: {method}")


def _tool_result_is_error(result: dict[str, Any]) -> bool:
    if "ok" in result:
        return not bool(result["ok"])
    if "success" in result:
        return not bool(result["success"])
    return bool(result.get("error"))


def _call_lab_tool(workspace: Path, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    try:
        return call_lab_mcp_tool(workspace, name, arguments)
    except LabMcpIpcError as exc:
        return {"ok": False, "error": str(exc), "tool": name}


def _result_response(request_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _error_response(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": code, "message": message},
    }


def _write_jsonrpc(stdout: TextIO, payload: dict[str, Any]) -> None:
    stdout.write(json.dumps(payload, separators=(",", ":")) + "\n")
    stdout.flush()
