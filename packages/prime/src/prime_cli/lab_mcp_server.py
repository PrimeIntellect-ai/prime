"""Minimal MCP stdio server for Prime Lab agent integrations."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, BinaryIO

JSON = dict[str, Any]


def run_lab_mcp_server(
    *,
    workspace: Path,
    stdin: BinaryIO | None = None,
    stdout: BinaryIO | None = None,
) -> int:
    """Serve a small MCP endpoint over stdio."""

    server = _LabMcpServer(workspace=workspace.expanduser().resolve())
    reader = stdin or sys.stdin.buffer
    writer = stdout or sys.stdout.buffer
    while True:
        message = _read_message(reader)
        if message is None:
            return 0
        response = server.handle(message)
        if response is not None:
            _write_message(writer, response)


class _LabMcpServer:
    def __init__(self, *, workspace: Path) -> None:
        self.workspace = workspace

    def handle(self, message: JSON) -> JSON | None:
        if "id" not in message:
            return None
        method = message.get("method")
        if method == "initialize":
            return self._response(
                message,
                {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": "prime-lab", "version": "0.1.0"},
                },
            )
        if method == "ping":
            return self._response(message, {})
        if method == "tools/list":
            return self._response(message, {"tools": []})
        if method == "tools/call":
            return self._error(message, -32601, "Prime Lab MCP tools are unavailable.")
        return self._error(message, -32601, f"Unknown method: {method}")

    def _response(self, request: JSON, result: JSON) -> JSON:
        return {"jsonrpc": "2.0", "id": request["id"], "result": result}

    def _error(self, request: JSON, code: int, message: str) -> JSON:
        return {
            "jsonrpc": "2.0",
            "id": request["id"],
            "error": {"code": code, "message": message},
        }


def _read_message(stream: BinaryIO) -> JSON | None:
    headers: dict[str, str] = {}
    while True:
        line = stream.readline()
        if not line:
            return None
        if line in {b"\r\n", b"\n"}:
            break
        key, _, value = line.decode("ascii").partition(":")
        headers[key.strip().lower()] = value.strip()
    length = int(headers.get("content-length", "0"))
    if length <= 0:
        return None
    payload = stream.read(length)
    return json.loads(payload.decode("utf-8"))


def _write_message(stream: BinaryIO, message: JSON) -> None:
    payload = json.dumps(message, separators=(",", ":")).encode("utf-8")
    stream.write(f"Content-Length: {len(payload)}\r\n\r\n".encode("ascii"))
    stream.write(payload)
    stream.flush()
