"""Workspace-native Lab MCP config writers."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


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
    """Write a standard MCP config containing the Prime Lab server."""

    payload = _read_json_object(path)
    servers = payload.get("mcpServers")
    if not isinstance(servers, dict):
        servers = {}
    servers["prime_lab"] = lab_mcp_server_config(workspace)
    payload["mcpServers"] = servers
    _write_json_object(path, payload)
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
    _write_json_object(path, payload)
    return path


def write_factory_mcp_config(workspace: Path, path: Path | None = None) -> Path:
    """Write Factory Droid's project MCP config."""

    path = path or workspace / ".factory" / "mcp.json"
    payload = _read_json_object(path)
    servers = payload.get("mcpServers")
    if not isinstance(servers, dict):
        servers = {}
    server = lab_mcp_server_config(workspace)
    servers["prime_lab"] = {
        "type": "stdio",
        "command": server["command"],
        "args": server["args"],
        "disabled": False,
    }
    payload["mcpServers"] = servers
    _write_json_object(path, payload)
    return path


def write_amp_mcp_config(workspace: Path, path: Path | None = None) -> Path:
    """Write Amp workspace settings with the Prime Lab MCP server."""

    path = path or workspace / ".amp" / "settings.json"
    payload = _read_json_object(path)
    servers = payload.get("amp.mcpServers")
    if not isinstance(servers, dict):
        servers = {}
    servers["prime_lab"] = lab_mcp_server_config(workspace)
    payload["amp.mcpServers"] = servers
    _write_json_object(path, payload)
    return path


def write_hermes_mcp_config(workspace: Path, path: Path | None = None) -> Path:
    """Write the Hermes user config that exposes Prime Lab MCP tools."""

    path = path or Path.home() / ".hermes" / "config.yaml"
    server = lab_mcp_server_config(workspace)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.read_text(encoding="utf-8") if path.is_file() else ""
    path.write_text(_upsert_hermes_prime_lab_server(existing, server), encoding="utf-8")
    return path


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    except json.JSONDecodeError:
        loaded = {}
    return loaded if isinstance(loaded, dict) else {}


def _write_json_object(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _upsert_hermes_prime_lab_server(existing: str, server: dict[str, Any]) -> str:
    lines = existing.splitlines()
    block = [
        "  prime_lab:",
        f"    command: {json.dumps(server['command'])}",
        f"    args: {json.dumps(server['args'])}",
        "    enabled: true",
    ]
    section_start = _top_level_section_index(lines, "mcp_servers")
    if section_start is None:
        prefix = [*lines, ""] if lines else []
        return "\n".join([*prefix, "mcp_servers:", *block]) + "\n"

    section_end = _next_top_level_index(lines, section_start + 1)
    body = lines[section_start + 1 : section_end]
    body = _remove_yaml_child_block(body, "prime_lab")
    updated = [
        *lines[: section_start + 1],
        *block,
        *body,
        *lines[section_end:],
    ]
    return "\n".join(updated) + "\n"


def _top_level_section_index(lines: list[str], section: str) -> int | None:
    target = f"{section}:"
    for index, line in enumerate(lines):
        if line == target:
            return index
    return None


def _next_top_level_index(lines: list[str], start: int) -> int:
    for index in range(start, len(lines)):
        line = lines[index]
        if line and not line.startswith((" ", "\t")):
            return index
    return len(lines)


def _remove_yaml_child_block(lines: list[str], child: str) -> list[str]:
    target = f"  {child}:"
    result: list[str] = []
    index = 0
    while index < len(lines):
        if lines[index] != target:
            result.append(lines[index])
            index += 1
            continue
        index += 1
        while index < len(lines) and (not lines[index].strip() or lines[index].startswith("    ")):
            index += 1
    return result
