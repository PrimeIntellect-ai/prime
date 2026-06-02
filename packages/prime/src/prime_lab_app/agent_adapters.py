"""Coding-agent command adapters for Lab."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .agent_mcp_bridge import (
    write_amp_mcp_config,
    write_droid_mcp_config,
    write_grok_mcp_config,
    write_hermes_mcp_config,
    write_lab_mcp_config,
    write_opencode_mcp_config,
)
from .agent_widgets import LAB_WIDGET_TOOLS

_LAB_MCP_ALLOWED_TOOLS = ",".join(f"mcp__prime_lab__{tool.name}" for tool in LAB_WIDGET_TOOLS)

AgentTransport = Literal[
    "stdio-jsonl",
    "stdio-jsonrpc",
    "codex-app-stdio",
    "resumable-cli",
    "letta-bidirectional",
    "websocket",
    "http",
    "mcp-stdio",
    "acp-stdio",
    "acp-http",
    "one-shot",
]
LabWidgetContract = Literal[
    "codex-dynamic-tools",
    "mcp-stdio-tools",
    "pi-extension-tools",
    "letta-external-tools",
    "not-supported",
]


@dataclass(frozen=True)
class AgentServerSpec:
    """Long-running process surface for a coding agent."""

    command: tuple[str, ...]
    transport: AgentTransport
    description: str


@dataclass(frozen=True)
class AgentAdapter:
    """Command mapping for one user-facing coding agent."""

    name: str
    label: str
    prompt_prefix: tuple[str, ...]
    server_prefix: tuple[str, ...]
    server_transport: AgentTransport
    server_description: str
    stream_prefix: tuple[str, ...] = ()
    resume_flag: str | None = None
    session_dir_flag: str | None = None
    mcp_config_flag: str | None = None
    server_workspace_flag: str | None = None
    workspace_flag: str | None = None
    prompt_flag: str | None = None
    prompt_separator: str | None = None
    aliases: tuple[str, ...] = ()
    lab_widget_contract: LabWidgetContract = "not-supported"

    def prompt_command(self, prompt: str) -> list[str]:
        return [*self.prompt_prefix, prompt]

    def server_spec(self, workspace: Path) -> AgentServerSpec:
        command = list(self.server_prefix)
        if command and self.session_dir_flag is not None:
            command.extend([self.session_dir_flag, str(agent_session_dir(workspace, self.name))])
        if command and self.server_workspace_flag is not None:
            command.extend([self.server_workspace_flag, str(workspace.expanduser().resolve())])
        return AgentServerSpec(
            command=tuple(command),
            transport=self.server_transport,
            description=self.server_description,
        )

    def stream_command(
        self,
        prompt: str,
        session_id: str = "",
        *,
        workspace: Path | None = None,
    ) -> list[str]:
        command = list(self.stream_prefix or self.prompt_prefix)
        if self.mcp_config_flag is not None and workspace is not None:
            command.extend([self.mcp_config_flag, str(agent_mcp_config_path(workspace, self.name))])
        if self.workspace_flag is not None and workspace is not None:
            command.extend([self.workspace_flag, str(workspace.expanduser().resolve())])
        if self.session_dir_flag is not None and workspace is not None:
            command.extend([self.session_dir_flag, str(agent_session_dir(workspace, self.name))])
        if session_id and self.resume_flag is not None:
            command.extend([self.resume_flag, session_id])
        if self.prompt_flag is not None:
            command.extend([self.prompt_flag, prompt])
        else:
            if self.prompt_separator is not None:
                command.append(self.prompt_separator)
            command.append(prompt)
        return command


@dataclass(frozen=True)
class AgentAction:
    """Action surface exposed by the agent chat screen."""

    key: str
    label: str
    detail: str


AGENT_ACTIONS = (
    AgentAction("prompt", "Run prompt", "Send one prompt to the selected coding agent."),
    AgentAction("server", "Start server", "Start the selected agent's server transport."),
)

KNOWN_AGENT_ADAPTERS = {
    "codex": AgentAdapter(
        name="codex",
        label="Codex",
        prompt_prefix=("codex", "exec"),
        server_prefix=("codex", "app-server", "--listen", "stdio://"),
        server_transport="codex-app-stdio",
        server_description="Codex app-server JSON-RPC transport.",
        lab_widget_contract="codex-dynamic-tools",
    ),
    "claude": AgentAdapter(
        name="claude",
        label="Claude",
        prompt_prefix=("claude", "-p"),
        server_prefix=(),
        server_transport="resumable-cli",
        server_description="Claude Code headless CLI with resumable stream-json sessions.",
        stream_prefix=(
            "claude",
            "-p",
            "--output-format",
            "stream-json",
            "--verbose",
            "--include-partial-messages",
            "--allowedTools",
            _LAB_MCP_ALLOWED_TOOLS,
        ),
        resume_flag="--resume",
        mcp_config_flag="--mcp-config",
        prompt_separator="--",
        aliases=("claude-code", "claude-cli"),
        lab_widget_contract="mcp-stdio-tools",
    ),
    "cursor": AgentAdapter(
        name="cursor",
        label="Cursor",
        prompt_prefix=("cursor-agent", "-p"),
        server_prefix=(),
        server_transport="resumable-cli",
        server_description="Cursor Agent headless CLI with resumable stream-json sessions.",
        stream_prefix=(
            "cursor-agent",
            "-p",
            "--output-format",
            "stream-json",
            "--stream-partial-output",
            "--trust",
            "--approve-mcps",
            "--force",
        ),
        resume_flag="--resume",
        workspace_flag="--workspace",
        lab_widget_contract="mcp-stdio-tools",
    ),
    "opencode": AgentAdapter(
        name="opencode",
        label="OpenCode",
        prompt_prefix=("opencode", "run"),
        server_prefix=("opencode", "acp"),
        server_transport="acp-stdio",
        server_description="OpenCode Agent Client Protocol stdio transport.",
        stream_prefix=("opencode", "run", "--format", "json"),
        resume_flag="--session",
        server_workspace_flag="--cwd",
        workspace_flag="--dir",
        lab_widget_contract="mcp-stdio-tools",
    ),
    "pi": AgentAdapter(
        name="pi",
        label="Pi Coding Agent",
        prompt_prefix=("pi", "--print"),
        server_prefix=(),
        server_transport="resumable-cli",
        server_description="Pi headless CLI with project-local extension tools.",
        stream_prefix=("pi", "--print", "--mode", "json", "--no-session"),
        lab_widget_contract="pi-extension-tools",
    ),
    "grok": AgentAdapter(
        name="grok",
        label="Grok Build",
        prompt_prefix=("grok", "--no-auto-update", "-p"),
        server_prefix=("grok", "--no-auto-update", "agent", "stdio"),
        server_transport="acp-stdio",
        server_description="Grok Build Agent Client Protocol stdio transport.",
        stream_prefix=("grok", "--no-auto-update", "--output-format", "streaming-json"),
        resume_flag="--resume",
        workspace_flag="--cwd",
        prompt_flag="-p",
        lab_widget_contract="mcp-stdio-tools",
    ),
    "hermes": AgentAdapter(
        name="hermes",
        label="Hermes Agent",
        prompt_prefix=("hermes", "--oneshot"),
        server_prefix=("hermes", "acp", "--accept-hooks"),
        server_transport="acp-stdio",
        server_description="Hermes Agent Client Protocol stdio transport.",
        stream_prefix=("hermes", "chat", "--quiet", "--accept-hooks", "--source", "prime-lab"),
        resume_flag="--resume",
        prompt_flag="--query",
        lab_widget_contract="mcp-stdio-tools",
        aliases=("hermes-agent",),
    ),
    "letta": AgentAdapter(
        name="letta",
        label="Letta Code",
        prompt_prefix=("letta", "-p", "--skills", ".agents/skills"),
        server_prefix=(
            "letta",
            "-p",
            "--skills",
            ".agents/skills",
            "--input-format",
            "stream-json",
            "--output-format",
            "stream-json",
            "--include-partial-messages",
        ),
        server_transport="letta-bidirectional",
        server_description="Letta Code bidirectional stream-json CLI with Lab external tools.",
        stream_prefix=(
            "letta",
            "-p",
            "--skills",
            ".agents/skills",
            "--output-format",
            "stream-json",
            "--include-partial-messages",
        ),
        aliases=("letta-code",),
        lab_widget_contract="letta-external-tools",
    ),
    "droid": AgentAdapter(
        name="droid",
        label="Factory Droid Agent",
        prompt_prefix=("droid", "exec"),
        server_prefix=(),
        server_transport="resumable-cli",
        server_description="Factory Droid Agent headless CLI with resumable JSON sessions.",
        stream_prefix=("droid", "exec", "--output-format", "stream-json"),
        workspace_flag="--cwd",
        aliases=("factory", "factory-droid"),
        lab_widget_contract="mcp-stdio-tools",
    ),
    "amp": AgentAdapter(
        name="amp",
        label="Amp Code",
        prompt_prefix=("amp", "--execute"),
        server_prefix=(),
        server_transport="resumable-cli",
        server_description="Amp Code headless CLI with stream-json output.",
        stream_prefix=("amp", "--stream-json"),
        mcp_config_flag="--mcp-config",
        prompt_flag="--execute",
        aliases=("amp-code",),
        lab_widget_contract="mcp-stdio-tools",
    ),
}
_AGENT_ALIASES = {
    alias: adapter.name for adapter in KNOWN_AGENT_ADAPTERS.values() for alias in adapter.aliases
}


def agent_session_dir(workspace: Path, agent: str) -> Path:
    """Workspace-scoped storage for persistent agent sessions."""

    return workspace / ".prime" / "lab" / "agent-sessions" / agent


def agent_mcp_config_path(workspace: Path, agent: str) -> Path:
    """Workspace-scoped MCP config file for an agent."""

    return workspace / ".prime" / "lab" / "agent-mcp" / f"{agent}.json"


def write_agent_mcp_config(workspace: Path, agent: str) -> Path:
    """Write the Prime Lab MCP server config for a native-MCP agent."""

    if agent == "cursor":
        return write_lab_mcp_config(workspace, workspace / ".cursor" / "mcp.json")
    path = agent_mcp_config_path(workspace, agent)
    return write_lab_mcp_config(workspace, path)


def write_agent_native_surface(workspace: Path, agent: str) -> tuple[Path, ...]:
    """Write the native Lab control surface for a supported coding agent."""

    adapter = agent_adapter(agent)
    if adapter.lab_widget_contract == "pi-extension-tools":
        return (_write_pi_lab_extension(workspace),)
    if adapter.lab_widget_contract == "mcp-stdio-tools":
        if adapter.name == "amp":
            return (write_amp_mcp_config(workspace),)
        if adapter.name == "droid":
            return (write_droid_mcp_config(workspace),)
        if adapter.name == "opencode":
            return (write_opencode_mcp_config(workspace),)
        if adapter.name == "hermes":
            return (write_hermes_mcp_config(workspace),)
        if adapter.name == "grok":
            return (write_grok_mcp_config(workspace),)
        return (write_agent_mcp_config(workspace, adapter.name),)
    return ()


def pi_lab_extension_path(workspace: Path) -> Path:
    """Project-local Pi extension that exposes Prime Lab native controls."""

    return workspace / ".pi" / "extensions" / "prime-lab" / "index.ts"


def _write_pi_lab_extension(workspace: Path) -> Path:
    path = pi_lab_extension_path(workspace)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_pi_lab_extension_source(), encoding="utf-8")
    return path


def _pi_lab_extension_source() -> str:
    specs = [
        {
            "name": tool.name,
            "label": f"Lab {tool.name.replace('_', ' ')}",
            "description": tool.description,
            "parameters": {
                "type": "object",
                "properties": tool.properties,
                "required": list(tool.required),
                "additionalProperties": True,
            },
        }
        for tool in LAB_WIDGET_TOOLS
    ]
    specs_json = json.dumps(specs, indent=2, sort_keys=True)
    return f"""import net from "node:net";
import os from "node:os";
import path from "node:path";
import crypto from "node:crypto";
import type {{ ExtensionAPI }} from "@mariozechner/pi-coding-agent";

const TOOLS = {specs_json};

function labSocketPath(cwd: string): string {{
  const workspace = path.resolve(cwd);
  const digest = crypto.createHash("sha256").update(workspace).digest("hex").slice(0, 24);
  const root = process.env.PRIME_LAB_RUNTIME_DIR || os.tmpdir();
  return path.join(root, `prime-lab-${{os.userInfo().uid}}`, digest, "lab.sock");
}}

function callLab(
  workspace: string,
  tool: string,
  args: Record<string, unknown>,
): Promise<unknown> {{
  const socketPath = labSocketPath(workspace);
  return new Promise((resolve, reject) => {{
    const client = net.createConnection(socketPath);
    let data = "";
    client.setTimeout(5000);
    client.on("connect", () => {{
      client.write(JSON.stringify({{
        request_id: crypto.randomUUID(),
        tool,
        arguments: args || {{}},
      }}) + "\\n");
    }});
    client.on("data", chunk => {{
      data += chunk.toString("utf8");
      if (data.includes("\\n")) {{
        client.end();
      }}
    }});
    client.on("timeout", () => {{
      client.destroy(new Error("Prime Lab IPC timed out."));
    }});
    client.on("error", reject);
    client.on("close", () => {{
      if (!data.trim()) {{
        reject(new Error("Prime Lab IPC returned no response."));
        return;
      }}
      try {{
        const response = JSON.parse(data.trim());
        if (!response.ok) {{
          reject(new Error(String(response.error || "Prime Lab tool call failed.")));
          return;
        }}
        resolve(response.result);
      }} catch (error) {{
        reject(error);
      }}
    }});
  }});
}}

export default function primeLabExtension(pi: ExtensionAPI) {{
  for (const tool of TOOLS) {{
    pi.registerTool({{
      name: tool.name,
      label: tool.label,
      description: tool.description,
      parameters: tool.parameters as any,
      async execute(_toolCallId, params, _signal, _onUpdate, ctx) {{
        const result = await callLab(ctx.cwd, tool.name, params as Record<string, unknown>);
        return {{
          content: [{{ type: "text", text: JSON.stringify(result) }}],
          details: result,
        }};
      }},
    }});
  }}
}}
"""


def agent_adapter(name: str) -> AgentAdapter:
    """Return a known or generic command adapter."""

    normalized = (name.strip() or "codex").lower()
    normalized = _AGENT_ALIASES.get(normalized, normalized)
    adapter = KNOWN_AGENT_ADAPTERS.get(normalized)
    if adapter is not None:
        return adapter
    return AgentAdapter(
        name=normalized,
        label=normalized,
        prompt_prefix=(normalized,),
        server_prefix=(),
        server_transport="one-shot",
        server_description="Generic one-shot prompt execution.",
    )


def agent_select_options(active_agent: str) -> list[tuple[str, str]]:
    """Build stable Select options with the active agent first."""

    from .agent_capabilities import agent_select_options as _agent_select_options

    return _agent_select_options(active_agent)
