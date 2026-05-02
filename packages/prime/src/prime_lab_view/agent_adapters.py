"""Coding-agent command adapters for Lab."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

AgentTransport = Literal[
    "stdio-jsonl",
    "stdio-jsonrpc",
    "codex-app-stdio",
    "websocket",
    "http",
    "mcp-stdio",
    "acp-stdio",
    "acp-http",
    "one-shot",
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
    session_dir_flag: str | None = None
    aliases: tuple[str, ...] = ()

    def prompt_command(self, prompt: str) -> list[str]:
        return [*self.prompt_prefix, prompt]

    def server_spec(self, workspace: Path) -> AgentServerSpec:
        command = list(self.server_prefix)
        if self.session_dir_flag is not None:
            command.extend([self.session_dir_flag, str(agent_session_dir(workspace, self.name))])
        return AgentServerSpec(
            command=tuple(command),
            transport=self.server_transport,
            description=self.server_description,
        )


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
    ),
    "claude": AgentAdapter(
        name="claude",
        label="Claude Code",
        prompt_prefix=("claude", "-p"),
        server_prefix=(),
        server_transport="one-shot",
        server_description="One-shot prompt execution.",
    ),
    "opencode": AgentAdapter(
        name="opencode",
        label="OpenCode",
        prompt_prefix=("opencode", "run"),
        server_prefix=("opencode", "acp", "--hostname", "127.0.0.1", "--port", "0"),
        server_transport="acp-http",
        server_description="OpenCode Agent Client Protocol server.",
    ),
    "pi": AgentAdapter(
        name="pi",
        label="Pi Coding Agent",
        prompt_prefix=("pi", "-p"),
        server_prefix=("pi", "--mode", "rpc"),
        server_transport="stdio-jsonrpc",
        server_description="Pi RPC mode over LF-delimited JSON records.",
        session_dir_flag="--session-dir",
    ),
    "hermes-agent": AgentAdapter(
        name="hermes-agent",
        label="Hermes Agent",
        prompt_prefix=("hermes", "--oneshot"),
        server_prefix=("hermes", "acp"),
        server_transport="acp-stdio",
        server_description="Hermes Agent Client Protocol server.",
        aliases=("hermes",),
    ),
}
_AGENT_ALIASES = {
    alias: adapter.name for adapter in KNOWN_AGENT_ADAPTERS.values() for alias in adapter.aliases
}


def agent_session_dir(workspace: Path, agent: str) -> Path:
    """Workspace-scoped storage for persistent agent sessions."""

    return workspace / ".prime" / "lab" / "agent-sessions" / agent


def agent_adapter(name: str) -> AgentAdapter:
    """Return a known or generic command adapter."""

    normalized = name.strip() or "codex"
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

    names = [active_agent, "codex", "opencode", "pi", "hermes-agent"]
    seen: set[str] = set()
    options: list[tuple[str, str]] = []
    for name in names:
        normalized = name.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        adapter = agent_adapter(normalized)
        options.append((adapter.label, adapter.name))
    return options
