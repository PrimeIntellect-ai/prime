"""Prime Lab coding-agent metadata and local setup surfaces."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

AgentCapabilityStatus = Literal["supported", "not_supported"]
AgentNativeSurface = Literal[
    "codex_app_server",
    "pi_acp",
    "none",
]
AgentTransport = Literal[
    "codex-app-stdio",
    "claude-agent-sdk",
    "resumable-cli",
    "acp-stdio",
    "one-shot",
]
LabWidgetContract = Literal[
    "codex-dynamic-tools",
    "mcp-stdio-tools",
    "not-supported",
]


@dataclass(frozen=True)
class AgentInstallRequirement:
    """One machine-level executable required for a Lab agent surface."""

    binary: str
    install_command: tuple[str, ...] = ()
    description: str = ""

    def installed(self) -> bool:
        return shutil.which(self.binary) is not None


@dataclass(frozen=True)
class AgentAdapter:
    """Command mapping for one user-facing coding agent."""

    name: str
    label: str
    prompt_prefix: tuple[str, ...]
    server_prefix: tuple[str, ...] = ()
    server_transport: AgentTransport = "one-shot"
    server_description: str = "Generic one-shot prompt execution."
    stream_prefix: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()
    lab_widget_contract: LabWidgetContract = "not-supported"

    def prompt_command(self, prompt: str) -> list[str]:
        return [*self.prompt_prefix, prompt]


@dataclass(frozen=True)
class AgentCapability:
    """Lab readiness and setup metadata for one coding agent."""

    name: str
    label: str
    native_surface: AgentNativeSurface
    requirements: tuple[AgentInstallRequirement, ...] = ()
    expected_surface_paths: tuple[str, ...] = ()
    project_skill_roots: tuple[str, ...] = ()
    user_skill_root: str = ""
    status: AgentCapabilityStatus = "supported"
    unsupported_reason: str = ""

    @property
    def adapter(self) -> AgentAdapter:
        return agent_adapter(self.name)

    def missing_requirements(self) -> tuple[AgentInstallRequirement, ...]:
        return tuple(
            requirement for requirement in self.requirements if not requirement.installed()
        )

    def resolved_surface_paths(self, workspace: Path) -> tuple[Path, ...]:
        workspace = workspace.expanduser().resolve()
        paths: list[Path] = []
        for raw_path in self.expected_surface_paths:
            path = Path(raw_path).expanduser()
            paths.append(path if path.is_absolute() else workspace / path)
        return tuple(paths)


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
        stream_prefix=("claude", "-p", "--output-format", "stream-json"),
        aliases=("claude-code", "claude-cli"),
        lab_widget_contract="mcp-stdio-tools",
    ),
    "cursor": AgentAdapter(
        name="cursor",
        label="Cursor",
        prompt_prefix=("cursor-agent", "-p"),
        stream_prefix=("cursor-agent", "-p", "--output-format", "stream-json"),
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
        lab_widget_contract="mcp-stdio-tools",
    ),
    "pi": AgentAdapter(
        name="pi",
        label="Pi Coding Agent",
        prompt_prefix=("pi", "--print"),
        server_prefix=("pi-acp",),
        server_transport="acp-stdio",
        server_description="Pi Agent Client Protocol stdio transport.",
        stream_prefix=("pi", "--print", "--mode", "json"),
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
        aliases=("hermes-agent",),
        lab_widget_contract="mcp-stdio-tools",
    ),
    "droid": AgentAdapter(
        name="droid",
        label="Factory Droid Agent",
        prompt_prefix=("droid", "exec"),
        stream_prefix=("droid", "exec", "--output-format", "stream-json"),
        aliases=("factory", "factory-droid"),
        lab_widget_contract="mcp-stdio-tools",
    ),
    "amp": AgentAdapter(
        name="amp",
        label="Amp Code",
        prompt_prefix=("amp", "--execute"),
        stream_prefix=("amp", "--execute", "--stream-json"),
        aliases=("amp-code",),
        lab_widget_contract="mcp-stdio-tools",
    ),
}
_AGENT_ALIASES = {
    alias: adapter.name for adapter in KNOWN_AGENT_ADAPTERS.values() for alias in adapter.aliases
}


_CAPABILITIES: dict[str, AgentCapability] = {
    "codex": AgentCapability(
        name="codex",
        label="Codex",
        native_surface="codex_app_server",
        requirements=(AgentInstallRequirement("codex", description="Codex CLI"),),
        project_skill_roots=(".agents/skills",),
        user_skill_root="~/.agents/skills",
    ),
    "claude": AgentCapability(
        name="claude",
        label="Claude",
        native_surface="none",
        requirements=(AgentInstallRequirement("claude", description="Claude CLI"),),
        project_skill_roots=(".claude/skills",),
        user_skill_root="~/.claude/skills",
    ),
    "cursor": AgentCapability(
        name="cursor",
        label="Cursor",
        native_surface="none",
        requirements=(AgentInstallRequirement("cursor-agent", description="Cursor Agent CLI"),),
        project_skill_roots=(".cursor/skills",),
        user_skill_root="~/.cursor/skills",
    ),
    "opencode": AgentCapability(
        name="opencode",
        label="OpenCode",
        native_surface="none",
        requirements=(AgentInstallRequirement("opencode", description="OpenCode CLI"),),
        project_skill_roots=(".opencode/skills",),
        user_skill_root="~/.config/opencode/skills",
    ),
    "pi": AgentCapability(
        name="pi",
        label="Pi Coding Agent",
        native_surface="pi_acp",
        project_skill_roots=(".pi/skills",),
        user_skill_root="~/.pi/agent/skills",
        requirements=(
            AgentInstallRequirement("pi", description="Pi Coding Agent CLI"),
            AgentInstallRequirement(
                "pi-acp",
                install_command=("npm", "install", "-g", "pi-acp"),
                description="Pi ACP bridge",
            ),
        ),
    ),
    "hermes": AgentCapability(
        name="hermes",
        label="Hermes Agent",
        native_surface="none",
        requirements=(AgentInstallRequirement("hermes", description="Hermes Agent CLI"),),
        user_skill_root="~/.hermes/skills",
    ),
    "droid": AgentCapability(
        name="droid",
        label="Factory Droid Agent",
        native_surface="none",
        requirements=(
            AgentInstallRequirement(
                "droid",
                install_command=("npm", "install", "-g", "@factory/cli"),
                description="Factory Droid Agent CLI",
            ),
        ),
        project_skill_roots=(".factory/skills",),
        user_skill_root="~/.factory/skills",
    ),
    "amp": AgentCapability(
        name="amp",
        label="Amp Code",
        native_surface="none",
        requirements=(
            AgentInstallRequirement(
                "amp",
                install_command=("npm", "install", "-g", "@sourcegraph/amp@latest"),
                description="Amp Code CLI",
            ),
        ),
        project_skill_roots=(".agents/skills",),
        user_skill_root="~/.config/agents/skills",
    ),
}
AGENT_DISPLAY_ORDER = (
    "amp",
    "claude",
    "codex",
    "cursor",
    "droid",
    "hermes",
    "opencode",
    "pi",
)


def known_agent_names() -> tuple[str, ...]:
    """Return Lab-supported agent names in stable display order."""

    return AGENT_DISPLAY_ORDER


def agent_capability(name: str) -> AgentCapability:
    """Return the declared Lab capability for a coding agent."""

    normalized = _normalize_agent_name(name)
    capability = _CAPABILITIES.get(normalized)
    if capability is not None:
        return capability
    adapter = agent_adapter(normalized)
    return AgentCapability(
        name=adapter.name,
        label=adapter.label,
        native_surface="none",
        status="not_supported",
        unsupported_reason="This command does not expose a native Lab tool surface.",
    )


def agent_adapter(name: str) -> AgentAdapter:
    """Return a known or generic command adapter."""

    normalized = _normalize_agent_name(name)
    adapter = KNOWN_AGENT_ADAPTERS.get(normalized)
    if adapter is not None:
        return adapter
    return AgentAdapter(name=normalized, label=normalized, prompt_prefix=(normalized,))


def write_agent_native_surface(workspace: Path, agent: str) -> tuple[Path, ...]:
    """Write the native Lab control surface for a supported coding agent."""

    _ = workspace, agent
    return ()


def agent_user_skills_dir(agent: str) -> Path | None:
    """Return the user-level skill root for a supported agent."""

    root = agent_capability(agent).user_skill_root
    return Path(root).expanduser() if root else None


def agent_project_skills_dirs(agent: str, workspace: Path) -> tuple[Path, ...]:
    """Return project-local skill roots for a supported agent."""

    workspace = workspace.expanduser().resolve()
    return tuple(workspace / root for root in agent_capability(agent).project_skill_roots)


def _normalize_agent_name(name: str) -> str:
    raw = name.strip().lower() or "codex"
    return _AGENT_ALIASES.get(raw, raw)
