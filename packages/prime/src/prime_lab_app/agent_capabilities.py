"""Central registry for Lab-supported coding agent capabilities."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .agent_adapters import (
    AgentAdapter,
    agent_adapter,
    agent_mcp_config_path,
)

AgentCapabilityStatus = Literal["supported", "needs_setup", "not_supported"]
AgentNativeSurface = Literal[
    "codex_app_server",
    "claude_sdk",
    "mcp_config",
    "acp_mcp",
    "pi_acp",
    "none",
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
class AgentCapability:
    """Lab readiness and setup metadata for one coding agent."""

    name: str
    label: str
    native_surface: AgentNativeSurface
    requirements: tuple[AgentInstallRequirement, ...] = ()
    expected_surface_paths: tuple[str, ...] = ()
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
            if raw_path == "{claude_code_mcp}":
                paths.append(agent_mcp_config_path(workspace, "claude-code"))
            elif raw_path == "{cursor_mcp}":
                paths.append(workspace / ".cursor" / "mcp.json")
            elif raw_path == "{opencode_config}":
                paths.append(workspace / "opencode.json")
            elif raw_path == "{hermes_config}":
                paths.append(Path.home() / ".hermes" / "config.yaml")
            else:
                path = Path(raw_path).expanduser()
                paths.append(path if path.is_absolute() else workspace / path)
        return tuple(paths)


_CAPABILITIES: dict[str, AgentCapability] = {
    "codex": AgentCapability(
        name="codex",
        label="Codex",
        native_surface="codex_app_server",
        requirements=(AgentInstallRequirement("codex", description="Codex CLI"),),
    ),
    "claude": AgentCapability(
        name="claude",
        label="Claude",
        native_surface="claude_sdk",
    ),
    "claude-code": AgentCapability(
        name="claude-code",
        label="Claude Code",
        native_surface="mcp_config",
        requirements=(AgentInstallRequirement("claude", description="Claude Code CLI"),),
        expected_surface_paths=("{claude_code_mcp}",),
    ),
    "cursor": AgentCapability(
        name="cursor",
        label="Cursor",
        native_surface="mcp_config",
        requirements=(AgentInstallRequirement("cursor-agent", description="Cursor Agent CLI"),),
        expected_surface_paths=("{cursor_mcp}",),
    ),
    "opencode": AgentCapability(
        name="opencode",
        label="OpenCode",
        native_surface="acp_mcp",
        requirements=(AgentInstallRequirement("opencode", description="OpenCode CLI"),),
        expected_surface_paths=("{opencode_config}",),
    ),
    "pi": AgentCapability(
        name="pi",
        label="Pi Coding Agent",
        native_surface="pi_acp",
        requirements=(
            AgentInstallRequirement("pi", description="Pi Coding Agent CLI"),
            AgentInstallRequirement(
                "pi-acp",
                install_command=("npm", "install", "-g", "pi-acp"),
                description="Pi ACP bridge",
            ),
        ),
    ),
    "hermes-agent": AgentCapability(
        name="hermes-agent",
        label="Hermes Agent",
        native_surface="acp_mcp",
        requirements=(AgentInstallRequirement("hermes", description="Hermes Agent CLI"),),
        expected_surface_paths=("{hermes_config}",),
    ),
}
_ALIASES = {
    "claude-cli": "claude-code",
    "hermes": "hermes-agent",
}


def known_agent_names() -> tuple[str, ...]:
    """Return Lab-supported agent names in stable display order."""

    return tuple(_CAPABILITIES)


def agent_capability(name: str) -> AgentCapability:
    """Return the declared Lab capability for a coding agent."""

    normalized = _ALIASES.get(name.strip() or "codex", name.strip() or "codex")
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


def agent_select_options(active_agent: str) -> list[tuple[str, str]]:
    """Build stable Select options with the active agent first."""

    names = [active_agent, *known_agent_names()]
    seen: set[str] = set()
    options: list[tuple[str, str]] = []
    for name in names:
        normalized = _ALIASES.get(name.strip(), name.strip())
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        capability = agent_capability(normalized)
        options.append((capability.label, capability.name))
    return options
