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
    pi_lab_extension_path,
)

AgentCapabilityStatus = Literal["supported", "needs_setup", "not_supported"]
AgentNativeSurface = Literal[
    "codex_app_server",
    "mcp_config",
    "acp_mcp",
    "droid_mcp_config",
    "pi_extension",
    "letta_external_tools",
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
            if raw_path == "{claude_mcp}":
                paths.append(agent_mcp_config_path(workspace, "claude"))
                continue
            if raw_path == "{amp_mcp}":
                paths.append(agent_mcp_config_path(workspace, "amp"))
                continue
            if raw_path == "{cursor_mcp}":
                paths.append(workspace / ".cursor" / "mcp.json")
                continue
            if raw_path == "{droid_mcp}":
                paths.append(workspace / ".factory" / "mcp.json")
                continue
            if raw_path == "{grok_config}":
                paths.append(workspace / ".grok" / "config.toml")
                continue
            if raw_path == "{opencode_config}":
                paths.append(workspace / "opencode.json")
                continue
            if raw_path == "{hermes_config}":
                paths.append(Path.home() / ".hermes" / "config.yaml")
                continue
            if raw_path == "{pi_extension}":
                paths.append(pi_lab_extension_path(workspace))
                continue
            path = Path(raw_path).expanduser()
            paths.append(path if path.is_absolute() else workspace / path)
        return tuple(paths)


_CAPABILITIES: dict[str, AgentCapability] = {
    "amp": AgentCapability(
        name="amp",
        label="Amp Code",
        native_surface="mcp_config",
        requirements=(
            AgentInstallRequirement(
                "amp",
                install_command=("npm", "install", "-g", "@sourcegraph/amp@latest"),
                description="Amp Code CLI",
            ),
        ),
        expected_surface_paths=("{amp_mcp}",),
    ),
    "claude": AgentCapability(
        name="claude",
        label="Claude",
        native_surface="mcp_config",
        requirements=(AgentInstallRequirement("claude", description="Claude Code CLI"),),
        expected_surface_paths=("{claude_mcp}",),
    ),
    "codex": AgentCapability(
        name="codex",
        label="Codex",
        native_surface="codex_app_server",
        requirements=(AgentInstallRequirement("codex", description="Codex CLI"),),
    ),
    "cursor": AgentCapability(
        name="cursor",
        label="Cursor",
        native_surface="mcp_config",
        requirements=(AgentInstallRequirement("cursor-agent", description="Cursor Agent CLI"),),
        expected_surface_paths=("{cursor_mcp}",),
    ),
    "droid": AgentCapability(
        name="droid",
        label="Factory Droid Agent",
        native_surface="droid_mcp_config",
        requirements=(
            AgentInstallRequirement(
                "droid",
                install_command=("npm", "install", "-g", "@factory/cli"),
                description="Factory Droid Agent CLI",
            ),
        ),
        expected_surface_paths=("{droid_mcp}",),
    ),
    "grok": AgentCapability(
        name="grok",
        label="Grok Build",
        native_surface="acp_mcp",
        requirements=(
            AgentInstallRequirement(
                "grok",
                install_command=("curl", "-fsSL", "https://x.ai/cli/install.sh", "|", "bash"),
                description="Grok Build CLI",
            ),
        ),
        expected_surface_paths=("{grok_config}",),
    ),
    "hermes": AgentCapability(
        name="hermes",
        label="Hermes Agent",
        native_surface="acp_mcp",
        requirements=(AgentInstallRequirement("hermes", description="Hermes Agent CLI"),),
        expected_surface_paths=("{hermes_config}",),
    ),
    "letta": AgentCapability(
        name="letta",
        label="Letta Code",
        native_surface="letta_external_tools",
        requirements=(
            AgentInstallRequirement(
                "letta",
                install_command=("npm", "install", "-g", "@letta-ai/letta-code"),
                description="Letta Code CLI",
            ),
        ),
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
        native_surface="pi_extension",
        requirements=(AgentInstallRequirement("pi", description="Pi Coding Agent CLI"),),
        expected_surface_paths=("{pi_extension}",),
    ),
}
_ALIASES = {
    "amp-code": "amp",
    "claude-cli": "claude",
    "claude-code": "claude",
    "factory": "droid",
    "factory-droid": "droid",
    "hermes-agent": "hermes",
    "letta-code": "letta",
}
AGENT_DISPLAY_ORDER = (
    "amp",
    "claude",
    "codex",
    "cursor",
    "droid",
    "grok",
    "hermes",
    "letta",
    "opencode",
    "pi",
)


def known_agent_names() -> tuple[str, ...]:
    """Return known Lab agent names in stable display order."""

    return AGENT_DISPLAY_ORDER


def agent_capability(name: str) -> AgentCapability:
    """Return the declared Lab capability for a coding agent."""

    raw = (name.strip() or "codex").lower()
    normalized = _ALIASES.get(raw, raw)
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
        raw = name.strip().lower()
        normalized = _ALIASES.get(raw, raw)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        capability = agent_capability(normalized)
        options.append((capability.label, capability.name))
    return options
