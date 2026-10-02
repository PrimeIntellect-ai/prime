"""Prime Lab coding-agent metadata used by workspace setup."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

AgentCapabilityStatus = Literal["supported", "not_supported"]


@dataclass(frozen=True)
class AgentInstallRequirement:
    """One machine-level executable a Lab agent needs."""

    binary: str
    install_command: tuple[str, ...] = ()
    description: str = ""

    def installed(self) -> bool:
        return shutil.which(self.binary) is not None


@dataclass(frozen=True)
class AgentCapability:
    """Setup metadata for one coding agent."""

    name: str
    label: str
    requirements: tuple[AgentInstallRequirement, ...] = ()
    status: AgentCapabilityStatus = "supported"
    unsupported_reason: str = ""

    def missing_requirements(self) -> tuple[AgentInstallRequirement, ...]:
        return tuple(
            requirement for requirement in self.requirements if not requirement.installed()
        )


_CAPABILITIES: dict[str, AgentCapability] = {
    "amp": AgentCapability(
        name="amp",
        label="Amp Code",
        requirements=(
            AgentInstallRequirement(
                "amp",
                install_command=("npm", "install", "-g", "@sourcegraph/amp@latest"),
                description="Amp Code CLI",
            ),
        ),
    ),
    "claude": AgentCapability(
        name="claude",
        label="Claude",
        requirements=(AgentInstallRequirement("claude", description="Claude Code CLI"),),
    ),
    "codex": AgentCapability(
        name="codex",
        label="Codex",
        requirements=(AgentInstallRequirement("codex", description="Codex CLI"),),
    ),
    "cursor": AgentCapability(
        name="cursor",
        label="Cursor",
        requirements=(AgentInstallRequirement("cursor-agent", description="Cursor Agent CLI"),),
    ),
    "droid": AgentCapability(
        name="droid",
        label="Factory Droid Agent",
        requirements=(
            AgentInstallRequirement(
                "droid",
                install_command=("npm", "install", "-g", "@factory/cli"),
                description="Factory Droid Agent CLI",
            ),
        ),
    ),
    "grok": AgentCapability(
        name="grok",
        label="Grok Build",
        requirements=(
            AgentInstallRequirement(
                "grok",
                install_command=("curl", "-fsSL", "https://x.ai/cli/install.sh", "|", "bash"),
                description="Grok Build CLI",
            ),
        ),
    ),
    "hermes": AgentCapability(
        name="hermes",
        label="Hermes Agent",
        requirements=(AgentInstallRequirement("hermes", description="Hermes Agent CLI"),),
    ),
    "letta": AgentCapability(
        name="letta",
        label="Letta Code",
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
        requirements=(AgentInstallRequirement("opencode", description="OpenCode CLI"),),
    ),
    "pi": AgentCapability(
        name="pi",
        label="Pi Coding Agent",
        requirements=(AgentInstallRequirement("pi", description="Pi Coding Agent CLI"),),
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

_AGENT_USER_SKILL_ROOTS = {
    "amp": "~/.config/agents/skills",
    "claude": "~/.claude/skills",
    "codex": "~/.agents/skills",
    "cursor": "~/.cursor/skills",
    "droid": "~/.factory/skills",
    "grok": "~/.grok/skills",
    "hermes": "~/.hermes/skills",
    "opencode": "~/.config/opencode/skills",
    "pi": "~/.pi/agent/skills",
}
_AGENT_PROJECT_SKILL_ROOTS = {
    "amp": (".agents/skills",),
    "claude": (".claude/skills",),
    "codex": (".agents/skills",),
    "cursor": (".cursor/skills",),
    "droid": (".factory/skills",),
    "grok": (".grok/skills",),
    "letta": (".agents/skills",),
    "opencode": (".opencode/skills",),
    "pi": (".pi/skills",),
}


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
    return AgentCapability(
        name=normalized,
        label=normalized,
        status="not_supported",
        unsupported_reason="Choose a Lab-supported coding agent.",
    )


def agent_user_skills_dir(agent: str) -> Path | None:
    """Return the user-level skill root for a supported agent."""

    root = _AGENT_USER_SKILL_ROOTS.get(agent_capability(agent).name)
    return Path(root).expanduser() if root else None


def agent_project_skills_dirs(agent: str, workspace: Path) -> tuple[Path, ...]:
    """Return project-local skill roots for a supported agent."""

    workspace = workspace.expanduser().resolve()
    return tuple(
        workspace / root
        for root in _AGENT_PROJECT_SKILL_ROOTS.get(agent_capability(agent).name, ())
    )


__all__ = [
    "AgentCapability",
    "AgentInstallRequirement",
    "agent_capability",
    "agent_project_skills_dirs",
    "agent_user_skills_dir",
    "known_agent_names",
]
