"""Prime Lab coding-agent metadata and local setup surfaces."""

from __future__ import annotations

from pathlib import Path

from prime_lab_app.agent_adapters import (
    AgentAdapter,
    agent_adapter,
    write_agent_native_surface,
)
from prime_lab_app.agent_capabilities import (
    AgentCapability,
    AgentInstallRequirement,
    agent_capability,
    known_agent_names,
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
    "AgentAdapter",
    "AgentCapability",
    "AgentInstallRequirement",
    "agent_adapter",
    "agent_capability",
    "agent_project_skills_dirs",
    "agent_user_skills_dir",
    "known_agent_names",
    "write_agent_native_surface",
]
