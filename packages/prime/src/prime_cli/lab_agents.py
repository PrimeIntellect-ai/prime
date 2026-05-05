"""Prime Lab coding-agent metadata and local setup surfaces."""

from __future__ import annotations

import shutil
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

_AGENT_SKILL_ROOTS = {
    "amp": "~/.config/amp/skills",
    "claude": "~/.claude/skills",
    "codex": "~/.agents/skills",
    "cursor": "~/.cursor/skills",
    "droid": "~/.factory/skills",
    "hermes": "~/.hermes/skills",
    "opencode": "~/.opencode/skills",
    "pi": "~/.pi/skills",
}


def agent_user_skills_dir(agent: str) -> Path | None:
    """Return the user-level skill root for a supported agent."""

    root = _AGENT_SKILL_ROOTS.get(agent_capability(agent).name)
    return Path(root).expanduser() if root else None


__all__ = [
    "AgentAdapter",
    "AgentCapability",
    "AgentInstallRequirement",
    "agent_adapter",
    "agent_capability",
    "agent_user_skills_dir",
    "known_agent_names",
    "shutil",
    "write_agent_native_surface",
]
