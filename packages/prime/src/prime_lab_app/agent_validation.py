"""Live validation helpers for Lab-supported agent surfaces."""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .agent_capabilities import agent_capability, known_agent_names

CommandRunner = Callable[..., Any]


@dataclass(frozen=True)
class AgentValidationResult:
    agent: str
    label: str
    ok: bool
    message: str
    checked_paths: tuple[Path, ...] = ()


def validate_agent_surfaces(
    workspace: Path,
    *,
    agents: tuple[str, ...] | None = None,
    runner: CommandRunner = subprocess.run,
    timeout_seconds: float = 10.0,
) -> tuple[AgentValidationResult, ...]:
    """Validate configured native-agent binaries and Lab surface files."""

    workspace = workspace.expanduser().resolve()
    results: list[AgentValidationResult] = []
    for agent in agents or known_agent_names():
        capability = agent_capability(agent)
        if capability.status == "not_supported":
            results.append(
                AgentValidationResult(
                    agent=capability.name,
                    label=capability.label,
                    ok=False,
                    message=capability.unsupported_reason
                    or "This agent does not expose native Lab tools.",
                    checked_paths=capability.resolved_surface_paths(workspace),
                )
            )
            continue
        missing = capability.missing_requirements()
        if missing:
            results.append(
                AgentValidationResult(
                    agent=capability.name,
                    label=capability.label,
                    ok=False,
                    message="Missing binaries: "
                    + ", ".join(requirement.binary for requirement in missing),
                    checked_paths=capability.resolved_surface_paths(workspace),
                )
            )
            continue
        binary_error = _validate_binaries(capability.requirements, runner, timeout_seconds)
        if binary_error:
            results.append(
                AgentValidationResult(
                    agent=capability.name,
                    label=capability.label,
                    ok=False,
                    message=binary_error,
                    checked_paths=capability.resolved_surface_paths(workspace),
                )
            )
            continue
        paths = capability.resolved_surface_paths(workspace)
        missing_paths = tuple(path for path in paths if not path.exists())
        if missing_paths:
            results.append(
                AgentValidationResult(
                    agent=capability.name,
                    label=capability.label,
                    ok=False,
                    message="Missing native surface files: "
                    + ", ".join(str(path) for path in missing_paths),
                    checked_paths=paths,
                )
            )
            continue
        results.append(
            AgentValidationResult(
                agent=capability.name,
                label=capability.label,
                ok=True,
                message="Native Lab surface is available.",
                checked_paths=paths,
            )
        )
    return tuple(results)


def _validate_binaries(
    requirements: tuple[Any, ...],
    runner: CommandRunner,
    timeout_seconds: float,
) -> str:
    for requirement in requirements:
        try:
            completed = runner(
                [requirement.binary, "--version"],
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            return f"{requirement.binary} --version timed out"
        except OSError as exc:
            return f"{requirement.binary} failed to start: {exc}"
        returncode = int(getattr(completed, "returncode", 0))
        if returncode != 0:
            return f"{requirement.binary} --version exited with {returncode}"
    return ""
