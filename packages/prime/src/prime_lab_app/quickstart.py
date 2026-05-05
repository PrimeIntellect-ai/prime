"""Quickstart launch actions for the Lab TUI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import toml

from .config_factory import evaluation_config, format_lab_config, rl_config
from .models import LabItem
from .palette import PRIMARY


@dataclass(frozen=True)
class PromptTemplate:
    """A prompt shortcut shown in agent-assisted Lab flows."""

    label: str
    prompt: str


def coding_agent_item(workspace: Path, *, agent: str) -> LabItem:
    """Create the general coding-agent chat item for the active workspace."""

    templates = (
        PromptTemplate(
            "Build env",
            "\n".join(
                [
                    "Help me build a new verifiers environment in this Lab workspace.",
                    "",
                    "Task idea:",
                    "- ",
                    "",
                    "Inspect the existing environments first. If there are multiple plausible",
                    "implementation paths, ask me to choose before editing files.",
                ]
            ),
        ),
        PromptTemplate(
            "Plan project",
            (
                "Help me plan the next research iteration in this Lab workspace. Inspect the local "
                "environments, configs, and recent runs first, then propose the smallest useful "
                "next step."
            ),
        ),
        PromptTemplate(
            "Edit config",
            (
                "Help me modify a Lab config in this workspace. If there are multiple plausible "
                "configs, show me the choices before making changes."
            ),
        ),
        PromptTemplate(
            "Fix issue",
            "Help me debug the current Lab workspace. Check setup, configs, generated outputs, and "
            "recent errors before proposing a fix.",
        ),
    )
    return LabItem(
        key=f"quickstart:agent:chat:{workspace}",
        section="workspace",
        title="Agent",
        subtitle="Chat with the configured coding agent in this workspace.",
        status=agent or "agent",
        status_style=PRIMARY,
        metadata=(
            ("Workspace", str(workspace)),
            ("Agent", agent or "not configured"),
        ),
        raw={
            "type": "agent_chat",
            "workspace": str(workspace),
            "agent": agent or "codex",
            "prompt_templates": tuple(
                {"label": template.label, "prompt": template.prompt} for template in templates
            ),
        },
    )


def evaluation_config_item(workspace: Path) -> LabItem:
    """Create a starter hosted-evaluation config item."""

    path = workspace / ".prime" / "lab" / "configs" / "eval" / "new-evaluation.toml"
    toml_text = format_lab_config(evaluation_config())
    return _config_item(
        key=f"quickstart:config:eval:{workspace}",
        title="Run Evaluation",
        subtitle="Create a hosted evaluation config.",
        config_kind="eval",
        workspace=workspace,
        path=path,
        toml_text=toml_text,
    )


def training_config_item(workspace: Path) -> LabItem:
    """Create a starter training config item."""

    path = workspace / ".prime" / "lab" / "configs" / "rl" / "new-training.toml"
    toml_text = format_lab_config(rl_config())
    return _config_item(
        key=f"quickstart:config:rl:{workspace}",
        title="Launch Training",
        subtitle="Create a training config.",
        config_kind="rl",
        workspace=workspace,
        path=path,
        toml_text=toml_text,
    )


def _config_item(
    *,
    key: str,
    title: str,
    subtitle: str,
    config_kind: str,
    workspace: Path,
    path: Path,
    toml_text: str,
) -> LabItem:
    return LabItem(
        key=key,
        section="workspace",
        title=title,
        subtitle=subtitle,
        status=config_kind,
        status_style=PRIMARY,
        metadata=(
            ("Kind", "Training config" if config_kind == "rl" else "Evaluation config"),
            ("Path", str(path)),
        ),
        raw={
            "type": "config_file",
            "config_kind": config_kind,
            "workspace": str(workspace),
            "path": str(path),
            "relative_path": str(path),
            "toml": toml_text,
            "parsed": _safe_toml_loads(toml_text),
        },
    )


def _safe_toml_loads(value: str) -> dict[str, Any]:
    try:
        parsed = toml.loads(value)
    except toml.TomlDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}
