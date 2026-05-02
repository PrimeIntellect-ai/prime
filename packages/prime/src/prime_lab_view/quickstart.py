"""Quickstart launch actions for the Lab TUI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import toml

from .models import LabItem
from .palette import PRIMARY


@dataclass(frozen=True)
class PromptTemplate:
    """A prompt shortcut shown in agent-assisted Lab flows."""

    label: str
    prompt: str


def build_environment_item(workspace: Path, *, agent: str) -> LabItem:
    """Create the agent-chat item for building a new local environment."""

    templates = (
        PromptTemplate(
            "New task",
            "\n".join(
                [
                    "Help me build a new verifiers environment in this Lab workspace.",
                    "",
                    "Task idea:",
                    "- ",
                    "",
                    "Please inspect the existing environments and propose the smallest scaffold.",
                ]
            ),
        ),
        PromptTemplate(
            "From benchmark",
            "\n".join(
                [
                    "Create a new verifiers environment from an existing benchmark or dataset.",
                    "",
                    "Benchmark or dataset:",
                    "- ",
                    "",
                    "Use the local workspace conventions and include a README.",
                    "Also add a quick eval config.",
                ]
            ),
        ),
        PromptTemplate(
            "Review scaffold",
            "\n".join(
                [
                    "Review the active Lab workspace and suggest a clean environment scaffold.",
                    "Focus on source layout, pyproject metadata, README content,",
                    "and evaluation hooks.",
                ]
            ),
        ),
    )
    return LabItem(
        key=f"quickstart:agent:build-environment:{workspace}",
        section="workspace",
        title="Build Environment",
        subtitle="Use the configured coding agent to create or refine a local environment.",
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
    toml_text = _space_toml_blocks(
        toml.dumps(
            {
                "model": "",
                "save_results": True,
                "eval": [
                    {
                        "env_id": "primeintellect/gsm8k",
                        "rollouts_per_example": 1,
                        "sampling_args": {"max_tokens": 512},
                    }
                ],
            }
        )
    ).rstrip()
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
    toml_text = _space_toml_blocks(
        toml.dumps(
            {
                "model": "",
                "max_steps": 100,
                "batch_size": 256,
                "rollouts_per_example": 8,
                "sampling": {"max_tokens": 512},
                "env": [{"id": "primeintellect/gsm8k"}],
            }
        )
    ).rstrip()
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


def _space_toml_blocks(value: str) -> str:
    lines = value.splitlines()
    spaced: list[str] = []
    for line in lines:
        if line.startswith("[[") and spaced and spaced[-1] != "":
            spaced.append("")
        spaced.append(line)
    return "\n".join(spaced) + ("\n" if spaced else "")
