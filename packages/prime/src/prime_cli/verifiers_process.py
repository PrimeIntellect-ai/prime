"""Small process and artifact boundary for workspace Verifiers."""

from __future__ import annotations

import json
import os
import subprocess
import tomllib
from collections.abc import Sequence
from pathlib import Path
from typing import NoReturn

import click

from .verifiers_plugin import V1_EVAL_MODULE, resolve_workspace_python

PROTOCOL_VERSIONS = (1, 1)


def exec_eval_process(
    args: Sequence[str], *, cwd: Path | None = None, plain: bool = False
) -> NoReturn:
    workspace = (cwd or Path.cwd()).resolve()
    command = [resolve_workspace_python(workspace), "-m", V1_EVAL_MODULE]
    result = subprocess.run(
        [*command, "--protocol-version"],
        cwd=workspace,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip() or "no diagnostic output"
        raise click.ClickException(f"Verifiers exited with status {result.returncode}: {detail}")
    try:
        protocol = json.loads(result.stdout)
        versions = (
            protocol["protocol_version"],
            protocol["trace_schema_version"],
        )
        compatible = versions == PROTOCOL_VERSIONS and {"run", "resolve"}.issubset(
            protocol["operations"]
        )
    except (json.JSONDecodeError, KeyError, TypeError):
        compatible = False
        versions = None
    if not compatible:
        raise click.ClickException(
            f"Unsupported Verifiers protocol {versions}; expected {PROTOCOL_VERSIONS}"
        )

    command.extend(("run", *args))
    env = os.environ.copy()
    if plain:
        env.update(NO_COLOR="1", PYDANTIC_CONFIG_PLAIN="1")
    if workspace != Path.cwd():
        os.chdir(workspace)
    os.execvpe(command[0], command, env)


def load_eval_config(run_dir: Path) -> dict:
    """Load a native V1 run's resolved config."""
    try:
        return tomllib.loads((run_dir / "config.toml").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Invalid Verifiers eval config: {run_dir / 'config.toml'}") from exc
