"""Small process and artifact boundary for workspace Verifiers."""

from __future__ import annotations

import json
import os
import subprocess
import tomllib
from collections.abc import Sequence
from pathlib import Path
from typing import Any, NoReturn

import click

from .verifiers_plugin import V1_EVAL_MODULE, resolve_workspace_python

PROTOCOL_VERSIONS = (1, 1, 1)
MANIFEST_SCHEMA = "verifiers.eval-run/v1"
ARTIFACTS = {
    "config": "config.toml",
    "results": "results.jsonl",
    "log": "eval.log",
}


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
            protocol["manifest_schema_version"],
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


def load_eval_artifact(run_dir: Path) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Load a native V1 config and its optional versioned manifest."""
    manifest_path = run_dir / "manifest.json"
    try:
        manifest = None
        config_path = run_dir / ARTIFACTS["config"]
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            versions = (manifest["protocol_version"], manifest["trace_schema_version"])
            if (
                manifest["schema"] != MANIFEST_SCHEMA
                or versions != PROTOCOL_VERSIONS[:2]
                or manifest["artifacts"] != ARTIFACTS
            ):
                raise ValueError
            config_path = run_dir / manifest["artifacts"]["config"]
        config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"Invalid Verifiers run artifact: {run_dir}") from exc
    return manifest, config
