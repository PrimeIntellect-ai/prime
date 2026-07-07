"""Locate a verifiers-capable Python and build verifiers CLI commands."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Sequence

V1_EVAL_MODULE = "verifiers.v1.cli.eval.main"
V1_INIT_MODULE = "verifiers.v1.cli.init"
V1_VALIDATE_MODULE = "verifiers.v1.cli.validate"
# `python -m` needs a `__main__` guard the v1 modules only gained recently; these -c
# shims call `main()` directly so they work across verifiers versions.
V1_ENTRYPOINTS = {
    V1_EVAL_MODULE: f"import sys; sys.argv[0] = 'eval'; from {V1_EVAL_MODULE} import main; main()",
    V1_INIT_MODULE: f"import sys; sys.argv[0] = 'init'; from {V1_INIT_MODULE} import main; main()",
    V1_VALIDATE_MODULE: (
        f"import sys; sys.argv[0] = 'validate'; from {V1_VALIDATE_MODULE} import main; main()"
    ),
}


def _venv_python(venv_root: Path) -> Path:
    if os.name == "nt":
        return venv_root / "Scripts" / "python.exe"
    return venv_root / "bin" / "python"


@lru_cache(maxsize=32)
def _python_can_import_module(python_executable: str, module_name: str, cwd: str) -> bool:
    probe = (
        "import importlib.util, sys; "
        "raise SystemExit(0 if importlib.util.find_spec(sys.argv[1]) else 1)"
    )
    try:
        result = subprocess.run(
            [python_executable, "-c", probe, module_name],
            cwd=cwd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except Exception:
        return False
    return result.returncode == 0


def resolve_workspace_python(cwd: Path | None = None) -> str:
    """Prefer workspace Python only when it can run verifiers command modules."""
    workspace = (cwd or Path.cwd()).resolve()
    workspace_str = str(workspace)
    module = V1_EVAL_MODULE

    def _usable(candidate: Path) -> bool:
        return candidate.exists() and _python_can_import_module(
            str(candidate), module, workspace_str
        )

    uv_project_env = os.environ.get("UV_PROJECT_ENVIRONMENT")
    if uv_project_env:
        candidate = _venv_python(Path(uv_project_env))
        if _usable(candidate):
            return str(candidate)

    virtual_env = os.environ.get("VIRTUAL_ENV")
    if virtual_env:
        candidate = _venv_python(Path(virtual_env))
        if _usable(candidate):
            return str(candidate)

    for directory in [workspace, *workspace.parents]:
        if (directory / "pyproject.toml").is_file():
            candidate = _venv_python(directory / ".venv")
            if _usable(candidate):
                return str(candidate)

    return sys.executable


@dataclass(frozen=True)
class PrimeVerifiersPlugin:
    """The verifiers CLI modules prime shells out to."""

    eval_module: str = V1_EVAL_MODULE
    init_module: str = V1_INIT_MODULE
    validate_module: str = V1_VALIDATE_MODULE
    gepa_module: str = "verifiers.scripts.gepa"

    def build_module_command(
        self, module_name: str, args: Sequence[str] | None = None
    ) -> list[str]:
        python = resolve_workspace_python()
        entrypoint = V1_ENTRYPOINTS.get(module_name)
        command = [python, "-c", entrypoint] if entrypoint else [python, "-m", module_name]
        if args:
            command.extend(args)
        return command


def load_verifiers_prime_plugin() -> PrimeVerifiersPlugin:
    """The static verifiers command map (verifiers stopped exporting a plugin module)."""
    return PrimeVerifiersPlugin()
