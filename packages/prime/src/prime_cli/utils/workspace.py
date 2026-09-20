"""Locate the Python interpreter of the caller's project."""

import os
import sys
from pathlib import Path


def _venv_python(venv_root: Path) -> Path:
    if os.name == "nt":
        return venv_root / "Scripts" / "python.exe"
    return venv_root / "bin" / "python"


def resolve_workspace_python(cwd: Path | None = None) -> str:
    """Prefer the project's virtual environment over the interpreter running `prime`.

    `prime` is usually installed as an isolated tool, so packages must land in the
    environment the user actually works in. Checks `UV_PROJECT_ENVIRONMENT`,
    `VIRTUAL_ENV`, then the nearest `.venv` next to a `pyproject.toml`.
    """
    workspace = (cwd or Path.cwd()).resolve()

    for env_var in ("UV_PROJECT_ENVIRONMENT", "VIRTUAL_ENV"):
        venv_root = os.environ.get(env_var)
        if venv_root:
            candidate = _venv_python(Path(venv_root))
            if candidate.exists():
                return str(candidate)

    for directory in [workspace, *workspace.parents]:
        if (directory / "pyproject.toml").is_file():
            candidate = _venv_python(directory / ".venv")
            if candidate.exists():
                return str(candidate)

    return sys.executable
