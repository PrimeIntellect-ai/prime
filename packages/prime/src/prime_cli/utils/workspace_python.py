"""Pick the interpreter that `prime env install` installs into."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _venv_python(venv_root: Path) -> Path:
    if os.name == "nt":
        return venv_root / "Scripts" / "python.exe"
    return venv_root / "bin" / "python"


def resolve_workspace_python(cwd: Path | None = None) -> str:
    """The active or project virtualenv's interpreter, else the one running prime."""
    candidates: list[Path] = []
    uv_project_env = os.environ.get("UV_PROJECT_ENVIRONMENT")
    if uv_project_env:
        candidates.append(_venv_python(Path(uv_project_env)))
    virtual_env = os.environ.get("VIRTUAL_ENV")
    if virtual_env:
        candidates.append(_venv_python(Path(virtual_env)))
    workspace = (cwd or Path.cwd()).resolve()
    for directory in [workspace, *workspace.parents]:
        if (directory / "pyproject.toml").is_file():
            candidates.append(_venv_python(directory / ".venv"))
            break

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return sys.executable
