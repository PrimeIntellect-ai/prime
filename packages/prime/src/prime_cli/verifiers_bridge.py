"""Process boundary between Prime's router and Verifiers-owned commands."""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Sequence
from functools import lru_cache
from importlib import import_module
from pathlib import Path
from typing import NoReturn

from prime_cli.core import Config


def venv_python(venv_root: Path) -> Path:
    if os.name == "nt":
        return venv_root / "Scripts" / "python.exe"
    return venv_root / "bin" / "python"


@lru_cache(maxsize=32)
def python_can_import_verifiers(python_executable: str, cwd: str) -> bool:
    probe = "; ".join(
        (
            "import importlib.util",
            "spec = importlib.util.find_spec('verifiers')",
            "raise SystemExit(0 if spec else 1)",
        )
    )
    result = subprocess.run(
        [python_executable, "-c", probe],
        cwd=cwd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def resolve_workspace_python(cwd: Path | None = None) -> str:
    """Prefer a nearby project interpreter that can import Verifiers."""
    workspace = (cwd or Path.cwd()).resolve()
    candidates = [
        *(
            venv_python(directory / ".venv")
            for directory in [workspace, *workspace.parents]
            if (directory / "pyproject.toml").is_file()
        ),
        *(
            venv_python(Path(value))
            for value in (
                os.environ.get("UV_PROJECT_ENVIRONMENT"),
                os.environ.get("VIRTUAL_ENV"),
            )
            if value
        ),
    ]
    for candidate in candidates:
        if candidate.exists() and python_can_import_verifiers(str(candidate), str(workspace)):
            return str(candidate)
    return sys.executable


def build_verifiers_command(name: str, args: Sequence[str] = ()) -> list[str]:
    modules = getattr(import_module("verifiers.cli"), "CLI_MODULES")
    return [resolve_workspace_python(), "-m", modules[name], *args]


def verifiers_environment(*, plain: bool = False) -> dict[str, str]:
    """Materialize Prime's selected context for a child Verifiers process."""
    config = Config()
    env = os.environ.copy()
    resolved = {
        "PRIME_API_KEY": config.api_key,
        "PRIME_TEAM_ID": config.team_id,
        "PRIME_USER_ID": config.user_id,
        "PRIME_API_BASE_URL": config.base_url,
        "PRIME_INFERENCE_URL": config.inference_url,
    }
    env.update({key: str(value) for key, value in resolved.items() if value})
    if plain:
        env.update(NO_COLOR="1", PYDANTIC_CONFIG_PLAIN="1")
    return env


def exec_verifiers_process(name: str, args: Sequence[str], *, plain: bool = False) -> NoReturn:
    """Replace Prime with a Verifiers-owned CLI process."""
    command = build_verifiers_command(name, args)
    os.execvpe(command[0], command, verifiers_environment(plain=plain))


def run_eval_view(env_dir: str | None, outputs_dir: str | None, limit: int = 50) -> None:
    from prime_lab_app import run_eval_view as run_prime_eval_view

    run_prime_eval_view(
        limit=limit,
        env_dir=env_dir or "./environments",
        outputs_dir=outputs_dir or "./outputs",
        workspace=Path.cwd(),
    )
