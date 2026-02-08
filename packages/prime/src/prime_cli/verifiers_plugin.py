"""Compatibility layer for verifiers prime plugin loading."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from rich.console import Console

EXPECTED_PLUGIN_API_VERSION = 1


def _venv_python(venv_root: Path) -> Path:
    if os.name == "nt":
        return venv_root / "Scripts" / "python.exe"
    return venv_root / "bin" / "python"


def resolve_workspace_python(cwd: Path | None = None) -> str:
    """Prefer the workspace interpreter over the prime tool interpreter."""
    uv_project_env = os.environ.get("UV_PROJECT_ENVIRONMENT")
    if uv_project_env:
        candidate = _venv_python(Path(uv_project_env))
        if candidate.exists():
            return str(candidate)

    virtual_env = os.environ.get("VIRTUAL_ENV")
    if virtual_env:
        candidate = _venv_python(Path(virtual_env))
        if candidate.exists():
            return str(candidate)

    start = cwd or Path.cwd()
    for directory in [start, *start.parents]:
        if (directory / "pyproject.toml").is_file():
            candidate = _venv_python(directory / ".venv")
            if candidate.exists():
                return str(candidate)

    return sys.executable


@dataclass(frozen=True)
class PrimeVerifiersPlugin:
    """Local fallback contract used when verifiers plugin loading fails."""

    api_version: int = EXPECTED_PLUGIN_API_VERSION
    eval_module: str = "verifiers.cli.commands.eval"
    gepa_module: str = "verifiers.cli.commands.gepa"
    install_module: str = "verifiers.cli.commands.install"
    init_module: str = "verifiers.cli.commands.init"
    setup_module: str = "verifiers.cli.commands.setup"
    build_module: str = "verifiers.cli.commands.build"
    tui_module: str = "verifiers.cli.tui"

    def build_module_command(
        self, module_name: str, args: Sequence[str] | None = None
    ) -> list[str]:
        command = [resolve_workspace_python(), "-m", module_name]
        if args:
            command.extend(args)
        return command


def load_verifiers_prime_plugin(console: Console | None = None) -> PrimeVerifiersPlugin:
    """Load plugin exported by verifiers with fallback behavior."""
    sink = console or Console(stderr=True)
    try:
        from verifiers.cli.plugins.prime import get_plugin  # type: ignore
    except Exception as exc:
        sink.print(
            "[yellow]Warning:[/yellow] Could not import verifiers prime plugin "
            f"({exc}). Falling back to built-in command mapping."
        )
        return PrimeVerifiersPlugin()

    try:
        plugin = get_plugin()
    except Exception as exc:
        sink.print(
            "[yellow]Warning:[/yellow] Failed to load verifiers plugin "
            f"({exc}). Falling back to built-in command mapping."
        )
        return PrimeVerifiersPlugin()

    api_version = getattr(plugin, "api_version", None)
    if api_version != EXPECTED_PLUGIN_API_VERSION:
        sink.print(
            "[yellow]Warning:[/yellow] verifiers plugin API version mismatch "
            f"(got {api_version}, expected {EXPECTED_PLUGIN_API_VERSION}). "
            "Continuing with compatibility behavior."
        )

    return PrimeVerifiersPlugin(
        api_version=int(api_version or EXPECTED_PLUGIN_API_VERSION),
        eval_module=getattr(plugin, "eval_module", "verifiers.cli.commands.eval"),
        gepa_module=getattr(plugin, "gepa_module", "verifiers.cli.commands.gepa"),
        install_module=getattr(plugin, "install_module", "verifiers.cli.commands.install"),
        init_module=getattr(plugin, "init_module", "verifiers.cli.commands.init"),
        setup_module=getattr(plugin, "setup_module", "verifiers.cli.commands.setup"),
        build_module=getattr(plugin, "build_module", "verifiers.cli.commands.build"),
        tui_module=getattr(plugin, "tui_module", "verifiers.cli.tui"),
    )
