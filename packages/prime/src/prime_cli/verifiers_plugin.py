"""Lazy access to command integration exported by Verifiers."""

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Protocol

V1_INIT_MODULE = "verifiers.v1.cli.init"


class PrimeVerifiersPlugin(Protocol):
    eval_module: str
    gepa_module: str
    install_module: str
    init_module: str
    setup_module: str
    build_module: str

    def build_module_command(
        self, module_name: str, args: Sequence[str] | None = None
    ) -> list[str]: ...


def load_verifiers_prime_plugin() -> PrimeVerifiersPlugin:
    from verifiers.cli.plugins.prime import get_plugin

    # The Prime lock predates Verifiers exporting its V1 init module.
    return replace(get_plugin(), init_module=V1_INIT_MODULE)


def resolve_workspace_python(cwd: Path | None = None) -> str:
    from verifiers.cli.plugins.prime import _resolve_workspace_python

    return _resolve_workspace_python(cwd)
