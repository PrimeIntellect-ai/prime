"""GEPA passthrough."""

from ..utils import is_plain_mode
from ..verifiers_bridge import exec_verifiers_process


def run_gepa_cmd(argv: list[str]) -> None:
    """Run Verifiers' native GEPA command."""
    exec_verifiers_process("gepa", argv, plain=is_plain_mode())
