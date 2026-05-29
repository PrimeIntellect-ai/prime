"""Shared TOML formatting helpers for Lab UI and launch flows."""

from __future__ import annotations


def format_toml_blocks(value: str) -> str:
    """Format TOML with readable spacing before array table blocks."""

    lines = value.splitlines()
    spaced: list[str] = []
    for line in lines:
        if line.startswith("[[") and spaced and spaced[-1] != "":
            spaced.append("")
        spaced.append(line)
    return "\n".join(spaced) + ("\n" if spaced else "")
