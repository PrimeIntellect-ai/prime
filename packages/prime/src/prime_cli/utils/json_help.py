"""Helpers for documenting JSON CLI output shapes in command help."""

from __future__ import annotations


def _build_json_help(title: str, *lines: str) -> str:
    body = "\n".join(f"  {line.rstrip()}" for line in lines if line.strip())
    return f"{title}\n{body}" if body else title


def json_output_help(*lines: str) -> str:
    """Build help text for commands with an explicit ``--output json`` mode."""
    return _build_json_help("JSON output (--output json):", *lines)


def json_help(*lines: str) -> str:
    """Build help text for commands that always print JSON."""
    return _build_json_help("JSON output:", *lines)
