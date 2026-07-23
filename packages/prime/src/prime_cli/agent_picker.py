"""Interactive coding-agent picker used by prime lab setup."""

from __future__ import annotations

from typing import Any

from questionary import Choice

from .utils.prompt import ask_checkbox

AgentMenu = list[tuple[str, str, bool]]


def _choices(menu: AgentMenu, default_index: int) -> list[Choice]:
    choices = []
    for index, (name, label, installed) in enumerate(menu):
        status = "installed" if installed else "not installed"
        title = [("class:text", f"{label}  ({status})")]
        choices.append(Choice(title=title, value=name, checked=index == default_index))
    return choices


def _require_one(selected: list[str]) -> bool | str:
    return bool(selected) or "Select at least one agent (space to toggle)."


def select_agents(
    menu: AgentMenu, default_index: int, **prompt_kwargs: Any
) -> tuple[str, ...] | None:
    """Prompt for coding agents inline; return the picks in menu order (first is primary)."""

    answer = ask_checkbox(
        "Select coding agents (first is primary)",
        _choices(menu, default_index),
        instruction="↑/↓ move · space toggle · enter confirm",
        validate=_require_one,
        **prompt_kwargs,
    )
    if not answer:
        return None
    chosen = set(answer)
    return tuple(name for name, _, _ in menu if name in chosen)
