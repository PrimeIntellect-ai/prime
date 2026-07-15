from __future__ import annotations

from prime_cli.agent_picker import select_agents
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

MENU = [
    ("amp", "Amp Code", False),
    ("claude", "Claude", True),
    ("codex", "Codex", True),
    ("cursor", "Cursor", False),
    ("grok", "Grok Build", True),
]

DOWN = "\x1b[B"
UP = "\x1b[A"
SPACE = " "
ENTER = "\r"
CTRL_C = "\x03"


def _pick(menu: list[tuple[str, str, bool]], default_index: int, keys: str) -> object:
    with create_pipe_input() as pipe:
        pipe.send_text(keys)
        return select_agents(menu, default_index, input=pipe, output=DummyOutput())


def test_confirm_returns_default_selection() -> None:
    assert _pick(MENU, 2, ENTER) == ("codex",)


def test_space_toggles_extra_agent() -> None:
    assert _pick(MENU, 2, DOWN * 4 + SPACE + ENTER) == ("codex", "grok")


def test_result_follows_menu_order_not_pick_order() -> None:
    assert _pick(MENU, 4, DOWN + SPACE + ENTER) == ("claude", "grok")


def test_can_deselect_default() -> None:
    assert _pick(MENU, 2, DOWN * 2 + SPACE + UP + SPACE + ENTER) == ("claude",)


def test_confirm_requires_at_least_one() -> None:
    keys = DOWN * 2 + SPACE + ENTER + SPACE + ENTER
    assert _pick(MENU, 2, keys) == ("codex",)


def test_ctrl_c_cancels() -> None:
    assert _pick(MENU, 2, CTRL_C) is None
