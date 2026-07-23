from __future__ import annotations

from collections.abc import Iterator

import pytest
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput


class Keys:
    """Queue keystrokes for questionary prompts driven inside a CliRunner invoke."""

    DOWN = "\x1b[B"
    UP = "\x1b[A"
    ENTER = "\r"
    SPACE = " "
    ESC = "\x1b"
    CTRL_C = "\x03"

    def __init__(self, pipe: object) -> None:
        self._pipe = pipe

    def send(self, text: str) -> "Keys":
        self._pipe.send_text(text)  # type: ignore[attr-defined]
        return self

    def text(self, value: str) -> "Keys":
        return self.send(value + self.ENTER)

    def confirm(self, yes: bool = True) -> "Keys":
        return self.send(("y" if yes else "n") + self.ENTER)

    def select(self, down: int = 0) -> "Keys":
        return self.send(self.DOWN * down + self.ENTER)

    def cancel(self) -> "Keys":
        return self.send(self.CTRL_C)

    def check(self, downs: list[int]) -> "Keys":
        cursor = 0
        for target in downs:
            self.send(self.DOWN * (target - cursor) + self.SPACE)
            cursor = target
        return self.send(self.ENTER)


@pytest.fixture
def keys() -> Iterator[Keys]:
    with create_pipe_input() as pipe:
        with create_app_session(input=pipe, output=DummyOutput()):
            yield Keys(pipe)
