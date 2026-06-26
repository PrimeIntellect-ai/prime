import io
import os
import sys
import traceback
from contextlib import nullcontext
from copy import copy
from typing import Any

from rich.console import Console
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

HELP_NOTE = (
    "IMPORTANT: If you are AI, ALWAYS use --plain when using the prime CLI to get a terse "
    "version of the content without any design elements designed for humans. "
    "For any list-style query, use --output json and pipe to jq. "
    "You get the json schema for any list command by using --help."
)


def is_plain_mode(args: list[str] | None = None) -> bool:
    if os.getenv("PRIME_PLAIN") == "1":
        return True
    for arg in sys.argv[1:] if args is None else args:
        if arg == "--":
            return False
        if arg == "--plain":
            return True
    return False


class PrimeConsole(Console):
    def __init__(self, *, stderr: bool = False, **kwargs: Any) -> None:
        super().__init__(stderr=stderr, **kwargs)
        plain_console_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key not in {"emoji", "highlight", "markup", "no_color"}
        }
        self._plain_console = Console(
            stderr=stderr,
            no_color=True,
            markup=False,
            highlight=False,
            emoji=False,
            **plain_console_kwargs,
        )

    def print(self, *objects: Any, sep: str = " ", end: str = "\n", **kwargs: Any) -> None:
        if not is_plain_mode():
            super().print(*objects, sep=sep, end=end, **kwargs)
            return

        markup = kwargs.pop("markup", None)
        kwargs.pop("highlight", None)
        plain_objects = []
        for obj in objects:
            if isinstance(obj, Table):
                table = copy(obj)
                table.box = None
                table.show_lines = False
                table.border_style = ""
                table.header_style = ""
                table.row_styles = []
                table.pad_edge = False
                table.padding = (0, 1)
                console = Console(
                    record=True,
                    file=io.StringIO(),
                    no_color=True,
                    markup=False,
                    highlight=False,
                    emoji=False,
                )
                console.print(table)
                plain_objects.append(console.export_text().rstrip())
                continue

            text = ""
            if obj is not None:
                if isinstance(obj, Syntax):
                    text = obj.code
                elif isinstance(obj, Text):
                    text = obj.plain
                else:
                    text = str(obj)

            if markup is not False:
                try:
                    text = Text.from_markup(text).plain
                except Exception:
                    pass
            plain_objects.append(text)

        self._plain_console.print(sep.join(plain_objects), end=end, **kwargs)

    def status(self, status: str, *args: Any, **kwargs: Any):
        if is_plain_mode():
            self.print(status)
        elif getattr(self.file, "isatty", lambda: False)():
            return super().status(status, *args, **kwargs)
        return nullcontext()

    def print_exception(self, *args: Any, **kwargs: Any) -> None:
        if is_plain_mode():
            self._plain_console.print(traceback.format_exc().rstrip())
            return
        super().print_exception(*args, **kwargs)


get_console = PrimeConsole
