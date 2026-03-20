import io
import sys
import traceback
from contextlib import nullcontext
from copy import copy

import click
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

HELP_NOTE = (
    "IMPORTANT: If you are AI, ALWAYS use --plain when using the prime CLI to get a terse version\
of the content without any design elements designed for humans.\
For any list-style query, use --output json and pipe to jq.\
You get the json schema for any list command by using --help."
)


def is_plain_mode(args=None):
    args = list(sys.argv[1:] if args is None else args)
    args = args[: args.index("--")] if "--" in args else args
    ctx = click.get_current_context(silent=True)
    while ctx:
        if ctx.meta.get("plain"):
            return True
        ctx = ctx.parent
    return "--plain" in args


def _plain(obj, collapse=False):
    if isinstance(obj, Table):
        obj = copy(obj)
        obj.box = None
        obj.show_lines = False
        obj.border_style = ""
        obj.header_style = ""
        obj.row_styles = []
        obj.pad_edge = False
        obj.padding = (0, 1)
        console = Console(
            record=True,
            file=io.StringIO(),
            no_color=True,
            markup=False,
            highlight=False,
            emoji=False,
        )
        console.print(obj)
        text = console.export_text().rstrip()
    else:
        text = (
            ""
            if obj is None
            else obj.code
            if isinstance(obj, Syntax)
            else obj.plain
            if isinstance(obj, Text)
            else str(obj)
        )
        try:
            text = Text.from_markup(text).plain
        except Exception:
            pass
    return (
        text
        if not collapse
        else " ".join(part.strip() for part in text.splitlines() if part.strip())
    )


class PrimeConsole:
    def __init__(self, *, stderr=False, **kwargs):
        self.rich = Console(stderr=stderr, **kwargs)
        self.plain = Console(
            stderr=stderr,
            no_color=True,
            markup=False,
            highlight=False,
            emoji=False,
            **kwargs,
        )

    def print(self, *objects, sep=" ", end="\n", **kwargs):
        if not is_plain_mode():
            self.rich.print(*objects, sep=sep, end=end, **kwargs)
            return
        kwargs.pop("markup", None)
        kwargs.pop("highlight", None)
        self.plain.print(sep.join(_plain(obj) for obj in objects), end=end, **kwargs)

    def status(self, status, *args, **kwargs):
        if is_plain_mode():
            self.print(status)
            return nullcontext()
        if not self.rich.file.isatty():
            return nullcontext()
        return self.rich.status(status, *args, **kwargs)

    def print_exception(self, *args, **kwargs):
        if is_plain_mode():
            self.plain.print(traceback.format_exc().rstrip())
        else:
            self.rich.print_exception(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.plain if is_plain_mode() else self.rich, name)


def get_console(*, stderr=False, **kwargs):
    return PrimeConsole(stderr=stderr, **kwargs)


def help_note(obj, ctx):
    if ctx.parent is None and getattr(obj, "name", None) == "prime":
        return HELP_NOTE
    return None


def rich_help(obj, ctx, markup_mode):
    from typer import rich_utils

    note = help_note(obj, ctx)
    if note:
        rich_utils._get_rich_console().print(Text(note, style="dim"))
    if not (obj.epilog and obj.epilog.lstrip().startswith("JSON output")):
        rich_utils.rich_format_help(obj=obj, ctx=ctx, markup_mode=markup_mode)
        return
    epilog = obj.epilog
    obj.epilog = None
    try:
        rich_utils.rich_format_help(obj=obj, ctx=ctx, markup_mode=markup_mode)
    finally:
        obj.epilog = epilog
    lines = [line.strip() for line in epilog.splitlines() if line.strip()]
    grid = Table.grid(expand=True, padding=(0, 1))
    grid.add_column(ratio=1)
    for line in lines[1:] or lines:
        grid.add_row(rich_utils._make_rich_text(text=line, markup_mode=markup_mode))
    rich_utils._get_rich_console().print(
        Panel(
            grid,
            border_style=rich_utils.STYLE_OPTIONS_PANEL_BORDER,
            title=lines[0].removesuffix(":"),
            title_align="left",
        )
    )
