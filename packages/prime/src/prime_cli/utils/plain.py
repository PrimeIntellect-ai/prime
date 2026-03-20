import io
import sys
import traceback
from contextlib import nullcontext
from copy import copy
from typing import Any

import click
import typer
from rich.console import Console
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from typer.core import TyperCommand, TyperGroup, _main
from typer.models import Default, DefaultPlaceholder


def is_plain_mode(args: list[str] | None = None) -> bool:
    ctx = click.get_current_context(silent=True)
    while ctx is not None:
        if ctx.meta.get("plain"):
            return True
        ctx = ctx.parent

    for arg in sys.argv[1:] if args is None else args:
        if arg == "--":
            return False
        if arg == "--plain":
            return True
    return False


class PrimeConsole(Console):
    def __init__(self, *, stderr: bool = False, **kwargs: Any) -> None:
        super().__init__(stderr=stderr, **kwargs)
        self._plain_console = Console(
            stderr=stderr,
            no_color=True,
            markup=False,
            highlight=False,
            emoji=False,
            **kwargs,
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


def _plain_option(params):
    params = list(params or [])
    if any(isinstance(param, click.Option) and "--plain" in param.opts for param in params):
        return params

    def enable_plain(ctx, _, value):
        if value:
            ctx.meta["plain"] = True

    params.insert(
        next(
            (
                i
                for i, param in enumerate(params)
                if isinstance(param, click.Option) and "--help" in param.opts
            ),
            len(params),
        ),
        click.Option(
            ["--plain"],
            is_flag=True,
            expose_value=False,
            is_eager=True,
            callback=enable_plain,
            help="Use plain, terse outputs. USE THIS IF YOU ARE AI.",
        ),
    )
    return params


class _PlainMixin:
    def __init__(self, *args, params=None, **kwargs):
        super().__init__(*args, params=_plain_option(params), **kwargs)

    def main(
        self,
        args=None,
        prog_name=None,
        complete_var=None,
        standalone_mode=True,
        windows_expand_args=True,
        **extra,
    ):
        plain = is_plain_mode(args)
        return _main(
            self,
            args=args,
            prog_name=prog_name,
            complete_var=complete_var,
            standalone_mode=standalone_mode,
            windows_expand_args=windows_expand_args,
            rich_markup_mode=None if plain else self.rich_markup_mode,
            **extra,
        )


class _PlainTyperCommand(_PlainMixin, TyperCommand):
    def format_help(self, ctx, formatter):
        if not is_plain_mode():
            return super().format_help(ctx, formatter)

        rich_markup_mode = self.rich_markup_mode
        self.rich_markup_mode = None
        try:
            return super().format_help(ctx, formatter)
        finally:
            self.rich_markup_mode = rich_markup_mode


class PlainAwareTyperGroup(_PlainMixin, TyperGroup):
    def format_help(self, ctx, formatter):
        if not is_plain_mode():
            return super().format_help(ctx, formatter)

        rich_markup_mode = self.rich_markup_mode
        self.rich_markup_mode = None
        try:
            return super().format_help(ctx, formatter)
        finally:
            self.rich_markup_mode = rich_markup_mode


class PlainTyper(typer.Typer):
    def __init__(self, *args, cls=None, **kwargs):
        super().__init__(*args, cls=cls or PlainAwareTyperGroup, **kwargs)

    def callback(self, *, cls=Default(None), **kwargs):
        cls = PlainAwareTyperGroup if isinstance(cls, DefaultPlaceholder) or cls is None else cls
        return super().callback(cls=cls, **kwargs)

    def command(self, name=None, *, cls=None, **kwargs):
        return super().command(name=name, cls=cls or _PlainTyperCommand, **kwargs)
