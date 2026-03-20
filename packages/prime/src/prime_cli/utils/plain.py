"""Plain-mode runtime and rendering helpers."""

from __future__ import annotations

import sys
import traceback
from contextlib import AbstractContextManager
from typing import Any, Optional, Sequence, Type

import click
import typer
from rich.console import Console
from rich.markup import MarkupError
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from typer.core import TyperCommand, TyperGroup, _main
from typer.models import Default, DefaultPlaceholder

_PLAIN_MODE = False

_MARKUP_HINTS = (
    "[red]",
    "[/red]",
    "[yellow]",
    "[/yellow]",
    "[green]",
    "[/green]",
    "[blue]",
    "[/blue]",
    "[cyan]",
    "[/cyan]",
    "[magenta]",
    "[/magenta]",
    "[white]",
    "[/white]",
    "[dim]",
    "[/dim]",
    "[bold]",
    "[/bold]",
    "[bold ",
    "[b]",
    "[/b]",
    "[i]",
    "[/i]",
    "[u]",
    "[/u]",
    "[link=",
    "[/link]",
)


def set_plain_mode(enabled: bool) -> None:
    global _PLAIN_MODE
    _PLAIN_MODE = enabled


def is_plain_mode() -> bool:
    return _PLAIN_MODE


def _detect_plain_flag(args: Optional[Sequence[str]]) -> bool:
    if args is None:
        args = sys.argv[1:]

    for arg in args:
        if arg == "--":
            return False
        if arg == "--plain":
            return True
    return False


def _plain_option_callback(
    ctx: click.Context,  # noqa: ARG001
    param: click.Parameter,  # noqa: ARG001
    value: bool,
) -> None:
    if value:
        set_plain_mode(True)


def _create_plain_option() -> click.Option:
    return click.Option(
        param_decls=["--plain"],
        is_flag=True,
        default=False,
        expose_value=False,
        is_eager=True,
        callback=_plain_option_callback,
        help="Use plain text layout without Rich borders or colors.",
    )


def _ensure_plain_option(params: Optional[list[click.Parameter]]) -> list[click.Parameter]:
    resolved_params = list(params or [])
    if any(
        isinstance(param, click.Option) and "--plain" in getattr(param, "opts", [])
        for param in resolved_params
    ):
        return resolved_params

    insert_idx = len(resolved_params)
    for idx, param in enumerate(resolved_params):
        if isinstance(param, click.Option) and "--help" in getattr(param, "opts", []):
            insert_idx = idx
            break

    resolved_params.insert(insert_idx, _create_plain_option())
    return resolved_params


def _strip_markup(text: str, *, force: bool = False) -> str:
    if not force and not any(hint in text for hint in _MARKUP_HINTS):
        return text
    try:
        return Text.from_markup(text).plain
    except MarkupError:
        return text


def _plain_text(value: Any, *, preserve_lines: bool = True) -> str:
    if value is None:
        return ""
    if isinstance(value, Text):
        text = value.plain
    elif isinstance(value, Syntax):
        text = value.code
    elif isinstance(value, str):
        text = _strip_markup(value)
    else:
        text = _strip_markup(str(value))

    if preserve_lines:
        return text
    return " ".join(part.strip() for part in text.splitlines() if part.strip())


def _align(text: str, width: int, justify: str) -> str:
    if justify == "right":
        return text.rjust(width)
    if justify == "center":
        return text.center(width)
    return text.ljust(width)


def _render_plain_key_values(table: Table, headers: list[str], rows: list[list[str]]) -> str:
    lines: list[str] = []
    title = _plain_text(table.title) if table.title else ""
    if title:
        lines.append(title)

    key_width = max((len(row[0]) for row in rows if row and row[0]), default=0)
    if not rows and table.show_header:
        rows = [headers]

    for idx, row in enumerate(rows):
        key = row[0] if row else ""
        value = row[1] if len(row) > 1 else ""
        if key:
            if key_width:
                lines.append(f"{key.ljust(key_width)}: {value}")
            else:
                lines.append(f"{key}: {value}")
        elif value:
            lines.append(value)
        else:
            lines.append("")

        if idx < len(table.rows) and table.rows[idx].end_section and idx != len(rows) - 1:
            lines.append("")

    return "\n".join(lines)


def render_plain_table(table: Table) -> str:
    headers = [_plain_text(column.header, preserve_lines=False) for column in table.columns]
    rows: list[list[str]] = []
    for row_idx in range(table.row_count):
        row: list[str] = []
        for column in table.columns:
            cells = getattr(column, "_cells", [])
            cell = cells[row_idx] if row_idx < len(cells) else ""
            row.append(_plain_text(cell, preserve_lines=False))
        rows.append(row)

    if len(table.columns) == 2 and (
        not table.show_header
        or headers[1].lower() == "value"
        or headers[0].lower() in {"field", "property"}
    ):
        return _render_plain_key_values(table, headers, rows)

    widths = [0] * len(table.columns)
    if table.show_header:
        for idx, header in enumerate(headers):
            widths[idx] = max(widths[idx], len(header))
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    lines: list[str] = []
    title = _plain_text(table.title) if table.title else ""
    if title:
        lines.append(title)

    if table.show_header and headers:
        header_row = "  ".join(
            _align(headers[idx], widths[idx], getattr(column, "justify", "left"))
            for idx, column in enumerate(table.columns)
        )
        lines.append(header_row.rstrip())

    for row_idx, row in enumerate(rows):
        line = "  ".join(
            _align(row[idx], widths[idx], getattr(column, "justify", "left"))
            for idx, column in enumerate(table.columns)
        )
        lines.append(line.rstrip())

        if (
            row_idx < len(table.rows)
            and table.rows[row_idx].end_section
            and row_idx != len(rows) - 1
        ):
            lines.append("")

    return "\n".join(lines)


def render_plain(value: Any) -> str:
    if isinstance(value, Table):
        return render_plain_table(value)
    return _plain_text(value)


class _PlainStatus(AbstractContextManager[None]):
    def __init__(self, console: "PrimeConsole", status: Any):
        self._console = console
        self._status = status

    def __enter__(self) -> None:
        text = render_plain(self._status)
        if text:
            self._console.print(text)
        return None

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        return None


class PrimeConsole:
    """Small console proxy that renders dense plain text when requested."""

    def __init__(self, *, stderr: bool = False, **kwargs: Any) -> None:
        self._stderr = stderr
        self._kwargs = kwargs
        self._rich_console = Console(stderr=stderr, **kwargs)
        plain_kwargs = dict(kwargs)
        plain_kwargs["color_system"] = None
        plain_kwargs["force_terminal"] = False
        plain_kwargs["no_color"] = True
        plain_kwargs["markup"] = False
        plain_kwargs["emoji"] = False
        plain_kwargs["highlight"] = False
        self._plain_console = Console(
            stderr=stderr,
            **plain_kwargs,
        )

    def print(self, *objects: Any, sep: str = " ", end: str = "\n", **kwargs: Any) -> None:
        if not is_plain_mode():
            self._rich_console.print(*objects, sep=sep, end=end, **kwargs)
            return

        markup = kwargs.pop("markup", True)
        kwargs.pop("highlight", None)

        rendered = [
            _plain_text(obj, preserve_lines=True) if markup is False else render_plain(obj)
            for obj in objects
        ]
        text = sep.join(rendered)
        self._plain_console.print(text, sep=sep, end=end, markup=False, highlight=False, **kwargs)

    def status(self, status: Any, *args: Any, **kwargs: Any) -> AbstractContextManager[Any]:
        if is_plain_mode():
            return _PlainStatus(self, status)
        return self._rich_console.status(status, *args, **kwargs)

    def print_exception(self, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002
        if not is_plain_mode():
            self._rich_console.print_exception(*args, **kwargs)
            return

        text = traceback.format_exc().rstrip()
        if text:
            self._plain_console.print(text, markup=False, highlight=False)

    def __getattr__(self, name: str) -> Any:
        active_console = self._plain_console if is_plain_mode() else self._rich_console
        return getattr(active_console, name)


def get_console(*, stderr: bool = False, **kwargs: Any) -> PrimeConsole:
    return PrimeConsole(stderr=stderr, **kwargs)


def _print_rich_epilog_panel(epilog: str, markup_mode: Any) -> None:
    from typer import rich_utils

    raw_lines = [line.rstrip() for line in epilog.splitlines()]
    while raw_lines and not raw_lines[0].strip():
        raw_lines.pop(0)
    while raw_lines and not raw_lines[-1].strip():
        raw_lines.pop()

    if not raw_lines:
        return

    title = raw_lines[0].strip()
    if title.endswith(":"):
        title = title[:-1]

    body_lines = [line.strip() for line in raw_lines[1:] if line.strip()]
    if not body_lines:
        body_lines = [line.strip() for line in raw_lines if line.strip()]
        title = "Notes"

    epilog_table = Table.grid(expand=True, padding=(0, 1))
    epilog_table.add_column(ratio=1)
    for line in body_lines:
        epilog_table.add_row(rich_utils._make_rich_text(text=line, markup_mode=markup_mode))

    console = rich_utils._get_rich_console()
    console.print(
        Panel(
            epilog_table,
            border_style=rich_utils.STYLE_OPTIONS_PANEL_BORDER,
            title=title,
            title_align="left",
        )
    )


def _rich_format_help_with_panels(
    obj: click.Command | click.Group,
    ctx: click.Context,
    markup_mode: Any,
) -> None:
    from typer import rich_utils

    epilog = obj.epilog
    if not epilog or not epilog.lstrip().startswith("JSON output"):
        rich_utils.rich_format_help(obj=obj, ctx=ctx, markup_mode=markup_mode)
        return

    obj.epilog = None
    try:
        rich_utils.rich_format_help(obj=obj, ctx=ctx, markup_mode=markup_mode)
    finally:
        obj.epilog = epilog

    _print_rich_epilog_panel(epilog, markup_mode)


class PlainAwareTyperCommand(TyperCommand):
    def __init__(
        self,
        *args: Any,
        params: Optional[list[click.Parameter]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, params=_ensure_plain_option(params), **kwargs)

    def main(
        self,
        args: Optional[Sequence[str]] = None,
        prog_name: Optional[str] = None,
        complete_var: Optional[str] = None,
        standalone_mode: bool = True,
        windows_expand_args: bool = True,
        **extra: Any,
    ) -> Any:
        set_plain_mode(_detect_plain_flag(args))
        rich_markup_mode = None if is_plain_mode() else self.rich_markup_mode
        return _main(
            self,
            args=args,
            prog_name=prog_name,
            complete_var=complete_var,
            standalone_mode=standalone_mode,
            windows_expand_args=windows_expand_args,
            rich_markup_mode=rich_markup_mode,
            **extra,
        )

    def format_help(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        if is_plain_mode():
            return click.core.Command.format_help(self, ctx, formatter)
        _rich_format_help_with_panels(self, ctx, self.rich_markup_mode)
        return None


class PlainAwareTyperGroup(TyperGroup):
    def __init__(
        self,
        *args: Any,
        params: Optional[list[click.Parameter]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, params=_ensure_plain_option(params), **kwargs)

    def main(
        self,
        args: Optional[Sequence[str]] = None,
        prog_name: Optional[str] = None,
        complete_var: Optional[str] = None,
        standalone_mode: bool = True,
        windows_expand_args: bool = True,
        **extra: Any,
    ) -> Any:
        set_plain_mode(_detect_plain_flag(args))
        rich_markup_mode = None if is_plain_mode() else self.rich_markup_mode
        return _main(
            self,
            args=args,
            prog_name=prog_name,
            complete_var=complete_var,
            standalone_mode=standalone_mode,
            windows_expand_args=windows_expand_args,
            rich_markup_mode=rich_markup_mode,
            **extra,
        )

    def format_help(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        if is_plain_mode():
            return click.core.Group.format_help(self, ctx, formatter)
        _rich_format_help_with_panels(self, ctx, self.rich_markup_mode)
        return None


class PlainTyper(typer.Typer):
    def __init__(self, *args: Any, cls: Optional[Type[TyperGroup]] = None, **kwargs: Any) -> None:
        super().__init__(*args, cls=cls or PlainAwareTyperGroup, **kwargs)

    def callback(
        self,
        *,
        cls: Optional[Type[TyperGroup]] = Default(None),
        invoke_without_command: bool = Default(False),
        no_args_is_help: bool = Default(False),
        subcommand_metavar: Optional[str] = Default(None),
        chain: bool = Default(False),
        result_callback: Optional[Any] = Default(None),
        context_settings: Optional[dict[Any, Any]] = Default(None),
        help: Optional[str] = Default(None),
        epilog: Optional[str] = Default(None),
        short_help: Optional[str] = Default(None),
        options_metavar: str = Default("[OPTIONS]"),
        add_help_option: bool = Default(True),
        hidden: bool = Default(False),
        deprecated: bool = Default(False),
        rich_help_panel: Optional[str] = Default(None),
    ) -> Any:
        resolved_cls = (
            PlainAwareTyperGroup if isinstance(cls, DefaultPlaceholder) or cls is None else cls
        )
        return super().callback(
            cls=resolved_cls,
            invoke_without_command=invoke_without_command,
            no_args_is_help=no_args_is_help,
            subcommand_metavar=subcommand_metavar,
            chain=chain,
            result_callback=result_callback,
            context_settings=context_settings,
            help=help,
            epilog=epilog,
            short_help=short_help,
            options_metavar=options_metavar,
            add_help_option=add_help_option,
            hidden=hidden,
            deprecated=deprecated,
            rich_help_panel=rich_help_panel,
        )

    def command(
        self,
        name: Optional[str] = None,
        *,
        cls: Optional[Type[TyperCommand]] = None,
        context_settings: Optional[dict[Any, Any]] = None,
        help: Optional[str] = None,
        epilog: Optional[str] = None,
        short_help: Optional[str] = None,
        options_metavar: str = "[OPTIONS]",
        add_help_option: bool = True,
        no_args_is_help: bool = False,
        hidden: bool = False,
        deprecated: bool = False,
        rich_help_panel: Optional[str] = None,
    ) -> Any:
        return super().command(
            name=name,
            cls=cls or PlainAwareTyperCommand,
            context_settings=context_settings,
            help=help,
            epilog=epilog,
            short_help=short_help,
            options_metavar=options_metavar,
            add_help_option=add_help_option,
            no_args_is_help=no_args_is_help,
            hidden=hidden,
            deprecated=deprecated,
            rich_help_panel=rich_help_panel,
        )
