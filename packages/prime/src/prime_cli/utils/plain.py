from typing import Any, cast

import click
import typer
from typer.core import TyperCommand, TyperGroup, _main
from typer.models import Default, DefaultPlaceholder

from . import plain_support

get_console = plain_support.get_console
is_plain_mode = plain_support.is_plain_mode
rich_help = plain_support.rich_help


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
        cast(Any, super()).__init__(*args, params=_plain_option(params), **kwargs)

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
            cast(click.Command, self),
            args=args,
            prog_name=prog_name,
            complete_var=complete_var,
            standalone_mode=standalone_mode,
            windows_expand_args=windows_expand_args,
            rich_markup_mode=None if plain else cast(Any, self).rich_markup_mode,
            **extra,
        )


class PlainAwareTyperCommand(_PlainMixin, TyperCommand):
    def format_help(self, ctx, formatter):
        note = plain_support.help_note(self, ctx)
        if is_plain_mode():
            if note:
                formatter.write_text(f"Note: {note}\n")
            return click.core.Command.format_help(self, ctx, formatter)
        return rich_help(self, ctx, self.rich_markup_mode)


class PlainAwareTyperGroup(_PlainMixin, TyperGroup):
    def format_help(self, ctx, formatter):
        note = plain_support.help_note(self, ctx)
        if is_plain_mode():
            if note:
                formatter.write_text(f"Note: {note}\n")
            return click.core.Group.format_help(self, ctx, formatter)
        return rich_help(self, ctx, self.rich_markup_mode)


class PlainTyper(typer.Typer):
    def __init__(self, *args, cls=None, **kwargs):
        super().__init__(*args, cls=cls or PlainAwareTyperGroup, **kwargs)

    def callback(self, *, cls=Default(None), **kwargs):
        cls = PlainAwareTyperGroup if isinstance(cls, DefaultPlaceholder) or cls is None else cls
        return super().callback(cls=cls, **kwargs)

    def command(self, name=None, *, cls=None, **kwargs):
        return super().command(name=name, cls=cls or PlainAwareTyperCommand, **kwargs)
