"""Small longest-path dispatcher for the Prime CLI."""

from __future__ import annotations

import importlib
import os
import sys
from typing import Any, Sequence

from pydantic_config import ConfigFileError, cli
from rich import box
from rich.panel import Panel
from rich.table import Table

from prime_cli import __version__
from prime_cli.command_registry import GROUPS, ROOT_GROUP, SECTION_ORDER, Command, command_map
from prime_cli.core import Config as PrimeConfig
from prime_cli.utils.plain import HELP_NOTE, get_console


def _global_options(argv: list[str]) -> tuple[list[str], bool]:
    """Consume root-only options before the command path."""
    plain = False
    while argv and argv[0].startswith("-"):
        arg = argv.pop(0)
        if arg in ("-h", "--help"):
            return ["--help", *argv], plain
        if arg in ("-v", "--version"):
            print(f"Prime CLI version: {__version__}")
            raise SystemExit(0)
        if arg == "--plain":
            plain = True
            continue
        if arg.startswith("--context="):
            _select_context(arg.split("=", 1)[1])
            continue
        if arg in ("-c", "--context"):
            if not argv:
                raise ConfigFileError(f"{arg} requires a value")
            _select_context(argv.pop(0))
            continue
        argv.insert(0, arg)
        break
    return argv, plain


def _select_context(context: str) -> None:
    config = PrimeConfig()
    if context.lower() != "production" and context not in config.list_environments():
        available = ", ".join(config.list_environments()) or "none"
        raise ConfigFileError(f"Unknown context {context!r}. Available contexts: {available}")
    os.environ["PRIME_CONTEXT"] = context


def _positionals(argv: list[str], fields: tuple[str, ...]) -> list[str]:
    """Turn leading positional values into ordinary Pydantic CLI flags."""
    if not fields or not argv or argv[0].startswith(("-", "@")):
        return argv
    values: list[str] = []
    while argv and not argv[0].startswith(("-", "@")):
        values.append(argv.pop(0))
    if len(values) > len(fields):
        last = fields[-1]
        leading = [
            item
            for pair in zip(fields[:-1], values[: len(fields) - 1])
            for item in (f"--{pair[0].replace('_', '-')}", pair[1])
        ]
        return [*leading, f"--{last.replace('_', '-')}", *values[len(fields) - 1 :], *argv]
    leading = [
        item for pair in zip(fields, values) for item in (f"--{pair[0].replace('_', '-')}", pair[1])
    ]
    return [*leading, *argv]


def _load_ref(ref: str) -> Any:
    module_name, name = ref.split(":", 1)
    return getattr(importlib.import_module(module_name), name)


def _root_help(commands: dict[tuple[str, ...], Command], prefix: tuple[str, ...] = ()) -> None:
    console = get_console()
    name = "prime" + (" " + " ".join(prefix) if prefix else "")
    group = GROUPS.get(prefix) or (ROOT_GROUP if not prefix else None)
    usage = group.usage if group else "[OPTIONS] COMMAND [ARGS]..."
    description = group.summary if group else None
    note = group.note if group else None
    children: dict[str, tuple[str, str]] = {}
    for path, command in commands.items():
        if path[: len(prefix)] != prefix or len(path) <= len(prefix):
            continue
        child = path[len(prefix)]
        is_group = len(path) > len(prefix) + 1
        child_group = GROUPS.get((*prefix, child))
        summary = child_group.summary if is_group and child_group else command.summary
        children.setdefault(child, (summary, command.section))

    console.print(f"[yellow]Usage:[/yellow] [bold]{name}[/bold] {usage}")
    if description:
        console.print(f"\n{description}")
    if note:
        console.print(f"\n[dim]{note}[/dim]")
    if not prefix:
        console.print(f"\n[dim]{HELP_NOTE}[/dim]")

    options = Table.grid(padding=(0, 2))
    options.add_column(style="green", no_wrap=True)
    options.add_column(style="yellow", no_wrap=True)
    options.add_column()
    if not prefix:
        options.add_row("--context", "-c NAME", "Select a saved Prime context")
        options.add_row("--version", "-v", "Show the Prime CLI version")
    options.add_row("--plain", "", "Use plain, terse outputs. USE THIS IF YOU ARE AI.")
    options.add_row("--help", "-h", "Show this message and exit.")

    console.print()
    console.print(
        Panel(
            options,
            title="Options",
            title_align="left",
            border_style="dim",
            box=box.ROUNDED,
        )
    )

    command_sections: dict[str, dict[str, tuple[str, str]]] = {}
    for child, value in children.items():
        title = value[1] if not prefix else "Commands"
        command_sections.setdefault(title, {})[child] = value
    section_titles = (
        list(SECTION_ORDER) + [title for title in command_sections if title not in SECTION_ORDER]
        if not prefix
        else ["Commands"]
    )
    for title in section_titles:
        rows = command_sections.get(title)
        if not rows:
            continue
        command_table = Table.grid(padding=(0, 3))
        command_table.add_column(style="green", no_wrap=True)
        command_table.add_column()
        for child, (summary, _) in rows.items():
            command_table.add_row(child, summary)
        console.print(
            Panel(
                command_table,
                title=title,
                title_align="left",
                border_style="dim",
                box=box.ROUNDED,
            )
        )


class Router:
    name = "prime"

    def main(
        self,
        args: Sequence[str] = (),
        prog_name: str | None = None,
        standalone_mode: bool = True,
        **_: Any,
    ) -> Any:
        del prog_name, standalone_mode
        argv = list(args)
        previous_plain = os.environ.get("PRIME_PLAIN")
        previous_context = os.environ.get("PRIME_CONTEXT")
        try:
            argv, plain = _global_options(argv)
            plain = plain or "--plain" in argv
            argv = [arg for arg in argv if arg != "--plain"]
            if plain:
                os.environ["PRIME_PLAIN"] = "1"

            commands = command_map()
            if argv[:1] in ([], ["-h"], ["--help"]):
                _root_help(commands)
                return None

            if argv[-1:] and argv[-1] in ("-h", "--help"):
                group = tuple(argv[:-1])
                if group not in commands and any(path[: len(group)] == group for path in commands):
                    _root_help(commands, group)
                    return None

            tokens = argv
            for group, group_info in GROUPS.items():
                if not group_info.default:
                    continue
                if tuple(tokens[: len(group)]) != group or len(tokens) <= len(group):
                    continue
                next_token = tokens[len(group)]
                if next_token.startswith("-") or (*group, next_token) in commands:
                    continue
                tokens = [*group, group_info.default, *tokens[len(group) :]]
                break
            command = next(
                (
                    commands[tuple(tokens[:length])]
                    for length in range(len(tokens), 0, -1)
                    if tuple(tokens[:length]) in commands
                ),
                None,
            )
            if command is None:
                group = tuple(tokens)
                if any(path[: len(group)] == group for path in commands):
                    _root_help(commands, group)
                    return None
                raise ConfigFileError(f"Unknown command: {' '.join(tokens)}")

            leaf_args = tokens[len(command.path) :]
            if command.raw:
                callback = _load_ref(command.callback)
                return callback(leaf_args)

            if command.config is None:
                raise RuntimeError(f"command {' '.join(command.path)} is missing config")
            parsed_args = _positionals(leaf_args, command.positionals)
            config_model = _load_ref(command.config)
            parse_config: Any = cli
            config = parse_config(  # ty: ignore[no-matching-overload]
                config_model,
                args=parsed_args,
                prog="prime " + " ".join(command.path),
                description=command.summary,
                plain=plain,
            )
            callback = _load_ref(command.callback)
            return callback(config)
        except (ConfigFileError, ValueError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            raise SystemExit(2) from exc
        finally:
            if previous_plain is None:
                os.environ.pop("PRIME_PLAIN", None)
            else:
                os.environ["PRIME_PLAIN"] = previous_plain
            if previous_context is None:
                os.environ.pop("PRIME_CONTEXT", None)
            else:
                os.environ["PRIME_CONTEXT"] = previous_context


app = Router()
