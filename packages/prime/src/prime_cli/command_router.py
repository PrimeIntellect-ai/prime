"""Small longest-path dispatcher for the Prime CLI."""

from __future__ import annotations

import importlib
import os
import sys
from typing import Any, Sequence

from pydantic_config import ConfigFileError, cli
from rich.box import ROUNDED
from rich.panel import Panel
from rich.table import Table

from prime_cli import __version__
from prime_cli.command_registry import GROUPS, Command, command_map
from prime_cli.core import Config as PrimeConfig
from prime_cli.utils.plain import HELP_NOTE, get_console, is_plain_mode

# Display order for the top-level command groups. The old ``rich_help_panel``
# grouping rendered these as titled boxes; we reproduce that layout from the
# registry's ``section`` field so the group-level help keeps the same panels
# the per-command help (pydantic_config) already shows.
SECTION_ORDER = ("Lab", "Compute", "Account")


def _print_help_group(title: str, rows: list[tuple[str, str]]) -> None:
    """Render one titled help group as a rounded panel (or plain text)."""
    rows = sorted(rows)
    if is_plain_mode():
        console = get_console()
        console.print(f"\n{title}:")
        for name, summary in rows:
            console.print(f"  {name}  {summary}".rstrip())
        return
    table = Table(box=None, show_header=False, padding=(0, 1))
    table.add_column(style="cyan", no_wrap=True)
    table.add_column()
    for name, summary in rows:
        table.add_row(name, summary)
    get_console().print(
        Panel(
            table,
            title=title,
            title_align="left",
            box=ROUNDED,
            border_style="dim",
            padding=(0, 1),
            width=get_console().width,
        )
    )


def _root_help(commands: dict[tuple[str, ...], Command], prefix: tuple[str, ...] = ()) -> None:
    console = get_console()
    name = "prime" + (" " + " ".join(prefix) if prefix else "")
    console.print(f"[bold]Usage:[/bold] {name} [OPTIONS] COMMAND [ARGS]...")
    if description := GROUPS.get(prefix):
        console.print(f"\n{description}")
    if not prefix:
        console.print("\nPrime Intellect CLI")
        console.print(f"\n[dim]{HELP_NOTE}[/dim]")
        _print_help_group(
            "Global options",
            [
                ("--context, -c NAME", "Select a saved Prime context"),
                ("--plain", "Use terse, unstyled output"),
                ("--version, -v", "Show the Prime CLI version"),
            ],
        )

    # Collect the direct children of ``prefix`` (deduped), keeping each
    # child's section so root help can group them into Lab/Compute/Account.
    children: dict[str, tuple[str, str]] = {}
    for path, command in commands.items():
        if path[: len(prefix)] != prefix or len(path) <= len(prefix):
            continue
        child = path[len(prefix)]
        is_group = len(path) > len(prefix) + 1
        summary = GROUPS.get((*prefix, child), f"{child} commands") if is_group else command.summary
        children.setdefault(child, (summary, command.section))

    if prefix:
        # Sub-group help lists its commands under a single "Commands" panel,
        # matching the old default subcommand rendering.
        _print_help_group("Commands", [(c, s) for c, (s, _) in children.items()])
        return

    by_section: dict[str, list[tuple[str, str]]] = {}
    for child, (summary, section) in children.items():
        by_section.setdefault(section, []).append((child, summary))
    for section in (*SECTION_ORDER, *(s for s in by_section if s not in SECTION_ORDER)):
        if section in by_section:
            _print_help_group(section, by_section[section])


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
            module = importlib.import_module(command.module)
            run_leaf = getattr(module, command.run_attr)
            if command.raw:
                return run_leaf(leaf_args)
            parsed_args = _positionals(leaf_args, command.positionals)
            config_model = getattr(module, command.config_attr)
            parse_config: Any = cli
            config = parse_config(  # ty: ignore[no-matching-overload]
                config_model,
                args=parsed_args,
                prog="prime " + " ".join(command.path),
                description=command.summary,
                plain=plain,
            )
            return run_leaf(config)
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
