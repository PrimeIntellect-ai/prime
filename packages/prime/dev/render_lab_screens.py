"""Print lightweight Lab TUI screen sketches for manual visual checks.

This is a developer utility, not an app dependency or test oracle. It writes
to stdout so quick checks can be redirected:

    uv run --project packages/prime python packages/prime/dev/render_lab_screens.py \
        --screen all > /tmp/lab-screens.txt
"""

from __future__ import annotations

import argparse
import importlib
import sys
import textwrap
from collections.abc import Iterable
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

LaunchBackdrop = importlib.import_module("prime_lab_app.launch_backdrop").LaunchBackdrop


ScreenName = str

SCREEN_CHOICES: tuple[ScreenName, ...] = (
    "all",
    "welcome",
    "settings",
    "environments",
    "environment",
    "training",
    "evaluations",
    "agent",
    "config",
    "rollouts",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--screen",
        choices=SCREEN_CHOICES,
        default="all",
        help="screen sketch to print",
    )
    parser.add_argument("--width", type=int, default=120, help="render width")
    parser.add_argument("--frame", type=int, default=9, help="welcome backdrop animation frame")
    parser.add_argument(
        "--ansi",
        action="store_true",
        help="emit ANSI color; omit for clean text files",
    )
    args = parser.parse_args()

    console = Console(
        width=max(80, args.width),
        force_terminal=args.ansi,
        color_system="truecolor" if args.ansi else None,
        soft_wrap=False,
    )
    screens = SCREEN_CHOICES[1:] if args.screen == "all" else (args.screen,)
    for index, screen in enumerate(screens):
        if index:
            console.print()
        _print_screen(console, screen, width=max(80, args.width), frame=args.frame)


def _print_screen(console: Console, screen: ScreenName, *, width: int, frame: int) -> None:
    title = screen.title()
    console.rule(f"[bold]{title}")
    renderer = {
        "welcome": _welcome,
        "settings": _settings,
        "environments": _environments,
        "environment": _environment,
        "training": _training,
        "evaluations": _evaluations,
        "agent": _agent,
        "config": _config,
        "rollouts": _rollouts,
    }[screen]
    renderer(console, width=width, frame=frame)


def _welcome(console: Console, *, width: int, frame: int) -> None:
    art_width = min(width - 10, 132)
    console.print(_center("PRIME Intellect", art_width), style="bold")
    console.print(_center("L A B / research control plane", art_width), style="bold")
    console.print(_center("Create. Evaluate. Train. Deploy.", art_width), style="dim")
    console.print()
    console.print(LaunchBackdrop(frame=frame).render_text(art_width, 12))
    console.print(_kv_line("ready", "~/dev/verifiers", "PI Applied Research", "Codex"))
    console.print(
        _button_grid(
            ("Explore Environments", "Train Models", "Run Evaluations", "Build with Codex")
        )
    )
    console.print(_footer("Enter Lab", "c Agent"))


def _settings(console: Console, *, width: int, frame: int) -> None:
    del width, frame
    console.print(_chrome("Settings", "~/dev/verifiers"))
    table = _three_column_table()
    table.add_row(
        _nav("Settings"),
        _panel_text(
            "Settings\n\nWorkspaces, profiles, local assets, setup.\n\n"
            "[Setup Lab workspace] [Sync assets] [Doctor]\n\n"
            "verifiers       active\n~/dev/verifiers\n\nprime-cli       inactive\n~/dev/prime-cli"
        ),
        _panel_text(
            "verifiers\n~/dev/verifiers\n\nStatus       active\nAuth         authenticated\n"
            "Team         PI Applied Research\nPrimary      Codex\n\nProfiles\nproduction   current"
        ),
    )
    console.print(table)
    console.print(_status("PI Applied Research", "~/dev/verifiers", "Codex"))


def _environments(console: Console, *, width: int, frame: int) -> None:
    del width, frame
    console.print(_chrome("Environments", "~/dev/verifiers"))
    table = _three_column_table()
    table.add_row(
        _nav("Environments"),
        _panel_text(
            "Environments\nLocal and platform environments. 33 shown\n\n"
            "primeintellect/alphabet-sort   LOCAL PUBLIC\n"
            "sentence-repeater              LOCAL\n"
            "primeintellect/math-python     PUBLIC\n"
            "research/private-env           PRIVATE"
        ),
        _panel_text(
            "primeintellect/alphabet-sort\nAlphabetized list updates across turns.\n\n"
            "Status       LOCAL PUBLIC\nSource       local, platform\nVersion      0.1.8\n"
            "Stars        6\nPath         environments/alphabet_sort\n\n"
            "[Open] [Train] [Evaluate] [Platform]"
        ),
    )
    console.print(table)
    console.print(_footer("Enter Open", "/ Filter", "Esc Back"))


def _environment(console: Console, *, width: int, frame: int) -> None:
    del width, frame
    console.print(_chrome("primeintellect/math-python  LOCAL PUBLIC", "~/dev/verifiers"))
    table = Table.grid(expand=True)
    table.add_column(ratio=3)
    table.add_column(ratio=7)
    table.add_column(ratio=3)
    table.add_row(
        Panel(
            "README.md       1.9 KB\nmath_python.py  1.9 KB\npyproject.toml    551 B\n\n"
            "[Train]\n[Evaluate]\n[Sync]\n[Platform]\n\nLinks\n[Source Code]",
            title="Files",
        ),
        Panel(
            "# math-python\n\nQuickstart\n\nprime eval run math-python\n\n"
            "Environment Arguments\n- dataset_name\n- dataset_split\n- num_train_examples",
            title="README",
        ),
        Panel(
            "About\nSolve math problems in a sandbox.\n\nVersion 0.1.10\nVisibility PUBLIC\n\n"
            "Dependencies\nverifiers>=0.1.8\nmath-verify>=0.8.0",
            title="About",
        ),
    )
    console.print(table)
    console.print(_footer("Enter Open", "Esc Back"))


def _training(console: Console, *, width: int, frame: int) -> None:
    del width, frame
    console.print(_chrome("wiki-search--qwen3-4b--tlabzb  COMPLETED", "~/dev/verifiers"))
    console.print("Overview   Data   System\n")
    console.print(
        "Run ID      tlabzbtv2f2ajhzp2hecm18i      Status      COMPLETED\n"
        "Model       Qwen/Qwen3.5-4B               Progress    100/100 steps\n"
        "Batch size  512                            Rollouts    16\n"
        "Progress    [########################################] 100.0%\n"
    )
    table = Table.grid(expand=True)
    table.add_column(ratio=7)
    table.add_column(ratio=3)
    table.add_row(
        Panel(
            "reward/all/mean\n\n  ···························◆\n  ·····························",
            title="Metric",
        ),
        Panel(
            '[Modify and run]\n\nmodel = "Qwen/Qwen3.5-4B"\nmax_steps = 100\nbatch_size = 512',
            title="Config",
        ),
    )
    console.print(table)


def _evaluations(console: Console, *, width: int, frame: int) -> None:
    del width, frame
    console.print(_chrome("Evaluations", "~/dev/verifiers"))
    table = _three_column_table()
    table.add_row(
        _nav("Evaluations"),
        _panel_text(
            "Evaluations\nLocal and platform evaluation runs.\n\n"
            "reverse-text--gpt-4.1-mini     COMPLETED  avg 0.98\n"
            "alphabet-sort--gpt-4.1-mini    RUNNING    42/50\n"
            "wiki-search--qwen3              FAILED     12/50"
        ),
        _panel_text(
            "Selection Details\n\nEnvironment   reverse-text\nModel         openai/gpt-4.1-mini\n"
            "Examples      50\nRollouts per example 3\nAverage score 0.98\n\n"
            "[Open] [Modify and run]"
        ),
    )
    console.print(table)


def _agent(console: Console, *, width: int, frame: int) -> None:
    del width, frame
    console.print(_chrome("Agent", "~/dev/verifiers"))
    console.print("│ › make an eval for reverse-text\n")
    console.print("· · ·\n")
    console.print("│ Ready to launch reverse-text.\n")
    console.print(
        Panel(
            "Evaluate reverse-text\n\n"
            "Environment  reverse-text\nModel        openai/gpt-4.1-mini\n"
            "Examples     50\nRollouts per example 3\nMax tokens   1024\nMax concurrent auto\n\n"
            "[Launch] [Stop]\n\nReady",
            border_style="green",
        )
    )
    console.print("\nAsk Codex  ·  /  ?  @")


def _config(console: Console, *, width: int, frame: int) -> None:
    del width, frame
    console.print(_chrome("Rerun with edits", "~/dev/verifiers"))
    table = _three_column_table()
    table.add_row(
        _panel_text("Source run\n\nRun ID abc123\nModel openai/gpt-oss-20b\nStatus COMPLETED"),
        _panel_text(
            "Edit configuration\n\nName        wiki-search-rerun-001\n"
            "Model       openai/gpt-oss-20b\n"
            "Max steps   150\nRollouts per example 16\nBatch size  512\nMax tokens  24576"
        ),
        _panel_text(
            "Validate and launch\n\n✓ Configuration is valid\n\n"
            "Changed fields\nmax_steps   100 -> 150\n"
            "max_tokens  16384 -> 24576\n\n[Launch]"
        ),
    )
    console.print(table)


def _rollouts(console: Console, *, width: int, frame: int) -> None:
    del width, frame
    console.print(_chrome("Training Data", "~/dev/verifiers"))
    table = _three_column_table()
    table.add_row(
        _panel_text("Samples\n\n#0 reward 0.000\n#1 reward 0.000\n#2 reward 1.000"),
        _panel_text(
            "Completion History\n\nSystem\nYou are solving a retrieval task.\n\n"
            "User\nFind the answer from the passage.\n\nAssistant\n<answer>hello endpoint</answer>"
        ),
        _panel_text(
            "Task\n\nEnvironment wiki-search\nRollouts per example 16\n\n"
            "State\ntool calls 2\nanswer found true"
        ),
    )
    console.print(table)


def _chrome(title: str, path: str) -> str:
    return f"L A B  {title:<48} {path:<26} PRIME Intellect\n" + "─" * 110


def _status(team: str, path: str, agent: str) -> str:
    return f"✓ {team} · {path} | ✓ {agent}"


def _footer(*items: str) -> str:
    return "  ·  ".join(items)


def _button_grid(labels: Iterable[str]) -> str:
    values = list(labels)
    rows = [values[index : index + 2] for index in range(0, len(values), 2)]
    return "\n".join(f"{left:^28}    {right:^28}" for left, right in rows)


def _kv_line(*items: str) -> str:
    return " · ".join(items)


def _center(value: str, width: int) -> str:
    return value.center(width)


def _panel_text(value: str) -> Panel:
    return Panel(textwrap.dedent(value).strip())


def _nav(active: str) -> Panel:
    items = ["Environments", "Training", "Evaluations"]
    lines = [f"> {item}" if item == active else f"  {item}" for item in items]
    return Panel("\n".join(lines), title="Sections")


def _three_column_table() -> Table:
    table = Table.grid(expand=True)
    table.add_column(ratio=2)
    table.add_column(ratio=5)
    table.add_column(ratio=3)
    return table


if __name__ == "__main__":
    main()
