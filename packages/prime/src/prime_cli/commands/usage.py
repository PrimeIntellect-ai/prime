"""Token usage and price commands.

Two entry points are exported:

- ``run_usage_command`` — registered as ``prime train usage`` in
  ``commands/rl.py`` (sits next to ``train logs``, ``train metrics``, etc.).
- ``summary_command`` — registered as the top-level ``prime usage`` so the
  account-wide billing roll-up has its own ergonomic command (no subgroup).

Both mirror the platform billing/analytics view, with sandbox excluded.
"""

import time
from typing import Any, Dict, Optional

import typer
from rich.live import Live
from rich.markup import escape as rich_escape
from rich.table import Table

from prime_cli.api.billing import (
    AreaUsage,
    BillingClient,
    RunUsage,
    UsageSummary,
)
from prime_cli.core import APIClient, APIError
from prime_cli.utils import (
    build_table,
    get_console,
    is_plain_mode,
    json_output_help,
    output_data_as_json,
    validate_output_format,
)
from prime_cli.utils.formatters import format_price_per_mtok

console = get_console()


VALID_PERIODS = ("7_days", "this_month", "last_month", "6_months")

RUN_USAGE_JSON_HELP = json_output_help(
    ". = {run_id, run_name?, base_model?, status?, total_tokens, "
    "total_cost_usd, record_count, "
    "training: {tokens, input_tokens, output_tokens, cost_usd}, "
    "inference: {tokens, input_tokens, output_tokens, cost_usd}, "
    "pricing: {training_per_mtok?, inference_input_per_mtok?, "
    "inference_output_per_mtok?}}"
)

USAGE_SUMMARY_JSON_HELP = json_output_help(
    ". = {period, start_date, end_date, wallet_id, team_id?, total_cost_usd, "
    "areas: [{area, total_cost_usd, training_tokens, inference_tokens, "
    "inference_requests}]}"
)


def _format_tokens(value: int) -> str:
    """Render token counts compactly: 1234 → '1.23K', 1_500_000 → '1.50M'."""
    if value <= 0:
        return "0"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.2f}K"
    return str(value)


def _format_usd(value: float) -> str:
    if value == 0:
        return "$0.00"
    if abs(value) < 0.01:
        return f"${value:.4f}"
    return f"${value:.2f}"


def _run_usage_json(usage: RunUsage) -> Dict[str, Any]:
    return usage.model_dump()


def _summary_json(summary: UsageSummary) -> Dict[str, Any]:
    return summary.model_dump()


def _build_run_usage_table(usage: RunUsage) -> Table:
    # Rich parses table titles as markup, so values from the API (run_name,
    # status) must be escaped before interpolation — otherwise a stray
    # "[bold]" or similar in those strings would be interpreted as markup.
    name = rich_escape(usage.run_name or usage.run_id)
    title = f"Run Usage — {name}"
    if usage.status:
        title += f"  [{rich_escape(usage.status)}]"

    table = build_table(
        title,
        [
            ("Bucket", "cyan"),
            ("Tokens", "white"),
            ("Input tokens", "white"),
            ("Output tokens", "white"),
            ("Cost", "green"),
            ("Price / Mtok", "magenta"),
        ],
        show_lines=False,
    )

    table.add_row(
        "Training",
        _format_tokens(usage.training.tokens),
        "-",
        "-",
        _format_usd(usage.training.cost_usd),
        format_price_per_mtok(usage.pricing.training_per_mtok) or "-",
    )
    table.add_row(
        "Inference (input)",
        "-",
        _format_tokens(usage.inference.input_tokens),
        "-",
        "",
        format_price_per_mtok(usage.pricing.inference_input_per_mtok) or "-",
    )
    table.add_row(
        "Inference (output)",
        "-",
        "-",
        _format_tokens(usage.inference.output_tokens),
        _format_usd(usage.inference.cost_usd),
        format_price_per_mtok(usage.pricing.inference_output_per_mtok) or "-",
    )
    table.add_row(
        "[bold]Total[/bold]",
        f"[bold]{_format_tokens(usage.total_tokens)}[/bold]",
        "",
        "",
        f"[bold]{_format_usd(usage.total_cost_usd)}[/bold]",
        "",
    )
    return table


def _build_summary_table(summary: UsageSummary) -> Table:
    # Escape API-supplied strings to keep them out of Rich's markup parser.
    period = rich_escape(summary.period)
    start = rich_escape(summary.start_date)
    end = rich_escape(summary.end_date)
    title = f"Usage Summary — {period} ({start} → {end})"
    table = build_table(
        title,
        [
            ("Area", "cyan"),
            ("Tokens / Requests", "white"),
            ("Cost", "green"),
        ],
        show_lines=False,
    )

    for area in summary.areas:
        table.add_row(
            area.area.title(),
            _format_area_metric(area),
            _format_usd(area.total_cost_usd),
        )

    table.add_row(
        "[bold]Total[/bold]",
        "",
        f"[bold]{_format_usd(summary.total_cost_usd)}[/bold]",
    )
    return table


def _format_area_metric(area: AreaUsage) -> str:
    if area.area == "training":
        train = _format_tokens(area.training_tokens)
        infer = _format_tokens(area.inference_tokens)
        return f"train {train} / inf {infer}"
    if area.area == "inference":
        return f"{area.inference_requests} req"
    return "-"


def run_usage_command(
    run_id: str = typer.Argument(..., help="RFT run ID (e.g. rft_..."),
    output: str = typer.Option("table", "--output", "-o", help="Output format: table or json"),
    watch: bool = typer.Option(
        False, "--watch", "-w", help="Poll continuously and update in place"
    ),
    interval: int = typer.Option(
        30,
        "--interval",
        "-n",
        min=2,
        help="Seconds between polls when --watch is set",
    ),
) -> None:
    """Show token usage and price for a single training run.

    Example:

        prime train usage <run_id>

        prime train usage <run_id> --watch --interval 15

        prime train usage <run_id> --output json
    """
    validate_output_format(output, console)

    billing = BillingClient(APIClient())

    if not watch:
        try:
            usage = billing.get_run_usage(run_id)
        except APIError as exc:
            console.print(f"[red]Error: {exc}[/red]")
            raise typer.Exit(1) from exc

        if output == "json":
            output_data_as_json(_run_usage_json(usage), console)
            return
        console.print(_build_run_usage_table(usage))
        return

    # Watch mode: the watcher does the first fetch itself — no eager call.
    if output == "json":
        _watch_json(billing.get_run_usage, run_id, interval, _run_usage_json)
        return

    if is_plain_mode():
        _watch_plain(billing.get_run_usage, run_id, interval, _build_run_usage_table)
        return

    _watch_live(billing.get_run_usage, run_id, interval, _build_run_usage_table)


def summary_command(
    period: str = typer.Option(
        "this_month",
        "--period",
        "-p",
        help=f"Time window. One of: {', '.join(VALID_PERIODS)}",
    ),
    team: Optional[str] = typer.Option(
        None, "--team", "-t", help="Team ID (defaults to personal wallet)"
    ),
    output: str = typer.Option("table", "--output", "-o", help="Output format: table or json"),
) -> None:
    """Account-wide billing summary (sandbox excluded).

    Mirrors the dashboard analytics view — total tokens + cost per area
    (training, inference, compute, disks, images) for the chosen period.

    Example:

        prime usage

        prime usage --period 7_days --output json
    """
    validate_output_format(output, console)

    if period not in VALID_PERIODS:
        console.print(
            f"[red]Error: invalid --period '{period}'. "
            f"Expected one of: {', '.join(VALID_PERIODS)}[/red]"
        )
        raise typer.Exit(1)

    billing = BillingClient(APIClient())

    try:
        result = billing.get_usage_summary(period=period, team_id=team)
    except APIError as exc:
        console.print(f"[red]Error: {exc}[/red]")
        raise typer.Exit(1) from exc

    if output == "json":
        output_data_as_json(_summary_json(result), console)
        return

    console.print(_build_summary_table(result))


def _watch_live(fetch, run_id, interval, render):
    try:
        with Live(render(fetch(run_id)), console=console, refresh_per_second=4) as live:
            while True:
                time.sleep(interval)
                live.update(render(fetch(run_id)))
    except KeyboardInterrupt:
        console.print("\n[dim]Stopped.[/dim]")
    except APIError as exc:
        console.print(f"[red]Error during watch: {exc}[/red]")
        raise typer.Exit(1) from exc


def _watch_plain(fetch, run_id, interval, render):
    try:
        while True:
            console.print(render(fetch(run_id)))
            time.sleep(interval)
    except KeyboardInterrupt:
        console.print("\nStopped.")
    except APIError as exc:
        console.print(f"[red]Error during watch: {exc}[/red]")
        raise typer.Exit(1) from exc


def _watch_json(fetch, run_id, interval, to_json):
    # JSON watch streams one object per tick to stdout — keep stdout strictly
    # JSON so agents can parse it with `jq -c` or similar. All diagnostics
    # (errors, interrupts) go to stderr to avoid corrupting the stream.
    err_console = get_console(stderr=True)
    try:
        while True:
            output_data_as_json(to_json(fetch(run_id)), console)
            time.sleep(interval)
    except KeyboardInterrupt:
        return
    except APIError as exc:
        err_console.print(f"[red]Error during watch: {exc}[/red]")
        raise typer.Exit(1) from exc
