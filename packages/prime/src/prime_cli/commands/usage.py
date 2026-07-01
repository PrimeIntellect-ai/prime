"""`prime train usage` — token usage and price for a single RFT run.

Registered as a subcommand of ``prime train`` in ``commands/rl.py`` (sits
next to ``train logs``, ``train metrics``, etc.). Mirrors the per-row data
the dashboard's billing page shows for a training run, with ``--watch`` for
live polling so an agent can monitor a run's running cost.
"""

import time

import typer
from rich.live import Live
from rich.markup import escape as rich_escape
from rich.table import Table

from prime_cli.api.billing import BillingClient, RunUsage
from prime_cli.core import APIClient, APIError
from prime_cli.utils import (
    build_table,
    get_console,
    is_plain_mode,
    json_output_help,
    output_data_as_json,
    validate_output_format,
)
from prime_cli.utils.formatters import format_price_per_mtok, format_usd

console = get_console()


RUN_USAGE_JSON_HELP = json_output_help(
    ". = {run_id, run_name?, base_model?, status?, total_tokens, "
    "total_cost_usd, record_count, "
    "training: {tokens, input_tokens, output_tokens, cost_usd}, "
    "inference: {tokens, input_tokens, output_tokens, cached_input_tokens?, cost_usd}, "
    "pricing: {training_per_mtok?, inference_input_per_mtok?, "
    "inference_output_per_mtok?, inference_cached_input_per_mtok?}}"
)


def _format_tokens(value: int) -> str:
    """Render token counts compactly: 1234 → '1.23K', 1_500_000 → '1.50M'.

    Promotes to M whenever rounding to 2dp would produce ≥ 1.00M, so values
    just below the boundary (e.g. 999_995) render as ``1.00M`` instead of
    the misleading ``1000.00K``.
    """
    if value <= 0:
        return "0"
    if round(value / 1_000_000, 2) >= 1.00:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.2f}K"
    return str(value)


def _derived_cost(tokens: int, price_per_mtok: float | None) -> float | None:
    """Compute cost in USD from tokens × snapshotted rate.

    Returns None when the rate is missing so the caller can render `-`
    (genuinely unknown) instead of `$0.00` (zero rate, which is a real value).
    """
    if price_per_mtok is None:
        return None
    return float(tokens) * price_per_mtok / 1_000_000


def _run_usage_json(usage: RunUsage) -> dict:
    # mode="json" emits canonical JSON (ISO-8601 datetimes etc.) instead of
    # Python-native types that fall back to repr() at serialization time.
    return usage.model_dump(mode="json")


def _build_run_usage_table(usage: RunUsage) -> Table:
    # Title stays short (run id + status) so long run names don't wrap and
    # mangle the table header. The descriptive name + base model live in
    # the caption below the table.
    title = f"Run Usage — {rich_escape(usage.run_id)}"
    if usage.status:
        title += f"  [{rich_escape(usage.status)}]"

    table = build_table(
        title,
        [
            ("Bucket", "cyan"),
            ("Tokens", "white"),
            ("Cost", "green"),
            ("Price / Mtok", "magenta"),
        ],
        show_lines=False,
    )

    # Each piece of metadata goes on its own caption line. Joining with a
    # separator lets Rich wrap mid-label (e.g. "model:" at the end of one
    # line, the value alone on the next) when the model name is long.
    caption_lines: list[str] = []
    if usage.run_name and usage.run_name != usage.run_id:
        caption_lines.append(rich_escape(usage.run_name))
    if usage.base_model:
        caption_lines.append(f"model: {rich_escape(usage.base_model)}")
    if caption_lines:
        table.caption = "[dim]" + "\n".join(caption_lines) + "[/dim]"

    # Per-row inference cost is derived from tokens × the *same* snapshotted
    # rate that produced RFTUsage.cost on the backend, so the two derived
    # halves sum to the combined inference cost in the response (modulo a
    # cent of rounding). The Total row keeps using the backend's exact sum
    # so what we show as "Total" is what was actually billed.
    #
    # Prefix-cache hits are a subset of input_tokens — the "Inference (input)"
    # row shows the non-cached remainder so the two rows sum to total input
    # tokens (and derived cost sums to combined inference cost). We fall back
    # to the full input rate for cached tokens when the backend hasn't sent a
    # discounted rate yet, so cost stays consistent with the billed total.
    cached_tokens = usage.inference.cached_input_tokens or 0
    cached_rate = usage.pricing.inference_cached_input_per_mtok
    effective_cached_rate = (
        cached_rate if cached_rate is not None else usage.pricing.inference_input_per_mtok
    )
    non_cached_input = max(0, usage.inference.input_tokens - cached_tokens)
    in_cost = _derived_cost(non_cached_input, usage.pricing.inference_input_per_mtok)
    cached_cost = _derived_cost(cached_tokens, effective_cached_rate)
    out_cost = _derived_cost(usage.inference.output_tokens, usage.pricing.inference_output_per_mtok)

    table.add_row(
        "Training",
        _format_tokens(usage.training.tokens),
        format_usd(usage.training.cost_usd),
        format_price_per_mtok(usage.pricing.training_per_mtok) or "-",
    )
    table.add_row(
        "Inference (input)",
        _format_tokens(non_cached_input),
        format_usd(in_cost) if in_cost is not None else "-",
        format_price_per_mtok(usage.pricing.inference_input_per_mtok) or "-",
    )
    # Only show the cached row when there's a signal it's relevant: either
    # the backend published a discounted rate, or the run actually consumed
    # cached tokens. Otherwise stay quiet for models without prefix caching.
    if cached_rate is not None or cached_tokens > 0:
        table.add_row(
            "Inference (cached input)",
            _format_tokens(cached_tokens),
            format_usd(cached_cost) if cached_cost is not None else "-",
            format_price_per_mtok(effective_cached_rate) or "-",
        )
    table.add_row(
        "Inference (output)",
        _format_tokens(usage.inference.output_tokens),
        format_usd(out_cost) if out_cost is not None else "-",
        format_price_per_mtok(usage.pricing.inference_output_per_mtok) or "-",
    )
    table.add_row(
        "[bold]Total[/bold]",
        f"[bold]{_format_tokens(usage.total_tokens)}[/bold]",
        f"[bold]{format_usd(usage.total_cost_usd)}[/bold]",
        "",
    )
    return table


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
        # In JSON mode, errors must go to stderr so stdout stays strictly
        # JSON for agents piping through `jq`. Watch-mode JSON already does
        # this — keep one-shot consistent.
        err_console = get_console(stderr=True) if output == "json" else console
        try:
            usage = billing.get_run_usage(run_id)
        except APIError as exc:
            err_console.print(f"[red]Error: {exc}[/red]")
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
