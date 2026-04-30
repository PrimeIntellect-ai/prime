"""`prime wallet` — balance + most recent billing rows.

Single one-shot command (no subcommands, no watch mode). Drives an agent's
audit of the billing flow: "balance dropped by exactly $X after run R logged
the charge in the Billing table".
"""

from typing import Optional

import typer
from rich.markup import escape as rich_escape

from prime_cli.api.wallet import BillingEntry, Wallet, WalletClient
from prime_cli.core import APIClient, APIError
from prime_cli.utils import (
    build_table,
    get_console,
    json_output_help,
    output_data_as_json,
    validate_output_format,
)

console = get_console()


WALLET_JSON_HELP = json_output_help(
    ". = {wallet_id, team_id?, balance_usd, currency, total_billings, "
    "recent_billings: [{id, created_at, updated_at, last_billed_at?, "
    "amount_usd, currency, resource_type, resource_id?}]}"
)


def _format_usd(value: float) -> str:
    if value == 0:
        return "$0.00"
    if abs(value) < 0.01:
        return f"${value:.4f}"
    return f"${value:.2f}"


def _format_when(entry: BillingEntry) -> str:
    when = entry.last_billed_at or entry.updated_at
    return when.strftime("%Y-%m-%d %H:%M")


def _format_resource(entry: BillingEntry) -> str:
    if entry.resource_id:
        return f"{entry.resource_type} ({rich_escape(entry.resource_id)})"
    return entry.resource_type


def _build_balance_table(wallet: Wallet) -> object:
    title = f"Wallet — {rich_escape(wallet.wallet_id)}" + (
        f" (team {rich_escape(wallet.team_id)})" if wallet.team_id else ""
    )
    table = build_table(
        title,
        [("Field", "cyan"), ("Value", "white")],
        show_lines=False,
    )
    table.add_row("Balance", f"[bold green]{_format_usd(wallet.balance_usd)}[/bold green]")
    table.add_row("Currency", wallet.currency)
    table.add_row("Total billing rows", str(wallet.total_billings))
    return table


def _build_billings_table(wallet: Wallet) -> object:
    title = f"Recent billings ({len(wallet.recent_billings)} of {wallet.total_billings})"
    table = build_table(
        title,
        [
            ("When", "dim"),
            ("Resource", "cyan"),
            ("Amount", "green"),
        ],
        show_lines=False,
    )
    for entry in wallet.recent_billings:
        table.add_row(
            _format_when(entry),
            _format_resource(entry),
            _format_usd(entry.amount_usd),
        )
    return table


def wallet_command(
    limit: int = typer.Option(
        20,
        "--limit",
        "-n",
        min=1,
        max=100,
        help="Number of recent billing rows to fetch (max 100)",
    ),
    team: Optional[str] = typer.Option(
        None, "--team", "-t", help="Team ID (defaults to personal wallet)"
    ),
    output: str = typer.Option("table", "--output", "-o", help="Output format: table or json"),
) -> None:
    """Show wallet balance and most recent billing rows.

    Lets an agent audit billing end-to-end: pair this with
    ``prime train usage <run_id>`` to confirm the wallet debit matches the
    run's reported cost.

    Example:

        prime wallet

        prime wallet --limit 50 --output json

        prime wallet --team team_abc
    """
    validate_output_format(output, console)

    client = WalletClient(APIClient())

    try:
        wallet = client.get(limit=limit, team_id=team)
    except APIError as exc:
        console.print(f"[red]Error: {exc}[/red]")
        raise typer.Exit(1) from exc

    if output == "json":
        output_data_as_json(wallet.model_dump(), console)
        return

    console.print(_build_balance_table(wallet))
    if wallet.recent_billings:
        console.print(_build_billings_table(wallet))
    else:
        console.print("[dim]No billing rows yet.[/dim]")
