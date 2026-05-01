"""`prime wallet` — balance + most recent billing rows.

Single one-shot command (no subcommands, no watch mode). Drives an agent's
audit of the billing flow: "balance dropped by exactly $X after run R logged
the charge in the Billing table".
"""

import typer
from rich.markup import escape as rich_escape

from prime_cli.api.wallet import BillingEntry, Wallet, WalletClient
from prime_cli.core import APIClient, APIError, Config
from prime_cli.utils import (
    build_table,
    get_console,
    json_output_help,
    output_data_as_json,
    validate_output_format,
)
from prime_cli.utils.formatters import format_usd

console = get_console()


WALLET_JSON_HELP = json_output_help(
    ". = {wallet_id, team_id?, balance_usd, currency, total_billings, "
    "recent_billings: [{id, created_at, updated_at, last_billed_at?, "
    "amount_usd, currency, resource_type, resource_id?}]}"
)


def _format_when(entry: BillingEntry) -> str:
    when = entry.last_billed_at or entry.updated_at
    return when.strftime("%Y-%m-%d %H:%M")


def _format_resource(entry: BillingEntry) -> str:
    if entry.resource_id:
        return f"{entry.resource_type} ({rich_escape(entry.resource_id)})"
    return entry.resource_type


def _print_header(wallet: Wallet, team_label: str) -> None:
    """Render the balance summary as plain left-aligned lines.

    Avoids stacking two centered tables of different widths, which makes
    the title + first column visibly jump as the eye moves down. Plain
    text keeps everything anchored to the left margin.
    """
    console.print(f"[bold]Wallet[/bold]  [dim]({team_label})[/dim]")
    console.print(
        f"  Balance:  [bold green]{format_usd(wallet.balance_usd)}[/bold green] "
        f"[dim]{wallet.currency}[/dim]"
    )
    console.print(f"  Billings: [bold]{wallet.total_billings}[/bold] total")
    console.print(f"  [dim]wallet id: {rich_escape(wallet.wallet_id)}[/dim]")
    console.print()


def _build_billings_table(wallet: Wallet) -> object:
    title = f"Recent billings — last {len(wallet.recent_billings)} of {wallet.total_billings}"
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
            format_usd(entry.amount_usd),
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
    output: str = typer.Option("table", "--output", "-o", help="Output format: table or json"),
) -> None:
    """Show wallet balance and most recent billing rows.

    Follows the team configured via ``prime switch`` / ``prime config``
    (personal wallet when no team is selected). Lets an agent audit billing
    end-to-end: pair this with ``prime train usage <run_id>`` to confirm
    the wallet debit matches the run's reported cost.

    Example:

        prime wallet

        prime wallet --limit 50 --output json
    """
    validate_output_format(output, console)

    config = Config()
    client = WalletClient(APIClient())

    try:
        wallet = client.get(limit=limit, team_id=config.team_id)
    except APIError as exc:
        console.print(f"[red]Error: {exc}[/red]")
        raise typer.Exit(1) from exc

    if output == "json":
        # mode="json" emits ISO-8601 datetime strings rather than the
        # space-separated repr that default=str produces.
        output_data_as_json(wallet.model_dump(mode="json"), console)
        return

    if config.team_id:
        team_label = f"team {rich_escape(config.team_name or config.team_id)}"
    else:
        team_label = "personal"
    _print_header(wallet, team_label)
    if wallet.recent_billings:
        console.print(_build_billings_table(wallet))
    else:
        console.print("[dim]No billing rows yet.[/dim]")
