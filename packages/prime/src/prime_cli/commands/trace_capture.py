"""`prime inference trace-capture`: per-route custom trace capture for the team."""

from typing import List, NoReturn

import typer
from rich.markup import escape
from rich.table import Table
from rich.text import Text

from ..api.trace_capture import (
    BASE_LORA_ID,
    TraceCaptureClient,
    TraceCaptureForbiddenError,
    TraceCaptureRoute,
    TraceCaptureUnavailableError,
)
from ..client import APIClient, APIError
from ..core import Config
from ..utils import (
    PlainTyper,
    confirm_or_skip,
    get_console,
    json_output_help,
    output_data_as_json,
    validate_output_format,
)

app = PlainTyper(
    help="Turn custom inference trace capture on or off per route for the active team.",
    no_args_is_help=True,
)
console = get_console()
error_console = get_console(stderr=True)

ROUTE_JSON = "{model, lora_id, name, run_id, enabled, active, inactive_reason}"
INACTIVE_REASONS = {
    "privacy_opt_out": "team privacy opt-out",
    "storage_limit": "trace storage limit reached",
    "route_unavailable": "route unavailable",
}
LORA_HELP = "RFT adapter ID; omit for the base model"


def _team_id() -> str:
    team_id = Config().team_id
    if not team_id:
        error_console.print(
            "[red]Error:[/red] trace capture is configured per team and no team is selected."
            " Use [bold]prime switch[/bold] or set PRIME_TEAM_ID."
        )
        raise typer.Exit(1)
    return team_id


def _fail(error: APIError) -> NoReturn:
    if isinstance(error, TraceCaptureForbiddenError):
        error_console.print(
            "[red]Not allowed:[/red] listing needs the traces:read scope;"
            " enable and disable need traces:write and team ADMIN."
        )
    elif isinstance(error, TraceCaptureUnavailableError):
        error_console.print(
            "[red]Not available:[/red] trace capture is not enabled for this team,"
            " you are not a member, or the route is not eligible."
        )
    else:
        error_console.print(f"[red]Error:[/red] {escape(str(error))}")
    raise typer.Exit(1)


def _status(route: TraceCaptureRoute) -> Text:
    if route.active:
        return Text("active", style="green")
    if route.enabled:
        reason = INACTIVE_REASONS.get(route.inactive_reason or "", route.inactive_reason or "")
        return Text(f"inactive: {reason}" if reason else "inactive", style="yellow")
    return Text("off", style="dim")


def _summary(routes: List[TraceCaptureRoute]) -> str:
    active = [route for route in routes if route.active]
    if not active:
        return "Custom trace capture is not active on any route."
    return f"Custom trace capture is active on {len(active)} of {len(routes)} route(s)."


@app.command("list", epilog=json_output_help(f".routes[] = {ROUTE_JSON}"))
def list_routes(
    output: str = typer.Option("table", "--output", "-o", help="Output format: table or json"),
) -> None:
    """List the team's capturable routes with whether capture is enabled and active.

    Example:

        prime inference trace-capture list
    """
    validate_output_format(output, error_console)
    try:
        routes = TraceCaptureClient(APIClient()).list_routes(_team_id())
    except APIError as e:
        _fail(e)
    if output == "json":
        output_data_as_json({"routes": [route.model_dump() for route in routes]}, console)
        return
    if not routes:
        console.print("[yellow]No routes this team can capture.[/yellow]")
        return

    table = Table(title="Inference trace capture")
    table.add_column("Model", style="cyan", overflow="fold")
    table.add_column("Adapter", overflow="fold")
    table.add_column("Enabled")
    table.add_column("Status")
    table.add_column("Run ID", overflow="fold")
    for route in routes:
        table.add_row(
            Text(route.name or route.model),
            Text(route.lora_id) if route.lora_id != BASE_LORA_ID else "-",
            "yes" if route.enabled else "no",
            _status(route),
            Text(route.run_id),
        )
    console.print(table)
    console.print(_summary(routes))
    if any(route.active for route in routes):
        console.print("[dim]View captures: prime traces list --run-id <run_id>[/dim]")


def _set(model: str, lora_id: str, enabled: bool, output: str) -> None:
    try:
        route = TraceCaptureClient(APIClient()).set_route(_team_id(), model, enabled, lora_id)
    except APIError as e:
        _fail(e)
    if output == "json":
        output_data_as_json(route.model_dump(), console)
        return
    label = escape(model) + (f" (adapter {escape(lora_id)})" if lora_id != BASE_LORA_ID else "")
    console.print(f"Trace capture {'enabled' if enabled else 'disabled'} for {label}")
    console.print(Text("Status: ") + _status(route))
    if route.active:
        console.print(Text(f"View captures: prime traces list --run-id {route.run_id}", "dim"))


@app.command("enable", epilog=json_output_help(f". = {ROUTE_JSON}"))
def enable(
    model: str = typer.Argument(..., help="Inference model ID"),
    lora_id: str = typer.Option(BASE_LORA_ID, "--lora-id", help=LORA_HELP),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation"),
    output: str = typer.Option("table", "--output", "-o", help="Output format: table or json"),
) -> None:
    """Capture the team's calls to one route as traces. Requires team ADMIN.

    Example:

        prime inference trace-capture enable <model> --yes
    """
    validate_output_format(output, error_console)
    prompt = "Store the prompts and completions of the team's calls to this route as traces?"
    if not confirm_or_skip(prompt, yes, err=True):
        raise typer.Exit(1)
    _set(model, lora_id, True, output)


@app.command("disable", epilog=json_output_help(f". = {ROUTE_JSON}"))
def disable(
    model: str = typer.Argument(..., help="Inference model ID"),
    lora_id: str = typer.Option(BASE_LORA_ID, "--lora-id", help=LORA_HELP),
    output: str = typer.Option("table", "--output", "-o", help="Output format: table or json"),
) -> None:
    """Stop capturing one route. Requires team ADMIN. Existing traces are kept.

    Example:

        prime inference trace-capture disable <model>
    """
    validate_output_format(output, error_console)
    _set(model, lora_id, False, output)
