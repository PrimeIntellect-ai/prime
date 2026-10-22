"""Self-service inference trace capture: turn it on per model for the team.

Registered under `prime inference` rather than `prime traces` on purpose. The
thing being configured is an inference model, and this group talks to the
platform API while `prime traces` talks to the Prime Traces service, whose URL
never falls back to base_url (see core/config.py traces_url).
"""

from typing import Any, Dict, List, Optional

import typer
from rich.markup import escape
from rich.table import Table
from rich.text import Text

from ..api.trace_capture import (
    BASE_LORA_ID,
    STATE_BLOCKED_PRIVACY_OPT_OUT,
    STATE_BLOCKED_STORAGE_FULL,
    STATE_INELIGIBLE,
    STATE_OFF,
    STATE_ON,
    STATES,
    TraceCaptureClient,
    TraceCaptureForbiddenError,
    TraceCaptureTarget,
    synthetic_run_id,
)
from ..client import APIClient, APIError
from ..core import Config
from ..utils import (
    PlainTyper,
    confirm_or_skip,
    get_console,
    json_output_help,
    optional_team_params,
    output_data_as_json,
    validate_output_format,
)
from ..utils.time_utils import format_time_ago

app = PlainTyper(
    help="Turn inference trace capture on or off for the active team.",
    no_args_is_help=True,
)
console = get_console()
error_console = get_console(stderr=True)

LIST_JSON_HELP = json_output_help(
    ".targets[] = {model, lora_id, display_name?, state, enabled, eligible,"
    " sample_rate?, updated_at?, run_id}",
    f".targets[].state = {' | '.join(STATES)}",
    ".targets[].run_id = the synthetic run captured traces carry",
    ".total = number",
)

SET_JSON_HELP = json_output_help(
    ". = the updated target {model, lora_id, state, enabled, eligible, sample_rate?, run_id}",
    "state is the effective state after every gate, so an enable can answer a blocked state",
)

# A word per state, never a color alone, so the table survives --plain / NO_COLOR.
# `blocked: storage full` must not read as `off`: "I turned it on and got nothing"
# is the failure this surface exists to prevent.
STATE_LABELS = {
    STATE_ON: ("on", "green"),
    STATE_OFF: ("off", "dim"),
    STATE_BLOCKED_PRIVACY_OPT_OUT: ("blocked: privacy opt-out", "yellow"),
    STATE_BLOCKED_STORAGE_FULL: ("blocked: storage full", "red"),
    STATE_INELIGIBLE: ("ineligible", "yellow"),
}

STATE_NOTES = {
    STATE_ON: "on: calls from the team to this target are being captured.",
    STATE_OFF: "off: capture is not enabled for this target.",
    STATE_BLOCKED_PRIVACY_OPT_OUT: (
        "blocked: privacy opt-out: capture is enabled, but the team's trace privacy"
        " setting vetoes it. Nothing is stored until that setting is cleared."
    ),
    STATE_BLOCKED_STORAGE_FULL: (
        "blocked: storage full: capture is enabled, but the team's trace storage limit"
        " is reached, so nothing is being stored. Delete traces or raise the limit."
    ),
    STATE_INELIGIBLE: (
        "ineligible: this route is not approved for capture, so it cannot be turned on."
        " A team's own deployments and adapters are approved; shared capacity needs an operator."
    ),
}

# The states whose cell alone would read as "not capturing" without saying why.
EXPLAINED_STATES = (
    STATE_BLOCKED_PRIVACY_OPT_OUT,
    STATE_BLOCKED_STORAGE_FULL,
    STATE_INELIGIBLE,
)


def _state_cell(state: str) -> Text:
    """The state as a word. An unrecognised state renders verbatim rather than
    collapsing into one of ours, so a new server state is visible, not silent."""
    label, style = STATE_LABELS.get(state, (state, ""))
    return Text(label, style=style)


def _format_sample_rate(rate: Optional[float]) -> str:
    """A sample rate as a percentage. None means the platform's default applies."""
    if rate is None:
        return "-"
    return f"{rate * 100:.4g}%"


def _team_params() -> Dict[str, Any]:
    """The active team as a platform-API query param.

    Capture policy is keyed by team, so a personal context has nothing to show
    or set. `optional_team_params` returns None exactly then.
    """
    params = optional_team_params(Config())
    if params is None:
        error_console.print(
            "[red]Error:[/red] trace capture is configured per team, and no team is selected."
        )
        error_console.print("Select one with [bold]prime switch[/bold], or set PRIME_TEAM_ID.")
        raise typer.Exit(1)
    return params


def _target_json(target: TraceCaptureTarget) -> Dict[str, Any]:
    """A target for --output json, with the run ID a reader needs next to it."""
    payload = target.model_dump()
    payload["run_id"] = target.run_id
    return payload


def _describe(model: str, lora_id: str) -> str:
    """One escaped label for a target. Model IDs and adapter IDs are untrusted."""
    if lora_id and lora_id != BASE_LORA_ID:
        return f"{escape(model)} (adapter {escape(lora_id)})"
    return escape(model)


def _print_forbidden(requirement: str, error: TraceCaptureForbiddenError) -> None:
    """Explain a 403 by the permission it needs, not by the status code."""
    error_console.print(f"[red]Not allowed:[/red] {escape(str(error))}")
    error_console.print(requirement)
    error_console.print("Check the active team with [bold]prime whoami[/bold].")


def _print_api_error(error: APIError) -> None:
    error_console.print(f"[red]Error:[/red] {escape(str(error))}")


def _print_state_notes(states: List[str]) -> None:
    """Explain the states the table cannot speak for itself.

    `on` and `off` read plainly in the cell; the rest say "enabled but not
    capturing", which is the whole reason the listing reports a state at all.
    """
    for state in STATES:
        if state in EXPLAINED_STATES and state in states:
            console.print(Text(STATE_NOTES[state], style="dim"))
    unknown = sorted({state for state in states if state not in STATE_NOTES})
    if unknown:
        console.print(
            Text(
                f"Unrecognised state(s) {', '.join(unknown)}: this CLI is older than the"
                " platform. Upgrade with 'prime upgrade'.",
                style="dim",
            )
        )


def _print_read_hint(targets: List[TraceCaptureTarget]) -> None:
    """Connect the two halves: where the traces this switch produces are read.

    Captured traces are grouped under a synthetic `inference/<model>` run, so
    `--run-id` is the exact filter for one target. That run ID contains a slash,
    which `prime traces search` and `get --runs` cannot address, so `--query` on
    the listing is the search that works here.
    """
    capturing = [target for target in targets if target.state == STATE_ON]
    if not capturing:
        return
    console.print(
        Text(f"Read its traces: prime traces list --run-id {capturing[0].run_id}", style="dim")
    )
    console.print(Text("Search them: add --query <text> to that command", style="dim"))
    if len(capturing) > 1:
        console.print(
            Text("All captured models: prime traces list --runs --run-type inference", style="dim")
        )


def _print_enable_hint(targets: List[TraceCaptureTarget]) -> None:
    """Name a target that can still be switched on, using one of its own IDs."""
    off = next((target for target in targets if target.state == STATE_OFF), None)
    if off is None:
        return
    command = f"prime inference trace-capture enable {off.model}"
    if off.lora_id != BASE_LORA_ID:
        command += f" --lora-id {off.lora_id}"
    console.print(Text(f"Turn one on: {command}", style="dim"))


@app.command("list", epilog=LIST_JSON_HELP)
def list_capture_targets(
    output: str = typer.Option("table", "--output", "-o", help="Output format: table or json"),
) -> None:
    """Show the team's inference trace-capture targets and their effective state.

    The state is the team's opt-in after the eligibility, privacy and storage
    gates, so a target can read `blocked` while capture is switched on.

    Example:

        prime inference trace-capture list

        prime inference trace-capture list -o json
    """
    validate_output_format(output, error_console)
    params = _team_params()

    try:
        targets = TraceCaptureClient(APIClient()).list_targets(params)
    except TraceCaptureForbiddenError as e:
        _print_forbidden("Reading trace capture needs membership of the team.", e)
        raise typer.Exit(1)
    except APIError as e:
        _print_api_error(e)
        raise typer.Exit(1)

    if output == "json":
        output_data_as_json(
            {"targets": [_target_json(target) for target in targets], "total": len(targets)},
            console,
        )
        return

    if not targets:
        console.print("[yellow]No capture targets for this team.[/yellow]")
        console.print(
            "[dim]Targets appear once the team has a route it can capture: a dedicated"
            " endpoint, an RFT adapter, or an approved shared model.[/dim]"
        )
        return

    table = Table(title="Inference trace capture")
    table.add_column("Model", style="cyan", overflow="fold")
    table.add_column("Adapter", overflow="fold")
    table.add_column("State")
    table.add_column("Sampling", justify="right")
    table.add_column("Updated", style="dim")

    for target in targets:
        # Model, adapter and display names are server data: Text cells render
        # them verbatim, so markup in an ID cannot restyle the table.
        table.add_row(
            Text(target.display_name or target.model),
            Text(target.lora_id) if target.lora_id != BASE_LORA_ID else "-",
            _state_cell(target.state),
            _format_sample_rate(target.sample_rate),
            format_time_ago(target.updated_at),
        )

    console.print(table)
    _print_state_notes([target.state for target in targets])
    _print_enable_hint(targets)
    _print_read_hint(targets)


def _apply(
    *,
    model: str,
    lora_id: str,
    enabled: bool,
    sample_rate: Optional[float],
    output: str,
) -> None:
    """Write one target's opt-in and report the state that came back."""
    params = _team_params()
    try:
        target = TraceCaptureClient(APIClient()).set_target(
            team_id=str(params["teamId"]),
            model=model,
            lora_id=lora_id,
            enabled=enabled,
            sample_rate=sample_rate,
        )
    except TraceCaptureForbiddenError as e:
        _print_forbidden("Changing trace capture needs team ADMIN.", e)
        raise typer.Exit(1)
    except APIError as e:
        _print_api_error(e)
        raise typer.Exit(1)

    if output == "json":
        if target is not None:
            output_data_as_json(_target_json(target), console)
        else:
            # The route acknowledged without echoing a target; report what was
            # asked for and leave `state` out rather than guessing it.
            output_data_as_json(
                {
                    "model": model,
                    "lora_id": lora_id,
                    "enabled": enabled,
                    "run_id": synthetic_run_id(model, lora_id),
                },
                console,
            )
        return

    verb = "enabled" if enabled else "disabled"
    console.print(f"[green]Trace capture {verb} for[/green] {_describe(model, lora_id)}")

    if target is None:
        console.print(
            "[dim]Run 'prime inference trace-capture list' to see the effective state.[/dim]"
        )
        return

    console.print(Text("Effective state: ") + _state_cell(target.state))
    if target.state != (STATE_ON if enabled else STATE_OFF):
        # The opt-in is stored but something else decides the outcome; say which.
        note = STATE_NOTES.get(target.state)
        console.print(
            Text(note, style="yellow")
            if note
            else Text(f"Unrecognised state {target.state}; upgrade with 'prime upgrade'.")
        )
    elif enabled:
        _print_read_hint([target])


@app.command("enable", epilog=SET_JSON_HELP)
def enable_capture(
    model: str = typer.Argument(..., help="Inference model ID (see `prime inference models`)"),
    lora_id: str = typer.Option(
        BASE_LORA_ID, "--lora-id", help="RFT adapter ID; omit for the base model"
    ),
    sample_rate: Optional[float] = typer.Option(
        None,
        "--sample-rate",
        min=0.0,
        max=1.0,
        help="Fraction of calls to capture, 0 to 1; the platform picks a conservative default",
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the retention confirmation"),
    output: str = typer.Option("table", "--output", "-o", help="Output format: table or json"),
) -> None:
    """Turn trace capture on for one model. Requires team ADMIN.

    Captured calls retain the prompts and completions of the team's traffic to
    this target, readable by anyone who can read the team's traces.

    Example:

        prime inference trace-capture enable <model>

        prime inference trace-capture enable <model> --sample-rate 0.1

        prime inference trace-capture enable <model> --lora-id <adapter_id> --yes
    """
    validate_output_format(output, error_console)
    if output == "json" and not yes:
        # Prompting would put the question on stdout and break the JSON.
        error_console.print("[red]Error:[/red] --output json requires --yes.")
        raise typer.Exit(1)
    if not yes:
        console.print(
            f"Enabling capture for {_describe(model, lora_id)} retains the prompts and"
            " completions of every team call to it."
        )
        console.print(
            "[dim]Anyone who can read the team's traces can read them."
            " Turn it off again with 'prime inference trace-capture disable'.[/dim]"
        )
    if not confirm_or_skip("Enable trace capture?", yes):
        console.print("Cancelled.")
        raise typer.Exit(0)

    _apply(model=model, lora_id=lora_id, enabled=True, sample_rate=sample_rate, output=output)


@app.command("disable", epilog=SET_JSON_HELP)
def disable_capture(
    model: str = typer.Argument(..., help="Inference model ID (see `prime inference models`)"),
    lora_id: str = typer.Option(
        BASE_LORA_ID, "--lora-id", help="RFT adapter ID; omit for the base model"
    ),
    output: str = typer.Option("table", "--output", "-o", help="Output format: table or json"),
) -> None:
    """Turn trace capture off for one model. Requires team ADMIN.

    Already-captured traces are kept; delete them with
    `prime traces delete --run-id inference/<model>`.

    Example:

        prime inference trace-capture disable <model>

        prime inference trace-capture disable <model> --lora-id <adapter_id>
    """
    validate_output_format(output, error_console)
    _apply(model=model, lora_id=lora_id, enabled=False, sample_rate=None, output=output)
