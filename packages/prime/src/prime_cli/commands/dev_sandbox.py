"""prime dev-sandbox command: manage per-task development environments.

A typed client of the platform's dev-sandboxes API only — long actions
(create/exec/delete) are admitted as operations and polled to a terminal
state. See MVP-CONTRACT.md (dev-sandboxes spike, 2026-10-14).
"""

import shlex
from typing import Any, Dict, List, Optional

import typer
from prime_dev_sandboxes import DevSandboxClient, DevSandboxError
from prime_dev_sandboxes.core import APIError
from prime_dev_sandboxes.models import (
    DevEnvironment,
    Operation,
    OperationState,
)
from rich.table import Table
from rich.text import Text

from ..utils import (
    PlainTyper,
    build_table,
    confirm_or_skip,
    get_console,
    human_age,
    iso_timestamp,
    json_output_help,
    output_data_as_json,
    status_color,
    validate_output_format,
)
from ..utils.time_utils import now_utc

app = PlainTyper(help="Manage per-task dev environments", no_args_is_help=True)
console = get_console()

DEFAULT_TEMPLATE = "platform-dev"
DEFAULT_TTL_MINUTES = 120
MAX_TTL_MINUTES = 480

ENV_STATE_COLORS = {
    "ALLOCATING": "yellow",
    "PREPARING": "yellow",
    "STARTING": "yellow",
    "VERIFYING": "yellow",
    "READY": "green",
    "FAILED": "red",
    "DELETING": "yellow",
    "DELETED": "white",
}

LIST_ENVIRONMENTS_JSON_HELP = json_output_help(
    ".environments[] = {id, state, template, ref, resolved_sha?, ttl_minutes, "
    "urls{}, created_at, expiry?}",
    ".total = number",
    ".page = number",
    ".per_page = number",
    ".has_next = boolean",
)

ENV_DETAIL_JSON_HELP = json_output_help(
    ". = {id, state, template, template_version?, ref, resolved_sha?, "
    "ttl_minutes, expiry?, urls{}, tunnel_ids[], sandbox_id?, created_at?, "
    "updated_at?, last_error?}",
)

UP_JSON_HELP = json_output_help(
    ". = {id, state, resolved_sha?, ttl_minutes, expiry?, urls{}, operation_id, operation_state}",
    "Printed once the create operation reaches a terminal state.",
)

EXEC_JSON_HELP = json_output_help(
    ". = {environment_id, operation_id, state, exit_code?, result{}}",
    "Printed once the exec operation reaches a terminal state.",
)

DOWN_JSON_HELP = json_output_help(
    ". = {environment_id, operation_id, operation_state, state}",
    "Printed once the delete operation reaches a terminal state.",
)


def _format_expiry(env: DevEnvironment) -> str:
    """Human-readable time left on the lease, or '-' when unset/elapsed."""
    if env.expiry is None or env.state in ("DELETED",):
        return "-"
    remaining = (env.expiry - now_utc()).total_seconds()
    if remaining <= 0:
        return "expiring"
    minutes = int(remaining // 60)
    if minutes < 60:
        return f"{minutes}m left"
    return f"{minutes // 60}h{minutes % 60:02d}m left"


def _format_env_for_list(env: DevEnvironment) -> Dict[str, Any]:
    """Format an environment for list display (both table and JSON)."""
    return {
        "id": env.id,
        "state": env.state.value,
        "template": env.template,
        "ref": env.ref,
        "resolved_sha": env.resolved_sha,
        "ttl_minutes": env.ttl_minutes,
        "team_id": env.team_id,
        "urls": dict(env.urls),
        "created_at": iso_timestamp(env.created_at) if env.created_at else None,
        "age": human_age(env.created_at) if env.created_at else "-",
        "expiry": iso_timestamp(env.expiry) if env.expiry else None,
        "expires": _format_expiry(env),
    }


def _format_env_for_details(env: DevEnvironment) -> Dict[str, Any]:
    """Format an environment for detail display (JSON-oriented)."""
    data: Dict[str, Any] = {
        "id": env.id,
        "state": env.state.value,
        "template": env.template,
        "template_version": env.template_version,
        "ref": env.ref,
        "resolved_sha": env.resolved_sha,
        "ttl_minutes": env.ttl_minutes,
        "team_id": env.team_id,
        "urls": dict(env.urls),
        "tunnel_ids": list(env.tunnel_ids),
        "sandbox_id": env.sandbox_id,
        "created_at": iso_timestamp(env.created_at) if env.created_at else None,
        "expiry": iso_timestamp(env.expiry) if env.expiry else None,
        "updated_at": iso_timestamp(env.updated_at) if env.updated_at else None,
        "last_error": env.last_error.model_dump() if env.last_error else None,
    }
    return data


def _print_operation_error(operation: Operation) -> None:
    """Print a failed/interrupted operation's structured error."""
    if operation.error is not None:
        console.print(f"[red]Error:[/red] {operation.error.code} ({operation.error.stage})")
        console.print(operation.error.message)
        if operation.error.diagnostic_path:
            console.print(f"[dim]diagnostics: {operation.error.diagnostic_path}[/dim]")
    else:
        console.print(
            f"[red]Error:[/red] operation {operation.operation_id} ended as {operation.state.value}"
        )


def _print_env_summary(env: DevEnvironment) -> None:
    """Print the post-boot summary: env id, state, ref, expiry, URLs."""
    console.print(f"Dev environment: [cyan]{env.id}[/cyan] [bold]{env.state.value}[/bold]")

    ref_line = env.ref
    if env.resolved_sha:
        ref_line = f"{env.ref} @ {env.resolved_sha}"
    console.print(f"  Ref: {ref_line}  Template: {env.template}")

    expiry_line = iso_timestamp(env.expiry) if env.expiry else "none"
    console.print(f"  Expiry: {expiry_line} ({_format_expiry(env)})")

    if env.urls:
        console.print("  URLs:")
        for name, url in env.urls.items():
            console.print(f"    {name}: {url}")
    else:
        console.print("  URLs: (none yet)")


def _client() -> DevSandboxClient:
    return DevSandboxClient()


def _poll_operation(
    client: DevSandboxClient,
    operation_id: str,
    poll_interval: float,
    quiet: bool,
    stream_diagnostics: bool = False,
) -> Operation:
    """Poll an operation to terminal, printing stage transitions.

    ``quiet`` suppresses progress output entirely (JSON mode); with
    ``stream_diagnostics`` new diagnostic lines are printed as they appear
    (exec tail streaming).
    """
    last_stage: Optional[str] = None
    seen_diagnostics = 0

    def on_update(operation: Operation) -> None:
        nonlocal last_stage, seen_diagnostics
        if quiet:
            return
        if operation.stage is not None and operation.stage != last_stage:
            last_stage = operation.stage
            console.print(f"[dim]{operation.kind.value}: {operation.stage}[/dim]")
        if stream_diagnostics and len(operation.diagnostics) > seen_diagnostics:
            for line in operation.diagnostics[seen_diagnostics:]:
                console.print(line, markup=False, highlight=False)
            seen_diagnostics = len(operation.diagnostics)

    return client.wait_for_operation(
        operation_id,
        poll_interval_seconds=poll_interval,
        on_update=on_update,
    )


def _check_operation_succeeded(operation: Operation) -> bool:
    if operation.state is OperationState.SUCCEEDED:
        return True
    _print_operation_error(operation)
    return False


def _validate_ttl(ttl_minutes: int) -> None:
    if not 1 <= ttl_minutes <= MAX_TTL_MINUTES:
        console.print(
            f"[red]Error:[/red] --ttl-minutes must be between 1 and {MAX_TTL_MINUTES} "
            f"(got {ttl_minutes})"
        )
        raise typer.Exit(1)


@app.command("up", epilog=UP_JSON_HELP)
def up(
    ref: str = typer.Option(..., "--ref", help="PR URL, branch name, or commit SHA to boot"),
    template: str = typer.Option(
        DEFAULT_TEMPLATE, "--template", help="Environment template (default: platform-dev)"
    ),
    team_id: Optional[str] = typer.Option(
        None, "--team-id", help="Team for the environment (defaults to token team context)"
    ),
    ttl_minutes: int = typer.Option(
        DEFAULT_TTL_MINUTES, "--ttl-minutes", help=f"Lease TTL in minutes (max {MAX_TTL_MINUTES})"
    ),
    poll_interval: float = typer.Option(
        5.0, "--poll-interval", help="Seconds between operation status polls"
    ),
    output: str = typer.Option("table", "--output", "-o", help="Output format: table or json"),
) -> None:
    """Create a dev environment from a template at a ref and wait for READY

    Admits the environment, polls the returned operation to a terminal
    state, then prints the environment id, URLs, and lease expiry.
    """
    validate_output_format(output, console)
    _validate_ttl(ttl_minutes)

    try:
        client = _client()
        created = client.create(
            template=template,
            ref=ref,
            ttl_minutes=ttl_minutes,
            team_id=team_id,
        )

        if output != "json":
            console.print(
                f"Admitted dev environment [cyan]{created.environment_id}[/cyan] "
                f"(operation {created.operation_id})"
            )

        operation = _poll_operation(
            client, created.operation_id, poll_interval, quiet=(output == "json")
        )

        if not _check_operation_succeeded(operation):
            raise typer.Exit(1)

        env = client.get(created.environment_id)

        if output == "json":
            data = _format_env_for_details(env)
            data["operation_id"] = operation.operation_id
            data["operation_state"] = operation.state.value
            output_data_as_json(data, console)
        else:
            _print_env_summary(env)
    except typer.Exit:
        raise
    except DevSandboxError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    except APIError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command("list", epilog=LIST_ENVIRONMENTS_JSON_HELP)
def list_cmd(
    page: int = typer.Option(1, "--page", "-p", help="Page number"),
    num: int = typer.Option(50, "--num", "-n", help="Items per page"),
    output: str = typer.Option("table", "--output", "-o", help="Output format: table or json"),
) -> None:
    """List your dev environments"""
    validate_output_format(output, console)

    if num < 1 or page < 1:
        console.print("[red]Error:[/red] --num and --page must be at least 1")
        raise typer.Exit(1)

    try:
        client = _client()
        result = client.list(page=page, per_page=num)

        if output == "json":
            environments = []
            for env in result.environments:
                env_data = _format_env_for_list(env)
                # JSON gets timestamps, not the humanized table columns.
                environments.append(
                    {
                        "id": env_data["id"],
                        "state": env_data["state"],
                        "template": env_data["template"],
                        "ref": env_data["ref"],
                        "resolved_sha": env_data["resolved_sha"],
                        "ttl_minutes": env_data["ttl_minutes"],
                        "team_id": env_data["team_id"],
                        "urls": env_data["urls"],
                        "created_at": env_data["created_at"],
                        "expiry": env_data["expiry"],
                    }
                )
            output_data_as_json(
                {
                    "environments": environments,
                    "total": result.total,
                    "page": result.page,
                    "per_page": result.per_page,
                    "has_next": result.has_next,
                },
                console,
            )
            return

        table = build_table(
            f"Dev Environments (Total: {result.total})",
            [
                ("ID", "cyan"),
                ("Template", "blue"),
                ("Ref", "green"),
                ("State", "yellow"),
                ("Age", "blue"),
                ("Expires", "yellow"),
            ],
        )

        for env in result.environments:
            env_data = _format_env_for_list(env)
            color = status_color(env_data["state"], ENV_STATE_COLORS)
            ref = env_data["ref"]
            if env_data["resolved_sha"]:
                ref = f"{ref} @ {env_data['resolved_sha']}"
            table.add_row(
                env_data["id"],
                env_data["template"],
                ref,
                Text(env_data["state"], style=color),
                env_data["age"],
                env_data["expires"],
            )

        console.print(table)

        if result.has_next:
            console.print(
                f"\n[yellow]Showing page {page} of results. "
                f"Use --page {page + 1} to see more.[/yellow]"
            )
    except typer.Exit:
        raise
    except DevSandboxError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    except APIError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command("status", epilog=ENV_DETAIL_JSON_HELP)
def status(
    environment_id: str = typer.Argument(..., help="Dev environment ID"),
    output: str = typer.Option("table", "--output", "-o", help="Output format: table or json"),
) -> None:
    """Show a dev environment's status"""
    validate_output_format(output, console)

    try:
        env = _client().get(environment_id)

        env_data = _format_env_for_details(env)
        if output == "json":
            output_data_as_json(env_data, console)
            return

        color = status_color(env.state.value, ENV_STATE_COLORS)
        ref = env.ref if not env.resolved_sha else f"{env.ref} @ {env.resolved_sha}"

        table = Table(title=f"Dev Environment: {environment_id}")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="white")
        table.add_row("ID", env.id)
        table.add_row("State", Text(env.state.value, style=color))
        table.add_row("Template", env.template)
        table.add_row("Ref", ref)
        table.add_row("TTL (minutes)", str(env.ttl_minutes))
        table.add_row("Expiry", env_data["expiry"] or "-")
        if env.sandbox_id:
            table.add_row("Sandbox", env.sandbox_id)
        if env.tunnel_ids:
            table.add_row("Tunnels", ", ".join(env.tunnel_ids))
        if env.team_id:
            table.add_row("Team", env.team_id)
        table.add_row("Created", env_data["created_at"] or "-")

        if env.urls:
            table.add_row("URLs", "\n".join(f"{k}: {v}" for k, v in env.urls.items()))

        if env.last_error is not None:
            table.add_row("Last error", env.last_error.model_dump_json())

        console.print(table)
    except typer.Exit:
        raise
    except DevSandboxError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    except APIError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command("exec", epilog=EXEC_JSON_HELP)
def exec_cmd(
    environment_id: str = typer.Argument(..., help="Dev environment ID"),
    command: List[str] = typer.Argument(
        ...,
        help="Command to run (use -- before any flags the command itself takes)",
    ),
    timeout_seconds: int = typer.Option(
        300, "--timeout-seconds", help="Server-side command timeout"
    ),
    poll_interval: float = typer.Option(
        2.0, "--poll-interval", help="Seconds between operation status polls"
    ),
    output: str = typer.Option("table", "--output", "-o", help="Output format: table or json"),
) -> None:
    """Run a command in a READY dev environment and stream its result tail

    Admits the exec as an operation, streams new operation diagnostics as
    they appear, then prints the final result (exit code and output tail).
    """
    validate_output_format(output, console)

    command_str = shlex.join(command)
    if not command_str:
        console.print("[red]Error:[/red] No command given")
        raise typer.Exit(1)

    try:
        client = _client()
        admitted = client.exec(environment_id, command_str, timeout_seconds=timeout_seconds)

        if output != "json":
            console.print(f"Admitted exec [cyan]{admitted.operation_id}[/cyan] on {environment_id}")

        operation = _poll_operation(
            client,
            admitted.operation_id,
            poll_interval,
            quiet=(output == "json"),
            stream_diagnostics=True,
        )

        if output == "json":
            output_data_as_json(
                {
                    "environment_id": environment_id,
                    "operation_id": operation.operation_id,
                    "state": operation.state.value,
                    "exit_code": operation.result.get("exit_code"),
                    "result": operation.result,
                },
                console,
            )
            if operation.state is not OperationState.SUCCEEDED:
                raise typer.Exit(1)
            raise typer.Exit(0)

        if not _check_operation_succeeded(operation):
            raise typer.Exit(1)

        exit_code = operation.result.get("exit_code")
        stdout = operation.result.get("stdout") or ""
        stderr = operation.result.get("stderr") or ""
        if stdout:
            console.print(stdout, markup=False, highlight=False)
        if stderr:
            console.print(stderr, markup=False, highlight=False, style="red")
        console.print(f"[dim]exit code: {exit_code}[/dim]")

        if exit_code not in (None, 0):
            raise typer.Exit(1)
    except typer.Exit:
        raise
    except DevSandboxError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    except APIError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command("down", epilog=DOWN_JSON_HELP)
def down(
    environment_id: str = typer.Argument(..., help="Dev environment ID"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
    poll_interval: float = typer.Option(
        5.0, "--poll-interval", help="Seconds between operation status polls"
    ),
    output: str = typer.Option("table", "--output", "-o", help="Output format: table or json"),
) -> None:
    """Delete a dev environment and wait for DELETED"""
    validate_output_format(output, console)

    if not confirm_or_skip(f"Delete dev environment {environment_id}?", yes):
        raise typer.Exit(0)

    try:
        client = _client()
        admitted = client.delete(environment_id)

        if output != "json":
            console.print(
                f"Deleting [cyan]{environment_id}[/cyan] (operation {admitted.operation_id})"
            )

        operation = _poll_operation(
            client, admitted.operation_id, poll_interval, quiet=(output == "json")
        )

        if output == "json":
            output_data_as_json(
                {
                    "environment_id": environment_id,
                    "operation_id": operation.operation_id,
                    "operation_state": operation.state.value,
                    "state": operation.result.get("state"),
                },
                console,
            )
            if operation.state is not OperationState.SUCCEEDED:
                raise typer.Exit(1)
            raise typer.Exit(0)

        if not _check_operation_succeeded(operation):
            raise typer.Exit(1)

        state = operation.result.get("state", "DELETED")
        console.print(f"Dev environment [cyan]{environment_id}[/cyan] {state}")
    except typer.Exit:
        raise
    except DevSandboxError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    except APIError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
