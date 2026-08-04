from pathlib import Path
from typing import Dict, List, Optional

import typer
from prime_traces import (
    APIError,
    Batch,
    LineFormat,
    PaymentRequiredError,
    PrimeTracesError,
    TracesClient,
    UnauthorizedError,
    UploadReceipt,
)
from rich.markup import escape
from rich.table import Table

from ..core import Config
from ..utils import (
    PlainTyper,
    get_console,
    json_output_help,
    output_data_as_json,
    validate_output_format,
)

app = PlainTyper(help="Upload and query traces (Prime Traces)", no_args_is_help=True)
console = get_console()


def _traces_client() -> TracesClient:
    """Build a client from the CLI config so `--context` is honored.

    The SDK's own Config reads only ~/.prime/config.json and env vars; the CLI
    Config additionally resolves PRIME_CONTEXT environments, so credentials and
    URLs must flow from here — the same injection pattern as the sandbox and
    evals commands.

    Every field is passed explicitly, never None: the SDK client treats None as
    "resolve from my static config", and a context whose api_key or team is
    unset must fail or go teamless rather than silently borrow the default
    context's credentials.
    """
    config = Config()
    return TracesClient(
        api_key=config.api_key,
        base_url=config.traces_url,
        team_id=config.team_id or "",
    )


UPLOAD_JSON_HELP = json_output_help(
    ".receipts[] = {upload_id, status}",
    ".num_batches = number",
)

LIST_TRACES_JSON_HELP = json_output_help(
    ".items[] = trace summary {trace_id, run_id, task_id, score, execution, ...}",
    ".next_cursor? = string",
)

GET_TRACE_JSON_HELP = json_output_help(
    ". = trace summary object; with --raw, the exact stored trace document",
)


def _parse_context(values: List[str]) -> Optional[Dict[str, str]]:
    if not values:
        return None
    context: Dict[str, str] = {}
    for item in values:
        key, sep, value = item.partition("=")
        if not sep or not key:
            console.print(f"[red]Invalid --context '{item}'; expected key=value[/red]")
            raise typer.Exit(1)
        context[key] = value
    return context


@app.command("upload", epilog=UPLOAD_JSON_HELP)
def upload_traces(
    file: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        readable=True,
        help="Completed JSONL file: one trace per line (or one episode with --episodes)",
    ),
    episodes: bool = typer.Option(
        False, "--episodes", help="Each line is one complete episode document"
    ),
    context: List[str] = typer.Option(
        [],
        "--context",
        "-c",
        help="Batch context as key=value, repeatable (e.g. -c source=hosted_eval)",
    ),
    no_compress: bool = typer.Option(
        False, "--no-compress", help="Skip gzip transport compression"
    ),
    output: str = typer.Option("table", "--output", "-o", help="Output format: table or json"),
) -> None:
    """Upload a JSONL file of traces. Safe to rerun after interruption:
    identical bytes replay their committed receipts without re-storing."""
    validate_output_format(output, console)
    line_format = LineFormat.EPISODE if episodes else LineFormat.TRACE

    def on_batch(batch: Batch, receipt: UploadReceipt) -> None:
        if output != "json":
            console.print(
                f"  batch {receipt.upload_id[:12]}… "
                f"({batch.num_lines} lines, {batch.size / (1024 * 1024):.1f} MiB) "
                f"[green]{receipt.status}[/green]"
            )

    try:
        client = _traces_client()
        receipts = client.upload_file(
            file,
            line_format=line_format,
            context=_parse_context(context),
            compress=not no_compress,
            on_batch=on_batch,
        )
    except typer.Exit:
        raise
    except UnauthorizedError as e:
        console.print(f"[red]Unauthorized:[/red] {str(e)}")
        raise typer.Exit(1)
    except PaymentRequiredError as e:
        console.print(f"[red]Payment Required:[/red] {str(e)}")
        raise typer.Exit(1)
    except PrimeTracesError as e:
        console.print(f"[red]Upload failed:[/red] {str(e)}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {escape(str(e))}")
        console.print_exception(show_locals=True)
        raise typer.Exit(1)

    if output == "json":
        output_data_as_json(
            {
                "receipts": [r.model_dump() for r in receipts],
                "num_batches": len(receipts),
            },
            console,
        )
    else:
        console.print(f"[green]Uploaded {len(receipts)} batch(es) from {file}[/green]")


@app.command("list", epilog=LIST_TRACES_JSON_HELP)
def list_traces(
    run_id: Optional[str] = typer.Option(None, "--run-id", help="Filter by run ID"),
    task_id: Optional[str] = typer.Option(None, "--task-id", help="Filter by task ID"),
    model_id: Optional[str] = typer.Option(None, "--model-id", help="Filter by model ID"),
    outcome: Optional[str] = typer.Option(None, "--outcome", help="Filter by outcome"),
    has_error: Optional[bool] = typer.Option(
        None, "--has-error/--no-has-error", help="Filter by error status"
    ),
    reward_min: Optional[float] = typer.Option(None, "--reward-min", help="Minimum reward"),
    reward_max: Optional[float] = typer.Option(None, "--reward-max", help="Maximum reward"),
    created_after: Optional[str] = typer.Option(
        None, "--created-after", help="ISO timestamp; also the cheapest filter"
    ),
    created_before: Optional[str] = typer.Option(None, "--created-before", help="ISO timestamp"),
    sort: Optional[str] = typer.Option(
        None, "--sort", help="Sort key: created_at (default, newest first), reward, duration_ms"
    ),
    limit: int = typer.Option(20, "--limit", help="Max results per page (up to 100)"),
    cursor: Optional[str] = typer.Option(None, "--cursor", help="Cursor from a previous page"),
    output: str = typer.Option("table", "--output", "-o", help="Output format: table or json"),
) -> None:
    """List trace summaries, newest first."""
    validate_output_format(output, console)
    try:
        client = _traces_client()
        page = client.list(
            run_id=run_id,
            task_id=task_id,
            model_id=model_id,
            outcome=outcome,
            has_error=has_error,
            reward_min=reward_min,
            reward_max=reward_max,
            created_after=created_after,
            created_before=created_before,
            sort=sort,
            limit=limit,
            cursor=cursor,
        )
    except typer.Exit:
        raise
    except UnauthorizedError as e:
        console.print(f"[red]Unauthorized:[/red] {str(e)}")
        raise typer.Exit(1)
    except PaymentRequiredError as e:
        console.print(f"[red]Payment Required:[/red] {str(e)}")
        raise typer.Exit(1)
    except APIError as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {escape(str(e))}")
        console.print_exception(show_locals=True)
        raise typer.Exit(1)

    if output == "json":
        output_data_as_json(page.model_dump(mode="json"), console)
        return

    table = Table(title="Traces")
    table.add_column("Trace ID", style="cyan", no_wrap=True)
    table.add_column("Run", style="green")
    table.add_column("Task")
    table.add_column("Reward", justify="right")
    table.add_column("Outcome")
    table.add_column("Created")

    for summary in page.items:
        reward = summary.score.reward if summary.score else None
        table.add_row(
            summary.trace_id[:16],
            summary.run_id or "-",
            summary.task_id or "-",
            "-" if reward is None else f"{reward:.2f}",
            (summary.score.outcome if summary.score else None) or "-",
            summary.created_at.isoformat() if summary.created_at else "-",
        )
    console.print(table)
    if page.next_cursor:
        console.print(f"[dim]More results: --cursor {page.next_cursor}[/dim]")


@app.command("get", epilog=GET_TRACE_JSON_HELP)
def get_trace(
    trace_id: str = typer.Argument(..., help="Trace ID"),
    raw: bool = typer.Option(False, "--raw", help="Fetch the exact stored trace document"),
    dest: Optional[Path] = typer.Option(
        None, "--dest", help="With --raw: stream the document to this file"
    ),
    output: str = typer.Option("table", "--output", "-o", help="Output format: table or json"),
) -> None:
    """Get one trace summary, or the raw trace document with --raw."""
    validate_output_format(output, console)
    try:
        client = _traces_client()
        if raw:
            if dest is not None:
                written = client.download_raw(trace_id, dest)
                console.print(f"[green]Wrote {written} bytes to {dest}[/green]")
                return
            # Raw documents can be tens of MiB; print verbatim, no re-encoding.
            print(client.get_raw(trace_id).decode("utf-8", errors="replace"))
            return
        summary = client.get(trace_id)
    except typer.Exit:
        raise
    except UnauthorizedError as e:
        console.print(f"[red]Unauthorized:[/red] {str(e)}")
        raise typer.Exit(1)
    except PaymentRequiredError as e:
        console.print(f"[red]Payment Required:[/red] {str(e)}")
        raise typer.Exit(1)
    except APIError as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {escape(str(e))}")
        console.print_exception(show_locals=True)
        raise typer.Exit(1)

    if output == "json":
        output_data_as_json(summary.model_dump(mode="json"), console)
        return

    table = Table(title=f"Trace {trace_id}")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="green")
    for field, value in summary.model_dump(mode="json").items():
        table.add_row(field, "-" if value is None else str(value))
    console.print(table)


@app.command("export")
def export_traces(
    dest: Path = typer.Argument(..., dir_okay=False, help="Destination file (JSONL)"),
    run_id: Optional[str] = typer.Option(None, "--run-id", help="Filter by run ID"),
    task_id: Optional[str] = typer.Option(None, "--task-id", help="Filter by task ID"),
    model_id: Optional[str] = typer.Option(None, "--model-id", help="Filter by model ID"),
    outcome: Optional[str] = typer.Option(None, "--outcome", help="Filter by outcome"),
    has_error: Optional[bool] = typer.Option(
        None, "--has-error/--no-has-error", help="Filter by error status"
    ),
    reward_min: Optional[float] = typer.Option(None, "--reward-min", help="Minimum reward"),
    reward_max: Optional[float] = typer.Option(None, "--reward-max", help="Maximum reward"),
    created_after: Optional[str] = typer.Option(None, "--created-after", help="ISO timestamp"),
    created_before: Optional[str] = typer.Option(None, "--created-before", help="ISO timestamp"),
) -> None:
    """Stream a filtered export to a file. Same filters as `list`;
    resumable by re-running. Exports are metered on bytes transferred."""
    try:
        client = _traces_client()
        written = client.export(
            dest,
            run_id=run_id,
            task_id=task_id,
            model_id=model_id,
            outcome=outcome,
            has_error=has_error,
            reward_min=reward_min,
            reward_max=reward_max,
            created_after=created_after,
            created_before=created_before,
        )
    except typer.Exit:
        raise
    except UnauthorizedError as e:
        console.print(f"[red]Unauthorized:[/red] {str(e)}")
        raise typer.Exit(1)
    except PaymentRequiredError as e:
        console.print(f"[red]Payment Required:[/red] {str(e)}")
        raise typer.Exit(1)
    except PrimeTracesError as e:
        console.print(f"[red]Export failed:[/red] {str(e)}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {escape(str(e))}")
        console.print_exception(show_locals=True)
        raise typer.Exit(1)
    console.print(f"[green]Wrote {written / (1024 * 1024):.1f} MiB to {dest}[/green]")


@app.command("delete")
def delete_traces(
    trace_id: Optional[str] = typer.Argument(None, help="Trace ID to delete"),
    run_id: Optional[str] = typer.Option(
        None, "--run-id", help="Delete every trace in this run instead"
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
) -> None:
    """Delete every stored copy of one trace, or a whole run with --run-id."""
    if bool(trace_id) == bool(run_id):
        console.print("[red]Provide exactly one of TRACE_ID or --run-id[/red]")
        raise typer.Exit(1)

    target = f"trace {trace_id}" if trace_id else f"every trace in run {run_id}"
    if not yes and not typer.confirm(f"Delete {target}?"):
        raise typer.Exit(0)

    try:
        client = _traces_client()
        if trace_id:
            client.delete(trace_id)
            console.print(f"[green]Deletion of {target} accepted[/green]")
        else:
            assert run_id is not None
            job_id = client.delete_run(run_id)
            suffix = f" (job {job_id})" if job_id else ""
            console.print(f"[green]Deletion of {target} accepted{suffix}[/green]")
    except typer.Exit:
        raise
    except UnauthorizedError as e:
        console.print(f"[red]Unauthorized:[/red] {str(e)}")
        raise typer.Exit(1)
    except PaymentRequiredError as e:
        console.print(f"[red]Payment Required:[/red] {str(e)}")
        raise typer.Exit(1)
    except APIError as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {escape(str(e))}")
        console.print_exception(show_locals=True)
        raise typer.Exit(1)
