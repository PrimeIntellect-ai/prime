from pathlib import Path
from typing import Callable, Dict, List, Optional

import click
import typer
from prime_traces import (
    APIError,
    Batch,
    LineFormat,
    NotFoundError,
    PaymentRequiredError,
    PrimeTracesError,
    TraceListPage,
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
error_console = get_console(stderr=True)


def _traces_client() -> TracesClient:
    """Build a client from the CLI config."""
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
    ". = trace summary object; with --raw and no --dest, the exact stored trace document",
    "with --raw --dest: {dest, bytes_written}",
)


def _parse_context(values: List[str]) -> Optional[Dict[str, str]]:
    if not values:
        return None
    context: Dict[str, str] = {}
    for item in values:
        key, sep, value = item.partition("=")
        if not sep or not key:
            error_console.print(
                f"[red]Invalid --context '{escape(item)}'; expected key=value[/red]"
            )
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
    validate_output_format(output, error_console)
    line_format = LineFormat.EPISODE if episodes else LineFormat.TRACE

    def on_batch(batch: Batch, receipt: UploadReceipt) -> None:
        if output != "json":
            console.print(
                f"  batch {escape(receipt.upload_id[:12])}… "
                f"({batch.num_lines} lines, {batch.size / (1024 * 1024):.1f} MiB) "
                f"[green]{escape(receipt.status)}[/green]"
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
        error_console.print(f"[red]Unauthorized:[/red] {escape(str(e))}")
        raise typer.Exit(1)
    except PaymentRequiredError as e:
        error_console.print(f"[red]Payment Required:[/red] {escape(str(e))}")
        raise typer.Exit(1)
    except PrimeTracesError as e:
        error_console.print(f"[red]Upload failed:[/red] {escape(str(e))}")
        raise typer.Exit(1)
    except Exception as e:
        error_console.print(f"[red]Unexpected error:[/red] {escape(str(e))}")
        error_console.print_exception()
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
        console.print(f"[green]Uploaded {len(receipts)} batch(es) from {escape(str(file))}[/green]")


def _list_page(
    fetch: Callable[[Optional[str]], TraceListPage],
    *,
    page: int,
    cursor: Optional[str],
) -> TraceListPage:
    """Fetch one page, walking the pages before it when it is not the first.

    The service paginates by cursor only, so page N costs N requests. Every
    hop asks for the same page size, which keeps the page boundaries lined
    up; a walk that runs out of pages early yields an empty page rather than
    the last real one.
    """
    for _ in range(page - 1):
        hop = fetch(cursor)
        if not hop.next_cursor:
            return TraceListPage(items=[], next_cursor=None)
        cursor = hop.next_cursor
    return fetch(cursor)


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
    page: int = typer.Option(
        1,
        "--page",
        "-p",
        help="Page number; later pages are reached by walking the pages before them",
    ),
    limit: int = typer.Option(20, "--limit", help="Max results per page (up to 100)"),
    cursor: Optional[str] = typer.Option(
        None,
        "--cursor",
        help="Resume from a cursor returned by a previous page (cannot be combined with --page)",
    ),
    output: str = typer.Option("table", "--output", "-o", help="Output format: table or json"),
) -> None:
    """List trace summaries, newest first."""
    validate_output_format(output, error_console)
    if page < 1:
        error_console.print("[red]Error:[/red] --page must be at least 1")
        raise typer.Exit(1)
    if cursor is not None and page > 1:
        error_console.print("[red]Error:[/red] --page cannot be combined with --cursor")
        raise typer.Exit(1)
    try:
        client = _traces_client()

        def fetch(page_cursor: Optional[str]) -> TraceListPage:
            return client.list(
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
                cursor=page_cursor,
            )

        result = _list_page(fetch, page=page, cursor=cursor)
    except typer.Exit:
        raise
    except UnauthorizedError as e:
        error_console.print(f"[red]Unauthorized:[/red] {escape(str(e))}")
        raise typer.Exit(1)
    except PaymentRequiredError as e:
        error_console.print(f"[red]Payment Required:[/red] {escape(str(e))}")
        raise typer.Exit(1)
    except APIError as e:
        error_console.print(f"[red]Error:[/red] {escape(str(e))}")
        raise typer.Exit(1)
    except Exception as e:
        error_console.print(f"[red]Unexpected error:[/red] {escape(str(e))}")
        error_console.print_exception()
        raise typer.Exit(1)

    if output == "json":
        output_data_as_json(result.model_dump(mode="json"), console)
        return

    table = Table(title="Traces")
    table.add_column("Trace ID", style="cyan", no_wrap=True)
    table.add_column("Run", style="green")
    table.add_column("Task")
    table.add_column("Reward", justify="right")
    table.add_column("Outcome")
    table.add_column("Created")

    for summary in result.items:
        reward = summary.score.reward
        table.add_row(
            escape(summary.trace_id),
            escape(summary.run_id or "-"),
            escape(summary.task_id or "-"),
            "-" if reward is None else f"{reward:.2f}",
            escape(summary.score.outcome or "-"),
            escape(summary.created_at.isoformat()),
        )
    if not result.items and page > 1:
        console.print(f"[yellow]No traces on page {page}.[/yellow]")
        console.print("Try [bold]--page 1[/bold] to start from the beginning.")
    console.print(table)

    # A cursor resume has no page number to report, so it keeps the raw
    # cursor hint; page mode mirrors the other list commands' footer.
    if cursor is not None:
        if result.next_cursor:
            console.print(f"[dim]More results: --cursor {escape(result.next_cursor)}[/dim]")
        return
    if result.items and (result.next_cursor or page > 1):
        start = (page - 1) * limit + 1
        end = (page - 1) * limit + len(result.items)
        console.print(f"[dim]Page {page} • showing {start}-{end}[/dim]")
    if result.next_cursor:
        console.print(f"[dim]Use --page {page + 1} to see more.[/dim]")


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
    validate_output_format(output, error_console)
    if dest is not None and not raw:
        error_console.print("[red]--dest requires --raw[/red]")
        raise typer.Exit(1)

    try:
        client = _traces_client()
        if raw:
            if dest is not None:
                written = client.download_raw(trace_id, dest)
                if output == "json":
                    output_data_as_json(
                        {"dest": str(dest), "bytes_written": written},
                        console,
                    )
                else:
                    console.print(f"[green]Wrote {written} bytes to {escape(str(dest))}[/green]")
                return
            # Raw documents can be tens of MiB. Preserve their exact bytes and
            # do not append a newline so redirected output is a faithful copy.
            stdout = click.get_binary_stream("stdout")
            stdout.write(client.get_raw(trace_id))
            stdout.flush()
            return
        summary = client.get(trace_id)
    except typer.Exit:
        raise
    except UnauthorizedError as e:
        error_console.print(f"[red]Unauthorized:[/red] {escape(str(e))}")
        raise typer.Exit(1)
    except PaymentRequiredError as e:
        error_console.print(f"[red]Payment Required:[/red] {escape(str(e))}")
        raise typer.Exit(1)
    except APIError as e:
        error_console.print(f"[red]Error:[/red] {escape(str(e))}")
        raise typer.Exit(1)
    except Exception as e:
        error_console.print(f"[red]Unexpected error:[/red] {escape(str(e))}")
        error_console.print_exception()
        raise typer.Exit(1)

    if output == "json":
        output_data_as_json(summary.model_dump(mode="json"), console)
        return

    table = Table(title=f"Trace {escape(trace_id)}")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="green")
    for field, value in summary.model_dump(mode="json").items():
        table.add_row(escape(field), "-" if value is None else escape(str(value)))
    console.print(table)


@app.command("delete")
def delete_traces(
    trace_id: Optional[str] = typer.Argument(None, help="Trace ID to delete"),
    run_id: Optional[str] = typer.Option(
        None, "--run-id", help="Delete every trace in this run instead"
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
) -> None:
    """Delete every stored copy of one trace, or a whole run with --run-id.

    202 confirms logical deletion, not physical reclamation. Deleting
    something the owner does not have is an error, not a no-op, so repeating
    a delete that already succeeded reports "not found".
    """
    if bool(trace_id) == bool(run_id):
        error_console.print("[red]Provide exactly one of TRACE_ID or --run-id[/red]")
        raise typer.Exit(1)

    target = f"trace {trace_id}" if trace_id else f"every trace in run {run_id}"
    if not yes and not typer.confirm(f"Delete {target}?"):
        raise typer.Exit(0)

    try:
        client = _traces_client()
        if trace_id:
            client.delete(trace_id)
        else:
            assert run_id is not None
            client.delete_run(run_id)
        console.print(f"[green]Deletion of {escape(target)} accepted[/green]")
    except typer.Exit:
        raise
    except NotFoundError as e:
        error_console.print(f"[red]Not found:[/red] {escape(str(e))}")
        raise typer.Exit(1)
    except UnauthorizedError as e:
        error_console.print(f"[red]Unauthorized:[/red] {escape(str(e))}")
        raise typer.Exit(1)
    except PaymentRequiredError as e:
        error_console.print(f"[red]Payment Required:[/red] {escape(str(e))}")
        raise typer.Exit(1)
    except APIError as e:
        error_console.print(f"[red]Error:[/red] {escape(str(e))}")
        raise typer.Exit(1)
    except Exception as e:
        error_console.print(f"[red]Unexpected error:[/red] {escape(str(e))}")
        error_console.print_exception()
        raise typer.Exit(1)
