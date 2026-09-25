import json
import os
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, TypeVar

import click
import typer
from prime_traces import (
    APIError,
    Batch,
    EpisodeListPage,
    LineFormat,
    NotFoundError,
    PaymentRequiredError,
    PrimeTracesError,
    TraceListPage,
    TracesClient,
    TraceSearchMatch,
    UnauthorizedError,
    UploadReceipt,
)
from rich.markup import escape
from rich.table import Table
from rich.text import Text

from ..core import Config
from ..utils import (
    PlainTyper,
    get_console,
    json_output_help,
    output_data_as_json,
    validate_output_format,
)
from ..utils.plain import is_plain_mode
from .traces_episodes import TIME_FORMAT, episode_view, episodes_table
from .traces_transcript import (
    Transcript,
    parse_node_range,
    select_nodes,
    summary_view,
    tools_only_table,
    transcript_from_document,
    transcript_header,
    transcript_lines,
)

app = PlainTyper(help="Upload and query traces (Prime Traces)", no_args_is_help=True)
episodes_app = PlainTyper(help="List and inspect episodes", no_args_is_help=True)
app.add_typer(episodes_app, name="episodes")
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

LIST_EPISODES_JSON_HELP = json_output_help(
    ".items[] = episode summary {episode_id, run_id, environment_id, outcome, has_error,"
    " error{type, message}, created_at, ...}",
    ".next_cursor? = string",
)

GET_EPISODE_JSON_HELP = json_output_help(
    ". = episode summary plus .traces = {trace_count, total_tokens, total_duration_ms,"
    " any_trace_error, agent_names[]}",
    "with --raw and no --dest: the exact stored episode (.traces = member trace IDs)",
    "with --raw --dest: {dest, bytes_written}",
)

# Member traces shown under `prime traces episodes get`; the rest are one
# `prime traces list --episode` away.
EPISODE_MEMBERS_SHOWN = 20

GET_TRACE_JSON_HELP = json_output_help(
    ". = trace summary object; with --raw and no --dest, the exact stored trace document",
    "with --raw --dest: {dest, bytes_written}",
)

TRANSCRIPT_JSON_HELP = json_output_help(
    ".trace_id = string; .source = index|document (document when the index cannot serve it)",
    ".nodes[] = {node_idx, parent_idx, timestamp, sampled, message{role, content, ...}}"
    " (only --node's range)",
    ".calls[] = {call_idx, node_idx, time_start, time_end, model, endpoint, finish_reason}",
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


SEARCH_ROLE_STYLES = {"system": "dim", "user": "green", "assistant": "blue", "tool": "yellow"}


def _search_excerpt(match: TraceSearchMatch, width: int) -> Text:
    """One line with the literal highlighted, trimmed to keep the hit inside `width`."""
    excerpt = match.excerpt
    start = match.match_start - match.excerpt_start
    end = match.match_end - match.excerpt_start
    if not 0 <= start < end <= len(excerpt):
        start = end = len(excerpt)  # offsets do not describe this excerpt: no highlight
    squash = lambda part: re.sub(r"\s+", " ", part.replace("\\n", " "))  # noqa: E731
    before = squash(excerpt[:start])
    lead = min(40, max(4, (width - (end - start)) * 2 // 5))
    text = Text()
    if match.excerpt_start > 0 or len(before) > lead:
        text.append("…", style="dim")
    text.append(before[-lead:])
    text.append(squash(excerpt[start:end]), style="bold black on yellow")
    text.append(squash(excerpt[end:]))
    if len(excerpt) >= (end - start) + 128:  # the server window is full
        text.append("…", style="dim")
    return text


def _search_table(matches: List[TraceSearchMatch], *, query: str, run_id: str, field: str) -> Table:
    # Node, role, borders and padding are fixed; the excerpt takes what is left and
    # the trace ID gives way first in a narrow terminal.
    remaining = console.width - 26
    width = max(16, remaining - 32)
    table = Table(
        title=f'Trace search: "{escape(query)}" in {escape(field)} · run {escape(run_id)}',
        title_justify="left",
    )
    table.add_column("Trace ID", style="cyan", no_wrap=True, width=max(8, remaining - width))
    table.add_column("Node", justify="right", style="dim", no_wrap=True, width=4)
    table.add_column("Role", no_wrap=True, width=9)
    table.add_column("Match", no_wrap=True, width=width)
    previous = None
    for match in matches:
        if previous is not None and match.trace_id != previous:
            table.add_section()
        table.add_row(
            escape(match.trace_id) if match.trace_id != previous else "",
            str(match.node_idx),
            Text(match.role, style=SEARCH_ROLE_STYLES.get(match.role, "")),
            _search_excerpt(match, width),
        )
        previous = match.trace_id
    return table


@app.command("search")
def search_traces(
    query: str = typer.Argument(..., help="Case-sensitive literal text (quote phrases)"),
    run_id: str = typer.Option(..., "--run-id", help="Required run scope"),
    field: str = typer.Option(
        "content", "--field", help="content, reasoning_content, or tool_calls"
    ),
    role: Optional[str] = typer.Option(None, "--role", help="Message role"),
    run_step: Optional[int] = typer.Option(None, "--run-step", min=0),
    has_error: Optional[bool] = typer.Option(None, "--has-error/--no-has-error"),
    reward_min: Optional[float] = typer.Option(None, "--reward-min"),
    reward_max: Optional[float] = typer.Option(None, "--reward-max"),
    limit: int = typer.Option(50, "--limit", min=1, max=100, help="Maximum matches returned"),
    cursor: Optional[str] = typer.Option(None, "--cursor", help="Continue an unfinished search"),
    output: str = typer.Option("table", "--output", "-o", help="table or json"),
) -> None:
    """Return one page of matches in indexed trace content.

    Follow next_cursor with unchanged filters until the search is exhausted.
    """
    validate_output_format(output, error_console)
    if field not in ("content", "reasoning_content", "tool_calls"):
        raise typer.BadParameter(
            "Choose content, reasoning_content, or tool_calls", param_hint="--field"
        )
    if "/" in run_id:
        # The run ID is one URL path segment; the SDK refuses what it cannot address.
        raise typer.BadParameter("Run IDs containing '/' cannot be searched", param_hint="--run-id")
    if not query.strip() or not 3 <= len(query) <= 256:
        raise typer.BadParameter("Provide 3–256 characters of nonblank text", param_hint="query")
    try:
        with _traces_client() as client:
            result = client.search(
                query,
                run_id=run_id,
                field=field,
                role=role,
                run_step=run_step,
                has_error=has_error,
                reward_min=reward_min,
                reward_max=reward_max,
                limit=limit,
                cursor=cursor,
            )
    except NotFoundError as exc:
        # A server without the route answers with a bare 404 and no error code.
        if exc.code is None:
            error_console.print(
                "[red]Search is unavailable on this server. "
                "It requires the Prime Traces search API.[/red]"
            )
        else:
            error_console.print(f"[red]Search failed:[/red] {escape(str(exc))}")
        raise typer.Exit(1)
    except PrimeTracesError as exc:
        error_console.print(f"[red]Search failed:[/red] {escape(str(exc))}")
        raise typer.Exit(1)

    if output == "json":
        output_data_as_json(result.model_dump(mode="json"), console)
        return
    console.print(_search_table(result.items, query=query, run_id=run_id, field=field))
    traces = len({match.trace_id for match in result.items})
    summary = f"{len(result.items)} matches in {traces} traces on this page"
    coverage = result.coverage
    if coverage is not None:
        summary += f" · {coverage.examined_traces} traces searched"
    console.print(f"[dim]{escape(summary)}[/dim]")
    if coverage is not None:
        if coverage.unindexed_trace_ids or coverage.partial_index:
            error_console.print(
                "[yellow]Incomplete index coverage: some traces are unindexed or capped. "
                "Restart after indexing to include pending traces.[/yellow]"
            )
    elif cursor is None:
        # Only resumed pages omit coverage by design; on a first page it means the
        # server could not compute it, which is unknown rather than complete.
        error_console.print(
            "[yellow]Index coverage is unknown: the server could not check whether every "
            "trace in the run was searchable. Matches may be incomplete.[/yellow]"
        )
    if result.next_cursor:
        console.print("Search has more pages. Continue with the same filters and:")
        console.print(f"--cursor {escape(result.next_cursor)}", soft_wrap=True)
    else:
        console.print("Search exhausted for the currently available index.")


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


PageT = TypeVar("PageT", TraceListPage, EpisodeListPage)


def _list_page(
    fetch: Callable[[Optional[str]], PageT],
    *,
    page: int,
    cursor: Optional[str],
) -> PageT:
    """Fetch one page, walking the pages before it when it is not the first.

    The service paginates by cursor only, so page N costs N requests. Every
    hop asks for the same page size, which keeps the page boundaries lined
    up; a walk that runs out of pages early yields an empty page rather than
    the last real one.
    """
    for _ in range(page - 1):
        hop = fetch(cursor)
        if not hop.next_cursor:
            return hop.model_copy(update={"items": [], "next_cursor": None})
        cursor = hop.next_cursor
    return fetch(cursor)


def _check_paging(page: int, cursor: Optional[str]) -> None:
    if page < 1:
        error_console.print("[red]Error:[/red] --page must be at least 1")
        raise typer.Exit(1)
    if cursor is not None and page > 1:
        error_console.print("[red]Error:[/red] --page cannot be combined with --cursor")
        raise typer.Exit(1)


def _print_empty_page(noun: str, page: int, count: int) -> None:
    if not count and page > 1:
        console.print(f"[yellow]No {noun} on page {page}.[/yellow]")
        console.print("Try [bold]--page 1[/bold] to start from the beginning.")


def _print_page_footer(
    *, count: int, next_cursor: Optional[str], page: int, limit: int, cursor: Optional[str]
) -> None:
    # A cursor resume has no page number to report, so it keeps the raw
    # cursor hint alone; page mode mirrors the other list commands' footer.
    if cursor is not None:
        if next_cursor:
            console.print(f"[dim]More results: --cursor {escape(next_cursor)}[/dim]")
        return
    if count and (next_cursor or page > 1):
        start = (page - 1) * limit + 1
        end = (page - 1) * limit + count
        console.print(f"[dim]Page {page} • showing {start}-{end}[/dim]")
    if next_cursor:
        # `--page N+1` re-walks from the current top in a fresh process, so a
        # row that arrives in between shifts every boundary and the row that
        # fell off this page shows up again on the next. The cursor pins the
        # boundary to this exact row, so it stays on offer.
        console.print(f"[dim]Use --page {page + 1} to see more.[/dim]")
        console.print(
            f"[dim]Or resume from this exact boundary: --cursor {escape(next_cursor)}[/dim]"
        )


def _episode_not_found(episode_id: str) -> None:
    """Explain a missing episode in terms of the account the lookup ran as.

    The service answers the same 404 for an episode that does not exist and
    for one another account owns, and the account is the active team when
    one is set, so the wrong team is the likeliest cause.
    """
    config = Config()
    if config.team_id:
        # The stored name belongs to the stored team, not one PRIME_TEAM_ID selects.
        name = None if config.team_id_from_env else config.team_name
        owner = f"team {name or config.team_id}"
    else:
        owner = "your personal account"
    error_console.print(
        f"[red]Not found:[/red] no episode {escape(episode_id)} in {escape(owner)}."
    )
    error_console.print("If another account owns it, switch with [bold]prime switch[/bold].")


def _is_episode_not_found(error: APIError) -> bool:
    return isinstance(error, NotFoundError) and error.code == "episode_not_found"


@app.command("list", epilog=LIST_TRACES_JSON_HELP)
def list_traces(
    run_id: Optional[str] = typer.Option(None, "--run-id", help="Filter by run ID"),
    episode_id: Optional[str] = typer.Option(
        None, "--episode", help="Only this episode's traces (newest first; no --sort)"
    ),
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
        help=(
            "Page number; each run walks the pages before it from the current top, so"
            " boundaries shift as traces arrive (use --cursor for a fixed boundary)"
        ),
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
    _check_paging(page, cursor)
    if episode_id is not None and sort is not None:
        error_console.print(
            "[red]Error:[/red] --sort cannot be combined with --episode;"
            " an episode's traces are listed newest first"
        )
        raise typer.Exit(1)
    try:
        client = _traces_client()

        def fetch(page_cursor: Optional[str]) -> TraceListPage:
            if episode_id is not None:
                return client.list_episode_traces(
                    episode_id,
                    run_id=run_id,
                    task_id=task_id,
                    model_id=model_id,
                    outcome=outcome,
                    has_error=has_error,
                    reward_min=reward_min,
                    reward_max=reward_max,
                    created_after=created_after,
                    created_before=created_before,
                    limit=limit,
                    cursor=page_cursor,
                )
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
        if episode_id is not None and _is_episode_not_found(e):
            _episode_not_found(episode_id)
        else:
            error_console.print(f"[red]Error:[/red] {escape(str(e))}")
        raise typer.Exit(1)
    except Exception as e:
        error_console.print(f"[red]Unexpected error:[/red] {escape(str(e))}")
        error_console.print_exception()
        raise typer.Exit(1)

    if output == "json":
        output_data_as_json(result.model_dump(mode="json"), console)
        return

    # An episode's traces share its run and task, so the agent that produced
    # each one takes their place.
    title = "Traces" if episode_id is None else f"Traces · episode {episode_id}"
    table = Table(title=Text(title), title_justify="left")
    table.add_column("Trace ID", style="cyan", no_wrap=True)
    if episode_id is None:
        table.add_column("Run", style="green")
        table.add_column("Task")
    else:
        table.add_column("Agent", style="green")
    table.add_column("Reward", justify="right")
    table.add_column("Outcome")
    table.add_column("Created", no_wrap=True)

    for summary in result.items:
        reward = summary.score.reward
        if episode_id is None:
            source = [escape(summary.run_id or "-"), escape(summary.task_id or "-")]
        else:
            source = [escape(summary.agent_name or "-")]
        table.add_row(
            escape(summary.trace_id),
            *source,
            "-" if reward is None else f"{reward:.2f}",
            escape(summary.score.outcome or "-"),
            escape(summary.created_at.strftime(TIME_FORMAT)),
        )
    _print_empty_page("traces", page, len(result.items))
    console.print(table)
    _print_page_footer(
        count=len(result.items),
        next_cursor=result.next_cursor,
        page=page,
        limit=limit,
        cursor=cursor,
    )


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

    for renderable in summary_view(summary):
        console.print(renderable)


def _load_transcript(client: TracesClient, trace_id: str) -> Tuple[Transcript, Optional[str]]:
    """Read nodes and calls from the index, or the raw document when the index can't serve it.

    Returns the transcript and, when it came from the document, why.
    """
    nodes: List[dict] = []
    calls: List[dict] = []
    try:
        cursor: Optional[str] = None
        while True:
            node_page = client.list_nodes(trace_id, limit=100, cursor=cursor)
            nodes.extend(n.model_dump(mode="json") for n in node_page.items)
            if node_page.partial_index:
                return _document_transcript(client, trace_id), (
                    "the trace is larger than the node index holds"
                )
            if not node_page.next_cursor:
                break
            cursor = node_page.next_cursor
        cursor = None
        while True:
            call_page = client.list_calls(trace_id, limit=100, cursor=cursor)
            calls.extend(c.model_dump(mode="json") for c in call_page.items)
            if call_page.partial_index:
                return _document_transcript(client, trace_id), (
                    "the trace is larger than the call index holds"
                )
            if not call_page.next_cursor:
                break
            cursor = call_page.next_cursor
    except APIError as e:
        # Matched by code rather than by `TraceNotIndexedError`, which only exists from
        # prime-traces 0.0.6: importing it would break every command on an older SDK.
        if e.code != "trace_not_indexed":
            raise
        return _document_transcript(client, trace_id), "the trace is still being indexed"
    return Transcript(trace_id=trace_id, source="index", nodes=nodes, calls=calls), None


def _document_transcript(client: TracesClient, trace_id: str) -> Transcript:
    try:
        document = json.loads(client.get_raw(trace_id))
    except ValueError as e:
        raise PrimeTracesError(f"the stored document is not valid JSON ({e})") from None
    try:
        return transcript_from_document(trace_id, document)
    except ValueError as e:
        raise PrimeTracesError(str(e)) from None


@app.command("transcript", epilog=TRANSCRIPT_JSON_HELP)
def transcript_command(
    trace_id: str = typer.Argument(..., help="Trace ID"),
    full: bool = typer.Option(
        False, "--full", help="Show every message in full, including the system prompt"
    ),
    node: Optional[str] = typer.Option(
        None,
        "--node",
        help="Node index or inclusive range: 12, 30:, :9 or 5:9. A single node is shown in full",
    ),
    tools_only: bool = typer.Option(
        False, "--tools-only", help="One row per tool call with the size of its result"
    ),
    no_pager: bool = typer.Option(
        False, "--no-pager", help="Print directly instead of opening a pager in a terminal"
    ),
    output: str = typer.Option("table", "--output", "-o", help="Output format: table or json"),
) -> None:
    """Show a trace's conversation: each model turn, its tool calls, and their results."""
    validate_output_format(output, error_console)
    try:
        node_range = parse_node_range(node) if node is not None else None
    except ValueError as e:
        error_console.print(f"[red]Error:[/red] {escape(str(e))}")
        raise typer.Exit(1)
    # A single node is what the truncation hints point at, so show all of it.
    if node_range is not None and node_range[0] is not None and node_range[0] == node_range[1]:
        full = True

    try:
        client = _traces_client()
        if not hasattr(client, "list_nodes"):
            error_console.print(
                "[red]Error:[/red] prime traces transcript needs prime-traces 0.0.6 or newer."
                " Upgrade the prime CLI and try again."
            )
            raise typer.Exit(1)
        summary = client.get(trace_id)
        transcript, fallback = _load_transcript(client, trace_id)
    except typer.Exit:
        raise
    except UnauthorizedError as e:
        error_console.print(f"[red]Unauthorized:[/red] {escape(str(e))}")
        raise typer.Exit(1)
    except PaymentRequiredError as e:
        error_console.print(f"[red]Payment Required:[/red] {escape(str(e))}")
        raise typer.Exit(1)
    except PrimeTracesError as e:
        error_console.print(f"[red]Error:[/red] {escape(str(e))}")
        raise typer.Exit(1)
    except Exception as e:
        error_console.print(f"[red]Unexpected error:[/red] {escape(str(e))}")
        error_console.print_exception()
        raise typer.Exit(1)

    nodes = select_nodes(transcript, node_range)

    if output == "json":
        output_data_as_json(
            {
                "trace_id": transcript.trace_id,
                "source": transcript.source,
                "nodes": nodes,
                "calls": transcript.calls,
            },
            console,
        )
        return

    plain = is_plain_mode()
    if not nodes:
        body = [Text("No nodes in that range.", style="yellow")]
    elif tools_only:
        body = [tools_only_table(transcript, nodes, width=console.width)]
    else:
        body = transcript_lines(transcript, nodes, full=full, plain=plain)

    renderables = [
        transcript_header(summary, transcript) if not plain else Text(f"trace {trace_id}")
    ]
    if fallback:
        renderables.append(Text(f"Read from the full document: {fallback}.", style="dim"))
    renderables.extend(body)

    use_pager = not no_pager and not plain and console.is_terminal
    if use_pager:
        # Keep colors, and quit straight away when everything fits on one screen.
        os.environ.setdefault("LESS", "-FRX")
        with console.pager(styles=True):
            for renderable in renderables:
                console.print(renderable)
    else:
        for renderable in renderables:
            console.print(renderable)


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


@episodes_app.command("list", epilog=LIST_EPISODES_JSON_HELP)
def list_episodes(
    run_id: Optional[str] = typer.Option(None, "--run-id", help="Filter by run ID"),
    environment_id: Optional[str] = typer.Option(
        None, "--environment-id", "--env", help="Filter by environment ID"
    ),
    outcome: Optional[str] = typer.Option(None, "--outcome", help="Filter by outcome"),
    has_error: Optional[bool] = typer.Option(
        None, "--has-error/--no-has-error", help="Filter by the episode's own error status"
    ),
    created_after: Optional[str] = typer.Option(None, "--created-after", help="ISO timestamp"),
    created_before: Optional[str] = typer.Option(None, "--created-before", help="ISO timestamp"),
    page: int = typer.Option(
        1,
        "--page",
        "-p",
        help=(
            "Page number; each run walks the pages before it from the current top, so"
            " boundaries shift as episodes arrive (use --cursor for a fixed boundary)"
        ),
    ),
    limit: int = typer.Option(20, "--limit", help="Max results per page (up to 100)"),
    cursor: Optional[str] = typer.Option(
        None,
        "--cursor",
        help="Resume from a cursor returned by a previous page (cannot be combined with --page)",
    ),
    output: str = typer.Option("table", "--output", "-o", help="Output format: table or json"),
) -> None:
    """List episode summaries, newest first."""
    validate_output_format(output, error_console)
    _check_paging(page, cursor)
    try:
        client = _traces_client()

        def fetch(page_cursor: Optional[str]) -> EpisodeListPage:
            return client.list_episodes(
                run_id=run_id,
                environment_id=environment_id,
                outcome=outcome,
                has_error=has_error,
                created_after=created_after,
                created_before=created_before,
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

    _print_empty_page("episodes", page, len(result.items))
    console.print(episodes_table(result, run_id=run_id))
    _print_page_footer(
        count=len(result.items),
        next_cursor=result.next_cursor,
        page=page,
        limit=limit,
        cursor=cursor,
    )
    if result.items:
        console.print(Text("Details: prime traces episodes get <episode_id>", style="dim"))


@episodes_app.command("get", epilog=GET_EPISODE_JSON_HELP)
def get_episode(
    episode_id: str = typer.Argument(..., help="Episode ID"),
    raw: bool = typer.Option(False, "--raw", help="Fetch the exact stored episode"),
    dest: Optional[Path] = typer.Option(
        None, "--dest", help="With --raw: write the episode to this file"
    ),
    output: str = typer.Option("table", "--output", "-o", help="Output format: table or json"),
) -> None:
    """Get one episode with its newest traces, or the stored episode with --raw."""
    validate_output_format(output, error_console)
    if dest is not None and not raw:
        error_console.print("[red]--dest requires --raw[/red]")
        raise typer.Exit(1)

    document: Optional[bytes] = None
    members: Optional[TraceListPage] = None
    try:
        client = _traces_client()
        if raw:
            document = client.get_episode_raw(episode_id)
        else:
            detail = client.get_episode(episode_id)
            if output != "json":
                members = client.list_episode_traces(episode_id, limit=EPISODE_MEMBERS_SHOWN)
    except typer.Exit:
        raise
    except UnauthorizedError as e:
        error_console.print(f"[red]Unauthorized:[/red] {escape(str(e))}")
        raise typer.Exit(1)
    except PaymentRequiredError as e:
        error_console.print(f"[red]Payment Required:[/red] {escape(str(e))}")
        raise typer.Exit(1)
    except APIError as e:
        if _is_episode_not_found(e):
            _episode_not_found(episode_id)
        else:
            error_console.print(f"[red]Error:[/red] {escape(str(e))}")
        raise typer.Exit(1)
    except Exception as e:
        error_console.print(f"[red]Unexpected error:[/red] {escape(str(e))}")
        error_console.print_exception()
        raise typer.Exit(1)

    if document is not None:
        # The stored episode lists its traces by ID only, so it is small enough
        # to read whole. Keep its exact bytes either way.
        if dest is None:
            stdout = click.get_binary_stream("stdout")
            stdout.write(document)
            stdout.flush()
            return
        try:
            dest.write_bytes(document)
        except OSError as e:
            error_console.print(
                f"[red]Error:[/red] could not write {escape(str(dest))}: {escape(str(e))}"
            )
            raise typer.Exit(1)
        if output == "json":
            output_data_as_json({"dest": str(dest), "bytes_written": len(document)}, console)
        else:
            console.print(f"[green]Wrote {len(document)} bytes to {escape(str(dest))}[/green]")
        return

    if output == "json":
        output_data_as_json(detail.model_dump(mode="json"), console)
        return
    assert members is not None
    for renderable in episode_view(detail, members):
        console.print(renderable)
