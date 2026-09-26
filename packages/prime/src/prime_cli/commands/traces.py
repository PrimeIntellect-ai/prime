import re
import tempfile
from pathlib import Path
from typing import Callable, Dict, List, Optional, TypeVar

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

from ..client import APIClient
from ..core import Config
from ..utils import (
    PlainTyper,
    get_console,
    json_output_help,
    output_data_as_json,
    validate_output_format,
)
from .teams import fetch_team_members
from .traces_views import episode_view, episodes_table, summary_view, traces_table

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

LIST_JSON_HELP = json_output_help(
    ".items[] = trace summary {trace_id, run_id, task_id, user_id, score, total_tokens,"
    " duration_ms, created_at, ingested_at, ...}",
    "with --episodes: .items[] = episode summary {episode_id, run_id, environment_id,"
    " outcome, has_error, error{type, message}, created_at, ...}",
    ".next_cursor? = string",
)

GET_JSON_HELP = json_output_help(
    ". = trace summary object; with --raw and no --dest, the exact stored trace document",
    "with --episodes: . = episode summary plus .traces = {trace_count, total_tokens,"
    " total_duration_ms, any_trace_error, agent_names[]}; with --raw and no --dest,"
    " the exact stored episode (.traces = member trace IDs)",
    "with --raw --dest: {dest, bytes_written}",
)

# Member traces shown under `prime traces get --episodes`; the rest are one
# `prime traces list --episode-id` away.
EPISODE_MEMBERS_SHOWN = 20


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
    query: str = typer.Argument(
        ...,
        help="Exact text to find: case-sensitive, 3-256 characters, no regex or wildcards",
    ),
    run_id: str = typer.Option(..., "--run-id", help="Run to search (required)"),
    field: str = typer.Option(
        "content",
        "--field",
        help=(
            "What to search: content (message text), reasoning_content, or tool_calls"
            " (the calls' JSON: tool names and arguments)"
        ),
    ),
    role: Optional[str] = typer.Option(
        None, "--role", help="Only messages from this role: system, user, assistant or tool"
    ),
    run_step: Optional[int] = typer.Option(
        None, "--run-step", min=0, help="Only traces from this training step"
    ),
    has_error: Optional[bool] = typer.Option(
        None, "--has-error/--no-has-error", help="Only traces with (or without) an error"
    ),
    reward_min: Optional[float] = typer.Option(None, "--reward-min", help="Minimum reward"),
    reward_max: Optional[float] = typer.Option(None, "--reward-max", help="Maximum reward"),
    limit: int = typer.Option(50, "--limit", min=1, max=100, help="Maximum matches per page"),
    cursor: Optional[str] = typer.Option(
        None, "--cursor", help="Continue a search from the cursor the previous page printed"
    ),
    output: str = typer.Option("table", "--output", "-o", help="table or json"),
) -> None:
    """Find exact text in the messages of one run's traces.

    The query is plain text matched exactly as typed.

    Each match is one message (a node), shown with the text around the hit.
    Results come one page at a time: to continue, rerun with the same query and
    filters plus the --cursor the page prints.

    \b
    Examples:
        prime traces search "Traceback" --run-id <run_id>
        prime traces search "rm -rf" --run-id <run_id> --field tool_calls
        prime traces search "I cannot" --run-id <run_id> --role assistant --has-error
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


def _write_atomically(dest: Path, data: bytes) -> None:
    """Replace ``dest`` only once ``data`` is fully written, like `get --raw --dest`."""
    with tempfile.NamedTemporaryFile(
        dir=dest.parent, prefix=".prime-traces-", suffix=".partial", delete=False
    ) as f:
        partial = Path(f.name)
    try:
        partial.write_bytes(data)
        partial.replace(dest)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


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
    if config.team_id_from_env:
        # `prime switch` refuses to run while the environment pins the team.
        error_console.print(
            "If another account owns it, change or unset [bold]PRIME_TEAM_ID[/bold]."
        )
    else:
        error_console.print("If another account owns it, switch with [bold]prime switch[/bold].")


def _is_episode_not_found(error: APIError) -> bool:
    return isinstance(error, NotFoundError) and error.code == "episode_not_found"


def _user_names() -> Optional[Dict[str, str]]:
    """Display names of the active team's members, keyed by user ID.

    None for a personal account, whose traces are all the caller's own. The
    lookup only decorates the table, so when it fails the IDs stand in.
    """
    config = Config()
    if not config.team_id:
        return None
    try:
        members = fetch_team_members(APIClient(), config.team_id)
    except Exception:
        return {}
    return {
        str(m["userId"]): str(m.get("userName") or m.get("userEmail") or m["userId"])
        for m in members
        if isinstance(m, dict) and m.get("userId")
    }


def _reject_options(mode: str, given: Dict[str, bool]) -> None:
    names = [name for name, present in given.items() if present]
    if names:
        error_console.print(f"[red]Error:[/red] {', '.join(names)} cannot be combined with {mode}")
        raise typer.Exit(1)


@app.command("list", epilog=LIST_JSON_HELP)
def list_traces(
    episodes: bool = typer.Option(False, "--episodes", help="List episodes instead of traces"),
    run_id: Optional[str] = typer.Option(None, "--run-id", help="Filter by run ID"),
    episode_id: Optional[str] = typer.Option(
        None, "--episode-id", help="Only this episode's traces (newest first; no --sort)"
    ),
    environment_id: Optional[str] = typer.Option(
        None, "--environment-id", "--env", help="Filter by environment ID"
    ),
    task_id: Optional[str] = typer.Option(
        None, "--task-id", help="Filter by task ID (traces only)"
    ),
    model_id: Optional[str] = typer.Option(
        None, "--model-id", help="Filter by model ID (traces only)"
    ),
    outcome: Optional[str] = typer.Option(None, "--outcome", help="Filter by outcome"),
    has_error: Optional[bool] = typer.Option(
        None, "--has-error/--no-has-error", help="Filter by error status"
    ),
    reward_min: Optional[float] = typer.Option(
        None, "--reward-min", help="Minimum reward (traces only)"
    ),
    reward_max: Optional[float] = typer.Option(
        None, "--reward-max", help="Maximum reward (traces only)"
    ),
    run_step: Optional[int] = typer.Option(
        None,
        "--run-step",
        min=0,
        help="With --episodes: only episodes with a trace from this training step",
    ),
    created_after: Optional[str] = typer.Option(
        None, "--created-after", help="ISO timestamp; also the cheapest filter"
    ),
    created_before: Optional[str] = typer.Option(None, "--created-before", help="ISO timestamp"),
    sort: Optional[str] = typer.Option(
        None,
        "--sort",
        help="Sort key (traces only): created_at (default, newest first), reward, duration_ms",
    ),
    page: int = typer.Option(
        1,
        "--page",
        "-p",
        help=(
            "Page number; each run walks the pages before it from the current top, so"
            " boundaries shift as new rows arrive (use --cursor for a fixed boundary)"
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
    """List trace summaries, or episode summaries with --episodes, newest first."""
    validate_output_format(output, error_console)
    _check_paging(page, cursor)
    if episodes:
        # Episodes carry no task, model, reward or sort key of their own.
        _reject_options(
            "--episodes",
            {
                "--episode-id": episode_id is not None,
                "--task-id": task_id is not None,
                "--model-id": model_id is not None,
                "--reward-min": reward_min is not None,
                "--reward-max": reward_max is not None,
                "--sort": sort is not None,
            },
        )
    else:
        _reject_options("a trace listing (add --episodes)", {"--run-step": run_step is not None})
    if episode_id is not None and sort is not None:
        error_console.print(
            "[red]Error:[/red] --sort cannot be combined with --episode-id;"
            " an episode's traces are listed newest first"
        )
        raise typer.Exit(1)
    try:
        client = _traces_client()

        def fetch_episodes(page_cursor: Optional[str]) -> EpisodeListPage:
            return client.list_episodes(
                run_id=run_id,
                environment_id=environment_id,
                outcome=outcome,
                has_error=has_error,
                run_step=run_step,
                created_after=created_after,
                created_before=created_before,
                limit=limit,
                cursor=page_cursor,
            )

        def fetch_traces(page_cursor: Optional[str]) -> TraceListPage:
            if episode_id is not None:
                return client.list_episode_traces(
                    episode_id,
                    run_id=run_id,
                    environment_id=environment_id,
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
                environment_id=environment_id,
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

        result = _list_page(fetch_episodes if episodes else fetch_traces, page=page, cursor=cursor)
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

    if isinstance(result, EpisodeListPage):
        _print_empty_page("episodes", page, len(result.items))
        table, note = episodes_table(
            result, run_id=run_id, run_step=run_step, environment_id=environment_id
        )
    else:
        _print_empty_page("traces", page, len(result.items))
        user_names = _user_names() if result.items else None
        table, note = traces_table(
            result,
            run_id=run_id,
            episode_id=episode_id,
            task_id=task_id,
            user_names=user_names,
            width=console.width,
        )
    console.print(table)
    if note is not None:
        console.print(note)
    _print_page_footer(
        count=len(result.items),
        next_cursor=result.next_cursor,
        page=page,
        limit=limit,
        cursor=cursor,
    )
    if isinstance(result, EpisodeListPage) and result.items:
        console.print(Text("Details: prime traces get <episode_id> --episodes", style="dim"))


@app.command("get", epilog=GET_JSON_HELP)
def get_trace(
    trace_id: str = typer.Argument(..., help="Trace ID, or episode ID with --episodes"),
    episodes: bool = typer.Option(
        False, "--episodes", help="Get an episode and its newest traces instead of a trace"
    ),
    raw: bool = typer.Option(False, "--raw", help="Fetch the exact stored document"),
    dest: Optional[Path] = typer.Option(
        None, "--dest", help="With --raw: write the document to this file"
    ),
    output: str = typer.Option("table", "--output", "-o", help="Output format: table or json"),
) -> None:
    """Get one trace, or one episode with --episodes; --raw fetches the stored document."""
    validate_output_format(output, error_console)
    if dest is not None and not raw:
        error_console.print("[red]--dest requires --raw[/red]")
        raise typer.Exit(1)
    if episodes:
        _get_episode(trace_id, raw=raw, dest=dest, output=output)
        return
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


def _get_episode(episode_id: str, *, raw: bool, dest: Optional[Path], output: str) -> None:
    """`get --episodes`: one episode with its newest traces, or the stored episode."""
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
            _write_atomically(dest, document)
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
