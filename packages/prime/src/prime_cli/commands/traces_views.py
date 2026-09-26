"""Rendering for `prime traces list` and `prime traces get`, for traces and episodes.

Pure functions over SDK models, so the command module keeps the I/O and the
tests can render without a network. Every producer-written string is wrapped
in `Text` rather than interpolated into markup, so it is always literal.
"""

from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from prime_traces import (
    EpisodeDetail,
    EpisodeError,
    EpisodeListPage,
    TraceListPage,
    TraceSummary,
)
from rich.console import RenderableType
from rich.table import Table
from rich.text import Text

from ..utils.time_utils import format_time_ago

TIME_FORMAT = "%Y-%m-%d %H:%M:%SZ"

TOOL_NAMES_CHARS = 60


def _duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    if seconds < 60:
        return f"{seconds}s"
    minutes, secs = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


def _size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KiB", "MiB"):
        if size < 1024 or unit == "MiB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GiB"


def _error_text(error: EpisodeError) -> str:
    return ": ".join(part for part in (error.type, error.message) if part)


def shared_value(values: Iterable[Optional[str]]) -> Optional[str]:
    """The one value every row has, or None when the rows differ or lack it."""
    distinct = set(values)
    return distinct.pop() if len(distinct) == 1 else None


def page_note(count: int, shared: Iterable[Optional[str]]) -> Optional[Text]:
    """The line under the table for values every row on the page happens to share.

    Returned beside the table rather than as its caption, which would wrap to
    the table's width, or in a Group, which would hide the table from plain
    mode's border stripping.

    They leave their columns to keep the table narrow, but they are what this
    page contains, not a filter, so they go on a line under the table and not in the title.
    """
    parts = [part for part in shared if part]
    if not parts:
        return None
    return Text(f"All {count} on this page: " + " · ".join(parts), style="dim")


def uploader_id(summary: TraceSummary) -> Optional[str]:
    """The user whose token uploaded the trace; null on rows older than the column."""
    value = (summary.model_extra or {}).get("user_id")
    return value if isinstance(value, str) and value else None


# ---------------------------------------------------------------------------
# prime traces list
# ---------------------------------------------------------------------------


# Free-text cells keep at least this many characters before a column is dropped.
FREE_TEXT_FLOOR = 8
# Columns dropped, in this order, while the terminal is too narrow for them all.
DROP_ORDER = ("Ingested", "Duration", "Tokens")


def _reward(summary: TraceSummary) -> Text:
    reward = summary.score.reward
    # With no Outcome column, a failed trace shows as a red reward.
    return Text(
        "-" if reward is None else f"{reward:.2f}",
        style="red" if summary.execution.has_error else "",
    )


# (header, right-aligned, cell). Relative times keep both timestamps narrow;
# `get` and JSON have the exact ones.
METRIC_COLUMNS: List[Tuple[str, bool, Callable[[TraceSummary], Text]]] = [
    ("Reward", True, _reward),
    ("Tokens", True, lambda s: Text(f"{s.total_tokens:,}")),
    ("Duration", True, lambda s: Text(_duration(s.duration_ms / 1000))),
    ("Created", False, lambda s: Text(format_time_ago(s.created_at))),
    ("Ingested", False, lambda s: Text(format_time_ago(s.ingested_at))),
]


def traces_table(
    page: TraceListPage,
    *,
    run_id: Optional[str] = None,
    episode_id: Optional[str] = None,
    task_id: Optional[str] = None,
    user_names: Optional[Dict[str, str]] = None,
    width: Optional[int] = None,
) -> Tuple[Table, Optional[Text]]:
    """One row per trace.

    The title names only the filters that were passed. A run, task or uploader
    that every row shares leaves its column for a line under the table, so the
    table keeps its width for what differs between traces. `user_names` is None
    for a personal account, where every trace is the caller's own and the User
    column would only repeat it.

    The trace ID is never shortened, since it is what the other commands take.
    When `width` cannot fit every column, the free-text columns shrink first
    and then the columns in DROP_ORDER go; `get` and JSON keep every field.
    """
    items = page.items
    # One row shares everything with itself, so it keeps its columns.
    several = len(items) > 1
    run = shared_value(s.run_id for s in items) if several and run_id is None else None
    task = shared_value(s.task_id for s in items) if several and task_id is None else None
    uploaders = {uploader_id(s) for s in items}
    show_users = user_names is not None and uploaders - {None} != set()
    user = shared_value(uploaders) if show_users and several else None

    def name(user_id: Optional[str]) -> str:
        if user_id is None:
            return "-"
        return (user_names or {}).get(user_id) or user_id

    title = " · ".join(
        part
        for part in (
            "Traces",
            run_id and f"run {run_id}",
            episode_id and f"episode {episode_id}",
            task_id and f"task {task_id}",
        )
        if part
    )
    note = page_note(
        len(items),
        (run and f"run {run}", task and f"task {task}", user and f"uploaded by {name(user)}"),
    )
    run_filtered_or_shared = run_id is not None or run is not None
    task_filtered_or_shared = task_id is not None or task is not None

    free_text: List[Tuple[str, str, Callable[[TraceSummary], Optional[str]]]] = []
    if not run_filtered_or_shared:
        free_text.append(("Run", "green", lambda s: s.run_id))
    # An episode's traces are its agents' turns, so each row names its agent.
    if episode_id is not None:
        free_text.append(("Agent", "green", lambda s: s.agent_name))
    if not task_filtered_or_shared:
        free_text.append(("Task", "", lambda s: s.task_id))
    if show_users and user is None:
        free_text.append(("User", "", lambda s: name(uploader_id(s))))

    id_width = max((len(s.trace_id) for s in items), default=len("Trace ID"))
    metrics = METRIC_COLUMNS
    if width is not None and items:

        def needed() -> int:
            # Every column costs its content plus three characters of border and padding.
            total = 4 + max(id_width, len("Trace ID"))
            total += (3 + FREE_TEXT_FLOOR) * len(free_text)
            for header, _, cell in metrics:
                total += 3 + max(len(header), *(len(cell(s).plain) for s in items))
            return total

        for header in DROP_ORDER:
            if needed() <= width:
                break
            metrics = [column for column in metrics if column[0] != header]

    table = Table(title=Text(title), title_justify="left")
    table.add_column("Trace ID", style="cyan", no_wrap=True, min_width=id_width)
    for header, style, _ in free_text:
        # Wrappable, so the column can shrink; each cell stays on one line.
        table.add_column(header, style=style)
    for header, right, _ in metrics:
        table.add_column(header, justify="right" if right else "left", no_wrap=True)
    for summary in items:
        table.add_row(
            Text(summary.trace_id),
            *(
                Text(value(summary) or "-", no_wrap=True, overflow="ellipsis")
                for *_, value in free_text
            ),
            *(cell(summary) for *_, cell in metrics),
        )
    return table, note


def episodes_table(
    page: EpisodeListPage,
    *,
    run_id: Optional[str],
    run_step: Optional[int] = None,
    environment_id: Optional[str] = None,
) -> Tuple[Table, Optional[Text]]:
    """One row per episode.

    The title names only the filters that were passed. A run or environment
    that every row shares leaves its column for a line under the table, so the
    table keeps its width for what differs between episodes.
    """
    several = len(page.items) > 1
    run = shared_value(e.run_id for e in page.items) if several and run_id is None else None
    environment = (
        shared_value(e.environment_id for e in page.items)
        if several and environment_id is None
        else None
    )
    title = " · ".join(
        part
        for part in (
            "Episodes",
            run_id and f"run {run_id}",
            run_step is not None and f"step {run_step}",
            environment_id and f"environment {environment_id}",
        )
        if part
    )
    note = page_note(
        len(page.items), (run and f"run {run}", environment and f"environment {environment}")
    )
    show_run = run_id is None and run is None
    show_environment = environment_id is None and environment is None
    table = Table(title=Text(title), title_justify="left")
    table.add_column("Episode ID", style="cyan", no_wrap=True)
    if show_run:
        table.add_column("Run", style="green", no_wrap=True)
    if show_environment:
        table.add_column("Environment", overflow="ellipsis")
    table.add_column("Outcome", no_wrap=True)
    # The type alone keeps every row on one line; `get` shows the message.
    table.add_column("Error", no_wrap=True)
    table.add_column("Created", no_wrap=True)
    for episode in page.items:
        row = [Text(episode.episode_id)]
        if show_run:
            row.append(Text(episode.run_id or "-"))
        if show_environment:
            row.append(Text(episode.environment_id or "-"))
        row += [
            Text(episode.outcome or "-", style="red" if episode.has_error else ""),
            Text(
                episode.error.type or ("yes" if episode.has_error else "-"),
                style="red" if episode.has_error else "",
            ),
            Text(format_time_ago(episode.created_at)),
        ]
        table.add_row(*row)
    return table, note


# ---------------------------------------------------------------------------
# prime traces get
# ---------------------------------------------------------------------------


def summary_view(summary: TraceSummary) -> List[RenderableType]:
    """The fixed-field summary: one grid, an optional metrics table, and a hint."""
    extra: Dict[str, Any] = summary.model_extra or {}
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="cyan", no_wrap=True)
    grid.add_column()

    def row(label: str, *parts: Tuple[str, str]) -> None:
        text = Text()
        for value, style in parts:
            text.append(value, style=style)
        grid.add_row(label, text)

    context = dict(summary.context)
    run_type = context.pop("run_type", None)
    run = [(summary.run_id or "-", "")]
    if run_type:
        run += [("  ·  ", "dim"), (run_type, "")]
    if extra.get("run_step") is not None:
        run += [("  ·  step ", "dim"), (str(extra["run_step"]), "")]
    row("trace", (summary.trace_id, "bold"))
    row("run", *run)
    if summary.episode_id:
        row("episode", (summary.episode_id, ""))
    row("environment", (summary.environment_id or "-", ""))
    row("task", (summary.task_id or "-", ""))
    model = "/".join(p for p in (summary.model.provider, summary.model.id) if p) or "-"
    row("model", (model, ""), ("  agent ", "dim"), (summary.agent_name or "-", ""))

    outcome_style = "red" if summary.execution.has_error else "green"
    reward = summary.score.reward
    row(
        "outcome",
        (summary.score.outcome or "-", outcome_style),
        ("  reward ", "dim"),
        ("-" if reward is None else f"{reward:g}", ""),
        ("  error ", "dim"),
        (
            "yes" if summary.execution.has_error else "no",
            "red" if summary.execution.has_error else "",
        ),
        ("  truncated ", "dim"),
        ("yes" if summary.execution.is_truncated else "no", ""),
    )
    row(
        "duration",
        (_duration(summary.duration_ms / 1000), ""),
        ("  created ", "dim"),
        (summary.created_at.strftime(TIME_FORMAT), ""),
        ("  ingested ", "dim"),
        (summary.ingested_at.strftime(TIME_FORMAT), ""),
    )
    row(
        "tokens",
        (f"{summary.total_tokens:,}", ""),
        ("  size ", "dim"),
        (_size(summary.size_bytes), ""),
    )

    activity = extra.get("activity")
    if isinstance(activity, dict):
        counts = [
            (activity.get("model_turns"), "model turns"),
            (activity.get("tool_calls"), "tool calls"),
            (activity.get("user_messages"), "user messages"),
            (activity.get("total_entries"), "entries"),
        ]
        shown = [f"{n:,} {label}" for n, label in counts if isinstance(n, int)]
        if shown:
            row("activity", (" · ".join(shown), ""))

    tools = extra.get("tool_definitions")
    if isinstance(tools, list) and tools:
        names = [_tool_name(t) for t in tools]
        listed: List[str] = []
        for name in names:
            if listed and len(", ".join(listed + [name])) > TOOL_NAMES_CHARS:
                break
            listed.append(name)
        rest = len(names) - len(listed)
        row(
            "tools",
            (f"{len(names)}  ", ""),
            (", ".join(listed), ""),
            (f"  +{rest} more" if rest else "", "dim"),
        )
    if context:
        row("context", (", ".join(f"{k}={v}" for k, v in context.items()), ""))

    out: List[RenderableType] = [grid]
    metrics = extra.get("metrics")
    if isinstance(metrics, dict) and metrics:
        table = Table(box=None, padding=(0, 2), header_style="dim", title_justify="left")
        table.add_column("metric")
        table.add_column("value", justify="right")
        for key, value in metrics.items():
            shown = f"{value:g}" if isinstance(value, (int, float)) else str(value)
            table.add_row(Text(str(key)), Text(shown))
        out += [Text(""), Text("metrics", style="bold"), table]
    out.append(Text(""))
    out.append(
        Text.assemble(
            ("Full document: ", "dim"),
            ("prime traces get ", "dim"),
            (summary.trace_id, "dim"),
            (" --raw", "dim"),
        )
    )
    return out


def _tool_name(tool: Any) -> str:
    if not isinstance(tool, dict):
        return "?"
    function = tool.get("function") if isinstance(tool.get("function"), dict) else {}
    return str(tool.get("name") or function.get("name") or "?")


def episode_view(detail: EpisodeDetail, members: TraceListPage) -> List[RenderableType]:
    """The episode's own fields, its member aggregate, and its newest member traces."""
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="cyan", no_wrap=True)
    grid.add_column()

    def row(label: str, *parts: Tuple[str, str]) -> None:
        grid.add_row(label, Text.assemble(*parts))

    aggregate = detail.traces
    row("episode", (detail.episode_id, "bold"))
    row("run", (detail.run_id or "-", ""))
    row("environment", (detail.environment_id or "-", ""))
    # Members of one episode normally share a task; show it only when every one
    # of them records the same task.
    task = shared_value(t.task_id for t in members.items)
    if task is not None and not members.next_cursor:
        row("task", (task, ""))
    # The episode's own error and its traces' errors are separate: an environment
    # hook can fail after every trace succeeded, so both are always shown.
    row(
        "outcome",
        (detail.outcome or "-", "red" if detail.has_error else "green"),
        ("  episode error ", "dim"),
        ("yes" if detail.has_error else "no", "red" if detail.has_error else ""),
        ("  trace errors ", "dim"),
        (
            "yes" if aggregate.any_trace_error else "no",
            "red" if aggregate.any_trace_error else "",
        ),
    )
    error = _error_text(detail.error)
    if error:
        row("error", (error, "red"))
    row(
        "traces",
        (f"{aggregate.trace_count:,}", ""),
        ("  tokens ", "dim"),
        (f"{aggregate.total_tokens:,}", ""),
        ("  duration ", "dim"),
        (_duration(aggregate.total_duration_ms / 1000), ""),
    )
    if aggregate.agent_names:
        row("agents", (", ".join(aggregate.agent_names), ""))
    row("created", (detail.created_at.strftime(TIME_FORMAT), ""))

    out: List[RenderableType] = [grid, Text("")]
    if not members.items:
        out.append(Text("No traces were recorded for this episode.", style="dim"))
        return out

    # Oldest first, so the table reads in the order the agents ran.
    out.append(members_table(members.items[::-1]))
    shown = len(members.items)
    if members.next_cursor:
        out.append(
            Text.assemble(
                (f"Showing the newest {shown} of {aggregate.trace_count:,} traces. All: ", "dim"),
                ("prime traces list --episode-id ", "dim"),
                (detail.episode_id, "dim"),
            )
        )
    out.append(Text("Trace details: prime traces get <trace_id>", style="dim"))
    return out


def members_table(members: List[TraceSummary]) -> Table:
    table = Table(title="Traces", title_justify="left")
    table.add_column("Trace ID", style="cyan", no_wrap=True)
    table.add_column("Agent")
    table.add_column("Reward", justify="right")
    table.add_column("Outcome")
    table.add_column("Turns", justify="right")
    table.add_column("Tokens", justify="right")
    table.add_column("Duration", justify="right", no_wrap=True)
    for trace in members:
        reward = trace.score.reward
        activity = (trace.model_extra or {}).get("activity")
        turns = activity.get("model_turns") if isinstance(activity, dict) else None
        error = trace.execution.has_error
        table.add_row(
            Text(trace.trace_id),
            Text(trace.agent_name or "-"),
            Text("-" if reward is None else f"{reward:.2f}"),
            Text(trace.score.outcome or ("error" if error else "-"), style="red" if error else ""),
            Text(f"{turns:,}" if isinstance(turns, int) else "-"),
            Text(f"{trace.total_tokens:,}"),
            Text(_duration(trace.duration_ms / 1000)),
        )
    return table
