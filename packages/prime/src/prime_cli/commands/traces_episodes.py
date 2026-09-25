"""Rendering for `prime traces episodes list` and `prime traces episodes get`.

Pure functions over SDK models, so the command module keeps the I/O and the
tests can render without a network. Every producer-written string is wrapped
in `Text` rather than interpolated into markup, so it is always literal.
"""

from typing import Iterable, List, Optional, Tuple

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

from .traces_transcript import _duration

TIME_FORMAT = "%Y-%m-%d %H:%M:%SZ"


def _error_text(error: EpisodeError) -> str:
    return ": ".join(part for part in (error.type, error.message) if part)


def shared_value(values: Iterable[Optional[str]]) -> Optional[str]:
    """The one value every row has, or None when the rows differ or lack it."""
    distinct = set(values)
    return distinct.pop() if len(distinct) == 1 else None


def episodes_table(page: EpisodeListPage, *, run_id: Optional[str]) -> Table:
    """One row per episode.

    A run or environment that every row shares moves from its column into the
    title, so the table keeps its width for what differs between episodes.
    """
    run = run_id or shared_value(e.run_id for e in page.items)
    environment = shared_value(e.environment_id for e in page.items)
    title = " · ".join(part for part in ("Episodes", run and f"run {run}", environment) if part)
    table = Table(title=Text(title), title_justify="left")
    table.add_column("Episode ID", style="cyan", no_wrap=True)
    if run is None:
        table.add_column("Run", style="green", no_wrap=True)
    if environment is None:
        table.add_column("Environment", overflow="ellipsis")
    table.add_column("Outcome", no_wrap=True)
    # The type alone keeps every row on one line; `get` shows the message.
    table.add_column("Error", no_wrap=True)
    table.add_column("Created", no_wrap=True)
    for episode in page.items:
        row = [Text(episode.episode_id)]
        if run is None:
            row.append(Text(episode.run_id or "-"))
        if environment is None:
            row.append(Text(episode.environment_id or "-"))
        row += [
            Text(episode.outcome or "-", style="red" if episode.has_error else ""),
            Text(
                episode.error.type or ("yes" if episode.has_error else "-"),
                style="red" if episode.has_error else "",
            ),
            Text(episode.created_at.strftime(TIME_FORMAT)),
        ]
        table.add_row(*row)
    return table


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
                ("prime traces list --episode ", "dim"),
                (detail.episode_id, "dim"),
            )
        )
    out.append(Text("Transcript: prime traces transcript <trace_id>", style="dim"))
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
