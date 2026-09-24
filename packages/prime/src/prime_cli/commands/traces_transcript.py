"""Rendering for `prime traces get` and `prime traces transcript`.

Pure functions over SDK models and plain dicts, so the command module keeps
the I/O (fetching, fallbacks, paging) and the tests can render without a
network. Every producer-written string is wrapped in `Text` rather than
interpolated into markup, so trace content is always literal.
"""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

from prime_traces import TraceSummary
from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

ROLE_STYLES = {"system": "dim", "user": "blue", "assistant": "magenta", "tool": "green"}

# Truncation budgets for the default (non --full) view.
USER_CHARS = 400
ASSISTANT_CHARS = 1200
THINKING_CHARS = 200
TOOL_LINES = 4
ARG_CHARS = 150
TOOL_NAME_WIDTH = 32

# Argument keys that best describe a tool call in one line, most specific first.
ARG_SUMMARY_KEYS = ("command", "file_path", "path", "pattern", "query", "url", "description")


@dataclass
class Transcript:
    """A trace's nodes and calls, from the index or the raw document.

    Nodes and calls are plain dicts in the index's response shape, so both
    sources render through one path. Raw-document calls also carry `usage`.
    """

    trace_id: str
    source: str  # "index" or "document"
    nodes: List[Dict[str, Any]]
    calls: List[Dict[str, Any]]
    turn_of: Dict[int, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        turn = 0
        for node in self.nodes:
            if _role(node) == "assistant":
                turn += 1
                self.turn_of[node["node_idx"]] = turn


def transcript_from_document(trace_id: str, document: Any) -> Transcript:
    """Build a transcript from a raw trace document (`nodes` + `calls` arrays)."""
    if not isinstance(document, dict) or not isinstance(document.get("nodes"), list):
        raise ValueError("the stored document has no `nodes` array to render")
    nodes = []
    for idx, raw in enumerate(document["nodes"]):
        raw = raw if isinstance(raw, dict) else {}
        parents = raw.get("semantic_parents") or []
        nodes.append(
            {
                "node_idx": idx,
                "parent_idx": parents[0] if parents and isinstance(parents[0], int) else None,
                "timestamp": raw.get("timestamp"),
                "sampled": bool(raw.get("sampled")),
                "message": raw.get("message") if isinstance(raw.get("message"), dict) else {},
            }
        )
    calls = []
    for idx, raw in enumerate(document.get("calls") or []):
        if not isinstance(raw, dict):
            continue
        time = raw.get("time") if isinstance(raw.get("time"), dict) else {}
        calls.append(
            {
                "call_idx": idx,
                "node_idx": raw.get("node"),
                "time_start": time.get("start"),
                "time_end": time.get("end"),
                "model": raw.get("model"),
                "endpoint": raw.get("endpoint"),
                "finish_reason": raw.get("finish_reason"),
                "usage": raw.get("usage") if isinstance(raw.get("usage"), dict) else None,
            }
        )
    return Transcript(trace_id=trace_id, source="document", nodes=nodes, calls=calls)


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def parse_node_range(value: str) -> Tuple[Optional[int], Optional[int]]:
    """`12` -> (12, 12); `30:` -> (30, None); `:9` -> (None, 9); `5:9` -> (5, 9).

    Both ends are inclusive, because they name node indexes, not offsets.
    """
    start, sep, end = value.partition(":")
    try:
        lo = int(start) if start.strip() else None
        hi = int(end) if end.strip() else None
    except ValueError:
        raise ValueError(f"invalid node range {value!r}; use N, N:, :M or N:M") from None
    if not sep:
        hi = lo
    if lo is None and hi is None and not sep:
        raise ValueError(f"invalid node range {value!r}; use N, N:, :M or N:M")
    if (lo is not None and lo < 0) or (hi is not None and hi < 0):
        raise ValueError("node indexes cannot be negative")
    if lo is not None and hi is not None and lo > hi:
        raise ValueError(f"invalid node range {value!r}; the start is after the end")
    return lo, hi


def select_nodes(
    transcript: Transcript,
    *,
    node_range: Optional[Tuple[Optional[int], Optional[int]]] = None,
    turn: Optional[int] = None,
    roles: Optional[Iterable[str]] = None,
) -> List[Dict[str, Any]]:
    """Filter nodes by index range, one model turn (plus what follows it), and role."""
    nodes = transcript.nodes
    if node_range is not None:
        lo, hi = node_range
        nodes = [
            n
            for n in nodes
            if (lo is None or n["node_idx"] >= lo) and (hi is None or n["node_idx"] <= hi)
        ]
    if turn is not None:
        selected, inside = [], False
        for n in nodes:
            if _role(n) == "assistant":
                inside = transcript.turn_of.get(n["node_idx"]) == turn
            if inside:
                selected.append(n)
        nodes = selected
    if roles:
        wanted = set(roles)
        nodes = [n for n in nodes if _role(n) in wanted]
    return nodes


# ---------------------------------------------------------------------------
# Message helpers
# ---------------------------------------------------------------------------


def _role(node: Dict[str, Any]) -> str:
    return str((node.get("message") or {}).get("role") or "?")


def message_text(content: Any) -> str:
    """Flatten producer content (a string or a list of parts) to text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                text = part.get("text")
                parts.append(text if isinstance(text, str) else f"<{part.get('type', 'part')}>")
        return "\n".join(parts)
    return json.dumps(content, ensure_ascii=False)


def tool_call_parts(call: Any) -> Tuple[str, str]:
    """(name, one-line argument summary) for OpenAI- or verifiers-shaped calls."""
    if not isinstance(call, dict):
        return "?", ""
    function = call.get("function") if isinstance(call.get("function"), dict) else {}
    name = call.get("name") or function.get("name") or "?"
    args = call.get("arguments", function.get("arguments"))
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            return str(name), _one_line(args)
    if isinstance(args, dict):
        for key in ARG_SUMMARY_KEYS:
            if isinstance(args.get(key), str) and args[key]:
                return str(name), _one_line(args[key])
    return str(name), _one_line(json.dumps(args, ensure_ascii=False) if args else "")


def _one_line(value: str) -> str:
    return " ".join(value.split())


def _clip(value: str, limit: Optional[int]) -> str:
    if limit is None or len(value) <= limit:
        return value
    return f"{value[:limit]} … [{len(value) - limit:,} more chars]"


def _looks_like_error(text: str) -> bool:
    # The index has no error flag for tool results; tool runners conventionally
    # prefix failures with "Error", so this only picks the highlight color.
    return text.lstrip().lower().startswith("error")


def _tool_calls(node: Dict[str, Any]) -> List[Any]:
    calls = (node.get("message") or {}).get("tool_calls")
    return calls if isinstance(calls, list) else []


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


def _span(call: Dict[str, Any]) -> Optional[float]:
    start, end = call.get("time_start"), call.get("time_end")
    if isinstance(start, (int, float)) and isinstance(end, (int, float)) and end >= start:
        return end - start
    return None


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
        (summary.created_at.strftime("%Y-%m-%d %H:%M:%SZ"), ""),
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
        row("tools", (f"{len(tools)} defined", ""), ("  (--tools to list)", "dim"))
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
            ("Transcript: ", "dim"), ("prime traces transcript ", "dim"), (summary.trace_id, "dim")
        )
    )
    return out


def tools_view(summary: TraceSummary) -> Table:
    """Tool names with the first line of each description."""
    tools = (summary.model_extra or {}).get("tool_definitions")
    table = Table(title=Text(f"Tools for {summary.trace_id}"), title_justify="left")
    table.add_column("Tool", style="yellow", no_wrap=True)
    table.add_column("Description")
    for tool in tools if isinstance(tools, list) else []:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function") if isinstance(tool.get("function"), dict) else {}
        name = tool.get("name") or function.get("name") or "?"
        description = tool.get("description") or function.get("description") or ""
        first = next((line for line in str(description).splitlines() if line.strip()), "")
        table.add_row(Text(str(name)), Text(_clip(first.strip(), 160)))
    return table


# ---------------------------------------------------------------------------
# prime traces transcript
# ---------------------------------------------------------------------------


def transcript_header(summary: TraceSummary, transcript: Transcript) -> RenderableType:
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="cyan", no_wrap=True)
    grid.add_column()
    model = "/".join(p for p in (summary.model.provider, summary.model.id) if p) or "-"
    outcome_style = "red" if summary.execution.has_error else "green"
    tool_results = sum(1 for n in transcript.nodes if _role(n) == "tool")
    grid.add_row("task", Text(summary.task_id or "-"))
    grid.add_row("model", Text(model))
    grid.add_row(
        "outcome",
        Text.assemble(
            (summary.score.outcome or "-", outcome_style),
            ("  duration ", "dim"),
            (_duration(summary.duration_ms / 1000), ""),
        ),
    )
    grid.add_row(
        "turns",
        Text(
            f"{len(transcript.turn_of)} model turns · {tool_results} tool results"
            f" · {len(transcript.nodes)} nodes"
        ),
    )
    return Panel(
        grid, title=Text(f"trace {summary.trace_id}"), title_align="left", border_style="cyan"
    )


def transcript_lines(
    transcript: Transcript,
    nodes: List[Dict[str, Any]],
    *,
    full: bool = False,
    show_system: bool = False,
    plain: bool = False,
) -> List[RenderableType]:
    """The conversation, one block per node."""
    limit = (lambda n: None) if full else (lambda n: n)  # noqa: E731
    calls_by_node = {c.get("node_idx"): c for c in transcript.calls}
    stamps = [
        n["timestamp"] for n in transcript.nodes if isinstance(n.get("timestamp"), (int, float))
    ]
    origin = min(stamps) if stamps else None
    out: List[RenderableType] = []

    for node in nodes:
        idx, message, role = node["node_idx"], node.get("message") or {}, _role(node)
        ts = node.get("timestamp")
        at = f"+{ts - origin:.0f}s" if origin is not None and isinstance(ts, (int, float)) else ""
        where = f"#{idx} {at}".strip()

        if role == "system":
            body = message_text(message.get("content"))
            if not show_system:
                out.append(
                    Text(
                        f"{where}  system prompt · {len(body):,} chars · hidden (--system to show)",
                        "dim",
                    )
                )
                continue
            out.append(_block(f"system {where}", body, "dim", plain))
        elif role == "user":
            body = _clip(message_text(message.get("content")), limit(USER_CHARS))
            out.append(_block(f"user {where}", body, "blue", plain))
        elif role == "assistant":
            out.append(_assistant_header(transcript, node, where, calls_by_node.get(idx)))
            thinking = message.get("reasoning_content")
            if isinstance(thinking, str) and thinking.strip():
                shown = thinking if full else _one_line(thinking)
                out.append(
                    Text.assemble(
                        ("  thinking  ", "dim"), (_clip(shown, limit(THINKING_CHARS)), "dim italic")
                    )
                )
            content = message_text(message.get("content"))
            if content.strip():
                out.append(Text(_indent(_clip(content, limit(ASSISTANT_CHARS)), "  ")))
            for call in _tool_calls(node):
                name, summary = tool_call_parts(call)
                out.append(
                    Text.assemble(
                        ("  → ", "yellow"),
                        (name, "bold yellow"),
                        ("  " + _clip(summary, limit(ARG_CHARS)), ""),
                    )
                )
        elif role == "tool":
            body = message_text(message.get("content"))
            lines = body.splitlines() or [""]
            shown = lines if full else lines[:TOOL_LINES]
            text = "\n".join(shown)
            if len(lines) > len(shown):
                text += f"\n… {len(lines) - len(shown)} more lines (--full, or --node {idx})"
            style = "red" if _looks_like_error(body) else "green"
            out.append(Text("    ⎿ " + _indent(text, "      "), style=style))
        else:
            body = _clip(message_text(message.get("content")), limit(USER_CHARS))
            out.append(_block(f"{role} {where}", body, "", plain))
    return out


def _assistant_header(
    transcript: Transcript, node: Dict[str, Any], where: str, call: Optional[Dict[str, Any]]
) -> Text:
    turn = transcript.turn_of.get(node["node_idx"])
    text = Text.assemble((f"● turn {turn}", "bold magenta"), (f"  {where}", "dim"))
    meta = []
    if call:
        span = _span(call)
        if span is not None:
            meta.append(f"{span:.1f}s")
        usage = call.get("usage") or {}
        if isinstance(usage.get("completion_tokens"), int):
            meta.append(f"{usage['completion_tokens']:,} tok out")
        if call.get("finish_reason"):
            meta.append(str(call["finish_reason"]))
    if meta:
        text.append("  · " + " · ".join(meta), style="dim")
    return text


def _block(title: str, body: str, style: str, plain: bool) -> RenderableType:
    if plain:
        return Group(Text(title, style="bold"), Text(_indent(body, "  ")))
    return Panel(Text(body), title=Text(title, style=style), title_align="left", border_style=style)


def _indent(text: str, prefix: str) -> str:
    return text.replace("\n", "\n" + prefix)


def tools_only_table(
    transcript: Transcript, nodes: List[Dict[str, Any]], width: int = 100
) -> Table:
    """One row per tool call: turn, tool, input, and the size of its result.

    Turn, node, tool and result have bounded widths; the input gets the rest
    of `width` and is cut with an ellipsis, so every row stays one line.
    """
    results = {
        (n.get("message") or {}).get("tool_call_id"): n
        for n in transcript.nodes
        if _role(n) == "tool"
    }
    rows = []
    for node in nodes:
        if _role(node) != "assistant":
            continue
        for call in _tool_calls(node):
            name, summary = tool_call_parts(call)
            call_id = call.get("id") if isinstance(call, dict) else None
            result = results.get(call_id) if call_id else None
            if result is None:
                outcome = Text("no result", style="dim")
            else:
                body = message_text((result.get("message") or {}).get("content"))
                lines = len(body.splitlines())
                outcome = (
                    Text("error", style="red")
                    if _looks_like_error(body)
                    else Text(f"{lines:,} line{'s' if lines != 1 else ''}", style="green")
                )
            turn = str(transcript.turn_of.get(node["node_idx"], ""))
            rows.append((turn, str(node["node_idx"]), Text(name), Text(summary), outcome))

    tool_width = min(TOOL_NAME_WIDTH, max((len(r[2]) for r in rows), default=4))
    # turn + node + result columns, plus one space of padding either side of five columns.
    fixed = 4 + 5 + 10 + 10
    table = Table(box=None, padding=(0, 1), header_style="dim")
    table.add_column("turn", justify="right", no_wrap=True)
    table.add_column("node", justify="right", style="dim", no_wrap=True)
    table.add_column(
        "tool", style="bold yellow", no_wrap=True, width=tool_width, overflow="ellipsis"
    )
    table.add_column(
        "input", no_wrap=True, overflow="ellipsis", max_width=max(16, width - fixed - tool_width)
    )
    table.add_column("result", no_wrap=True)
    for row in rows:
        table.add_row(*row)
    return table


def calls_table(transcript: Transcript) -> Table:
    """One row per model call: timing, a latency bar, finish reason, and usage when known."""
    calls = transcript.calls
    has_usage = any(isinstance(c.get("usage"), dict) for c in calls)
    models = {c.get("model") for c in calls if c.get("model")}
    starts = [c["time_start"] for c in calls if isinstance(c.get("time_start"), (int, float))]
    origin = min(starts) if starts else None
    spans = [s for s in (_span(c) for c in calls) if s is not None]
    longest = max(spans) if spans else 0

    table = Table(box=None, padding=(0, 1), header_style="dim")
    table.add_column("turn", justify="right", no_wrap=True)
    table.add_column("node", justify="right", style="dim", no_wrap=True)
    table.add_column("start", justify="right", no_wrap=True)
    table.add_column("time", justify="right", no_wrap=True)
    table.add_column("", no_wrap=True)
    if len(models) > 1:
        table.add_column("model", no_wrap=True)
    if has_usage:
        for label in ("in", "cached", "out", "think"):
            table.add_column(label, justify="right", no_wrap=True)
    table.add_column("finish", no_wrap=True)

    for call in calls:
        span = _span(call)
        start = call.get("time_start")
        bar = "█" * max(1, round(span / longest * 24)) if span and longest else ""
        node_idx = call.get("node_idx")
        cells: List[Any] = [
            str(transcript.turn_of.get(node_idx, "")) if isinstance(node_idx, int) else "",
            "" if node_idx is None else str(node_idx),
            f"+{start - origin:.0f}s"
            if origin is not None and isinstance(start, (int, float))
            else "",
            "" if span is None else f"{span:.1f}s",
            Text(bar, style="magenta"),
        ]
        if len(models) > 1:
            cells.append(Text(str(call.get("model") or "")))
        if has_usage:
            usage = call.get("usage") or {}
            for key in (
                "prompt_tokens",
                "cached_input_tokens",
                "completion_tokens",
                "reasoning_tokens",
            ):
                value = usage.get(key)
                cells.append(f"{value:,}" if isinstance(value, int) else "")
        finish = str(call.get("finish_reason") or "")
        cells.append(Text(finish, style="green" if finish == "stop" else "yellow"))
        table.add_row(*cells)
    return table
