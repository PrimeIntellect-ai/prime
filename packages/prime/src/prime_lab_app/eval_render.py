"""Renderable helpers for local eval rollout viewing."""

from __future__ import annotations

import ast
import json
from collections import defaultdict
from typing import Any

from rich import box
from rich.console import Group
from rich.table import Table
from rich.text import Text

from .eval_records import (
    LocalEvalRun,
    MetricSummary,
    RunOverviewStats,
    legacy_row,
    parse_log_header,
)
from .palette import (
    STATUS_ERROR,
    STATUS_INFO,
    STATUS_ROLLOUT_SUCCESS,
    STATUS_ROLLOUT_WARNING,
    STATUS_SUCCESS,
    STATUS_WARNING,
)

IGNORED_ENV_SETTING_KEYS = {"version", "version_id", "versionId", "visibility"}


def format_numeric(value: float | int | str) -> str:
    if isinstance(value, float):
        if value == int(value):
            return str(int(value))
        if abs(value) < 0.01:
            return f"{value:.4f}"
        return f"{value:.3f}"
    return str(value)


def stringify_message_content(content: Any) -> str:
    """Render message content into readable plain text."""

    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, dict):
                item_type = item.get("type")
                if item_type == "text":
                    chunks.append(str(item.get("text", "")))
                elif item_type in {"input_audio", "audio"}:
                    chunks.append("[audio]")
                elif item_type in {"image", "image_url"}:
                    chunks.append("[image]")
                elif item_type in {"thinking", "redacted_thinking"}:
                    continue
                else:
                    chunks.append(pretty_json_or_str(item))
            else:
                chunks.append(str(item))
        return "\n".join(chunk for chunk in chunks if chunk)
    if isinstance(content, dict):
        return pretty_json_or_str(content)
    return str(content)


def stringify_message_reasoning(message: Any) -> str:
    if not isinstance(message, dict):
        return ""

    parts: list[str] = []

    def add_part(value: str) -> None:
        text = value.strip()
        if text and text not in parts:
            parts.append(text)

    reasoning_content = message.get("reasoning_content")
    if isinstance(reasoning_content, str):
        add_part(reasoning_content)

    thinking_blocks = message.get("thinking_blocks")
    if isinstance(thinking_blocks, list):
        for block in thinking_blocks:
            add_part(_thinking_block_to_text(block))

    content = message.get("content")
    if isinstance(content, list):
        for item in content:
            add_part(_thinking_block_to_text(item))

    return "\n\n".join(parts)


def stringify_message(message: Any) -> str:
    if not isinstance(message, dict):
        return stringify_message_content(message)

    content = stringify_message_content(message.get("content", ""))
    reasoning = stringify_message_reasoning(message)
    if reasoning and content:
        return f"Reasoning\n{reasoning}\n\n{content}"
    if reasoning:
        return f"Reasoning\n{reasoning}"
    return content


def parse_tool_calls(tool_calls: Any) -> list[Any]:
    parsed = parse_jsonish_string(tool_calls)
    if isinstance(parsed, dict):
        return [parsed]
    if not isinstance(parsed, list):
        return []
    return [parse_jsonish_string(tool_call) for tool_call in parsed]


def tool_call_parts(tool_call: Any) -> tuple[str, str, str | None]:
    if not isinstance(tool_call, dict):
        return str(tool_call), "", None

    function = tool_call.get("function")
    payload = function if isinstance(function, dict) else tool_call
    name = str(payload.get("name") or tool_call.get("name") or "")
    raw_arguments = payload.get("arguments", tool_call.get("arguments", ""))
    parsed_arguments = parse_jsonish_string(raw_arguments)
    if isinstance(parsed_arguments, dict):
        arguments = (
            str(parsed_arguments["code"])
            if set(parsed_arguments.keys()) == {"code"}
            else pretty_json_or_str(parsed_arguments)
        )
    elif isinstance(parsed_arguments, list):
        arguments = pretty_json_or_str(parsed_arguments)
    else:
        arguments = str(raw_arguments) if raw_arguments not in (None, "") else ""
    call_id = tool_call.get("id")
    return name, arguments, str(call_id) if call_id not in (None, "") else None


def format_prompt_or_completion(prompt_or_completion: Any) -> Text:
    out = Text()
    if not isinstance(prompt_or_completion, list):
        out.append(str(prompt_or_completion))
        return out

    for message in prompt_or_completion:
        if not isinstance(message, dict):
            out.append(str(message))
            out.append("\n\n")
            continue
        role = str(message.get("role", ""))
        content = stringify_message_content(message.get("content", ""))
        reasoning = stringify_message_reasoning(message)
        if role == "assistant":
            out.append("assistant: ", style="bold")
        elif role == "tool":
            out.append("tool result: ", style="bold dim")
        else:
            out.append(f"{role}: ", style="bold dim")
        if reasoning:
            out.append("\n")
            out.append("reasoning:\n", style="dim")
            out.append(reasoning, style="dim")
            out.append("\n")
            if content:
                out.append("\n")
        out.append(content)
        out.append("\n")

        for tool_call in parse_tool_calls(message.get("tool_calls")):
            name, arguments, _ = tool_call_parts(tool_call)
            out.append("\ntool call: ", style="bold")
            out.append(name)
            out.append("\n")
            out.append(arguments)
            out.append("\n")

        out.append("\n")

    return out


def format_message_preview(message: Any) -> str:
    if not isinstance(message, dict):
        return ""
    content = stringify_message_content(message.get("content", ""))
    reasoning = stringify_message_reasoning(message)
    tool_calls = parse_tool_calls(message.get("tool_calls"))
    if content:
        return truncate_preview(content, 56)
    if reasoning:
        return f"reasoning: {truncate_preview(reasoning, 45)}"
    if tool_calls:
        first = tool_calls[0]
        if isinstance(first, dict):
            function = first.get("function", {})
            name = function.get("name") if isinstance(function, dict) else None
            name = name or first.get("name") or ""
            return f"calls {name}" if name else ""
        return f"calls {truncate_preview(str(first), 48)}"
    return ""


def record_preview(record: dict[str, Any]) -> str:
    completion = record.get("completion")
    if isinstance(completion, list) and completion:
        for group in reversed(history_groups(completion)):
            message = group["message"]
            if group.get("kind") == "assistant-tools":
                preview = tool_group_preview(message, group["tool_outputs"])
            else:
                preview = format_message_preview(message)
            if preview:
                return preview
    completion_preview = raw_preview(completion, limit=56)
    if completion_preview:
        return completion_preview

    error = error_preview(record.get("error"))
    if error:
        return error

    prompt = record.get("prompt")
    if isinstance(prompt, list) and prompt:
        if isinstance(prompt[-1], dict):
            preview = format_message_preview(prompt[-1])
            if preview:
                return preview
        prompt_preview = raw_preview(prompt[-1], limit=56)
        if prompt_preview:
            return prompt_preview
    prompt_preview = raw_preview(prompt, limit=56)
    if prompt_preview:
        return prompt_preview
    return ""


def history_groups(completion: list[Any]) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    idx = 0
    while idx < len(completion):
        message = completion[idx]
        if not isinstance(message, dict):
            idx += 1
            continue
        if message.get("role") == "assistant":
            tool_calls = parse_tool_calls(message.get("tool_calls"))
            if tool_calls:
                tool_outputs: list[Any] = []
                next_idx = idx + 1
                while next_idx < len(completion):
                    next_message = completion[next_idx]
                    if not isinstance(next_message, dict) or next_message.get("role") != "tool":
                        break
                    tool_outputs.append(next_message)
                    next_idx += 1
                groups.append(
                    {
                        "kind": "assistant-tools",
                        "message": message,
                        "tool_calls": tool_calls,
                        "tool_outputs": tool_outputs,
                    }
                )
                idx = next_idx
                continue
        groups.append({"kind": "message", "message": message})
        idx += 1
    return groups


def build_rollout_prompt(idx: int, record: dict[str, Any] | None = None) -> Text:
    label = Text()
    label.append(f"#{idx}", style="bold")
    if not record:
        return label

    reward = record.get("reward")
    label.append("  ")
    label.append("reward ", style="dim")
    label.append(format_reward_value(reward), style=reward_style(reward, subdued=True))
    label.append("\n")
    label.append(truncate_preview(record_preview(record), 38), style="dim")
    return label


def build_reward_text(
    record: dict[str, Any],
    *,
    heading: str,
    multiline: bool,
    limit: int | None = None,
) -> Text:
    reward = record.get("reward")
    out = Text()
    out.append(f"{heading}\n", style="bold dim")
    out.append(format_reward_value(reward), style=reward_style(reward))

    breakdown = sorted(extract_numeric_metric_values(record).items())
    if breakdown:
        breakdown = breakdown[:limit] if limit is not None else breakdown
        if multiline:
            out.append("\n\nBreakdown\n", style="bold dim")
            width = max(len(name) for name, _ in breakdown)
            for name, value in breakdown:
                out.append(name.ljust(width), style="bold")
                out.append("  ")
                out.append(format_reward_value(value), style=reward_style(value))
                out.append("\n")
        else:
            out.append("\n")
            for idx, (name, value) in enumerate(breakdown):
                if idx:
                    out.append("   ")
                out.append(f"{name} ", style="bold")
                out.append(format_reward_value(value), style=reward_style(value))
    return out


def build_score_text(record: dict[str, Any]) -> Text:
    out = build_reward_text(record, heading="Reward", multiline=True)
    record_metrics = record.get("metrics")
    if isinstance(record_metrics, dict) and record_metrics:
        out.append("\nRecord metrics\n", style="bold dim")
        for key in sorted(record_metrics.keys()):
            out.append(f"{key}: ", style="bold")
            out.append(format_compact_metric(record_metrics[key]))
            out.append("\n")
    return out


def build_task_text(record: dict[str, Any], metadata: dict[str, Any] | None = None) -> Text:
    out = Text()
    append_context_section(out, "Environment", record.get("env_id"))
    append_context_section(out, "Answer", record.get("answer"))
    append_context_section(out, "Stop condition", record.get("stop_condition"))
    error = record.get("error")
    if error not in (None, ""):
        append_context_section(out, "Error", error)
    info = record.get("info")
    if info not in (None, {}, ""):
        append_context_section(out, "Info", format_info_for_details(info))
    return out


def build_usage_text(record: dict[str, Any]) -> Text:
    out = Text()
    token_usage = record.get("token_usage")
    if isinstance(token_usage, dict):
        usage_lines = []
        for key in (
            "input_tokens",
            "output_tokens",
            "final_input_tokens",
            "final_output_tokens",
        ):
            value = token_usage.get(key)
            if value is not None:
                usage_lines.append(f"{key}: {format_numeric(value)}")
        append_context_section(out, "Tokens", "\n".join(usage_lines))

    timing = record.get("timing")
    if isinstance(timing, dict):
        timing_lines = []
        for key in ("generation_ms", "scoring_ms", "total_ms"):
            value = timing.get(key)
            if value is not None:
                timing_lines.append(f"{key}: {format_compact_metric(value)}")
        append_context_section(out, "Timing", "\n".join(timing_lines))
    return out


def build_state_text(record: dict[str, Any], metadata: dict[str, Any]) -> Text:
    out = Text()

    state_columns = metadata.get("state_columns")
    if isinstance(state_columns, list):
        for column in state_columns:
            if not isinstance(column, str) or not column:
                continue
            value = record.get(column)
            if value in (None, "", {}):
                continue
            append_context_section(out, column, format_info_for_details(value))
    return out


def build_run_summary_text(
    run: LocalEvalRun,
    *,
    record_progress_label: str,
) -> Text:
    metadata = run.load_metadata()
    lines: list[Text] = [Text("Run Summary", style="bold dim")]

    identity = Text()
    identity.append("Environment: ", style="bold")
    identity.append(str(run.env_id))
    identity.append("   ")
    identity.append("Model: ", style="bold")
    identity.append(str(run.model))
    identity.append("   ")
    identity.append("Run ID: ", style="bold")
    identity.append(str(run.run_id))
    lines.append(identity)

    progress = Text()
    progress.append("Record: ", style="bold")
    progress.append(record_progress_label)
    progress.append("   ")
    progress.append("Examples: ", style="bold")
    progress.append(str(metadata.get("num_examples", "")))
    progress.append("   ")
    progress.append("Rollouts/ex: ", style="bold")
    progress.append(str(metadata.get("rollouts_per_example", "")))
    date_text = f"{str(metadata.get('date', ''))} {str(metadata.get('time', ''))}".strip()
    if date_text:
        progress.append("   ")
        progress.append("Date: ", style="bold")
        progress.append(date_text)
    lines.append(progress)

    usage = metadata.get("usage")
    sampling_args = metadata.get("sampling_args", {})
    usage_items: list[tuple[str, str]] = []
    if isinstance(usage, dict):
        input_tok = usage.get("input_tokens")
        output_tok = usage.get("output_tokens")
        if input_tok is not None:
            usage_items.append(("Avg input tokens", format_numeric(input_tok)))
        if output_tok is not None:
            usage_items.append(("Avg output tokens", format_numeric(output_tok)))
    if isinstance(sampling_args, dict):
        for key in ("max_tokens", "temperature"):
            value = sampling_args.get(key)
            if value not in (None, ""):
                usage_items.append((key.replace("_", " ").capitalize(), str(value)))

    if usage_items:
        usage_line = Text()
        for idx, (label, value) in enumerate(usage_items):
            if idx:
                usage_line.append("   ")
            usage_line.append(f"{label}: ", style="bold")
            usage_line.append(value)
        lines.append(usage_line)

    return Text("\n").join(lines)


def build_run_metric_text(run: LocalEvalRun) -> Text:
    metadata = run.load_metadata()
    stats: list[tuple[str, Any]] = []

    pass_at_k = metadata.get("pass_at_k")
    if isinstance(pass_at_k, dict):
        for key in sorted(pass_at_k.keys(), key=int_like_sort_key):
            stats.append((f"pass@{key}", pass_at_k[key]))

    pass_all_k = metadata.get("pass_all_k")
    if isinstance(pass_all_k, dict):
        for key in sorted(pass_all_k.keys(), key=int_like_sort_key):
            stats.append((f"pass-all@{key}", pass_all_k[key]))

    avg_metrics = metadata.get("avg_metrics")
    preferred_metric_keys = [
        ("evaluate_tau2_task", "task"),
        ("num_turns", "turns"),
        ("total_tool_calls", "tools"),
        ("num_steps", "steps"),
        ("num_errors", "errors"),
    ]
    if isinstance(avg_metrics, dict):
        for key, label in preferred_metric_keys:
            if key in avg_metrics:
                stats.append((label, avg_metrics[key]))

    if not stats:
        return Text()

    out = Text()
    out.append("Run Metrics\n", style="bold dim")
    for idx, (label, value) in enumerate(stats[:6]):
        if idx and idx % 3 == 0:
            out.append("\n")
        elif idx:
            out.append("   ")
        out.append(f"{label} ", style="bold")
        out.append(format_compact_metric(value))

    pass_threshold = metadata.get("pass_threshold")
    if pass_threshold not in (None, ""):
        out.append("\n")
        out.append("threshold ", style="bold")
        out.append(format_compact_metric(pass_threshold))
    return out


def compute_run_overview_stats(run: LocalEvalRun) -> RunOverviewStats:
    rewards: list[float] = []
    metric_values: dict[str, list[float]] = defaultdict(list)
    try:
        with (run.path / "results.jsonl").open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(record, dict):
                    continue
                record = legacy_row(record)
                reward = numeric_reward(record.get("reward"))
                if reward is not None:
                    rewards.append(reward)
                for name, value in extract_numeric_metric_values(record).items():
                    metric_values[name].append(value)
    except OSError:
        pass

    return RunOverviewStats(
        rewards=rewards,
        metric_summaries=[
            MetricSummary(
                name=name,
                count=len(values),
                avg=sum(values) / len(values),
                min_value=min(values),
                max_value=max(values),
            )
            for name, values in sorted(metric_values.items())
            if values
        ],
        metric_values=dict(metric_values),
    )


def build_reward_distribution_table(values: list[float], heading: str) -> Group | Text:
    if not values:
        return Text()

    avg_reward = sum(values) / len(values)
    summary = Text()
    summary.append(heading, style="bold dim")
    summary.append("\n")
    summary.append("count ", style="bold")
    summary.append(f"{len(values):,}")
    summary.append("   avg ", style="bold")
    summary.append(f"{avg_reward:.3f}", style=reward_style(avg_reward))
    summary.append("   min ", style="bold")
    summary.append(f"{min(values):.3f}", style=reward_style(min(values)))
    summary.append("   max ", style="bold")
    summary.append(f"{max(values):.3f}", style=reward_style(max(values)))

    bucket_counts = reward_bucket_counts(values)
    peak_count = max(count for _, count, _ in bucket_counts) or 1

    table = Table(
        box=box.SIMPLE_HEAD,
        expand=True,
        show_edge=False,
        pad_edge=False,
        padding=(0, 1),
        collapse_padding=True,
        row_styles=["none", "dim"],
    )
    table.add_column("Range", style="dim", width=10, no_wrap=True)
    table.add_column("Count", justify="right", width=8)
    table.add_column("Share", justify="right", width=8)
    table.add_column("Distribution", ratio=1, min_width=24)

    for label, count, style in bucket_counts:
        share = (count / len(values)) if values else 0.0
        fraction = count / peak_count if peak_count else 0.0
        filled_cells = round(max(0.0, min(1.0, fraction)) * 24)
        bar = Text()
        if filled_cells:
            bar.append("#" * filled_cells, style=style)
        if filled_cells < 24:
            bar.append("." * (24 - filled_cells), style="dim")
        table.add_row(label, f"{count:,}", f"{share:.1%}", bar)

    return Group(summary, table)


def build_metric_summary_table(metric_summaries: list[MetricSummary]) -> Table | Text:
    if not metric_summaries:
        return Text()

    counts = {summary.count for summary in metric_summaries}
    show_count_column = len(counts) > 1
    title_suffix = f" (n={next(iter(counts)):,})" if len(counts) == 1 else ""

    prepared: list[tuple[int, str, str, MetricSummary]] = []
    category_order = {
        "Tokens": 0,
        "Calls": 1,
        "Flow": 2,
        "Errors": 3,
        "Timing": 4,
        "Scores": 5,
        "Other": 6,
    }
    for summary in metric_summaries:
        lowered = summary.name.lower()
        if "token" in lowered:
            category = "Tokens"
        elif "call" in lowered:
            category = "Calls"
        elif "turn" in lowered or "step" in lowered or "batch" in lowered:
            category = "Flow"
        elif "error" in lowered:
            category = "Errors"
        elif "time" in lowered or lowered.endswith("_ms"):
            category = "Timing"
        elif "reward" in lowered or "score" in lowered or "task" in lowered:
            category = "Scores"
        else:
            category = "Other"
        prepared.append(
            (category_order.get(category, 99), summary.name.replace("_", " "), category, summary)
        )

    table = Table(
        title=f"Rollout metrics{title_suffix}",
        title_style="bold dim",
        box=box.SIMPLE_HEAD,
        expand=True,
        show_edge=False,
        pad_edge=False,
        padding=(0, 1),
        collapse_padding=True,
        row_styles=["none", "dim"],
    )
    table.add_column("Metric", style=f"bold {STATUS_INFO}", ratio=1, min_width=24, no_wrap=True)
    table.add_column("Average", justify="right", no_wrap=True)
    table.add_column("Min", justify="right", no_wrap=True)
    table.add_column("Max", justify="right", no_wrap=True)
    if show_count_column:
        table.add_column("N", justify="right", style="dim", no_wrap=True)

    previous_category: str | None = None
    for _, display_name, category, summary in sorted(prepared):
        if previous_category is not None and category != previous_category:
            table.add_section()
        row = [
            display_name,
            format_metric_stat_value(summary.avg),
            format_metric_stat_value(summary.min_value),
            format_metric_stat_value(summary.max_value),
        ]
        if show_count_column:
            row.append(f"{summary.count:,}")
        table.add_row(*row)
        previous_category = category
    return table


def append_styled_log_line(log_text: Text, line: str) -> None:
    parsed = parse_log_header(line)
    if parsed is None:
        log_text.append(line, style="dim")
        return
    timestamp, source, level, message = parsed
    level_style = {
        "DEBUG": f"dim {STATUS_INFO}",
        "INFO": STATUS_SUCCESS,
        "WARNING": STATUS_WARNING,
        "ERROR": STATUS_ERROR,
        "CRITICAL": f"{STATUS_ERROR} reverse",
    }.get(level, "dim")
    log_text.append(timestamp, style="bold dim")
    log_text.append(" - ", style="dim")
    log_text.append(source, style=f"dim {STATUS_INFO}")
    log_text.append(" - ", style="dim")
    log_text.append(level, style=level_style)
    log_text.append(message, style="dim")


def format_info_for_details(info: Any) -> str:
    info_value = parse_jsonish_string(info)
    if isinstance(info_value, (dict, list)):
        return pretty_json_or_str(info_value)
    return str(info_value)


def extract_numeric_metric_values(record: dict[str, Any]) -> dict[str, float]:
    metric_values: dict[str, float] = {}

    metrics = record.get("metrics")
    if isinstance(metrics, dict):
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                metric_values[key] = float(value)

    info = parse_jsonish_string(record.get("info"))
    if isinstance(info, dict):
        reward_signals = info.get("reward_signals")
        if isinstance(reward_signals, dict):
            for key, value in reward_signals.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    metric_values.setdefault(key, float(value))

    for key, value in record.items():
        if key in _STANDARD_NUMERIC_FIELDS:
            continue
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            metric_values.setdefault(key, float(value))

    return metric_values


def parse_jsonish_string(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped:
        return ""
    if stripped[0] not in "[{":
        return value
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(stripped)
        except (SyntaxError, ValueError):
            return value


def normalize_message_sequence(value: Any) -> Any:
    parsed = parse_jsonish_string(value)
    if isinstance(parsed, dict):
        return [normalize_message(parsed)] if parsed.get("role") else parsed
    if not isinstance(parsed, list):
        return parsed
    return [
        normalize_message(message) if isinstance(message, dict) else message for message in parsed
    ]


def normalize_message(message: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(message)
    tool_calls = normalized.get("tool_calls")
    parsed_tool_calls = parse_jsonish_string(tool_calls)
    if isinstance(parsed_tool_calls, dict):
        normalized["tool_calls"] = [parsed_tool_calls]
    elif isinstance(parsed_tool_calls, list):
        normalized["tool_calls"] = [
            parse_jsonish_string(tool_call) for tool_call in parsed_tool_calls
        ]
    return normalized


def normalize_rollout_record(record: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(record)
    for key in ("prompt", "completion", "trajectory"):
        if key in normalized:
            normalized[key] = normalize_message_sequence(normalized[key])
    for key in ("info", "metrics", "timing", "token_usage"):
        if key in normalized:
            normalized[key] = parse_jsonish_string(normalized[key])
    return normalized


def pretty_json_or_str(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, indent=2)
    except (TypeError, ValueError):
        return str(value)


def compact_json_or_str(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        return str(value)


def format_run_datetime(metadata: dict[str, Any]) -> str:
    return f"{metadata.get('date', '')} {metadata.get('time', '')}".strip()


def format_setting_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return format_compact_metric(value)
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        if not value:
            return "[]"
        if all(
            isinstance(item, (str, int, float, bool)) and not isinstance(item, dict)
            for item in value
        ):
            return ", ".join(format_setting_value(item) for item in value)
    return compact_json_or_str(value)


def tool_name(tool: Any) -> str:
    if not isinstance(tool, dict):
        return str(getattr(tool, "name", "") or "")
    function = tool.get("function")
    if isinstance(function, dict):
        name = function.get("name")
        if isinstance(name, str):
            return name
    name = tool.get("name")
    return name if isinstance(name, str) else ""


def run_setting_rows(metadata: dict[str, Any]) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []

    ordered_settings: list[tuple[str, Any]] = []
    if metadata.get("base_url") not in (None, ""):
        ordered_settings.append(("endpoint", metadata["base_url"]))
    if metadata.get("num_examples") not in (None, ""):
        ordered_settings.append(("examples", metadata["num_examples"]))
    if metadata.get("rollouts_per_example") not in (None, ""):
        ordered_settings.append(("rollouts/example", metadata["rollouts_per_example"]))
    if metadata.get("pass_threshold") not in (None, ""):
        ordered_settings.append(("pass threshold", metadata["pass_threshold"]))

    sampling_args = metadata.get("sampling_args")
    if isinstance(sampling_args, dict):
        for key in sorted(sampling_args):
            value = sampling_args[key]
            if value not in (None, ""):
                ordered_settings.append((f"sampling.{key}", value))

    env_args = metadata.get("env_args")
    if isinstance(env_args, dict):
        for key in sorted(env_args):
            if key in IGNORED_ENV_SETTING_KEYS:
                continue
            value = env_args[key]
            if value not in (None, ""):
                ordered_settings.append((f"env.{key}", value))

    state_columns = metadata.get("state_columns")
    if isinstance(state_columns, list) and state_columns:
        ordered_settings.append(("state columns", state_columns))

    tools = metadata.get("tools")
    if isinstance(tools, list):
        tool_names = sorted(name for name in (tool_name(tool) for tool in tools) if name)
        if tool_names:
            ordered_settings.append(("tools", tool_names))

    for label, value in ordered_settings:
        rows.append((label, format_setting_value(value)))

    return rows


def build_settings_table(
    rows: list[tuple[str, str]],
    heading: str,
    *,
    value_header: str = "Value",
) -> Group | Text:
    if not rows:
        return Text()

    title = Text()
    title.append(heading, style="bold dim")

    table = Table(
        box=box.SIMPLE_HEAD,
        expand=True,
        show_edge=False,
        pad_edge=False,
        padding=(0, 1),
        collapse_padding=True,
        row_styles=["none", "dim"],
    )
    table.add_column("Setting", style="dim", width=20, no_wrap=True)
    table.add_column(value_header, ratio=1)

    for setting, value in rows:
        table.add_row(setting, value)

    return Group(title, table)


def run_setting_variation_rows(
    metadatas: list[dict[str, Any]],
    *,
    max_rows: int = 8,
) -> tuple[list[tuple[str, str]], int]:
    if not metadatas:
        return [], 0

    setting_maps = [dict(run_setting_rows(metadata)) for metadata in metadatas]

    ordered_keys: list[str] = []
    for setting_map in setting_maps:
        for key in setting_map:
            if key not in ordered_keys:
                ordered_keys.append(key)

    rows: list[tuple[str, str]] = []
    for key in ordered_keys:
        counts: dict[str, int] = defaultdict(int)
        for setting_map in setting_maps:
            counts[setting_map.get(key, "(unset)")] += 1
        if len(counts) <= 1:
            continue
        parts = [
            f"{value} ({count} run{'s' if count != 1 else ''})"
            for value, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        ]
        rows.append((key, ", ".join(parts)))

    hidden_rows = max(0, len(rows) - max_rows)
    return rows[:max_rows], hidden_rows


def truncate_preview(text: str, limit: int = 72) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 1].rstrip() + "..."


def reward_style(value: Any, *, subdued: bool = False) -> str:
    success_style = STATUS_ROLLOUT_SUCCESS if subdued else STATUS_SUCCESS
    warning_style = STATUS_ROLLOUT_WARNING if subdued else STATUS_WARNING
    if isinstance(value, (int, float)):
        if value >= 0.9:
            return success_style
        if value >= 0.5:
            return warning_style
        return STATUS_ERROR
    return "bold"


def format_reward_value(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.3f}"
    return str(value)


def format_compact_metric(value: Any) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        number = float(value)
        if number.is_integer():
            return str(int(number))
        return f"{number:.2f}".rstrip("0").rstrip(".")
    return str(value)


def numeric_reward(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def format_metric_stat_value(value: float) -> str:
    if float(value).is_integer():
        return f"{int(value):,}"

    magnitude = abs(value)
    if magnitude >= 1000:
        precision = 1
    elif magnitude >= 100:
        precision = 2
    elif magnitude >= 1:
        precision = 3
    elif magnitude >= 0.01:
        precision = 3
    else:
        precision = 4
    return f"{value:,.{precision}f}".rstrip("0").rstrip(".")


def append_context_section(out: Text, title: str, value: Any) -> None:
    if value in (None, "", {}):
        return
    if out.plain:
        out.append("\n\n")
    out.append(f"{title}\n", style="bold dim")
    out.append(str(value))


def tool_output_preview(message: Any) -> str:
    if not isinstance(message, dict):
        return truncate_preview(str(message), 44)
    content = stringify_message(message)
    for line in content.splitlines():
        if line.strip():
            return truncate_preview(line.strip(), 44)
    return truncate_preview(content, 44)


def tool_group_preview(message: Any, tool_outputs: list[Any]) -> str:
    base = format_message_preview(message)
    if not tool_outputs:
        return base
    output_preview = tool_output_preview(tool_outputs[0])
    if not base:
        return output_preview
    return truncate_preview(f"{base} ... {output_preview}", 68)


def raw_preview(value: Any, *, limit: int = 56) -> str:
    if value in (None, "", [], {}):
        return ""
    if isinstance(value, str):
        return truncate_preview(value, limit)
    if isinstance(value, list):
        for item in value:
            preview = raw_preview(item, limit=limit)
            if preview:
                return preview
        return ""
    if isinstance(value, dict):
        content = stringify_message_content(value.get("content", ""))
        if content:
            return truncate_preview(content, limit)
        reasoning = stringify_message_reasoning(value)
        if reasoning:
            return truncate_preview(reasoning, limit)
        for key in ("text", "message", "error", "detail", "details", "type", "name"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return truncate_preview(candidate, limit)
        return ""
    return truncate_preview(str(value), limit)


def error_preview(error: Any) -> str:
    parsed = parse_jsonish_string(error)
    if isinstance(parsed, dict):
        for key in ("error_chain_str", "error", "error_chain_repr"):
            value = parsed.get(key)
            if isinstance(value, str) and value.strip():
                return truncate_preview(value, 56)
    return raw_preview(parsed, limit=56)


def reward_bucket_counts(values: list[float]) -> list[tuple[str, int, str]]:
    bucket_counts = [
        ("<0", 0, STATUS_ERROR),
        ("=0", 0, STATUS_ERROR),
        ("0-<0.25", 0, STATUS_ERROR),
        ("0.25-<0.5", 0, STATUS_WARNING),
        ("0.5-<0.75", 0, STATUS_WARNING),
        ("0.75-<1", 0, STATUS_SUCCESS),
        ("=1", 0, STATUS_SUCCESS),
        (">1", 0, STATUS_SUCCESS),
    ]

    for reward in values:
        if reward < 0:
            bucket_idx = 0
        elif reward == 0:
            bucket_idx = 1
        elif reward < 0.25:
            bucket_idx = 2
        elif reward < 0.5:
            bucket_idx = 3
        elif reward < 0.75:
            bucket_idx = 4
        elif reward < 1.0:
            bucket_idx = 5
        elif reward == 1.0:
            bucket_idx = 6
        else:
            bucket_idx = 7
        label, count, style = bucket_counts[bucket_idx]
        bucket_counts[bucket_idx] = (label, count + 1, style)

    return [
        (label, count, style)
        for label, count, style in bucket_counts
        if not (label in {"<0", ">1"} and count == 0)
    ]


def text_to_plain(text: Text) -> str:
    return text.plain


def indent_block(text: str, prefix: str) -> str:
    return "\n".join(f"{prefix}{line}" if line else prefix.rstrip() for line in text.splitlines())


def int_like_sort_key(value: Any) -> tuple[int, int, str]:
    text = str(value)
    try:
        return (0, int(text), text)
    except (TypeError, ValueError):
        return (1, 0, text)


def _thinking_block_to_text(block: Any) -> str:
    if isinstance(block, dict):
        block_type = block.get("type")
        if block_type == "thinking":
            thinking = block.get("thinking")
            return str(thinking).strip() if thinking else ""
        if block_type == "redacted_thinking":
            return "[reasoning redacted]"
        return ""

    block_type = getattr(block, "type", None)
    if block_type == "thinking":
        thinking = getattr(block, "thinking", None)
        return str(thinking).strip() if thinking else ""
    if block_type == "redacted_thinking":
        return "[reasoning redacted]"
    return ""


_STANDARD_NUMERIC_FIELDS = {
    "example_id",
    "prompt",
    "completion",
    "answer",
    "env_id",
    "info",
    "reward",
    "error",
    "timing",
    "is_completed",
    "is_truncated",
    "stop_condition",
    "metrics",
    "tool_defs",
    "token_usage",
    "error_chain",
    "long_error_chain",
}
