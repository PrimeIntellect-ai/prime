"""Evaluation browser indexing, labels, and selection details."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from rich.console import Group
from rich.table import Table
from rich.text import Text

from .eval_records import RunOverviewStats
from .eval_render import (
    build_metric_summary_table,
    build_reward_distribution_table,
    build_settings_table,
    format_reward_value,
    format_run_datetime,
    int_like_sort_key,
    numeric_reward,
    reward_style,
    run_setting_rows,
    run_setting_variation_rows,
)
from .models import LabItem
from .widgets import EvaluationNodeData

EvaluationView = str
EVALUATION_VIEWS: tuple[tuple[EvaluationView, str], ...] = (
    ("runs", "By run"),
    ("env", "By env"),
)


def evaluation_index(items: list[LabItem]) -> dict[str, dict[str, list[LabItem]]]:
    index: dict[str, dict[str, list[LabItem]]] = {}
    for item in items:
        env_id = evaluation_env_id(item)
        model = evaluation_model(item)
        index.setdefault(env_id, {}).setdefault(model, []).append(item)
    return index


def evaluation_env_id(item: LabItem) -> str:
    raw = item.raw
    for value in (
        raw.get("env_id"),
        raw.get("environment"),
        raw.get("environment_name"),
        raw.get("environmentName"),
        raw.get("environment_names"),
        raw.get("environmentNames"),
        raw.get("environment_id"),
        raw.get("environmentId"),
        raw.get("environment_ids"),
        raw.get("environmentIds"),
        metadata_value(item, "Environment"),
        metadata_value(item, "Env"),
    ):
        text = first_text_value(value)
        if text and text != "-":
            return text
    return "Unknown environment"


def evaluation_env_label(item: LabItem) -> Text:
    label = Text(evaluation_env_id(item))
    version = evaluation_env_version(item)
    if version:
        label.append("  ")
        label.append(f"({version})", style="dim")
    return label


def evaluation_env_version(item: LabItem) -> str:
    raw = item.raw
    metadata = evaluation_item_metadata(item)
    for value in (
        raw.get("environment_version"),
        raw.get("environmentVersion"),
        raw.get("version"),
        metadata.get("version"),
        metadata.get("environment_version"),
        metadata.get("environmentVersion"),
        _env_arg_value(metadata, "version"),
    ):
        text = first_text_value(value)
        if text and text != "-":
            return text
    return ""


def evaluation_model(item: LabItem) -> str:
    raw = item.raw
    for value in (
        raw.get("model"),
        raw.get("model_name"),
        raw.get("modelName"),
        raw.get("base_model"),
        raw.get("baseModel"),
        metadata_value(item, "Model"),
    ):
        text = first_text_value(value)
        if text and text != "-":
            return text
    return "Unknown model"


def evaluation_run_id(item: LabItem) -> str:
    raw = item.raw
    for value in (
        raw.get("run_id"),
        raw.get("evaluation_id"),
        raw.get("evaluationId"),
        raw.get("id"),
        item.title,
    ):
        text = first_text_value(value)
        if text and text != "-":
            return text
    return item.key


def evaluation_reward(item: LabItem) -> float | None:
    raw = item.raw
    metadata = raw.get("metadata")
    metadata_dict = metadata if isinstance(metadata, dict) else {}
    for value in (
        metadata_dict.get("avg_reward"),
        metadata_dict.get("avgReward"),
        raw.get("avg_reward"),
        raw.get("avgReward"),
        raw.get("avg_score"),
        raw.get("avgScore"),
        raw.get("score"),
        metadata_value(item, "Avg reward"),
        metadata_value(item, "Avg score"),
        item.status,
    ):
        parsed = numeric_reward(value)
        if parsed is not None:
            return parsed
    return None


def sorted_evaluation_runs(items: list[LabItem]) -> list[LabItem]:
    return sorted(
        items,
        key=lambda item: (evaluation_sort_time(item), evaluation_run_id(item)),
        reverse=True,
    )


def evaluation_env_tree_label(
    env_id: str, models: dict[str, list[LabItem]], total_runs: int
) -> Text:
    label = Text()
    label.append(env_id, style="bold")
    label.append("  ")
    label.append(f"{len(models)} models", style="dim")
    label.append("  ")
    label.append(f"{total_runs} runs", style="dim")
    return label


def evaluation_model_tree_label(model: str, runs: list[LabItem]) -> Text:
    label = Text()
    label.append(model, style="bold")
    label.append("  ")
    label.append(f"{len(runs)} runs", style="dim")
    return label


def evaluation_run_tree_label(item: LabItem) -> Text:
    label = Text()
    label.append(evaluation_run_id(item), style="bold")
    reward = evaluation_reward(item)
    if reward is not None:
        label.append("  ")
        label.append(format_reward_value(reward), style=reward_style(reward))
    if item.status and numeric_reward(item.status) is None:
        label.append("  ")
        label.append(item.status, style=item.status_style)
    return label


def evaluation_group_selection_details(
    payload: EvaluationNodeData,
    index: dict[str, dict[str, list[LabItem]]],
) -> Group:
    if payload.kind == "env":
        return evaluation_env_selection_details(payload.env_id, index)
    return evaluation_model_selection_details(payload.env_id, payload.model, index)


def evaluation_run_selection_details(
    item: LabItem,
    stats: RunOverviewStats | None = None,
    *,
    platform_detail_chunks: Callable[[dict[str, Any]], list[Any]] | None = None,
) -> Group:
    reward = evaluation_reward(item)
    summary = Text()
    summary.append("Run\n", style="bold dim")
    summary.append(evaluation_run_id(item), style="bold")
    summary.append("\n")
    env_model = Text()
    env_model.append_text(evaluation_env_label(item))
    env_model.append("   ")
    env_model.append(evaluation_model(item), style="dim")
    summary.append_text(env_model)

    is_local_eval = item.raw.get("type") == "local_eval"
    summary_parts: list[tuple[str, str, str | None]] = []
    if not is_local_eval and item.status and numeric_reward(item.status) is None:
        summary_parts.append(("status", item.status, item.status_style))
    if reward is not None:
        summary_parts.append(("avg reward", f"{reward:.3f}", reward_style(reward)))

    metadata = evaluation_item_metadata(item)
    if is_local_eval:
        created = format_run_datetime(metadata)
        if created:
            summary_parts.insert(0, ("created", created, None))
        if metadata.get("num_examples") not in (None, ""):
            summary_parts.append(("examples", str(metadata.get("num_examples")), None))
        if metadata.get("rollouts_per_example") not in (None, ""):
            summary_parts.append(("rollouts/ex", str(metadata.get("rollouts_per_example")), None))
        if stats is not None and stats.rewards:
            summary_parts.append(("records", str(len(stats.rewards)), None))
    else:
        for label, value in (
            ("samples", metadata_value(item, "Samples")),
            ("examples", metadata_value(item, "Examples")),
            ("rollouts", metadata_value(item, "Rollouts")),
            ("updated", metadata_value(item, "Updated")),
        ):
            if value and value != "-":
                summary_parts.append((label, value, None))
    if summary_parts:
        summary.append("\n\n")
        for idx, (label, value, style) in enumerate(summary_parts):
            if idx:
                summary.append("   ")
            summary.append(f"{label} ", style="bold")
            summary.append(value, style=style or "")

    items: list[Any] = [summary]
    if is_local_eval:
        setting_rows = run_setting_rows(metadata)
        if setting_rows:
            items.extend([Text(""), build_settings_table(setting_rows, "Run settings")])

        pass_rate_text = evaluation_pass_rates_text(metadata)

        if stats is None:
            items.extend(
                [
                    Text(""),
                    Text("Loading rollout metrics ...", style="dim"),
                    Text("Enter opens the rollout viewer immediately.", style="dim"),
                ]
            )
        else:
            reward_summary = build_reward_distribution_table(stats.rewards, "Rollout rewards")
            metric_summary = build_metric_summary_table(stats.metric_summaries)
            if getattr(reward_summary, "plain", None) != "":
                items.extend([Text(""), reward_summary])
            if getattr(metric_summary, "plain", None) != "":
                items.extend([Text(""), metric_summary])
        if pass_rate_text.plain:
            items.extend([Text(""), pass_rate_text])
        return Group(*items)

    settings = evaluation_run_settings_table(item)
    if settings.row_count:
        items.extend([Text(""), settings])

    detail_chunks = platform_detail_chunks(item.raw) if platform_detail_chunks is not None else []
    if detail_chunks:
        items.extend(detail_chunks)
    return Group(*items)


def local_eval_stats_key(item: LabItem) -> str:
    path = item.raw.get("path")
    return str(path or item.key)


def metadata_value(item: LabItem, key: str) -> str | None:
    for metadata_key, value in item.metadata:
        if metadata_key == key:
            return value
    return None


def first_text_value(value: Any) -> str | None:
    if isinstance(value, list | tuple):
        for child in value:
            text = first_text_value(child)
            if text:
                return text
        return None
    if isinstance(value, dict):
        for key in ("slug", "id", "name", "environment", "env_id", "model"):
            text = first_text_value(value.get(key))
            if text:
                return text
        return None
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def evaluation_item_metadata(item: LabItem) -> dict[str, Any]:
    metadata = item.raw.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def _env_arg_value(metadata: dict[str, Any], key: str) -> Any:
    env_args = metadata.get("env_args")
    if not isinstance(env_args, dict):
        return None
    return env_args.get(key)


def evaluation_pass_rates_text(metadata: dict[str, Any]) -> Text:
    pass_rates: list[tuple[str, float]] = []
    for key, prefix in (("pass_at_k", "pass@"), ("pass_all_k", "pass-all@")):
        values = metadata.get(key)
        if isinstance(values, dict):
            for bucket, value in sorted(
                values.items(),
                key=lambda item: int_like_sort_key(item[0]),
            ):
                numeric = numeric_reward(value)
                if numeric is not None:
                    pass_rates.append((f"{prefix}{bucket}", numeric))

    text = Text()
    if not pass_rates:
        return text
    text.append("Pass rates\n", style="bold dim")
    for idx, (label, value) in enumerate(pass_rates[:6]):
        if idx and idx % 3 == 0:
            text.append("\n")
        elif idx:
            text.append("   ")
        text.append(f"{label} ", style="bold")
        text.append(f"{value:.3f}", style=reward_style(value))
    return text


def evaluation_sort_time(item: LabItem) -> str:
    raw = item.raw
    for value in (
        raw.get("updated_at"),
        raw.get("updatedAt"),
        raw.get("created_at"),
        raw.get("createdAt"),
        metadata_value(item, "Updated"),
        metadata_value(item, "Created"),
    ):
        text = first_text_value(value)
        if text and text != "-":
            return text
    return ""


def evaluation_env_selection_details(
    env_id: str, index: dict[str, dict[str, list[LabItem]]]
) -> Group:
    models = index.get(env_id, {})
    runs = [run for model_runs in models.values() for run in model_runs]
    rewards = evaluation_rewards(runs)

    summary = Text()
    summary.append("Environment\n", style="bold dim")
    summary.append(env_id, style="bold")
    summary.append("\n")
    summary.append(f"{len(models)} models   {len(runs)} runs", style="dim")
    items: list[Any] = [summary]
    if rewards:
        items.extend([Text(""), build_reward_distribution_table(rewards, "Run avg rewards")])

    if models:
        ranked_models = sorted(models.items(), key=lambda row: (-len(row[1]), row[0]))[:4]
        activity = Text()
        activity.append("Model activity\n", style="bold dim")
        for model, model_runs in ranked_models:
            model_rewards = evaluation_rewards(model_runs)
            activity.append(model, style="bold")
            activity.append(f"  {len(model_runs)} runs", style="dim")
            if model_rewards:
                avg_reward = sum(model_rewards) / len(model_rewards)
                activity.append("  avg ", style="dim")
                activity.append(f"{avg_reward:.3f}", style=reward_style(avg_reward))
            activity.append("\n")
        items.extend([Text(""), activity])
    return Group(*items)


def evaluation_model_selection_details(
    env_id: str,
    model: str,
    index: dict[str, dict[str, list[LabItem]]],
) -> Group:
    runs = sorted_evaluation_runs(index.get(env_id, {}).get(model, []))
    rewards = evaluation_rewards(runs)

    summary = Text()
    summary.append("Model\n", style="bold dim")
    summary.append(model, style="bold")
    summary.append("\n")
    summary.append(f"{env_id}   {len(runs)} runs", style="dim")
    items: list[Any] = [summary]
    if rewards:
        items.extend([Text(""), build_reward_distribution_table(rewards, "Run avg rewards")])

    if runs:
        best = max(runs, key=evaluation_reward_sort_key)
        recent = Text()
        recent.append("Recent runs\n", style="bold dim")
        for label, item in (("latest", runs[0]), ("best", best)):
            reward = evaluation_reward(item)
            recent.append(label, style="bold")
            recent.append("  ")
            recent.append(evaluation_run_id(item))
            if reward is not None:
                recent.append("  reward ", style="dim")
                recent.append(f"{reward:.3f}", style=reward_style(reward))
            recent.append("\n")
        items.extend([Text(""), recent])

        local_metadatas = [
            metadata
            for run in runs
            for metadata in [evaluation_item_metadata(run)]
            if run.raw.get("type") == "local_eval" and metadata
        ]
        variation_rows, hidden_variations = run_setting_variation_rows(local_metadatas)
        if variation_rows:
            items.extend(
                [
                    Text(""),
                    build_settings_table(
                        variation_rows,
                        "Setting variations",
                        value_header="Across runs",
                    ),
                ]
            )
            if hidden_variations:
                items.extend(
                    [
                        Text(""),
                        Text(f"{hidden_variations} more varied settings not shown", style="dim"),
                    ]
                )

        items.extend([Text(""), Text("Enter opens highlighted run", style="dim")])
    return Group(*items)


def evaluation_run_settings_table(item: LabItem) -> Table:
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold dim", no_wrap=True)
    table.add_column()
    skipped = {
        "Environment",
        "Model",
        "Avg reward",
        "Samples",
        "Examples",
        "Rollouts",
        "Updated",
    }
    for key, value in item.metadata:
        if key in skipped:
            continue
        if value != "-":
            table.add_row(key, value)
    path = item.raw.get("path")
    if isinstance(path, str) and path:
        table.add_row("Path", path)
    return table


def evaluation_rewards(items: list[LabItem]) -> list[float]:
    return [reward for item in items for reward in [evaluation_reward(item)] if reward is not None]


def evaluation_reward_sort_key(item: LabItem) -> float:
    reward = evaluation_reward(item)
    return reward if reward is not None else float("-inf")
