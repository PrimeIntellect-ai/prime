"""Renderable training run widgets shared by selector details and run screens."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rich.console import Group
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widget import Widget
from textual.widgets import Button, Static

from .eval_screen import RolloutViewer
from .logs import visible_system_renderable
from .models import LabItem
from .palette import CODE_THEME, PRIMARY, STATUS_ERROR
from .training_charts import (
    ChartMode,
    chart_fallback,
    chart_heading_from_specs,
    chart_specs,
    plot_widget,
    visible_metric_rows_for,
)
from .training_config import (
    training_config_item,
    training_config_toml,
    training_platform_url,
)
from .values import dict_value, int_value, list_value, metadata_value
from .widgets import LoadingChart, LoadingMessage

RUN_TABS = (
    ("overview", "Overview"),
    ("data", "Data"),
    ("system", "System"),
)


def adjacent_tab(active_tab: str, offset: int) -> str:
    keys = [key for key, _ in RUN_TABS]
    idx = keys.index(active_tab) if active_tab in keys else 0
    return keys[(idx + offset) % len(keys)]


def training_title(item: LabItem) -> Text:
    text = Text()
    name = metadata_value(item, "Name")
    heading = name if name and name != "-" else item.title
    text.append(heading, style="bold")
    text.append(f"  {item.status}", style=item.status_style)
    text.append(f"\nTraining run · {item.title}", style="dim")
    return text


def training_run_widgets(
    item: LabItem,
    *,
    include_logs: bool,
    active_tab: str = "overview",
    chart_mode: ChartMode = "metrics",
    metric_index: int = 0,
    distribution_index: int = 0,
    visible_metric_rows: int | None = None,
    visible_log_lines: int | None = None,
    loading_log_tail_lines: int | None = None,
    render_plots: bool = True,
    frontend_url: str = "",
    workspace: Path | None = None,
) -> list[Widget]:
    raw = item.raw
    progress = dict_value(raw.get("progress"))
    metrics = list_value(raw.get("recent_metrics"))
    environments = list_value(raw.get("environments"))
    environment_statuses = list_value(raw.get("environment_statuses"))
    reward_distribution = dict_value(raw.get("reward_distribution"))

    summary = training_progress_summary(raw, progress, metrics)

    if active_tab == "data":
        return training_data_page(raw, progress)

    if active_tab == "system":
        if not include_logs:
            return [Static(Text("Open the System tab or press l to load logs.", style="dim"))]
        logs = raw.get("logs_tail")
        if (not isinstance(logs, str) or not logs) and should_show_logs_loading(raw):
            return [LoadingChart("Loading logs ...")]
        widgets: list[Widget] = []
        current_tail = int_value(raw.get("log_tail_lines"))
        if (
            loading_log_tail_lines is not None
            and current_tail is not None
            and current_tail < loading_log_tail_lines
        ):
            widgets.append(LoadingMessage(f"Loading {loading_log_tail_lines:,} log lines ..."))
        widgets.append(
            Static(
                visible_system_renderable(
                    raw,
                    visible_log_lines=visible_log_lines,
                ),
                classes="run-log-table",
            )
        )
        return widgets

    return _training_overview_widgets(
        item,
        raw,
        summary,
        metrics,
        reward_distribution,
        environments,
        environment_statuses,
        chart_mode=chart_mode,
        metric_index=metric_index,
        distribution_index=distribution_index,
        visible_metric_rows=visible_metric_rows,
        render_plots=render_plots,
        frontend_url=frontend_url,
        workspace=workspace,
    )


def training_run_page(
    item: LabItem,
    *,
    include_logs: bool,
    active_tab: str = "overview",
    chart_mode: ChartMode = "metrics",
    metric_index: int = 0,
    distribution_index: int = 0,
) -> Group:
    widgets = training_run_widgets(
        item,
        include_logs=include_logs,
        active_tab=active_tab,
        chart_mode=chart_mode,
        metric_index=metric_index,
        distribution_index=distribution_index,
        visible_metric_rows=None,
        loading_log_tail_lines=None,
        render_plots=False,
        frontend_url="",
        workspace=None,
    )
    return Group(*(getattr(widget, "renderable", Text("")) for widget in widgets))


def run_loading_widgets(*, include_logs: bool) -> list[Widget]:
    widgets: list[Widget] = [
        LoadingMessage("Loading metadata ..."),
        LoadingMessage("Loading environments ..."),
        Static(loading_chart_heading("metrics"), classes="run-chart-heading"),
        LoadingChart("Loading metrics ..."),
    ]
    if include_logs:
        widgets.append(LoadingMessage("Loading logs ..."))
    return widgets


def loading_chart_heading(mode: ChartMode) -> Text:
    return Text("Metrics" if mode == "metrics" else "Reward Distribution", style="bold")


def should_show_chart_loading(raw: dict[str, Any], mode: ChartMode) -> bool:
    loaded_key = "metrics_loaded" if mode == "metrics" else "reward_distribution_loaded"
    if raw.get(loaded_key) is not True:
        return True
    return run_may_still_emit_charts(raw)


def should_show_logs_loading(raw: dict[str, Any]) -> bool:
    if raw.get("logs_loaded") is not True:
        return True
    return run_may_still_emit_charts(raw)


def run_may_still_emit_charts(raw: dict[str, Any]) -> bool:
    status = str(raw.get("status") or "").upper()
    return status in {"RUNNING", "PENDING", "STARTING", "QUEUED"}


def training_data_page(raw: dict[str, Any], progress: dict[str, Any]) -> list[Widget]:
    steps = progress.get("steps_with_samples") or progress.get("stepsWithSamples")
    step_count = len(steps) if isinstance(steps, list) else 0
    rollout_samples = dict_value(raw.get("rollout_samples"))
    samples = list_value(rollout_samples.get("samples"))
    step = raw.get("rollout_samples_step")

    text = Text()
    text.append("Rollouts", style="bold")
    text.append(f"\n{step_count} sample steps available", style="dim")
    if step is not None:
        text.append(f" · showing step {step}", style="dim")

    if raw.get("rollout_samples_loaded") is not True:
        return [Static(text), LoadingMessage("Loading rollout samples ...")]

    error = rollout_samples.get("error")
    if error:
        text.append(f"\n\nFailed to load rollout samples: {error}", style=STATUS_ERROR)
        return [Static(text)]

    records = [sample for sample in samples if isinstance(sample, dict)]
    if not records:
        text.append("\n\nNo rollout samples found for sampled steps.", style="dim")
        return [Static(text)]

    metadata = {
        "run_id": raw.get("id"),
        "step": step,
        "total": rollout_samples.get("total"),
        "page": rollout_samples.get("page"),
        "limit": rollout_samples.get("limit"),
    }
    return [
        Static(text),
        RolloutViewer(records, metadata=metadata, title="Samples", classes="training-data-viewer"),
    ]


def training_progress_summary(
    raw: dict[str, Any], progress: dict[str, Any], metrics: list[Any]
) -> dict[str, int | float | bool]:
    max_steps = int_value(raw.get("max_steps") or raw.get("maxSteps")) or 0
    status = str(raw.get("status") or "")
    latest_step = int_value(progress.get("latest_step") or progress.get("latestStep"))
    steps_with_samples = progress.get("steps_with_samples") or progress.get("stepsWithSamples")
    latest_sample_step = (
        max(step for step in steps_with_samples if isinstance(step, int))
        if isinstance(steps_with_samples, list)
        and any(isinstance(step, int) for step in steps_with_samples)
        else None
    )
    metric_steps = [
        metric.get("step")
        for metric in metrics
        if isinstance(metric, dict) and isinstance(metric.get("step"), int)
    ]
    latest_metric_step = max(metric_steps) if metric_steps else None
    has_stored_progress = any(
        step is not None for step in (latest_step, latest_sample_step, latest_metric_step)
    )
    is_completed = status == "COMPLETED"

    completed_candidates = []
    if latest_metric_step is not None:
        completed_candidates.append(latest_metric_step + 1)
    if latest_sample_step is not None:
        completed_candidates.append(latest_sample_step + 1)
    if latest_step is not None:
        completed_candidates.append(latest_step + 1 if latest_step > 0 else 0)

    current_candidates = [
        step for step in (latest_metric_step, latest_sample_step, latest_step) if step is not None
    ]
    current_step = (
        max(current_candidates)
        if current_candidates
        else max_steps - 1
        if is_completed and not has_stored_progress and max_steps > 0
        else 0
    )
    steps_completed = (
        max_steps if is_completed else max(completed_candidates) if completed_candidates else 0
    )
    progress_percent = (
        min(100.0, (steps_completed / max_steps) * 100)
        if max_steps > 0
        else 100.0
        if is_completed
        else 0.0
    )
    return {
        "current_step": current_step,
        "steps_completed": steps_completed,
        "total_steps": max_steps,
        "has_stored_progress": has_stored_progress,
        "has_progress": has_stored_progress or is_completed,
        "progress_percent": progress_percent,
    }


def _training_overview_widgets(
    item: LabItem,
    raw: dict[str, Any],
    summary: dict[str, int | float | bool],
    metrics: list[Any],
    reward_distribution: dict[str, Any],
    environments: list[Any],
    environment_statuses: list[Any],
    *,
    chart_mode: ChartMode,
    metric_index: int,
    distribution_index: int,
    visible_metric_rows: int | None,
    render_plots: bool,
    frontend_url: str,
    workspace: Path | None,
) -> list[Widget]:
    chart_widgets = _training_chart_widgets(
        raw,
        metrics,
        reward_distribution,
        chart_mode=chart_mode,
        metric_index=metric_index,
        distribution_index=distribution_index,
        visible_metric_rows=visible_metric_rows,
        render_plots=render_plots,
    )
    config_item = training_config_item(item, raw, workspace=workspace)
    chart_column_widgets: list[Widget] = [
        Static(_training_progress_panel(summary), classes="run-progress-panel")
    ]
    if config_item is not None:
        chart_column_widgets.append(
            Button("Train", id="run-train", classes="run-train-button", variant="primary")
        )
    chart_column_widgets.extend(chart_widgets)

    config = training_config_toml(raw)
    side_widgets: list[Widget] = [
        Static(
            _training_side_panel(
                item,
                raw,
                summary,
                environments,
                environment_statuses,
            )
        )
    ]
    if config:
        side_widgets.extend(
            [
                Horizontal(
                    Static("Config", classes="run-config-title", markup=False),
                    Button(
                        "Modify and run",
                        id="run-edit-config",
                        classes="run-config-action",
                        compact=True,
                    ),
                    classes="run-config-toolbar",
                ),
                Static(Syntax(config, "toml", theme=CODE_THEME), classes="run-config-panel"),
            ]
        )
    elif config_item is not None:
        side_widgets.append(
            Horizontal(
                Button(
                    "Modify and run",
                    id="run-edit-config",
                    classes="run-config-action",
                    compact=True,
                ),
                classes="run-config-toolbar",
            )
        )
    if training_platform_url(frontend_url, raw, fallback_id=item.title) is not None:
        side_widgets.append(
            Button(
                "View on platform",
                id="run-platform",
                classes="run-side-button",
                compact=True,
            )
        )
    return [
        Horizontal(
            Vertical(
                *chart_column_widgets,
                classes="run-chart-column",
            ),
            VerticalScroll(
                *side_widgets,
                classes="run-side-column",
            ),
            classes="run-overview-grid",
        )
    ]


def _training_chart_widgets(
    raw: dict[str, Any],
    metrics: list[Any],
    reward_distribution: dict[str, Any],
    *,
    chart_mode: ChartMode,
    metric_index: int,
    distribution_index: int,
    visible_metric_rows: int | None,
    render_plots: bool,
) -> list[Widget]:
    if chart_mode == "metrics":
        metrics = visible_metric_rows_for(metrics, visible_metric_rows)
    charts = chart_specs(metrics, reward_distribution, mode=chart_mode)
    widgets: list[Widget] = []
    if not charts:
        if should_show_chart_loading(raw, chart_mode):
            message = (
                "Loading metrics ..."
                if chart_mode == "metrics"
                else "Loading reward distribution ..."
            )
            widgets.extend(
                [
                    Static(loading_chart_heading(chart_mode), classes="run-chart-heading"),
                    LoadingChart(message),
                ]
            )
            return widgets
        message = (
            "No metrics recorded" if chart_mode == "metrics" else "No reward distribution recorded"
        )
        widgets.append(Static(Text(message, style="dim")))
        return widgets

    chart = charts[(metric_index if chart_mode == "metrics" else distribution_index) % len(charts)]
    heading = chart_heading_from_specs(
        charts,
        mode=chart_mode,
        metric_index=metric_index,
        distribution_index=distribution_index,
    )
    if not render_plots:
        widgets.extend(
            [
                Static(heading, classes="run-chart-heading"),
                Static(chart_fallback(chart), classes="run-chart-fallback"),
            ]
        )
        return widgets
    widgets.extend([Static(heading, classes="run-chart-heading"), plot_widget(chart)])
    return widgets


def _training_overview_table(
    item: LabItem, raw: dict[str, Any], summary: dict[str, int | float | bool]
) -> Table:
    table = Table.grid(padding=(0, 3))
    table.add_column(style="bold dim", no_wrap=True)
    table.add_column()
    table.add_column(style="bold dim", no_wrap=True)
    table.add_column()
    table.add_row("Run ID", item.title, "Status", Text(item.status, style=item.status_style))
    table.add_row(
        "Base model",
        str(raw.get("base_model") or raw.get("baseModel") or "-"),
        "Progress",
        f"{summary['steps_completed']}/{summary['total_steps']} steps",
    )
    table.add_row(
        "Batch size",
        str(raw.get("batch_size") or raw.get("batchSize") or "-"),
        "Rollouts per example",
        str(raw.get("rollouts_per_example") or raw.get("rolloutsPerExample") or "-"),
    )
    table.add_row(
        "Created",
        str(metadata_value(item, "Created") or "-"),
        "Updated",
        str(metadata_value(item, "Updated") or "-"),
    )
    return table


def _training_progress_panel(summary: dict[str, int | float | bool]) -> Group:
    return Group(Text("Progress", style="bold"), _progress_block(summary))


def _training_side_panel(
    item: LabItem,
    raw: dict[str, Any],
    summary: dict[str, int | float | bool],
    environments: list[Any],
    environment_statuses: list[Any],
) -> Group:
    chunks: list[Any] = [
        Text("Run", style="bold"),
        _training_overview_table(item, raw, summary),
    ]
    env_table = _training_environment_table(environments, environment_statuses)
    if env_table.row_count:
        chunks.extend([Text(""), Text("Environments", style="bold"), env_table])
    return Group(*chunks)


def _progress_block(summary: dict[str, int | float | bool]) -> Text:
    percent = float(summary["progress_percent"])
    width = 48
    filled = int(round((percent / 100) * width))
    text = Text()
    text.append(f"{percent:5.1f}% ", style="bold")
    text.append("[")
    text.append("#" * filled, style=PRIMARY)
    text.append("." * (width - filled), style="dim")
    text.append("]")
    text.append(
        f"  current step {summary['current_step']} · "
        f"{summary['steps_completed']}/{summary['total_steps']} steps",
        style="dim",
    )
    return text


def _training_environment_table(environments: list[Any], environment_statuses: list[Any]) -> Table:
    table = Table(show_header=True, header_style="bold dim")
    table.add_column("Environment")
    table.add_column("Action")
    status_by_slug: dict[str, dict[str, Any]] = {}
    for status in environment_statuses:
        if not isinstance(status, dict):
            continue
        slug = str(status.get("environment") or status.get("slug") or status.get("id") or "")
        if slug:
            status_by_slug[slug] = status
    if not environments:
        table.add_row("-", "-")
        return table
    for env in environments:
        if not isinstance(env, dict):
            table.add_row(str(env), "-")
            continue
        slug = str(env.get("slug") or env.get("id") or env.get("name") or "-")
        status = status_by_slug.get(slug, {})
        action = status.get("latest_ci_status") or status.get("status") or "-"
        table.add_row(_environment_version_label(slug, env), str(action))
    return table


def _environment_version_label(slug: str, env: dict[str, Any]) -> Text:
    label = Text(slug)
    version = env.get("version") or env.get("semantic_version") or env.get("semanticVersion")
    if version not in (None, "", "-"):
        label.append("  ")
        label.append(f"({version})", style="dim")
    return label
