"""Textual application for `prime lab view`."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

import toml
from rich.console import Group
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.screen import Screen
from textual.widget import Widget
from textual.widgets import Footer, Label, OptionList, Static, Tab, Tabs, Tree
from textual.widgets._option_list import Option
from textual_hires_canvas import Canvas, TextAlign
from textual_plot import HiResMode, PlotWidget
from textual_plot.axis_formatter import CategoricalAxisFormatter

from .data import LabLoadOptions
from .eval_records import LocalEvalRun, RunOverviewStats
from .eval_render import compute_run_overview_stats
from .eval_screen import LocalEvalRunScreen, RolloutViewer
from .evaluation_browser import (
    EVALUATION_VIEWS,
    evaluation_env_tree_label,
    evaluation_group_selection_details,
    evaluation_index,
    evaluation_model_tree_label,
    evaluation_run_selection_details,
    evaluation_run_tree_label,
    local_eval_stats_key,
    sorted_evaluation_runs,
)
from .filters import FilterChoice, FilterScreen
from .models import LabItem, LabSection, LabSnapshot
from .palette import (
    GRID,
    LAB_THEME,
    NEUTRAL,
    PRIMARY,
    STATUS_ERROR,
    STATUS_INFO,
    STATUS_SUCCESS,
    STATUS_WARNING,
)
from .widgets import (
    EvaluationNodeData,
    EvaluationTree,
    EvaluationViewToggle,
    HomeGroupToggle,
    LabInspector,
    LabOptionList,
    LoadingChart,
    LoadingMessage,
)


@dataclass(frozen=True)
class ChartSpec:
    title: str
    kind: Literal["line", "bar"]
    x: tuple[int | float | str, ...]
    y: tuple[float, ...]
    xlabel: str
    ylabel: str
    smooth_y: tuple[float, ...] = ()
    smooth_label: str = ""


ChartMode = Literal["metrics", "distribution"]
DetailLoader = Callable[[LabItem, bool, int, int, int | None], LabItem]
_RUN_METRIC_PREVIEW_LIMIT = 10
_RUN_METRIC_FULL_LIMIT = 500
_METRIC_REVEAL_INTERVAL_SECONDS = 0.04
_METRIC_REVEAL_BUFFER = 1.15
_METRIC_REVEAL_MIN_SECONDS = 0.1
_METRIC_REVEAL_MAX_SECONDS = 10.0
_METRIC_CACHED_REVEAL_SECONDS = 1.0
_LOG_PREVIEW_TAIL = 50
_LOG_DEFAULT_TAIL = 1000
_LOG_TAIL_STEPS = (_LOG_PREVIEW_TAIL, _LOG_DEFAULT_TAIL)
_LOG_RENDER_ROW_LIMIT = 5000
_LOG_REVEAL_BUFFER = 1.15
_LOG_REVEAL_MIN_SECONDS = 0.1
_LOG_REVEAL_MAX_SECONDS = 10.0


class LabPlotWidget(PlotWidget, can_focus=False):
    """Plot widget that draws after mount so textual-plot internals are ready."""

    BINDINGS = []

    def __init__(self, chart: ChartSpec) -> None:
        super().__init__(classes="chart-plot run-chart", allow_pan_and_zoom=False)
        self._chart = chart

    def on_mount(self) -> None:
        super().on_mount()
        self.call_after_refresh(self._draw_chart)

    def clear(self) -> None:
        try:
            super().clear()
        except NoMatches:
            self._datasets = []
            self._labels = []
            self._v_lines = []
            self._v_lines_labels = []
            self._rerender()

    def _update_legend(self) -> None:
        try:
            super()._update_legend()
        except NoMatches:
            return

    def _render_x_ticks(self) -> None:
        canvas = self.query_one("#plot", Canvas)
        bottom_margin = self.query_one("#margin-bottom", Canvas)
        bottom_margin.reset()

        if self._x_ticks is None:
            x_ticks, x_labels = self._x_formatter.get_ticks_and_labels(self._x_min, self._x_max)
        else:
            x_ticks = self._x_ticks
            x_labels = self._x_formatter.get_labels_for_ticks(x_ticks)

        for tick, label in zip(x_ticks, x_labels):
            if tick < self._x_min or tick > self._x_max:
                continue
            align = TextAlign.CENTER
            x, _ = self.get_pixel_from_coordinate(tick, 0.0)

            if not isinstance(self._x_formatter, CategoricalAxisFormatter):
                if tick == self._x_min:
                    x -= 1
                elif tick == self._x_max:
                    align = TextAlign.RIGHT

            new_pixel = self.combine_quad_with_pixel(
                (0, 0, 2, 0),
                canvas,
                x,
                self._scale_rectangle.bottom,
            )
            canvas.set_pixel(
                x,
                self._scale_rectangle.bottom,
                new_pixel,
                style=str(self.get_component_rich_style("plot--axis")),
            )
            bottom_margin.write_text(
                x + self.margin_left,
                0,
                f"[{self.get_component_rich_style('plot--tick')}]{label}",
                align,
            )

    def _render_y_ticks(self) -> None:
        canvas = self.query_one("#plot", Canvas)
        left_margin = self.query_one("#margin-left", Canvas)
        left_margin.reset()

        if self._y_ticks is None:
            y_ticks, y_labels = self._y_formatter.get_ticks_and_labels(self._y_min, self._y_max)
        else:
            y_ticks = self._y_ticks
            y_labels = self._y_formatter.get_labels_for_ticks(y_ticks)

        y_labels = [label[: self.margin_left - 1] for label in y_labels]
        for tick, label in zip(y_ticks, y_labels):
            if tick < self._y_min or tick > self._y_max:
                continue
            _, y = self.get_pixel_from_coordinate(0.0, tick)
            if tick == self._y_min:
                y += 1

            new_pixel = self.combine_quad_with_pixel((0, 0, 0, 2), canvas, 0, y)
            canvas.set_pixel(
                0,
                y,
                new_pixel,
                style=str(self.get_component_rich_style("plot--axis")),
            )
            left_margin.write_text(
                self.margin_left - 2,
                y,
                f"[{self.get_component_rich_style('plot--tick')}]{label}",
                TextAlign.RIGHT,
            )

    def _draw_chart(self) -> None:
        try:
            _draw_plot(self, self._chart)
        except Exception:
            self.clear()

    def update_chart(self, chart: ChartSpec) -> None:
        self._chart = chart
        self._draw_chart()


class TrainingRunScreen(Screen[None]):
    """Full-page training run view."""

    BINDINGS = [
        Binding("b,backspace", "back", "Back"),
        Binding("q", "quit", "Quit"),
        Binding("l", "toggle_logs", "Logs"),
        Binding("left", "previous_tab", "Prev tab"),
        Binding("right", "next_tab", "Next tab"),
        Binding("up", "previous_metric", "Prev chart"),
        Binding("down", "next_metric", "Next chart"),
        Binding("space", "pick_chart", "Pick"),
        Binding("m", "show_metric_charts", "Metrics"),
        Binding("d", "show_distribution_charts", "Reward dist"),
        Binding("g", "load_more_logs", "More logs"),
    ]

    CSS = """
    TrainingRunScreen {
        background: $background;
        color: $foreground;
        layout: vertical;
    }

    #run-topbar {
        height: 6;
        padding: 0 1;
        border-bottom: solid $primary;
        background: $background;
    }

    #run-title {
        height: 2;
    }

    #run-tabs {
        height: 3;
    }

    #run-body {
        height: 1fr;
        padding: 0 1;
    }

    #run-content {
        height: auto;
    }

    .chart-plot {
        height: 30;
        min-height: 18;
        border: round $primary;
        background: $surface;
    }

    .chart-loading {
        height: 30;
        min-height: 18;
        padding: 1 2;
        border: round $primary;
        background: $surface;
    }

    .training-data-viewer {
        height: 34;
        min-height: 24;
    }

    PlotWidget.chart-plot > .plot--axis {
        color: $primary;
    }

    PlotWidget.chart-plot > .plot--tick {
        color: $warning;
        text-style: bold;
    }

    PlotWidget.chart-plot > .plot--label {
        color: $primary;
        text-style: bold italic;
    }
    """

    def __init__(
        self,
        item: LabItem,
        detail_loader: DetailLoader,
        *,
        include_logs: bool = False,
    ) -> None:
        super().__init__()
        self._base_item = item
        self._detail_loader = detail_loader
        self._include_logs = include_logs
        self._initial_detail = item if _has_training_detail(item) else None
        self._detail: LabItem | None = None
        self._active_tab = "system" if include_logs else "overview"
        self._chart_mode: ChartMode = "metrics"
        self._metric_index = 0
        self._distribution_index = 0
        self._log_tail_lines = _LOG_TAIL_STEPS[0]
        self._log_default_requested = False
        self._loading_log_tail_lines: int | None = None
        self._visible_metric_rows = 0
        self._metric_reveal_target = 0
        self._metric_reveal_start_rows = 0
        self._metric_reveal_started_at = 0.0
        self._metric_reveal_deadline = 0.0
        self._visible_log_lines = 0
        self._log_reveal_target = 0
        self._log_reveal_start_lines = 0
        self._log_reveal_started_at = 0.0
        self._log_reveal_deadline = 0.0

    def _run_topbar_title(self, item: LabItem) -> Table:
        title = Table.grid(expand=True)
        title.add_column(ratio=1)
        title.add_column(justify="right", no_wrap=True)
        title.add_row(_training_title(item), _lab_logo_text())
        return title

    def compose(self) -> ComposeResult:
        with Vertical(id="run-topbar"):
            yield Static(self._run_topbar_title(self._base_item), id="run-title")
            yield Tabs(
                *(Tab(label, id=tab) for tab, label in _RUN_TABS),
                active=self._active_tab,
                id="run-tabs",
            )
        with VerticalScroll(id="run-body"):
            with Vertical(id="run-content"):
                yield from _run_loading_widgets(include_logs=self._include_logs)
        yield Footer()

    def on_mount(self) -> None:
        self.set_interval(_METRIC_REVEAL_INTERVAL_SECONDS, self._advance_metric_reveal)
        self.set_interval(_METRIC_REVEAL_INTERVAL_SECONDS, self._advance_log_reveal)
        if self._initial_detail is not None:
            self._set_detail(
                self._initial_detail,
                _cached_metric_reveal_seconds(self._initial_detail),
            )
            if self._include_logs and self._initial_detail.raw.get("logs_loaded") is not True:
                self._loading_log_tail_lines = self._log_tail_lines
                self._load(_current_metrics_limit(self._detail))
        else:
            self._load()

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        if action in {"previous_metric", "next_metric", "pick_chart"}:
            if self._active_tab != "overview" or self._detail is None:
                return False
            chart_count = _chart_count_for_mode(self._visible_chart_raw(), self._chart_mode)
            if action == "pick_chart":
                return chart_count > 0
            return chart_count > 1
        if action in {"show_metric_charts", "show_distribution_charts"}:
            if self._active_tab != "overview" or self._detail is None:
                return False
            if action == "show_metric_charts":
                return (
                    self._chart_mode != "metrics"
                    and _metric_chart_count(self._visible_chart_raw()) > 0
                )
            return (
                self._chart_mode != "distribution"
                and _distribution_chart_count(self._detail.raw) > 0
            )
        if action == "load_more_logs":
            return self._active_tab == "system" and self._include_logs
        return True

    def action_back(self) -> None:
        self.app.pop_screen()

    def action_toggle_logs(self) -> None:
        if not self._include_logs:
            self._active_tab = "system"
            self._include_logs = True
            self._set_run_body(LoadingMessage("Loading logs ..."))
            self._sync_tabs()
            self._load(_current_metrics_limit(self._detail))
            return
        self._active_tab = "overview" if self._active_tab == "system" else "system"
        self._render_detail()

    def action_previous_tab(self) -> None:
        self._active_tab = _adjacent_tab(self._active_tab, -1)
        self._load_logs_for_active_tab()
        self._render_detail()

    def action_next_tab(self) -> None:
        self._active_tab = _adjacent_tab(self._active_tab, 1)
        self._load_logs_for_active_tab()
        self._render_detail()

    def _load_logs_for_active_tab(self) -> None:
        if self._active_tab != "system" or self._include_logs:
            return
        self._include_logs = True
        self._set_run_body(LoadingMessage("Loading logs ..."))
        self._sync_tabs()
        self._load(_current_metrics_limit(self._detail))

    def action_show_logs(self) -> None:
        self._active_tab = "system"
        if not self._include_logs:
            self._include_logs = True
            self._set_run_body(LoadingMessage("Loading logs ..."))
            self._sync_tabs()
            self._load(_current_metrics_limit(self._detail))
            return
        self._render_detail()

    def action_load_more_logs(self) -> None:
        self._active_tab = "system"
        self._include_logs = True
        current_tail = _int_value((self._detail or self._base_item).raw.get("log_tail_lines"))
        self._log_tail_lines = _next_log_tail_lines(max(self._log_tail_lines, current_tail or 0))
        self._loading_log_tail_lines = self._log_tail_lines
        if self._detail is None:
            self._set_run_body(LoadingChart(f"Loading {self._log_tail_lines:,} log lines ..."))
        else:
            self._render_detail()
        self._sync_tabs()
        self._load(_current_metrics_limit(self._detail))

    def action_previous_metric(self) -> None:
        if self._detail is None:
            return
        count = _chart_count_for_mode(self._visible_chart_raw(), self._chart_mode)
        if count <= 1:
            return
        was_overview = self._active_tab == "overview"
        self._active_tab = "overview"
        if self._chart_mode == "metrics":
            self._metric_index = (self._metric_index - 1) % count
        else:
            self._distribution_index = (self._distribution_index - 1) % count
        self._render_chart_change(was_overview=was_overview)

    def action_next_metric(self) -> None:
        if self._detail is None:
            return
        count = _chart_count_for_mode(self._visible_chart_raw(), self._chart_mode)
        if count <= 1:
            return
        was_overview = self._active_tab == "overview"
        self._active_tab = "overview"
        if self._chart_mode == "metrics":
            self._metric_index = (self._metric_index + 1) % count
        else:
            self._distribution_index = (self._distribution_index + 1) % count
        self._render_chart_change(was_overview=was_overview)

    def action_pick_chart(self) -> None:
        if self._detail is None:
            return
        charts = _chart_specs_for_raw(self._visible_chart_raw(), mode=self._chart_mode)
        if not charts:
            return

        def select_chart(value: str | None) -> None:
            if value is None:
                return
            index = int(value)
            self._active_tab = "overview"
            if self._chart_mode == "metrics":
                self._metric_index = index
            else:
                self._distribution_index = index
            self._render_chart_change(was_overview=True)

        self.app.push_screen(
            FilterScreen(
                [
                    FilterChoice(
                        key=str(index),
                        label=chart.title,
                        search_text=chart.title.lower(),
                        value=str(index),
                    )
                    for index, chart in enumerate(charts)
                ],
                title="Metrics" if self._chart_mode == "metrics" else "Reward Distribution",
                placeholder="Filter charts",
                width=88,
                height=28,
            ),
            select_chart,
        )

    def action_show_metric_charts(self) -> None:
        if self._detail is None or not _metric_chart_count(self._visible_chart_raw()):
            return
        was_overview = self._active_tab == "overview"
        self._active_tab = "overview"
        self._chart_mode = "metrics"
        self._render_chart_change(was_overview=was_overview)

    def action_show_distribution_charts(self) -> None:
        if self._detail is None or not _distribution_chart_count(self._detail.raw):
            return
        was_overview = self._active_tab == "overview"
        self._active_tab = "overview"
        self._chart_mode = "distribution"
        self._render_chart_change(was_overview=was_overview)

    @work(thread=True, exclusive=True)
    def _load(
        self,
        metrics_limit: int = _RUN_METRIC_PREVIEW_LIMIT,
        metrics_min_step: int | None = None,
    ) -> None:
        started_at = time.perf_counter()
        try:
            detail = self._detail_loader(
                self._base_item,
                self._include_logs,
                self._log_tail_lines,
                metrics_limit,
                metrics_min_step,
            )
        except Exception as exc:
            detail = LabItem(
                key=self._base_item.key,
                section=self._base_item.section,
                title=self._base_item.title,
                subtitle=str(exc),
                status="error",
                status_style=STATUS_ERROR,
            )
        load_seconds = time.perf_counter() - started_at
        self.app.call_from_thread(self._set_detail, detail, load_seconds)

    def _set_detail(self, detail: LabItem, load_seconds: float) -> None:
        had_detail = self._detail is not None
        incoming_detail = detail
        detail = _merge_training_detail(self._detail, detail)
        self._detail = detail
        incoming_logs_loaded = (
            incoming_detail.raw.get("logs_loaded") is True and "logs_tail" in incoming_detail.raw
        )
        loaded_log_tail = _int_value(detail.raw.get("log_tail_lines"))
        if (
            loaded_log_tail is not None
            and self._loading_log_tail_lines is not None
            and loaded_log_tail >= self._loading_log_tail_lines
        ):
            self._loading_log_tail_lines = None
        loaded_page_limit = _int_value(
            incoming_detail.raw.get("metrics_page_limit")
            or incoming_detail.raw.get("metrics_limit")
        )
        loaded_page_count = _int_value(incoming_detail.raw.get("metrics_page_count"))
        if loaded_page_count is None:
            loaded_page_count = len(_list_value(incoming_detail.raw.get("recent_metrics")))
        metric_rows = len(_list_value(detail.raw.get("recent_metrics")))
        page_was_full = loaded_page_limit is not None and loaded_page_count >= loaded_page_limit
        next_metrics_min_step = (
            _next_metrics_min_step(detail.raw)
            if detail.raw.get("metrics_loaded") is True
            and page_was_full
            and metric_rows < _RUN_METRIC_FULL_LIMIT
            else None
        )
        next_metrics_limit = (
            _next_metrics_page_limit(metric_rows, loaded_page_limit)
            if next_metrics_min_step is not None
            else 0
        )
        self._sync_metric_reveal(
            detail,
            load_seconds,
            current_limit=loaded_page_limit or loaded_page_count or metric_rows,
            next_limit=next_metrics_limit or None,
        )
        self._sync_log_reveal(
            detail,
            load_seconds,
            incoming_logs_loaded=incoming_logs_loaded,
            current_tail=loaded_log_tail,
        )
        visible_raw = self._visible_chart_raw(detail.raw)
        metric_count = _metric_chart_count(visible_raw)
        distribution_count = _distribution_chart_count(detail.raw)
        if metric_count:
            self._metric_index %= metric_count
        else:
            self._metric_index = 0
        if distribution_count:
            self._distribution_index %= distribution_count
        else:
            self._distribution_index = 0
        if self._chart_mode == "metrics" and not metric_count and distribution_count:
            self._chart_mode = "distribution"
        elif self._chart_mode == "distribution" and not distribution_count and metric_count:
            self._chart_mode = "metrics"
        self.refresh_bindings()
        if had_detail and self._active_tab == "overview" and self._update_chart_panel():
            self.query_one("#run-title", Static).update(self._run_topbar_title(detail))
            self._sync_tabs()
        else:
            self._render_detail()
        needs_more_metrics = next_metrics_limit > 0 and next_metrics_min_step is not None
        needs_default_logs = (
            self._include_logs
            and loaded_log_tail is not None
            and loaded_log_tail < _LOG_DEFAULT_TAIL
            and not self._log_default_requested
        )
        metrics_limit = (
            next_metrics_limit
            if needs_more_metrics
            else max(_RUN_METRIC_PREVIEW_LIMIT, metric_rows)
        )
        metrics_min_step = next_metrics_min_step if needs_more_metrics else None
        if needs_default_logs:
            self._log_default_requested = True
            self._log_tail_lines = _LOG_DEFAULT_TAIL
            self._loading_log_tail_lines = _LOG_DEFAULT_TAIL
            if self._active_tab == "system":
                self._render_detail()
        if needs_more_metrics or needs_default_logs:
            self._load(metrics_limit, metrics_min_step=metrics_min_step)

    def _render_detail(self) -> None:
        detail = self._detail or self._base_item
        self.query_one("#run-title", Static).update(self._run_topbar_title(detail))
        self._sync_tabs()
        self._set_run_body(
            *_training_run_widgets(
                detail,
                include_logs=self._include_logs,
                active_tab=self._active_tab,
                chart_mode=self._chart_mode,
                metric_index=self._metric_index,
                distribution_index=self._distribution_index,
                visible_metric_rows=self._visible_metric_rows,
                visible_log_lines=self._visible_log_lines,
                loading_log_tail_lines=self._loading_log_tail_lines,
                render_plots=self._can_render_plots(),
            )
        )

    def _sync_metric_reveal(
        self,
        detail: LabItem,
        load_seconds: float,
        *,
        current_limit: int,
        next_limit: int | None,
    ) -> None:
        metric_rows = len(_list_value(detail.raw.get("recent_metrics")))
        if detail.raw.get("metrics_loaded") is not True or metric_rows <= 0:
            self._visible_metric_rows = 0
            self._metric_reveal_target = 0
            self._metric_reveal_start_rows = 0
            self._metric_reveal_started_at = 0.0
            self._metric_reveal_deadline = 0.0
            return
        if metric_rows <= self._metric_reveal_target and self._visible_metric_rows > 0:
            self._visible_metric_rows = min(self._visible_metric_rows, metric_rows)
            self._metric_reveal_target = metric_rows
            return
        if self._visible_metric_rows <= 0:
            self._visible_metric_rows = 1
        elif self._visible_metric_rows > metric_rows:
            self._visible_metric_rows = metric_rows
        self._metric_reveal_target = metric_rows
        self._metric_reveal_start_rows = self._visible_metric_rows
        self._metric_reveal_started_at = time.perf_counter()
        self._metric_reveal_deadline = self._metric_reveal_started_at + _metric_reveal_duration(
            load_seconds,
            current_limit=current_limit,
            next_limit=next_limit,
            start_rows=self._metric_reveal_start_rows,
            target_rows=self._metric_reveal_target,
        )

    def _advance_metric_reveal(self) -> None:
        if self._detail is None or self._visible_metric_rows >= self._metric_reveal_target:
            return
        next_visible_rows = _interpolated_metric_rows(
            now=time.perf_counter(),
            start_time=self._metric_reveal_started_at,
            deadline=self._metric_reveal_deadline,
            start_rows=self._metric_reveal_start_rows,
            target_rows=self._metric_reveal_target,
        )
        if next_visible_rows <= self._visible_metric_rows:
            return
        self._visible_metric_rows = next_visible_rows
        if self._active_tab != "overview":
            return
        if not self._update_chart_panel():
            self._render_detail()

    def _sync_log_reveal(
        self,
        detail: LabItem,
        load_seconds: float,
        *,
        incoming_logs_loaded: bool,
        current_tail: int | None,
    ) -> None:
        if not self._include_logs:
            self._visible_log_lines = 0
            self._log_reveal_target = 0
            self._log_reveal_start_lines = 0
            self._log_reveal_started_at = 0.0
            self._log_reveal_deadline = 0.0
            return
        logs = detail.raw.get("logs_tail")
        if not isinstance(logs, str):
            if incoming_logs_loaded:
                self._visible_log_lines = 0
                self._log_reveal_target = 0
            return
        line_count = len(logs.splitlines())
        if line_count <= 0:
            self._visible_log_lines = 0
            self._log_reveal_target = 0
            self._log_reveal_start_lines = 0
            self._log_reveal_started_at = 0.0
            self._log_reveal_deadline = 0.0
            return
        if not incoming_logs_loaded and self._log_reveal_target == line_count:
            return
        if line_count <= self._log_reveal_target and self._visible_log_lines > 0:
            self._visible_log_lines = min(self._visible_log_lines, line_count)
            self._log_reveal_target = line_count
            return
        if self._visible_log_lines <= 0:
            self._visible_log_lines = 1
        elif self._visible_log_lines > line_count:
            self._visible_log_lines = line_count
        self._log_reveal_target = line_count
        self._log_reveal_start_lines = self._visible_log_lines
        self._log_reveal_started_at = time.perf_counter()
        self._log_reveal_deadline = self._log_reveal_started_at + _log_reveal_duration(
            load_seconds,
            current_tail=current_tail or line_count,
            start_lines=self._log_reveal_start_lines,
            target_lines=self._log_reveal_target,
        )

    def _advance_log_reveal(self) -> None:
        if self._detail is None or self._visible_log_lines >= self._log_reveal_target:
            return
        next_visible_lines = _interpolated_metric_rows(
            now=time.perf_counter(),
            start_time=self._log_reveal_started_at,
            deadline=self._log_reveal_deadline,
            start_rows=self._log_reveal_start_lines,
            target_rows=self._log_reveal_target,
        )
        if next_visible_lines <= self._visible_log_lines:
            return
        self._visible_log_lines = next_visible_lines
        if self._active_tab != "system":
            return
        if not self._update_log_panel():
            self._render_detail()

    def _visible_chart_raw(self, raw: dict[str, Any] | None = None) -> dict[str, Any]:
        if raw is None:
            if self._detail is None:
                return {}
            raw = self._detail.raw
        return _raw_with_visible_metric_rows(raw, self._visible_metric_rows)

    def _set_run_body(self, *widgets: Widget) -> None:
        content = self.query_one("#run-content", Vertical)
        content.remove_children()
        content.mount(*widgets)

    def _update_log_panel(self) -> bool:
        if self._detail is None:
            return False
        try:
            self.query(".run-log-table").first(Static).update(
                _visible_system_renderable(
                    self._detail.raw,
                    visible_log_lines=self._visible_log_lines,
                )
            )
            return True
        except NoMatches:
            return False

    def _render_chart_change(self, *, was_overview: bool) -> None:
        self._sync_tabs()
        if was_overview and self._update_chart_panel():
            self.refresh_bindings()
            return
        self._render_detail()

    def _update_chart_panel(self) -> bool:
        if self._detail is None:
            return False
        chart = _selected_chart(
            self._visible_chart_raw(),
            mode=self._chart_mode,
            metric_index=self._metric_index,
            distribution_index=self._distribution_index,
        )
        if chart is None:
            return False
        try:
            self.query(".run-chart-heading").first(Static).update(
                _chart_heading(
                    self._visible_chart_raw(),
                    mode=self._chart_mode,
                    metric_index=self._metric_index,
                    distribution_index=self._distribution_index,
                )
            )
        except NoMatches:
            return False
        try:
            self.query(".run-chart").first(LabPlotWidget).update_chart(chart)
            return True
        except NoMatches:
            pass
        except Exception:
            return False
        try:
            self.query(".run-chart-fallback").first(Static).update(_chart_fallback(chart))
            return True
        except NoMatches:
            return False

    def _can_render_plots(self) -> bool:
        app_pseudo_classes = getattr(self.app, "pseudo_classes", set())
        screen_pseudo_classes = getattr(self, "pseudo_classes", set())
        return "nocolor" not in app_pseudo_classes and "nocolor" not in screen_pseudo_classes

    def _sync_tabs(self) -> None:
        tabs = self.query_one("#run-tabs", Tabs)
        if tabs.active != self._active_tab:
            tabs.active = self._active_tab
        self.refresh_bindings()

    def _active_chart_index(self) -> int:
        return self._metric_index if self._chart_mode == "metrics" else self._distribution_index

    @on(Tabs.TabActivated, "#run-tabs")
    def _tab_activated(self, event: Tabs.TabActivated) -> None:
        self._active_tab = event.tab.id or "overview"
        self._load_logs_for_active_tab()
        self._render_detail()


class PrimeLabView(App[None]):
    """Lab read-only terminal viewer."""

    ENABLE_COMMAND_PALETTE = False
    PRIME_THEME = LAB_THEME

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("r", "refresh", "Refresh"),
        Binding("enter", "load_detail", "Open", key_display="Enter"),
        Binding("g", "load_more_rows", "More rows"),
        Binding("/", "search", "Filter"),
        Binding("left", "previous_pane", "Prev pane", key_display="Left"),
        Binding("right", "next_pane", "Next pane", key_display="Right"),
        Binding("escape", "clear_filter", "Clear filter", key_display="Esc"),
        Binding("tab", "focus_next", "Next pane", key_display="Tab"),
        Binding("shift+tab", "focus_previous", "Prev pane", key_display="Shift+Tab"),
    ]

    CSS = """
    Screen {
        background: $background;
        color: $foreground;
        layout: vertical;
    }

    #topbar {
        height: 3;
        padding: 0 1;
        border-bottom: solid $primary;
        background: $background;
    }

    #body {
        height: 1fr;
    }

    .pane {
        border: round $primary;
        background: $surface;
        padding: 0 1;
    }

    #nav-pane {
        width: 28;
        min-width: 24;
    }

    #list-pane {
        width: 2fr;
        min-width: 42;
    }

    #inspector-pane {
        width: 1.35fr;
        min-width: 36;
    }

    .pane-title {
        text-style: bold;
        color: $foreground;
        height: 1;
    }

    .pane-subtitle {
        color: $text-muted;
        height: 1;
    }

    Tree {
        background: $surface;
        height: 1fr;
    }

    OptionList {
        background: $surface;
        height: 1fr;
    }

    #home-toggle {
        display: none;
        height: 1;
        color: $foreground;
    }

    #evaluation-toggle {
        display: none;
        height: 1;
        color: $foreground;
    }

    #evaluation-tree {
        display: none;
        background: $surface;
        height: 1fr;
    }

    OptionList > .option-list--option-highlighted {
        background: $primary 20%;
    }

    #inspector {
        height: 1fr;
        overflow-y: auto;
    }
    """

    def __init__(
        self,
        loader: Callable[[], LabSnapshot],
        detail_loader: DetailLoader | None = None,
        initial_loader: Callable[[], LabSnapshot] | None = None,
        ladder_loader: Callable[[int], LabSnapshot] | None = None,
        ladder_limits: tuple[int, ...] = (),
    ):
        super().__init__()
        self._loader = loader
        self._detail_loader = detail_loader
        self._initial_loader = initial_loader
        self._ladder_loader = ladder_loader
        self._ladder_limits = ladder_limits
        self._loaded_section_limit = 0
        self._requested_section_limit = max(ladder_limits, default=0)
        self._snapshot: LabSnapshot | None = None
        self._active_section_key = "workspace"
        self._filter = ""
        self._visible_items: list[LabItem] = []
        self._items_by_key: dict[str, LabItem] = {}
        self._selected_item: LabItem | None = None
        self._detail_cache: dict[str, LabItem] = {}
        self._prefetching_detail_key: str | None = None
        self._expand_ready_run_key: str | None = None
        self._home_group = "workspaces"
        self._evaluation_view = "runs"
        self._evaluation_tree_index: dict[str, dict[str, list[LabItem]]] = {}
        self._evaluation_click_selected_node: object | None = None
        self._local_eval_stats_cache: dict[str, RunOverviewStats] = {}
        self._loading_local_eval_stats: set[str] = set()

    def compose(self) -> ComposeResult:
        yield Static("Loading Lab ...", id="topbar", markup=False)
        with Horizontal(id="body"):
            with Vertical(id="nav-pane", classes="pane"):
                yield Tree("Sections", id="section-tree")
            with Vertical(id="list-pane", classes="pane"):
                yield Label("Loading", id="section-title", classes="pane-title")
                yield Static("", id="section-subtitle", classes="pane-subtitle", markup=False)
                yield HomeGroupToggle("", id="home-toggle", markup=False)
                yield EvaluationViewToggle("", id="evaluation-toggle", markup=False)
                yield LabOptionList(id="item-list")
                yield EvaluationTree("Evaluations", id="evaluation-tree")
            with Vertical(id="inspector-pane", classes="pane"):
                yield Label("Inspector", id="inspector-title", classes="pane-title")
                yield LabInspector("", id="inspector", markup=False)
        yield Footer()

    def on_mount(self) -> None:
        self.register_theme(self.PRIME_THEME)
        self.theme = "prime-lab"
        self.query_one("#section-tree", Tree).show_root = False
        evaluation_tree = self.query_one("#evaluation-tree", EvaluationTree)
        evaluation_tree.show_root = False
        evaluation_tree.auto_expand = False
        evaluation_tree.guide_depth = 2
        self._show_initial_snapshot()
        self._reload()

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        if action in {"search", "load_more_rows"} and isinstance(
            self.screen, TrainingRunScreen | FilterScreen | LocalEvalRunScreen
        ):
            return False
        if action == "load_detail":
            return self._main_open_action_available()
        if action == "clear_filter":
            return bool(self._filter)
        if action == "load_more_rows":
            return self._ladder_loader is not None and self._requested_section_limit > 0
        return True

    def action_refresh(self) -> None:
        self._loaded_section_limit = 0
        self._show_initial_snapshot()
        self._reload()

    def action_clear_filter(self) -> None:
        if not self._filter:
            return
        self._filter = ""
        self._render_active_section()

    def action_previous_pane(self) -> None:
        focused = self.focused
        if isinstance(focused, LabInspector):
            self._focus_main_pane(prefer_rows=True)
            return
        self.focus_nav_pane()

    def action_next_pane(self) -> None:
        focused = self.focused
        if isinstance(focused, Tree):
            self._focus_main_pane(prefer_rows=False)
            return
        if isinstance(focused, LabOptionList | HomeGroupToggle | EvaluationViewToggle):
            self._focus_inspector_pane()
            return
        self._focus_main_pane(prefer_rows=False)

    def action_back_from_list(self) -> None:
        if self._active_section_key == "workspace":
            toggle = self.query_one("#home-toggle", HomeGroupToggle)
            if toggle.display:
                toggle.focus()
                return
        if self._active_section_key == "evaluations":
            toggle = self.query_one("#evaluation-toggle", EvaluationViewToggle)
            if toggle.display:
                toggle.focus()
                return
        self.focus_nav_pane()

    def focus_nav_pane(self) -> None:
        self.query_one("#section-tree", Tree).focus()

    def focus_home_rows(self) -> None:
        self._focus_main_pane(prefer_rows=True)

    def focus_evaluation_rows(self) -> None:
        self._focus_main_pane(prefer_rows=True)

    def action_previous_home_group(self) -> None:
        self._switch_home_group(-1)

    def action_next_home_group(self) -> None:
        self._switch_home_group(1)

    def _switch_home_group(self, direction: int) -> None:
        groups = _workspace_home_groups(self._workspace_items())
        if not groups:
            return
        keys = [key for key, _, items in groups if items]
        if not keys:
            return
        try:
            index = keys.index(self._home_group)
        except ValueError:
            index = 0
        self._home_group = keys[(index + direction) % len(keys)]
        self._render_active_section()

    def set_home_group(self, group: str) -> None:
        groups = _workspace_home_groups(self._workspace_items())
        if group not in {key for key, _, items in groups if items}:
            return
        if group == self._home_group:
            return
        self._home_group = group
        self._render_active_section()

    def set_segmented_toggle(self, toggle_id: str, key: str) -> None:
        if toggle_id == "home-toggle":
            self.set_home_group(key)
            self.focus_home_rows()
        elif toggle_id == "evaluation-toggle":
            self.set_evaluation_view(key)
            self.focus_evaluation_rows()

    def action_previous_evaluation_view(self) -> None:
        self._switch_evaluation_view(-1)

    def action_next_evaluation_view(self) -> None:
        self._switch_evaluation_view(1)

    def _switch_evaluation_view(self, direction: int) -> None:
        keys = [key for key, _ in EVALUATION_VIEWS]
        try:
            index = keys.index(self._evaluation_view)
        except ValueError:
            index = 0
        self._evaluation_view = keys[(index + direction) % len(keys)]
        self._evaluation_click_selected_node = None
        self._render_active_section()

    def set_evaluation_view(self, view: str) -> None:
        if view not in {key for key, _ in EVALUATION_VIEWS}:
            return
        if view == self._evaluation_view:
            return
        self._evaluation_view = view
        self._evaluation_click_selected_node = None
        self._render_active_section()

    def _focus_main_pane(self, *, prefer_rows: bool) -> None:
        if self._active_section_key == "workspace" and not prefer_rows:
            toggle = self.query_one("#home-toggle", HomeGroupToggle)
            if toggle.display:
                toggle.focus()
                return
        if self._active_section_key == "evaluations":
            if not prefer_rows:
                toggle = self.query_one("#evaluation-toggle", EvaluationViewToggle)
                if toggle.display:
                    toggle.focus()
                    return
            if self._evaluation_view == "env":
                tree = self.query_one("#evaluation-tree", EvaluationTree)
                if tree.display:
                    tree.focus()
                    return
        self.query_one("#item-list", OptionList).focus()

    def _focus_inspector_pane(self) -> None:
        self.query_one("#inspector", LabInspector).focus()

    def _workspace_items(self) -> list[LabItem]:
        if self._snapshot is None:
            return []
        section = self._snapshot.section("workspace")
        if section is None:
            return []
        return [
            self._detail_cache.get(item.key, item)
            for item in section.items
            if not self._filter or self._matches_filter(item)
        ]

    def action_search(self) -> None:
        def apply_filter(value: str | None) -> None:
            if value is not None:
                self._filter = value.lower()
                self._render_active_section()

        self.push_screen(
            FilterScreen(
                self._filter_choices_for_active_section(),
                title="Filter",
                placeholder="environment, model, run, status",
                initial=self._filter,
            ),
            apply_filter,
        )

    def action_load_detail(self) -> None:
        if self._handle_evaluation_tree_enter():
            return
        self._load_selected_detail(include_logs=False)

    def action_load_logs(self) -> None:
        self._load_selected_detail(include_logs=True)

    def action_load_more_rows(self) -> None:
        if self._ladder_loader is None or self._requested_section_limit <= 0:
            return
        current_limit = max(self._loaded_section_limit, self._requested_section_limit)
        next_limit = current_limit * 2
        self._requested_section_limit = next_limit
        self._reload_limits((next_limit,))

    def _show_initial_snapshot(self) -> None:
        if self._initial_loader is None:
            return
        try:
            self._set_snapshot(self._initial_loader())
        except Exception as exc:
            text = Text()
            text.append("Loading Lab ...", style="bold")
            text.append(f"\nInitial workspace view unavailable: {exc}", style="dim")
            self.query_one("#topbar", Static).update(text)

    @work(thread=True, exclusive=True)
    def _reload(self) -> None:
        if self._ladder_loader is not None and self._ladder_limits:
            target_limit = max(
                self._requested_section_limit,
                max(self._ladder_limits, default=0),
            )
            self._load_snapshots_for_limits(_ladder_limits(target_limit))
            return
        snapshot = self._loader()
        self.call_from_thread(self._set_snapshot, snapshot)

    @work(thread=True, exclusive=True)
    def _reload_limits(self, limits: tuple[int, ...]) -> None:
        self._load_snapshots_for_limits(limits)

    def _load_snapshots_for_limits(self, limits: tuple[int, ...]) -> None:
        if self._ladder_loader is None:
            return
        for limit in limits:
            snapshot = self._ladder_loader(limit)
            self.call_from_thread(self._set_ladder_snapshot, snapshot, limit)

    def _set_ladder_snapshot(self, snapshot: LabSnapshot, limit: int) -> None:
        self._loaded_section_limit = max(self._loaded_section_limit, limit)
        self._set_snapshot(snapshot)

    def _set_snapshot(self, snapshot: LabSnapshot) -> None:
        self._snapshot = snapshot
        if snapshot.section(self._active_section_key) is None and snapshot.sections:
            self._active_section_key = snapshot.sections[0].key
        self._render_topbar()
        self._render_tree()
        self._render_active_section()

    @work(thread=True, exclusive=True, group="detail-load")
    def _load_detail_worker(self, item: LabItem, include_logs: bool) -> None:
        if self._detail_loader is None:
            return
        started_at = time.perf_counter()
        try:
            detailed = self._detail_loader(
                item,
                include_logs,
                _LOG_TAIL_STEPS[0],
                _RUN_METRIC_PREVIEW_LIMIT,
                None,
            )
            detailed = _with_metric_load_seconds(detailed, time.perf_counter() - started_at)
        except Exception as exc:
            detailed = LabItem(
                key=item.key,
                section=item.section,
                title=item.title,
                subtitle=str(exc),
                status="error",
                status_style=STATUS_ERROR,
                metadata=item.metadata,
                raw=item.raw,
            )
        self.call_from_thread(self._replace_selected_item, detailed)

    @work(thread=True, exclusive=True, group="detail-prefetch")
    def _prefetch_training_detail_worker(self, item: LabItem) -> None:
        if self._detail_loader is None:
            return
        started_at = time.perf_counter()
        try:
            detailed = self._detail_loader(
                item,
                False,
                _LOG_TAIL_STEPS[0],
                _RUN_METRIC_PREVIEW_LIMIT,
                None,
            )
            detailed = _with_metric_load_seconds(detailed, time.perf_counter() - started_at)
        except Exception:
            self.call_from_thread(self._finish_training_prefetch, item.key, None)
            return
        self.call_from_thread(self._finish_training_prefetch, item.key, detailed)

    def _render_topbar(self) -> None:
        if self._snapshot is None:
            return
        auth = "authenticated" if self._snapshot.authenticated else "public"
        team = self._snapshot.team or "personal"
        warnings = f"  {len(self._snapshot.warnings)} warnings" if self._snapshot.warnings else ""
        left = Text()
        left.append("Lab", style="bold")
        left.append(f"\n{auth} · {team} · {self._snapshot.workspace}{warnings}", style="dim")
        topbar = Table.grid(expand=True)
        topbar.add_column(ratio=1)
        topbar.add_column(justify="right", no_wrap=True)
        topbar.add_row(left, _lab_logo_text())
        self.query_one("#topbar", Static).update(topbar)

    def _render_tree(self) -> None:
        if self._snapshot is None:
            return
        tree = self.query_one("#section-tree", Tree)
        tree.clear()
        root = tree.root
        root.expand()
        selected_node = None
        for section in self._snapshot.sections:
            label = Text()
            label.append(section.title, style="bold")
            label.append(f"  {section.status}", style=section.status_style)
            node = root.add(label, data=section.key, allow_expand=False)
            if section.key == self._active_section_key:
                selected_node = node
        if selected_node is not None:
            self.call_after_refresh(lambda: tree.move_cursor(selected_node))

    def _render_active_section(self) -> None:
        if self._snapshot is None:
            return
        section = self._snapshot.section(self._active_section_key)
        if section is None:
            return
        selected_key = (
            self._selected_item.key
            if self._selected_item is not None and self._selected_item.section == section.key
            else None
        )

        title = section.title
        if self._filter:
            title = f"{title} / {self._filter}"
        self.query_one("#section-title", Label).update(title)
        self.query_one("#section-subtitle", Static).update(section.description)

        if section.key == "workspace":
            self.query_one("#evaluation-toggle", Static).display = False
            self.query_one("#evaluation-tree", EvaluationTree).display = False
            self._render_workspace_home(section, selected_key)
            return

        self.query_one("#home-toggle", Static).display = False

        if section.key == "evaluations":
            self._render_evaluations_section(section, selected_key)
            return

        self.query_one("#evaluation-toggle", Static).display = False
        self.query_one("#evaluation-tree", EvaluationTree).display = False
        option_list = self.query_one("#item-list", OptionList)
        option_list.display = True
        option_list.clear_options()
        self._items_by_key = {}
        self._visible_items = []
        for item in section.items:
            item = self._detail_cache.get(item.key, item)
            if not self._filter or self._matches_filter(item):
                self._visible_items.append(item)

        if not self._visible_items:
            empty = LabItem(
                key=f"{section.key}:empty",
                section=section.key,
                title="No rows",
                subtitle="Clear the filter or refresh the view.",
                status="empty",
                status_style="dim",
            )
            self._visible_items = [empty]

        for item in self._visible_items:
            self._items_by_key[item.key] = item
            option_list.add_option(Option(_item_label(item), id=item.key))

        self.call_after_refresh(lambda: self._highlight_item(option_list, selected_key))

    def _render_evaluations_section(self, section: LabSection, selected_key: str | None) -> None:
        toggle = self.query_one("#evaluation-toggle", EvaluationViewToggle)
        toggle.display = True
        toggle.update_views(EVALUATION_VIEWS, self._evaluation_view)

        self._items_by_key = {}
        self._visible_items = []
        for item in section.items:
            item = self._detail_cache.get(item.key, item)
            if not self._filter or self._matches_filter(item):
                self._visible_items.append(item)
                self._items_by_key[item.key] = item
        self._evaluation_tree_index = evaluation_index(self._visible_items)

        option_list = self.query_one("#item-list", OptionList)
        evaluation_tree = self.query_one("#evaluation-tree", EvaluationTree)
        if self._evaluation_view == "env":
            option_list.display = False
            option_list.clear_options()
            evaluation_tree.display = True
            self._populate_evaluation_tree(evaluation_tree, selected_key)
            return

        evaluation_tree.display = False
        evaluation_tree.clear()
        option_list.display = True
        option_list.clear_options()

        if not self._visible_items:
            empty = LabItem(
                key="evaluations:empty",
                section="evaluations",
                title="No rows",
                subtitle="Clear the filter or refresh the view.",
                status="empty",
                status_style="dim",
            )
            self._visible_items = [empty]
            self._items_by_key[empty.key] = empty

        for item in self._visible_items:
            self._items_by_key[item.key] = item
            option_list.add_option(Option(_item_label(item), id=item.key))

        self.call_after_refresh(lambda: self._highlight_item(option_list, selected_key))

    def _populate_evaluation_tree(self, tree: EvaluationTree, selected_key: str | None) -> None:
        tree.clear()
        root = tree.root
        root.expand()
        first_run_node = None
        selected_node = None
        if not self._evaluation_tree_index:
            root.add("No evaluation runs", allow_expand=False)
            self.query_one("#inspector-title", Label).update("Selection Details")
            self.query_one("#inspector", Static).update(Text("No evaluation runs", style="dim"))
            return

        for env_index, env_id in enumerate(sorted(self._evaluation_tree_index)):
            models = self._evaluation_tree_index[env_id]
            total_runs = sum(len(runs) for runs in models.values())
            env_node = root.add(
                evaluation_env_tree_label(env_id, models, total_runs),
                data=EvaluationNodeData(kind="env", env_id=env_id),
                expand=env_index == 0,
            )
            for model_index, model in enumerate(sorted(models)):
                runs = sorted_evaluation_runs(models[model])
                model_node = env_node.add(
                    evaluation_model_tree_label(model, runs),
                    data=EvaluationNodeData(kind="model", env_id=env_id, model=model),
                    expand=env_index == 0 and model_index == 0,
                )
                for item in runs:
                    self._items_by_key[item.key] = item
                    run_node = model_node.add(
                        evaluation_run_tree_label(item),
                        data=EvaluationNodeData(
                            kind="run",
                            env_id=env_id,
                            model=model,
                            item_key=item.key,
                        ),
                        allow_expand=False,
                    )
                    if first_run_node is None:
                        first_run_node = run_node
                    if selected_key == item.key:
                        selected_node = run_node

        target_node = selected_node or first_run_node
        if target_node is not None:
            self.call_after_refresh(lambda: tree.move_cursor(target_node))

    def _render_workspace_home(self, section: LabSection, selected_key: str | None) -> None:
        toggle = self.query_one("#home-toggle", HomeGroupToggle)
        toggle.display = True
        self._items_by_key = {}
        workspace_items: list[LabItem] = []
        for item in section.items:
            item = self._detail_cache.get(item.key, item)
            if not self._filter or self._matches_filter(item):
                workspace_items.append(item)
                self._items_by_key[item.key] = item

        groups = _workspace_home_groups(workspace_items)
        available_keys = [key for key, _, items in groups if items]
        if self._home_group not in available_keys and available_keys:
            self._home_group = available_keys[0]
        active_items = next(
            (items for key, _, items in groups if key == self._home_group),
            [],
        )
        self._visible_items = active_items
        toggle.update_groups(groups, self._home_group)

        option_list = self.query_one("#item-list", OptionList)
        option_list.display = True
        option_list.clear_options()
        if not self._visible_items:
            empty = LabItem(
                key="workspace:empty",
                section="workspace",
                title="No rows",
                subtitle="Switch category or refresh the view.",
                status="empty",
                status_style="dim",
            )
            self._visible_items = [empty]
            self._items_by_key[empty.key] = empty
        for item in self._visible_items:
            self._items_by_key[item.key] = item
            option_list.add_option(Option(_item_label(item), id=item.key))
        self.call_after_refresh(lambda: self._highlight_item(option_list, selected_key))

    def _highlight_item(self, option_list: OptionList, key: str | None) -> None:
        if not option_list.option_count:
            return
        index = 0
        if key is not None:
            for item_index, item in enumerate(self._visible_items):
                if item.key == key:
                    index = item_index
                    break
        option_list.highlighted = index
        self._show_item(self._visible_items[index])

    def _matches_filter(self, item: LabItem) -> bool:
        haystack = " ".join(
            [
                item.title,
                item.subtitle,
                item.status,
                *(value for _, value in item.metadata),
            ]
        ).lower()
        return self._filter in haystack

    def _show_item(self, item: LabItem) -> None:
        item = self._detail_cache.get(item.key, item)
        self._selected_item = item
        if item.section != "training" or self._expand_ready_run_key != item.key:
            self._expand_ready_run_key = None
        if item.section == "evaluations":
            self.query_one("#inspector-title", Label).update("Selection Details")
            self.query_one("#inspector", Static).update(self._evaluation_run_details(item))
        else:
            self.query_one("#inspector-title", Label).update(item.title)
            self.query_one("#inspector", Static).update(_item_details(item))
        self._prefetch_training_detail(item)

    def is_training_item_key(self, key: str) -> bool:
        item = self._items_by_key.get(key)
        return item is not None and item.section == "training"

    def is_guarded_option_key(self, key: str) -> bool:
        return self.is_training_item_key(key)

    def is_run_expand_ready(self, key: str) -> bool:
        return self._expand_ready_run_key == key

    def is_option_expand_ready(self, key: str) -> bool:
        return self.is_run_expand_ready(key)

    def arm_training_run_from_mouse(self, key: str) -> None:
        item = self._items_by_key.get(key)
        if item is None or item.section != "training":
            return
        self._show_item(item)
        self._expand_ready_run_key = item.key

    def arm_option_from_mouse(self, key: str) -> None:
        self.arm_training_run_from_mouse(key)

    def _evaluation_run_details(self, item: LabItem) -> Group:
        stats: RunOverviewStats | None = None
        if item.raw.get("type") == "local_eval":
            cache_key = local_eval_stats_key(item)
            stats = self._local_eval_stats_cache.get(cache_key)
            if stats is None and cache_key not in self._loading_local_eval_stats:
                self._loading_local_eval_stats.add(cache_key)
                self._load_local_eval_stats_worker(item)
        return evaluation_run_selection_details(
            item,
            stats,
            platform_detail_chunks=_evaluation_detail_chunks,
        )

    @work(thread=True, exclusive=False, group="local-eval-stats")
    def _load_local_eval_stats_worker(self, item: LabItem) -> None:
        cache_key = local_eval_stats_key(item)
        stats = compute_run_overview_stats(LocalEvalRun.from_item(item))
        self.call_from_thread(self._finish_local_eval_stats, item.key, cache_key, stats)

    def _finish_local_eval_stats(
        self, item_key: str, cache_key: str, stats: RunOverviewStats
    ) -> None:
        self._loading_local_eval_stats.discard(cache_key)
        self._local_eval_stats_cache[cache_key] = stats
        if self._selected_item is not None and self._selected_item.key == item_key:
            item = self._detail_cache.get(item_key, self._selected_item)
            self.query_one("#inspector-title", Label).update("Selection Details")
            self.query_one("#inspector", Static).update(
                evaluation_run_selection_details(
                    item,
                    stats,
                    platform_detail_chunks=_evaluation_detail_chunks,
                )
            )

    def _prefetch_training_detail(self, item: LabItem) -> None:
        if (
            item.section != "training"
            or self._detail_loader is None
            or item.raw.get("loading")
            or _has_training_detail(item)
            or (cached := self._detail_cache.get(item.key)) is not None
            and _has_training_detail(cached)
            or self._prefetching_detail_key == item.key
        ):
            return
        self._prefetching_detail_key = item.key
        self._prefetch_training_detail_worker(item)

    def _load_selected_detail(self, *, include_logs: bool) -> None:
        if self._selected_item is None:
            return
        item = self._detail_cache.get(self._selected_item.key, self._selected_item)
        if item.raw.get("loading"):
            return
        if item.section == "evaluations" and item.raw.get("type") == "local_eval":
            self.push_screen(LocalEvalRunScreen(LocalEvalRun.from_item(item)))
            return
        if self._detail_loader is None:
            return
        if item.section == "training":
            self.push_screen(
                TrainingRunScreen(item, self._detail_loader, include_logs=include_logs)
            )
            return
        if item.section not in {"training", "environments", "evaluations"}:
            return
        self.query_one("#inspector-title", Label).update(item.title)
        self.query_one("#inspector", Static).update(
            Text("Loading logs ..." if include_logs else "Loading details ...", style="dim")
        )
        self._load_detail_worker(item, include_logs)

    def _replace_selected_item(self, item: LabItem) -> None:
        if item.section == "training":
            self._detail_cache[item.key] = _merge_training_detail(
                self._detail_cache.get(item.key), item
            )
            item = self._detail_cache[item.key]
        elif item.section in {"environments", "evaluations"}:
            self._detail_cache[item.key] = item
        self._items_by_key[item.key] = item
        self._visible_items = [
            item if visible.key == item.key else visible for visible in self._visible_items
        ]
        self._show_item(item)

    def _finish_training_prefetch(self, key: str, item: LabItem | None) -> None:
        if self._prefetching_detail_key == key:
            self._prefetching_detail_key = None
        if item is None:
            return
        self._detail_cache[key] = _merge_training_detail(self._detail_cache.get(key), item)
        item = self._detail_cache[key]
        self._items_by_key[key] = item
        self._visible_items = [
            item if visible.key == key else visible for visible in self._visible_items
        ]
        if self._selected_item is not None and self._selected_item.key == key:
            self._show_item(item)

    @on(Tree.NodeSelected, "#section-tree")
    def _section_selected(self, event: Tree.NodeSelected) -> None:
        key = getattr(event.node, "data", None)
        if isinstance(key, str):
            self._active_section_key = key
            self._filter = ""
            self._render_active_section()

    @on(Tree.NodeHighlighted, "#section-tree")
    def _section_highlighted(self, event: Tree.NodeHighlighted) -> None:
        key = getattr(event.node, "data", None)
        if isinstance(key, str):
            self._active_section_key = key
            self._render_active_section()

    @on(Tree.NodeHighlighted, "#evaluation-tree")
    def _evaluation_tree_highlighted(self, event: Tree.NodeHighlighted) -> None:
        self._show_evaluation_tree_selection(getattr(event.node, "data", None))

    @on(Tree.NodeSelected, "#evaluation-tree")
    def _evaluation_tree_selected(self, event: Tree.NodeSelected) -> None:
        payload = getattr(event.node, "data", None)
        if not isinstance(payload, EvaluationNodeData):
            return
        self._show_evaluation_tree_selection(payload)
        if payload.kind == "run":
            if self._evaluation_click_selected_node is event.node:
                self._open_evaluation_tree_run(payload)
            else:
                self._evaluation_click_selected_node = event.node
            return
        self._evaluation_click_selected_node = None
        if event.node.allow_expand:
            event.node.toggle()

    @on(OptionList.OptionHighlighted, "#item-list")
    def _item_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        item = self._items_by_key.get(str(event.option.id))
        if item is not None:
            self._show_item(item)

    @on(OptionList.OptionSelected, "#item-list")
    def _item_selected(self, event: OptionList.OptionSelected) -> None:
        item = self._items_by_key.get(str(event.option.id))
        if item is not None:
            self._show_item(item)
            option_list = event.option_list
            mouse_option_id: str | None = None
            mouse_was_armed = False
            if isinstance(option_list, LabOptionList):
                mouse_option_id, mouse_was_armed = option_list.consume_mouse_selection()
            is_mouse_activation = mouse_option_id == item.key
            if is_mouse_activation and item.section == "training":
                if not mouse_was_armed:
                    self._prefetch_training_detail(item)
                    return
            self._load_selected_detail(include_logs=False)

    def _show_evaluation_tree_selection(self, payload: Any) -> None:
        if not isinstance(payload, EvaluationNodeData):
            return
        self.query_one("#inspector-title", Label).update("Selection Details")
        if payload.kind == "run":
            item = self._items_by_key.get(payload.item_key)
            if item is None:
                return
            self._selected_item = item
            self.query_one("#inspector", Static).update(self._evaluation_run_details(item))
            return
        self._selected_item = None
        self._expand_ready_run_key = None
        self.query_one("#inspector", Static).update(
            evaluation_group_selection_details(payload, self._evaluation_tree_index)
        )

    def _open_evaluation_tree_run(self, payload: EvaluationNodeData) -> None:
        item = self._items_by_key.get(payload.item_key)
        if item is None:
            return
        self._show_item(item)
        self._load_selected_detail(include_logs=False)

    def _handle_evaluation_tree_enter(self) -> bool:
        if self._active_section_key != "evaluations" or self._evaluation_view != "env":
            return False
        if not isinstance(self.focused, EvaluationTree):
            return False
        tree = self.query_one("#evaluation-tree", EvaluationTree)
        node = tree.cursor_node
        if node is None:
            return False
        payload = getattr(node, "data", None)
        if not isinstance(payload, EvaluationNodeData):
            return False
        if payload.kind == "run":
            self._open_evaluation_tree_run(payload)
            return True
        if node.allow_expand:
            node.toggle()
            return True
        return False

    def _main_open_action_available(self) -> bool:
        if self._active_section_key == "evaluations" and self._evaluation_view == "env":
            return isinstance(self.focused, EvaluationTree)
        return self._selected_item is not None and not self._selected_item.raw.get("loading")

    def _filter_choices_for_active_section(self) -> list[FilterChoice]:
        if self._snapshot is None:
            return []
        section = self._snapshot.section(self._active_section_key)
        if section is None:
            return []
        if section.key == "workspace":
            items = self._workspace_items()
        else:
            items = [self._detail_cache.get(item.key, item) for item in section.items]
        return [_filter_choice_for_item(item) for item in items]


def run_lab_view(
    *,
    limit: int = 100,
    env_dir: str = "./environments",
    outputs_dir: str = "./outputs",
    workspace: Path | None = None,
) -> None:
    workspace = workspace or Path.cwd()

    def options_for(current_limit: int) -> LabLoadOptions:
        return LabLoadOptions(
            limit=current_limit,
            workspace=workspace,
            env_dir=env_dir,
            outputs_dir=outputs_dir,
        )

    from .data import LabDataSource

    data_source = LabDataSource()
    PrimeLabView(
        lambda: data_source.load(options_for(limit)),
        lambda item, include_logs, log_tail_lines, metrics_limit, metrics_min_step: (
            data_source.load_item_detail(
                item,
                include_logs=include_logs,
                log_tail_lines=log_tail_lines,
                metrics_limit=metrics_limit,
                metrics_min_step=metrics_min_step,
            )
        ),
        lambda: data_source.load_initial(options_for(limit)),
        lambda current_limit: data_source.load(options_for(current_limit)),
        _ladder_limits(limit),
    ).run()


def _ladder_limits(max_limit: int, *, start: int = 5) -> tuple[int, ...]:
    if max_limit <= start:
        return (max_limit,)
    limits = []
    current = start
    while current < max_limit:
        limits.append(current)
        current *= 2
    limits.append(max_limit)
    return tuple(limits)


def _filter_choice_for_item(item: LabItem) -> FilterChoice:
    return FilterChoice(
        key=item.key,
        label=_item_label(item),
        search_text=_item_search_text(item),
        value=item.title,
    )


def _item_search_text(item: LabItem) -> str:
    return " ".join(
        [
            item.title,
            item.subtitle,
            item.status,
            *(value for _, value in item.metadata),
        ]
    ).lower()


def _item_label(item: LabItem) -> Text:
    text = Text()
    text.append(item.title, style="bold")
    if item.status:
        text.append("  ")
        text.append(item.status, style=item.status_style)
    if item.subtitle:
        text.append("\n")
        text.append(item.subtitle, style="dim")
    return text


def _workspace_home_groups(items: list[LabItem]) -> list[tuple[str, str, list[LabItem]]]:
    group_order = [
        ("workspaces", "Workspaces"),
        ("profiles", "Profiles"),
        ("environments", "Environments"),
        ("configs", "Configs"),
    ]
    groups = {key: [] for key, _ in group_order}
    for item in items:
        group = _workspace_home_group(item)
        groups.setdefault(group, []).append(item)
    return [(key, label, groups[key]) for key, label in group_order if groups.get(key)]


def _workspace_home_group(item: LabItem) -> str:
    item_type = item.raw.get("type")
    if item_type == "auth_profile":
        return "profiles"
    if item_type == "local_environment":
        return "environments"
    if item_type == "config_file":
        return "configs"
    return "workspaces"


def _lab_logo_text() -> Text:
    text = Text()
    text.append("PRIME", style="bold white")
    text.append(" Intellect", style="italic white")
    return text


def _item_details(item: LabItem) -> Group:
    metadata = Table.grid(padding=(0, 2))
    metadata.add_column(style="bold dim", no_wrap=True)
    metadata.add_column()
    if item.status:
        metadata.add_row("Status", Text(item.status, style=item.status_style))
    metadata_rows = item.metadata
    if item.section == "training":
        metadata_rows = tuple(
            (key, value) for key, value in item.metadata if key not in {"Model", "Environments"}
        )
    for key, value in metadata_rows:
        metadata.add_row(key, value)

    chunks: list[Any] = []
    if item.section == "training" and item.raw:
        chunks.append(_training_context_text(item))
        chunks.append(Text(""))
    elif item.subtitle:
        chunks.append(Text(item.subtitle))
        chunks.append(Text(""))
    chunks.append(metadata)

    if item.raw:
        if item.section == "training":
            config = _training_config_toml(item.raw)
            if config:
                chunks.extend(
                    [
                        Text(""),
                        Text("Config", style="bold dim"),
                        Syntax(config, "toml", theme="monokai"),
                    ]
                )
            return Group(*chunks)
        if item.section == "workspace":
            chunks.extend(_workspace_detail_chunks(item))
            return Group(*chunks)
        if item.section in {"evaluations", "local-evals"}:
            chunks.extend(_evaluation_detail_chunks(item.raw))
            return Group(*chunks)
        if item.section == "environments":
            chunks.extend(_environment_detail_chunks(item.raw))
            return Group(*chunks)

        histograms = _histogram_charts_from_raw(item.raw)
        if histograms:
            chunks.extend([Text(""), Text("Charts", style="bold dim")])
            for _, chart in histograms:
                chunks.append(chart)

        raw_data = dict(item.raw)
        logs_tail = raw_data.pop("logs_tail", None)
        raw = json.dumps(raw_data, indent=2, sort_keys=True, default=str)
        if len(raw) > 6000:
            raw = raw[:6000].rstrip() + "\n..."
        chunks.extend(
            [Text(""), Text("Raw", style="bold dim"), Syntax(raw, "json", theme="monokai")]
        )
        if isinstance(logs_tail, str) and logs_tail:
            logs = logs_tail
            if len(logs) > 12000:
                logs = logs[-12000:]
            chunks.extend([Text(""), Text("Logs", style="bold dim"), Text(logs)])

    return Group(*chunks)


def _training_context_text(item: LabItem) -> Text:
    raw = item.raw
    model = str(
        raw.get("base_model") or raw.get("baseModel") or _metadata_value(item, "Model") or "-"
    )
    envs = raw.get("environments")
    env_names = (
        [_environment_display_name(env) for env in envs]
        if isinstance(envs, list)
        else [str(_metadata_value(item, "Environments") or "-")]
    )
    env_names = [name for name in env_names if name]
    if not env_names:
        env_names = ["-"]

    text = Text()
    text.append("Model", style="bold dim")
    text.append(f"\n{model}")
    text.append("\n\nEnvironments", style="bold dim")
    for env_name in env_names:
        text.append(f"\n{env_name}")
    return text


def _workspace_detail_chunks(item: LabItem) -> list[Any]:
    raw = item.raw
    kind = raw.get("type")
    chunks: list[Any] = []
    if kind == "workspace_context":
        profiles = raw.get("profiles")
        if isinstance(profiles, list) and profiles:
            table = Table.grid(padding=(0, 2))
            table.add_column(style="bold")
            table.add_column(style="dim")
            current = str(raw.get("profile") or "production")
            for profile in profiles:
                profile_text = str(profile)
                table.add_row(
                    profile_text,
                    "current" if profile_text == current else f"prime config use {profile_text}",
                )
            chunks.extend([Text(""), Text("Auth Profiles", style="bold dim"), table])
        return chunks

    command = raw.get("command")
    if isinstance(command, str) and command:
        actions = Table.grid(padding=(0, 2))
        actions.add_column(style="bold dim", no_wrap=True)
        actions.add_column()
        actions.add_row("Run", command)
        path = raw.get("relative_path")
        if isinstance(path, str) and path:
            actions.add_row("Edit", f"$EDITOR {path}")
        chunks.extend([Text(""), Text("Actions", style="bold dim"), actions])

    if kind == "local_environment":
        files = raw.get("files")
        if isinstance(files, list) and files:
            table = Table.grid()
            for file_name in files:
                table.add_row(str(file_name))
            chunks.extend([Text(""), Text("Files", style="bold dim"), table])
        return chunks

    if kind == "config_file":
        toml_text = raw.get("toml")
        if isinstance(toml_text, str) and toml_text:
            chunks.extend(
                [
                    Text(""),
                    Text("Config", style="bold dim"),
                    Syntax(toml_text.rstrip(), "toml", theme="monokai"),
                ]
            )
        return chunks

    return chunks


def _environment_display_name(env: Any) -> str:
    if not isinstance(env, dict):
        return str(env)
    return str(env.get("slug") or env.get("id") or env.get("name") or "-")


def _evaluation_detail_chunks(raw: dict[str, Any]) -> list[Any]:
    chunks: list[Any] = []

    overview = _evaluation_overview_table(raw)
    if overview.row_count:
        chunks.extend([Text(""), Text("Evaluation", style="bold dim"), overview])

    histograms = _histogram_charts_from_raw(raw)
    for title, chart in histograms:
        if not any(isinstance(chunk, Text) and chunk.plain == "Charts" for chunk in chunks):
            chunks.extend([Text(""), Text("Charts", style="bold dim")])
        chunks.append(chart)
        stats = _evaluation_stats_table(raw, chart_title=title)
        if stats.row_count:
            chunks.append(stats)
    if not histograms:
        stats = _evaluation_stats_table(raw, chart_title="Stats")
        if stats.row_count:
            chunks.extend([Text(""), Text("Stats", style="bold dim"), stats])

    detailed = _detailed_metrics_table(raw)
    if detailed is not None:
        chunks.extend([Text(""), Text("Detailed Metrics", style="bold dim"), detailed])

    rubric = _rubric_table(raw)
    if rubric is not None:
        chunks.extend([Text(""), Text("Rubric", style="bold dim"), rubric])

    samples = _samples_preview_table(raw)
    if samples is not None:
        chunks.extend([Text(""), Text("Samples Preview", style="bold dim"), samples])

    return chunks


def _evaluation_overview_table(raw: dict[str, Any]) -> Table:
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold dim", no_wrap=True)
    table.add_column()
    rows = [
        ("Name", raw.get("name")),
        ("Type", raw.get("eval_type") or ("hosted" if raw.get("is_hosted") else None)),
        ("Dataset", raw.get("dataset")),
        ("Framework", raw.get("framework")),
        ("Task", raw.get("task_type")),
        ("Total samples", raw.get("total_samples") or raw.get("totalSamples")),
        ("Avg score", raw.get("avg_score") or raw.get("avgScore")),
        ("Min score", raw.get("min_score") or raw.get("minScore")),
        ("Max score", raw.get("max_score") or raw.get("maxScore")),
        ("Started", _format_detail_time(raw.get("started_at") or raw.get("startedAt"))),
        ("Completed", _format_detail_time(raw.get("completed_at") or raw.get("completedAt"))),
        ("Viewer", raw.get("viewer_url") or raw.get("viewerUrl")),
        ("Error", raw.get("error_message") or raw.get("errorMessage")),
    ]
    for key, value in rows:
        text = _display_value(value)
        if text != "-":
            table.add_row(key, text)
    return table


def _evaluation_stats_table(raw: dict[str, Any], *, chart_title: str) -> Table:
    stats = _chart_stats(raw)
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold dim", no_wrap=True)
    table.add_column()
    if not stats:
        return table

    preferred = (
        ("avgReward", "Mean"),
        ("medianReward", "Median"),
        ("minReward", "Min"),
        ("maxReward", "Max"),
        ("stdDev", "Std dev"),
        ("totalResults", "Results"),
        ("totalReward", "Total"),
    )
    table.add_row(chart_title, "")
    for key, label in preferred:
        if key in stats:
            table.add_row(label, _display_value(stats[key]))
    for key, value in stats.items():
        if key not in {stat_key for stat_key, _ in preferred}:
            table.add_row(_humanize_key(key), _display_value(value))
    return table


def _chart_stats(raw: dict[str, Any]) -> dict[str, Any]:
    chart_data = raw.get("chartData") or raw.get("chart_data")
    if isinstance(chart_data, dict) and isinstance(chart_data.get("stats"), dict):
        return chart_data["stats"]
    for key in ("statistics", "rewardStats", "reward_stats"):
        value = raw.get(key)
        if isinstance(value, dict):
            return value
    return {}


def _detailed_metrics_table(raw: dict[str, Any]) -> Table | None:
    detailed = raw.get("detailedMetrics") or raw.get("detailed_metrics")
    if not isinstance(detailed, dict):
        return None
    metric_keys = detailed.get("detectedMetrics") or detailed.get("detected_metrics")
    metrics_stats = detailed.get("metricsStats") or detailed.get("metrics_stats")
    if not isinstance(metric_keys, list) or not isinstance(metrics_stats, dict):
        return None

    table = Table(show_header=True, header_style="bold dim")
    table.add_column("Metric")
    table.add_column("Mean", justify="right")
    table.add_column("Std dev", justify="right")
    table.add_column("Count", justify="right")
    table.add_column("p50", justify="right")
    for metric_key in metric_keys:
        stats = metrics_stats.get(metric_key)
        if not isinstance(stats, dict):
            continue
        table.add_row(
            str(metric_key),
            _display_value(stats.get("mean")),
            _display_value(stats.get("stdDev") or stats.get("std_dev")),
            _display_value(stats.get("count")),
            _display_value(stats.get("p50")),
        )
    return table if table.row_count else None


def _rubric_table(raw: dict[str, Any]) -> Table | None:
    rubric = raw.get("rubricInfo") or raw.get("rubric_info")
    if not isinstance(rubric, dict):
        return None
    rows = {
        key: value
        for key, value in rubric.items()
        if value is not None and value != {} and value != []
    }
    if not rows:
        return None
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold dim", no_wrap=True)
    table.add_column()
    for key, value in rows.items():
        table.add_row(_humanize_key(key), _display_value(value))
    return table


def _samples_preview_table(raw: dict[str, Any]) -> Table | None:
    preview = raw.get("samples_preview")
    if not isinstance(preview, dict):
        return None
    samples = preview.get("samples")
    if not isinstance(samples, list) or not samples:
        return None

    table = Table(show_header=True, header_style="bold dim")
    keys = _sample_preview_keys(samples)
    for key in keys:
        table.add_column(_humanize_key(key))
    for sample in samples[:5]:
        if not isinstance(sample, dict):
            continue
        table.add_row(*(_display_value(sample.get(key)) for key in keys))
    return table if table.row_count else None


def _sample_preview_keys(samples: list[Any]) -> list[str]:
    preferred = ["reward", "score", "prompt", "answer", "completion", "output"]
    keys: list[str] = []
    for key in preferred:
        if any(isinstance(sample, dict) and key in sample for sample in samples):
            keys.append(key)
    if keys:
        return keys[:4]
    for sample in samples:
        if isinstance(sample, dict):
            return [str(key) for key in list(sample.keys())[:4]]
    return []


def _environment_detail_chunks(raw: dict[str, Any]) -> list[Any]:
    if raw.get("loading"):
        return []

    chunks: list[Any] = []
    detail = _environment_detail_table(raw)
    if detail.row_count:
        chunks.extend([Text(""), Text("Environment", style="bold dim"), detail])

    status = raw.get("status")
    if isinstance(status, dict):
        status_table = _environment_status_table(status)
        if status_table.row_count:
            chunks.extend([Text(""), Text("Status", style="bold dim"), status_table])

    install = _environment_install_table(raw)
    if install.row_count:
        chunks.extend([Text(""), Text("Install", style="bold dim"), install])

    return chunks


def _environment_detail_table(raw: dict[str, Any]) -> Table:
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold dim", no_wrap=True)
    table.add_column()
    for key, value in (
        ("Version", raw.get("semantic_version") or raw.get("latest_version")),
        ("Content hash", raw.get("content_hash")),
        ("Visibility", raw.get("visibility")),
        ("Description", raw.get("description")),
        ("Created", _format_detail_time(raw.get("created_at") or raw.get("createdAt"))),
        ("Updated", _format_detail_time(raw.get("updated_at") or raw.get("updatedAt"))),
    ):
        text = _display_value(value)
        if text != "-":
            table.add_row(key, text)
    return table


def _environment_status_table(status: dict[str, Any]) -> Table:
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold dim", no_wrap=True)
    table.add_column()
    for key, value in status.items():
        if isinstance(value, dict):
            version = value.get("semantic_version") or value.get("version")
            state = value.get("latest_ci_status") or value.get("status")
            text = " · ".join(
                part for part in (_display_value(version), _display_value(state)) if part != "-"
            )
        else:
            text = _display_value(value)
        if text != "-":
            table.add_row(_humanize_key(str(key)), text)
    return table


def _environment_install_table(raw: dict[str, Any]) -> Table:
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold dim", no_wrap=True)
    table.add_column()
    for key, value in (
        ("Wheel", raw.get("wheel_url") or raw.get("wheelUrl")),
        ("Index", raw.get("simple_index_url") or raw.get("simpleIndexUrl")),
    ):
        text = _display_value(value)
        if text != "-":
            table.add_row(key, text)
    return table


def _display_value(value: Any) -> str:
    if value is None or value == "" or value == [] or value == {}:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return _format_number(float(value))
    if isinstance(value, list):
        parts = [_display_value(child) for child in value]
        return ", ".join(part for part in parts if part != "-") or "-"
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True, default=str)
    return str(value)


def _format_detail_time(value: Any) -> str:
    if value is None:
        return "-"
    return str(value)


def _humanize_key(key: str) -> str:
    text = key.replace("_", " ")
    chars: list[str] = []
    for idx, char in enumerate(text):
        if idx > 0 and char.isupper() and text[idx - 1].islower():
            chars.append(" ")
        chars.append(char)
    return " ".join("".join(chars).split()).capitalize()


def _training_config_toml(raw: dict[str, Any]) -> str:
    config: dict[str, Any] = {}
    if name := raw.get("name"):
        config["name"] = name
    if model := raw.get("base_model") or raw.get("baseModel"):
        config["model"] = model

    for key in (
        "max_steps",
        "rollouts_per_example",
        "seq_len",
        "batch_size",
        "max_tokens",
        "learning_rate",
        "lora_alpha",
        "oversampling_factor",
        "max_async_level",
    ):
        if key in raw:
            config[key] = raw[key]

    environments = raw.get("environments")
    if isinstance(environments, list) and environments:
        config["environments"] = environments

    for key in ("run_config", "eval_config", "val_config", "buffer_config"):
        value = raw.get(key)
        if isinstance(value, dict):
            config[key] = value

    filtered = _filter_empty_values(config)
    return _space_toml_blocks(toml.dumps(filtered)).rstrip() if filtered else ""


def _space_toml_blocks(value: str) -> str:
    lines = value.splitlines()
    spaced: list[str] = []
    for line in lines:
        if line.startswith("[[") and spaced and spaced[-1] != "":
            spaced.append("")
        spaced.append(line)
    return "\n".join(spaced) + ("\n" if spaced else "")


def _filter_empty_values(value: Any) -> Any:
    if isinstance(value, dict):
        filtered = {
            str(key): _filter_empty_values(child)
            for key, child in value.items()
            if child is not None
        }
        return {
            key: child
            for key, child in filtered.items()
            if child is not None and child != {} and child != []
        }
    if isinstance(value, list):
        return [
            child
            for child in (_filter_empty_values(child) for child in value)
            if child is not None and child != {} and child != []
        ]
    return value


_RUN_TABS = (
    ("overview", "Overview"),
    ("data", "Data"),
    ("system", "System"),
)


def _adjacent_tab(active_tab: str, offset: int) -> str:
    keys = [key for key, _ in _RUN_TABS]
    idx = keys.index(active_tab) if active_tab in keys else 0
    return keys[(idx + offset) % len(keys)]


def _training_title(item: LabItem) -> Text:
    text = Text()
    name = _metadata_value(item, "Name")
    heading = name if name and name != "-" else item.title
    text.append(heading, style="bold")
    text.append(f"  {item.status}", style=item.status_style)
    text.append(f"\nTraining run · {item.title}", style="dim")
    return text


def _training_run_widgets(
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
) -> list[Widget]:
    raw = item.raw
    progress = _dict_value(raw.get("progress"))
    metrics = _list_value(raw.get("recent_metrics"))
    environments = _list_value(raw.get("environments"))
    environment_statuses = _list_value(raw.get("environment_statuses"))
    reward_distribution = _dict_value(raw.get("reward_distribution"))

    summary = _training_progress_summary(raw, progress, metrics)

    if active_tab == "data":
        return _training_data_page(raw, progress)

    if active_tab == "system":
        if not include_logs:
            return [Static(Text("Open the System tab or press l to load logs.", style="dim"))]
        logs = raw.get("logs_tail")
        if (not isinstance(logs, str) or not logs) and _should_show_logs_loading(raw):
            return [LoadingChart("Loading logs ...")]
        widgets: list[Widget] = []
        current_tail = _int_value(raw.get("log_tail_lines"))
        if (
            loading_log_tail_lines is not None
            and current_tail is not None
            and current_tail < loading_log_tail_lines
        ):
            widgets.append(LoadingMessage(f"Loading {loading_log_tail_lines:,} log lines ..."))
        widgets.append(
            Static(
                _visible_system_renderable(
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
    )


def _training_run_page(
    item: LabItem,
    *,
    include_logs: bool,
    active_tab: str = "overview",
    chart_mode: ChartMode = "metrics",
    metric_index: int = 0,
    distribution_index: int = 0,
) -> Group:
    widgets = _training_run_widgets(
        item,
        include_logs=include_logs,
        active_tab=active_tab,
        chart_mode=chart_mode,
        metric_index=metric_index,
        distribution_index=distribution_index,
        visible_metric_rows=None,
        loading_log_tail_lines=None,
        render_plots=False,
    )
    return Group(*(getattr(widget, "renderable", Text("")) for widget in widgets))


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
) -> list[Widget]:
    widgets: list[Widget] = [
        Static(
            _training_top_panel(
                item,
                raw,
                summary,
                environments,
                environment_statuses,
            )
        )
    ]
    widgets.extend(
        _training_chart_widgets(
            raw,
            metrics,
            reward_distribution,
            chart_mode=chart_mode,
            metric_index=metric_index,
            distribution_index=distribution_index,
            visible_metric_rows=visible_metric_rows,
            render_plots=render_plots,
        )
    )
    return widgets


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
        metrics = _visible_metric_rows(metrics, visible_metric_rows)
    charts = _chart_specs(metrics, reward_distribution, mode=chart_mode)
    widgets: list[Widget] = []
    if not charts:
        if _should_show_chart_loading(raw, chart_mode):
            message = (
                "Loading metrics ..."
                if chart_mode == "metrics"
                else "Loading reward distribution ..."
            )
            widgets.extend(
                [
                    Static(_loading_chart_heading(chart_mode), classes="run-chart-heading"),
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
    heading = _chart_heading_from_specs(
        charts,
        mode=chart_mode,
        metric_index=metric_index,
        distribution_index=distribution_index,
    )
    if not render_plots:
        widgets.extend(
            [
                Static(heading, classes="run-chart-heading"),
                Static(_chart_fallback(chart), classes="run-chart-fallback"),
            ]
        )
        return widgets
    widgets.extend([Static(heading, classes="run-chart-heading"), _plot_widget(chart)])
    return widgets


def _run_loading_widgets(*, include_logs: bool) -> list[Widget]:
    widgets: list[Widget] = [
        LoadingMessage("Loading metadata ..."),
        LoadingMessage("Loading environments ..."),
        Static(_loading_chart_heading("metrics"), classes="run-chart-heading"),
        LoadingChart("Loading metrics ..."),
    ]
    if include_logs:
        widgets.append(LoadingMessage("Loading logs ..."))
    return widgets


def _loading_chart_heading(mode: ChartMode) -> Text:
    return Text("Metrics" if mode == "metrics" else "Reward Distribution", style="bold")


def _should_show_chart_loading(raw: dict[str, Any], mode: ChartMode) -> bool:
    loaded_key = "metrics_loaded" if mode == "metrics" else "reward_distribution_loaded"
    if raw.get(loaded_key) is not True:
        return True
    return _run_may_still_emit_charts(raw)


def _run_may_still_emit_charts(raw: dict[str, Any]) -> bool:
    status = str(raw.get("status") or "").upper()
    return status in {"RUNNING", "PENDING", "STARTING", "QUEUED"}


def _next_log_tail_lines(current: int) -> int:
    if current < _LOG_DEFAULT_TAIL:
        return _LOG_DEFAULT_TAIL
    return current * 2


def _next_metrics_limit(current: int) -> int:
    if current >= _RUN_METRIC_FULL_LIMIT:
        return current
    return min(_RUN_METRIC_FULL_LIMIT, max(current + 1, current * 2))


def _next_metrics_page_limit(loaded_rows: int, page_limit: int | None) -> int:
    remaining = _RUN_METRIC_FULL_LIMIT - loaded_rows
    if remaining <= 0:
        return 0
    current_page_limit = page_limit or _RUN_METRIC_PREVIEW_LIMIT
    return min(remaining, _next_metrics_limit(current_page_limit))


def _metric_reveal_duration(
    load_seconds: float,
    *,
    current_limit: int,
    next_limit: int | None,
    start_rows: int,
    target_rows: int,
) -> float:
    if target_rows <= start_rows:
        return 0.0
    buffered = _estimated_metric_fetch_seconds(
        load_seconds,
        current_limit=current_limit,
        next_limit=next_limit,
    )
    return min(
        _METRIC_REVEAL_MAX_SECONDS,
        max(_METRIC_REVEAL_MIN_SECONDS, buffered),
    )


def _estimated_metric_fetch_seconds(
    load_seconds: float, *, current_limit: int, next_limit: int | None
) -> float:
    if next_limit is None or current_limit <= 0 or next_limit <= current_limit:
        return load_seconds * _METRIC_REVEAL_BUFFER
    ratio = next_limit / current_limit
    row_growth_factor = 1 + 0.35 * math.log2(ratio)
    return load_seconds * row_growth_factor * _METRIC_REVEAL_BUFFER


def _log_reveal_duration(
    load_seconds: float,
    *,
    current_tail: int,
    start_lines: int,
    target_lines: int,
) -> float:
    if target_lines <= start_lines:
        return 0.0
    line_growth = max(1.0, target_lines / max(current_tail, start_lines, 1))
    buffered = load_seconds * (1 + 0.2 * math.log2(line_growth)) * _LOG_REVEAL_BUFFER
    return min(_LOG_REVEAL_MAX_SECONDS, max(_LOG_REVEAL_MIN_SECONDS, buffered))


def _interpolated_metric_rows(
    *,
    now: float,
    start_time: float,
    deadline: float,
    start_rows: int,
    target_rows: int,
) -> int:
    if target_rows <= start_rows:
        return target_rows
    if deadline <= start_time or now >= deadline:
        return target_rows
    if now <= start_time:
        return start_rows
    progress = (now - start_time) / (deadline - start_time)
    visible_rows = start_rows + int((target_rows - start_rows) * progress)
    return min(target_rows, max(start_rows, visible_rows))


def _should_show_logs_loading(raw: dict[str, Any]) -> bool:
    if raw.get("logs_loaded") is not True:
        return True
    return _run_may_still_emit_charts(raw)


def _selected_chart(
    raw: dict[str, Any],
    *,
    mode: ChartMode,
    metric_index: int,
    distribution_index: int,
) -> ChartSpec | None:
    charts = _chart_specs_for_raw(raw, mode=mode)
    if not charts:
        return None
    selected_index = (metric_index if mode == "metrics" else distribution_index) % len(charts)
    return charts[selected_index]


def _chart_heading(
    raw: dict[str, Any],
    *,
    mode: ChartMode,
    metric_index: int,
    distribution_index: int,
) -> Text:
    return _chart_heading_from_specs(
        _chart_specs_for_raw(raw, mode=mode),
        mode=mode,
        metric_index=metric_index,
        distribution_index=distribution_index,
    )


def _chart_heading_from_specs(
    charts: list[ChartSpec],
    *,
    mode: ChartMode,
    metric_index: int,
    distribution_index: int,
) -> Text:
    heading = Text()
    if not charts:
        heading.append(
            "Metrics" if mode == "metrics" else "Reward Distribution",
            style="bold",
        )
        return heading

    selected_index = (metric_index if mode == "metrics" else distribution_index) % len(charts)
    label = "Metrics" if mode == "metrics" else "Reward Distribution"
    heading.append(
        f"{label} {selected_index + 1}/{len(charts)} | {charts[selected_index].title}",
        style="bold",
    )
    heading.append("\nUse up/down to switch charts, space to pick.", style="dim")
    return heading


def _training_data_page(raw: dict[str, Any], progress: dict[str, Any]) -> list[Widget]:
    steps = progress.get("steps_with_samples") or progress.get("stepsWithSamples")
    step_count = len(steps) if isinstance(steps, list) else 0
    rollout_samples = _dict_value(raw.get("rollout_samples"))
    samples = _list_value(rollout_samples.get("samples"))
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


def _chart_count(raw: dict[str, Any]) -> int:
    return len(_chart_specs_for_raw(raw))


def _chart_count_for_mode(raw: dict[str, Any], mode: ChartMode) -> int:
    return len(_chart_specs_for_raw(raw, mode=mode))


def _metric_chart_count(raw: dict[str, Any]) -> int:
    return len(_chart_specs_for_raw(raw, mode="metrics"))


def _distribution_chart_count(raw: dict[str, Any]) -> int:
    return len(_chart_specs_for_raw(raw, mode="distribution"))


def _chart_specs_for_raw(raw: dict[str, Any], *, mode: ChartMode | None = None) -> list[ChartSpec]:
    metrics = _list_value(raw.get("recent_metrics"))
    reward_distribution = _dict_value(raw.get("reward_distribution"))
    if mode is None:
        return [
            *_chart_specs(metrics, reward_distribution, mode="metrics"),
            *_chart_specs(metrics, reward_distribution, mode="distribution"),
        ]
    return _chart_specs(metrics, reward_distribution, mode=mode)


def _raw_with_visible_metric_rows(
    raw: dict[str, Any], visible_metric_rows: int | None
) -> dict[str, Any]:
    if visible_metric_rows is None:
        return raw
    metrics = _list_value(raw.get("recent_metrics"))
    visible_metrics = _visible_metric_rows(metrics, visible_metric_rows)
    if len(visible_metrics) == len(metrics):
        return raw
    return {**raw, "recent_metrics": visible_metrics}


def _visible_metric_rows(metrics: list[Any], visible_metric_rows: int | None) -> list[Any]:
    if visible_metric_rows is None or visible_metric_rows >= len(metrics):
        return metrics
    if visible_metric_rows <= 0:
        return []
    return metrics[:visible_metric_rows]


def _has_training_detail(item: LabItem) -> bool:
    return item.section == "training" and item.raw.get("metrics_loaded") is True


def _with_metric_load_seconds(item: LabItem, load_seconds: float) -> LabItem:
    if item.section != "training":
        return item
    return replace(item, raw={**item.raw, "metrics_load_seconds": load_seconds})


def _cached_metric_reveal_seconds(item: LabItem) -> float:
    value = item.raw.get("metrics_load_seconds")
    if isinstance(value, (int, float)) and value > 0:
        return max(_METRIC_CACHED_REVEAL_SECONDS, float(value))
    return _METRIC_CACHED_REVEAL_SECONDS


def _merge_training_detail(previous: LabItem | None, incoming: LabItem) -> LabItem:
    if (
        previous is None
        or previous.key != incoming.key
        or previous.section != "training"
        or incoming.section != "training"
    ):
        return incoming

    incoming_min_step = incoming.raw.get("metrics_min_step")
    previous_metrics = _list_value(previous.raw.get("recent_metrics"))
    incoming_metrics = _list_value(incoming.raw.get("recent_metrics"))
    if incoming_min_step is None:
        return incoming

    combined_metrics = _append_metric_rows(previous_metrics, incoming_metrics)
    raw = {
        **incoming.raw,
        "recent_metrics": combined_metrics,
        "metrics_limit": len(combined_metrics),
        "metrics_loaded": previous.raw.get("metrics_loaded") is True
        or incoming.raw.get("metrics_loaded") is True,
    }
    if "logs_tail" not in raw and "logs_tail" in previous.raw:
        raw["logs_tail"] = previous.raw["logs_tail"]
        raw["logs_loaded"] = previous.raw.get("logs_loaded")
        raw["log_tail_lines"] = previous.raw.get("log_tail_lines")
    return replace(incoming, raw=raw)


def _append_metric_rows(previous: list[Any], incoming: list[Any]) -> list[Any]:
    seen_steps = {
        metric.get("step")
        for metric in previous
        if isinstance(metric, dict) and metric.get("step") is not None
    }
    combined = [*previous]
    for metric in incoming:
        if isinstance(metric, dict) and metric.get("step") in seen_steps:
            continue
        combined.append(metric)
        if isinstance(metric, dict) and metric.get("step") is not None:
            seen_steps.add(metric.get("step"))
    return combined


def _next_metrics_min_step(raw: dict[str, Any]) -> int | None:
    steps = [
        int(metric["step"])
        for metric in _list_value(raw.get("recent_metrics"))
        if isinstance(metric, dict) and isinstance(metric.get("step"), int | float)
    ]
    if not steps:
        return None
    return max(steps) + 1


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
        str(_metadata_value(item, "Created") or "-"),
        "Updated",
        str(_metadata_value(item, "Updated") or "-"),
    )
    return table


def _training_top_panel(
    item: LabItem,
    raw: dict[str, Any],
    summary: dict[str, int | float | bool],
    environments: list[Any],
    environment_statuses: list[Any],
) -> Table:
    table = Table.grid(expand=True)
    table.add_column(ratio=3)
    table.add_column(ratio=2)
    table.add_row(
        Group(
            _training_overview_table(item, raw, summary),
            Text(""),
            Text("Progress", style="bold"),
            _progress_block(summary),
            Text(""),
        ),
        Group(
            Text("Environments", style="bold"),
            _training_environment_table(environments, environment_statuses),
        ),
    )
    return table


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


def _chart_specs(
    metrics: list[Any],
    reward_distribution: dict[str, Any],
    *,
    mode: ChartMode,
) -> list[ChartSpec]:
    if mode == "distribution":
        distribution = _reward_distribution_spec(reward_distribution)
        return [distribution] if distribution is not None else []

    numeric = _numeric_metric_series(metrics)
    keys = _ordered_metric_keys(numeric)
    return [_line_chart_spec(key, numeric[key]) for key in keys if len(numeric[key]) >= 1]


def _line_chart_spec(label: str, points: list[tuple[int, float]]) -> ChartSpec:
    values = tuple(value for _, value in points)
    smooth_y, smooth_label = _adaptive_ema(values)
    return ChartSpec(
        title=label,
        kind="line",
        x=tuple(step for step, _ in points),
        y=values,
        xlabel="Step",
        ylabel="Value",
        smooth_y=smooth_y,
        smooth_label=smooth_label,
    )


def _reward_distribution_spec(distribution: dict[str, Any]) -> ChartSpec | None:
    bins = _distribution_bins(distribution)
    if not bins:
        return None

    title = "Reward Distribution"
    step = distribution.get("step")
    if isinstance(step, int):
        title = f"{title} | step {step}"
    return ChartSpec(
        title=title,
        kind="bar",
        x=tuple(label for label, _ in bins),
        y=tuple(count for _, count in bins),
        xlabel="Range",
        ylabel="Count",
    )


def _chart_fallback(chart: ChartSpec) -> Text:
    text = Text("Chart rendering is unavailable in no-color mode.", style="dim")
    if chart.kind == "line" and chart.y:
        text.append(
            f"\n{len(chart.y)} points · y {min(chart.y):.4g} to {max(chart.y):.4g}",
            style="dim",
        )
    elif chart.kind == "bar" and chart.y:
        text.append(
            f"\n{len(chart.y)} bins · max count {max(chart.y):.4g}",
            style="dim",
        )
    return text


def _plot_widget(chart: ChartSpec) -> PlotWidget:
    return LabPlotWidget(chart)


def _draw_plot(plot: PlotWidget, chart: ChartSpec) -> None:
    plot.clear()
    if not chart.x or not chart.y or len(chart.x) != len(chart.y):
        return
    plot.set_xlabel(chart.xlabel)
    plot.set_ylabel(chart.ylabel)
    plot.set_xlimits(*_chart_x_limits(chart))
    plot.set_ylimits(*_chart_y_limits(chart))
    if chart.kind == "line":
        smooth_y = chart.smooth_y if len(chart.smooth_y) == len(chart.y) else chart.y
        plot.plot(
            list(chart.x),
            list(chart.y),
            line_style=GRID,
            hires_mode=HiResMode.BRAILLE,
        )
        plot.plot(
            list(chart.x),
            list(smooth_y or chart.y),
            line_style=f"{PRIMARY} bold",
            hires_mode=HiResMode.BRAILLE,
        )
        plot.scatter(
            list(chart.x),
            list(chart.y),
            marker="•",
            marker_style=NEUTRAL,
        )
    else:
        plot.bar(
            list(chart.x),
            list(chart.y),
            bar_style=PRIMARY,
            hires_mode=HiResMode.HALFBLOCK,
        )


def _chart_x_limits(chart: ChartSpec) -> tuple[float, float]:
    if chart.kind == "bar":
        return 0.5, max(1.5, len(chart.x) + 0.5)

    values = [float(value) for value in chart.x if isinstance(value, int | float)]
    return _padded_numeric_limits(values, pad_fraction=0.02, minimum_pad=1.0)


def _chart_y_limits(chart: ChartSpec) -> tuple[float, float]:
    values = [*chart.y, *chart.smooth_y]
    if chart.kind == "bar":
        upper = max(values, default=0.0)
        return 0.0, max(1.0, upper + max(upper * 0.08, 1.0))

    lower, upper = _padded_numeric_limits(
        [float(value) for value in values],
        pad_fraction=0.08,
        minimum_pad=0.05,
    )
    if values and min(values) >= 0 and lower < 0:
        lower = 0.0
    return lower, upper


def _padded_numeric_limits(
    values: list[float], *, pad_fraction: float, minimum_pad: float
) -> tuple[float, float]:
    if not values:
        return 0.0, 1.0

    lower = min(values)
    upper = max(values)
    if lower == upper:
        pad = max(abs(lower) * pad_fraction, minimum_pad)
    else:
        pad = max((upper - lower) * pad_fraction, minimum_pad)
    return lower - pad, upper + pad


def _adaptive_ema(values: tuple[float, ...]) -> tuple[tuple[float, ...], str]:
    if len(values) < 3:
        return values, "raw"
    span = max(3, min(25, round(len(values) * 0.1)))
    new_weight = 2 / (span + 1)
    retention = 1 - new_weight
    return _ema(values, retention=retention), f"EMA {span}"


def _ema(values: tuple[float, ...], *, retention: float) -> tuple[float, ...]:
    if not values:
        return ()
    smoothed = [values[0]]
    previous = values[0]
    for value in values[1:]:
        previous = retention * previous + (1 - retention) * value
        smoothed.append(previous)
    return tuple(smoothed)


def _histogram_charts_from_raw(raw: dict[str, Any]) -> list[tuple[str, Table]]:
    charts: list[tuple[str, Table]] = []

    chart_data = raw.get("chartData") or raw.get("chart_data")
    if isinstance(chart_data, dict):
        chart = _distribution_chart(
            {"histogramData": chart_data.get("histogramData")},
            title=_raw_distribution_title(raw),
        )
        if chart is not None:
            charts.append((_raw_distribution_title(raw), chart))

    for key, title in (
        ("reward_distribution", "Reward Distribution"),
        ("rewardDistribution", "Reward Distribution"),
        ("score_distribution", "Score Distribution"),
        ("scoreDistribution", "Score Distribution"),
    ):
        value = raw.get(key)
        if not isinstance(value, dict):
            continue
        chart = _distribution_chart(value, title=title)
        if chart is not None:
            charts.append((title, chart))

    return charts


def _raw_distribution_title(raw: dict[str, Any]) -> str:
    if "avg_score" in raw or "avgScore" in raw:
        return "Score Distribution"
    return "Reward Distribution"


def _ordered_metric_keys(numeric: dict[str, list[tuple[int, float]]]) -> list[str]:
    return sorted(numeric, key=_metric_sort_key)


def _metric_sort_key(key: str) -> tuple[int, str]:
    normalized = key.lower().replace("_", "/")
    if normalized == "reward/all/mean":
        return (0, key)
    if _is_reward_metric(key):
        return (1, key)
    return (2, key)


def _is_reward_metric(key: str) -> bool:
    normalized = key.lower().replace("_", "/")
    return (
        normalized == "reward"
        or normalized.startswith("reward/")
        or normalized.endswith("/reward")
        or "/reward/" in normalized
    )


def _numeric_metric_series(metrics: list[Any]) -> dict[str, list[tuple[int, float]]]:
    series: dict[str, list[tuple[int, float]]] = {}
    for idx, metric in enumerate(metrics):
        if not isinstance(metric, dict):
            continue
        raw_step = metric.get("step", idx)
        step = (
            int(raw_step) if isinstance(raw_step, int | float) and math.isfinite(raw_step) else idx
        )
        for key, value in metric.items():
            if key in {"step", "run_id", "runId", "timestamp"}:
                continue
            number = _finite_float(value)
            if number is not None:
                series.setdefault(key, []).append((step, number))
    return series


def _line_chart(label: str, points: list[tuple[int, float]]) -> Text:
    width = 56
    values = [value for _, value in points]
    sampled = _sample_values(values, width)
    lo = min(sampled)
    hi = max(sampled)
    levels = "._-~=+*#%@"
    chars = []
    for value in sampled:
        idx = 0 if hi == lo else round(((value - lo) / (hi - lo)) * (len(levels) - 1))
        chars.append(levels[idx])
    start_step = points[0][0]
    end_step = points[-1][0]
    latest = points[-1][1]
    text = Text()
    text.append(f"{label}\n", style="bold dim")
    text.append("".join(chars), style=PRIMARY)
    text.append(f"  latest {latest:.4g}  step {start_step}->{end_step}", style="dim")
    return text


def _reward_distribution_chart(distribution: dict[str, Any]) -> Table | None:
    return _distribution_chart(distribution, title="Reward Distribution")


def _distribution_chart(distribution: dict[str, Any], *, title: str) -> Table | None:
    bins = _distribution_bins(distribution)
    if not bins:
        return None

    max_count = max(count for _, count in bins) or 1.0
    step = distribution.get("step")
    if isinstance(step, int):
        title = f"{title} | step {step}"

    table = Table(title=title, show_header=True, header_style="bold dim")
    table.add_column("Range")
    table.add_column("Count", justify="right")
    table.add_column("Chart")
    for label, count in bins:
        width = max(1, int(round((count / max_count) * 36))) if count > 0 else 0
        table.add_row(label, _format_number(count), Text("#" * width, style=PRIMARY))
    return table


def _distribution_bins(distribution: dict[str, Any]) -> list[tuple[str, float]]:
    chart_data = distribution.get("chartData") or distribution.get("chart_data")
    raw_bins = (
        distribution.get("bins")
        or distribution.get("histogramData")
        or (chart_data.get("histogramData") if isinstance(chart_data, dict) else None)
        or distribution.get("data")
    )
    if not isinstance(raw_bins, list):
        return []

    bins: list[tuple[str, float]] = []
    for idx, raw_bin in enumerate(raw_bins):
        if isinstance(raw_bin, dict):
            count = _first_number(
                raw_bin.get("count"),
                raw_bin.get("value"),
                raw_bin.get("frequency"),
                raw_bin.get("total"),
                raw_bin.get("percentage"),
            )
            label = _distribution_bin_label(raw_bin, idx)
        elif isinstance(raw_bin, list | tuple) and len(raw_bin) >= 2:
            label = str(raw_bin[0])
            count = _first_number(raw_bin[1])
        else:
            continue

        if count is not None:
            bins.append((label, count))
    return bins


def _distribution_bin_label(raw_bin: dict[str, Any], idx: int) -> str:
    for key in ("range", "label", "bin"):
        value = raw_bin.get(key)
        if value is not None:
            return str(value)
    for lower_key, upper_key in (
        ("min", "max"),
        ("start", "end"),
        ("lower", "upper"),
    ):
        lower = raw_bin.get(lower_key)
        upper = raw_bin.get(upper_key)
        if lower is not None and upper is not None:
            return f"{lower}-{upper}"
    return str(idx)


def _first_number(*values: Any) -> float | None:
    for value in values:
        number = _finite_float(value)
        if number is not None:
            return number
    return None


def _finite_float(value: Any) -> float | None:
    if not isinstance(value, int | float):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _format_number(value: float) -> str:
    number = float(value)
    if number.is_integer():
        return str(int(number))
    return f"{number:.4g}"


def _dict_value(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list_value(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _sample_values(values: list[float], width: int) -> list[float]:
    if len(values) <= width:
        return values
    sampled = []
    for idx in range(width):
        source_idx = round(idx * (len(values) - 1) / (width - 1))
        sampled.append(values[source_idx])
    return sampled


def _training_environment_table(environments: list[Any], environment_statuses: list[Any]) -> Table:
    table = Table(show_header=True, header_style="bold dim")
    table.add_column("Environment")
    table.add_column("Version")
    table.add_column("Visibility")
    table.add_column("Action")
    status_by_slug: dict[str, dict[str, Any]] = {}
    for status in environment_statuses:
        if not isinstance(status, dict):
            continue
        slug = str(status.get("environment") or status.get("slug") or status.get("id") or "")
        if slug:
            status_by_slug[slug] = status
    if not environments:
        table.add_row("-", "-", "-", "-")
        return table
    for env in environments:
        if not isinstance(env, dict):
            table.add_row(str(env), "-", "-", "-")
            continue
        slug = str(env.get("slug") or env.get("id") or env.get("name") or "-")
        status = status_by_slug.get(slug, {})
        action = status.get("latest_ci_status") or status.get("status") or "-"
        table.add_row(
            slug,
            str(env.get("version") or env.get("version_id") or env.get("versionId") or "-"),
            str(env.get("visibility") or "-"),
            str(action),
        )
    return table


def _training_progress_summary(
    raw: dict[str, Any], progress: dict[str, Any], metrics: list[Any]
) -> dict[str, int | float | bool]:
    max_steps = _int_value(raw.get("max_steps") or raw.get("maxSteps")) or 0
    status = str(raw.get("status") or "")
    latest_step = _int_value(progress.get("latest_step") or progress.get("latestStep"))
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


def _int_value(value: Any) -> int | None:
    return value if isinstance(value, int) else None


def _current_metrics_limit(item: LabItem | None) -> int:
    if item is None:
        return _RUN_METRIC_PREVIEW_LIMIT
    return _int_value(item.raw.get("metrics_limit")) or _RUN_METRIC_PREVIEW_LIMIT


def _metadata_value(item: LabItem, key: str) -> str | None:
    for metadata_key, value in item.metadata:
        if metadata_key == key:
            return value
    return None


def _visible_system_renderable(raw: dict[str, Any], *, visible_log_lines: int | None) -> Group:
    logs = raw.get("logs_tail")
    current_tail = _int_value(raw.get("log_tail_lines"))
    visible_logs = _visible_log_text(
        logs if isinstance(logs, str) else "",
        visible_log_lines,
    )
    return _system_renderable(visible_logs, tail_lines=current_tail)


def _visible_log_text(value: str, visible_log_lines: int | None) -> str:
    if visible_log_lines is None:
        return value
    lines = value.splitlines()
    if visible_log_lines >= len(lines):
        return value
    if visible_log_lines <= 0:
        return ""
    return "\n".join(lines[-visible_log_lines:])


def _system_renderable(value: str, *, tail_lines: int | None = None) -> Group:
    log_tail_lines = tail_lines or 1000
    logs = _tail_text(
        value,
        max_chars=min(max(16000, log_tail_lines * 320), 2_000_000),
    )
    records = _parse_log_records(logs)
    if not records:
        return Group(Text("No logs available", style="dim"))
    title = Text("Logs", style="bold")
    title.append(
        f" · {log_tail_lines:,} line tail · "
        f"showing {min(len(records), _LOG_RENDER_ROW_LIMIT):,}/{len(records):,}",
        style="dim",
    )
    if not any(record.get("structured") for record in records):
        return Group(title, Text(""), Text(logs))

    table = Table(show_header=True, header_style="bold dim", expand=True)
    table.add_column("Time", no_wrap=True)
    table.add_column("Level", no_wrap=True)
    table.add_column("Step", no_wrap=True)
    table.add_column("Log", ratio=1)

    for record in records[-_LOG_RENDER_ROW_LIMIT:]:
        level = str(record.get("level") or "")
        table.add_row(
            str(record.get("time") or ""),
            Text(level, style=_log_level_style(level)),
            str(record.get("step") or ""),
            str(record.get("log") or ""),
        )
    return Group(title, Text(""), table)


def _parse_log_records(value: str) -> list[dict[str, Any]]:
    stripped = value.strip()
    if not stripped:
        return []

    parsed = _parse_json_value(stripped)
    if isinstance(parsed, list):
        return [_log_record(record) for record in parsed]
    if isinstance(parsed, dict):
        return [_log_record(parsed)]

    records = []
    for line in stripped.splitlines():
        line = line.strip()
        if not line:
            continue
        line_value = _parse_json_value(line)
        records.append(_log_record(line_value if isinstance(line_value, dict) else line))
    return records


def _parse_json_value(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def _log_record(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {"message": str(value), "structured": False}

    consumed = {
        "time",
        "timestamp",
        "created_at",
        "createdAt",
        "level",
        "levelname",
        "severity",
        "message",
        "msg",
        "event",
        "text",
        "desc",
        "step",
        "global_step",
        "iteration",
        "type",
        "current",
        "total",
        "percent",
    }
    if value.get("type") == "progress":
        message = _progress_log_message(value)
    else:
        message = _first_text_field(value, "message", "msg", "event", "text", "desc")
    if message is None:
        message = json.dumps(value, sort_keys=True, default=str)
    step = _first_text_field(value, "step", "global_step", "iteration")
    fields = {
        key: child
        for key, child in value.items()
        if key not in consumed and child is not None and child != {} and child != []
    }
    return {
        "time": _first_text_field(value, "time", "timestamp", "created_at", "createdAt"),
        "level": _first_text_field(value, "level", "levelname", "severity"),
        "step": step,
        "message": message,
        "fields": _compact_fields(fields),
        "log": _join_log_parts(message, _compact_fields(fields)),
        "structured": True,
    }


def _join_log_parts(message: str, fields: str) -> str:
    if fields:
        return f"{message}  {fields}"
    return message


def _progress_log_message(value: dict[str, Any]) -> str:
    desc = str(value.get("desc") or "Progress")
    current = value.get("current")
    total = value.get("total")
    percent = value.get("percent")
    if current is not None and total is not None and percent is not None:
        return f"{desc}  {current}/{total} ({percent}%)"
    if current is not None and total is not None:
        return f"{desc}  {current}/{total}"
    return desc


def _first_text_field(value: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        field = value.get(key)
        if field is not None:
            return str(field)
    return None


def _compact_fields(value: dict[str, Any]) -> str:
    if not value:
        return ""
    parts = []
    for key, child in value.items():
        rendered = (
            json.dumps(child, sort_keys=True, default=str)
            if isinstance(child, dict | list)
            else str(child)
        )
        parts.append(f"{key}={rendered}")
    text = " ".join(parts)
    return text if len(text) <= 240 else text[:237] + "..."


def _log_level_style(level: str) -> str:
    normalized = level.upper()
    if normalized in {"ERROR", "ERR", "CRITICAL", "FATAL"}:
        return STATUS_ERROR
    if normalized in {"WARNING", "WARN"}:
        return STATUS_WARNING
    if normalized in {"INFO", "NOTICE"}:
        return STATUS_SUCCESS
    if normalized in {"DEBUG", "TRACE"}:
        return "dim"
    return STATUS_INFO


def _tail_text(value: str, *, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value or "No logs available"
    return value[-max_chars:]
