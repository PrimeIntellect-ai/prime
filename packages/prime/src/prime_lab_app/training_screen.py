"""Training run screen state and background loading orchestration."""

from __future__ import annotations

import math
import time
import webbrowser
from dataclasses import replace
from pathlib import Path
from typing import Any

from rich.table import Table
from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.screen import Screen
from textual.widget import Widget
from textual.widgets import Button, Footer, Static, Tab, Tabs

from .config_screen import ConfigRunScreen
from .detail_loader import DetailLoader
from .filters import FilterChoice, FilterScreen
from .logs import visible_system_renderable as _visible_system_renderable
from .models import LabItem
from .palette import BUTTON_CSS, STATUS_ERROR
from .shell import lab_header
from .training_charts import (
    ChartMode,
    LabPlotWidget,
)
from .training_charts import (
    chart_count_for_mode as _chart_count_for_mode,
)
from .training_charts import (
    chart_fallback as _chart_fallback,
)
from .training_charts import (
    chart_heading as _chart_heading,
)
from .training_charts import (
    chart_identity as _chart_identity,
)
from .training_charts import (
    chart_specs_for_raw as _chart_specs_for_raw,
)
from .training_charts import (
    distribution_chart_count as _distribution_chart_count,
)
from .training_charts import (
    metric_chart_count as _metric_chart_count,
)
from .training_charts import (
    raw_with_visible_metric_rows as _raw_with_visible_metric_rows,
)
from .training_charts import (
    selected_chart as _selected_chart,
)
from .training_config import (
    training_config_item as _training_config_item,
)
from .training_config import (
    training_platform_url as _training_platform_url,
)
from .training_render import (
    RUN_TABS as _RUN_TABS,
)
from .training_render import (
    adjacent_tab as _adjacent_tab,
)
from .training_render import (
    run_loading_widgets as _run_loading_widgets,
)
from .training_render import (
    training_run_widgets as _training_run_widgets,
)
from .training_render import (
    training_title as _training_title,
)
from .values import int_value as _int_value
from .values import list_value as _list_value
from .widgets import LoadingChart, LoadingMessage

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
_LOG_REVEAL_BUFFER = 1.15
_LOG_REVEAL_MIN_SECONDS = 0.1
_LOG_REVEAL_MAX_SECONDS = 10.0


class TrainingRunScreen(Screen[None]):
    """Full-page training run view."""

    BINDINGS = [
        Binding("escape", "back", "Back", key_display="Esc"),
        Binding("b", "back", "Back", key_display="B"),
        Binding("l", "toggle_logs", "Logs"),
        Binding("left", "previous_tab", "Prev tab"),
        Binding("right", "next_tab", "Next tab"),
        Binding("up", "previous_metric", "Prev chart"),
        Binding("down", "next_metric", "Next chart"),
        Binding("space", "pick_chart", "Pick"),
        Binding("m", "show_metric_charts", "Metrics", show=False),
        Binding("d", "show_distribution_charts", "Reward dist", show=False),
        Binding("g", "load_more_logs", "More logs", show=False),
        Binding("e", "edit_config", "Edit config"),
        Binding("o", "open_platform", "Open platform", show=False),
    ]

    CSS = (
        BUTTON_CSS
        + """
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
        height: 1fr;
    }

    .run-overview-grid {
        height: auto;
    }

    .run-chart-column {
        width: 2fr;
        min-width: 58;
        margin-right: 1;
    }

    .run-side-column {
        width: 1fr;
        min-width: 38;
        border: round $primary;
        background: $surface;
        padding: 0 1;
    }

    .run-side-button {
        width: 1fr;
        margin: 1 0 0 0;
    }

    .run-train-button {
        width: 1fr;
        height: 3;
        margin: 0 0 1 0;
    }

    .run-config-toolbar {
        height: 1;
        align-horizontal: right;
        margin: 1 0 0 0;
    }

    .run-config-title {
        width: 1fr;
        content-align: left middle;
    }

    .run-config-action {
        width: auto;
        height: auto;
    }

    .run-config-panel {
        background: $surface;
    }

    .run-progress-panel {
        height: auto;
        border: round $primary;
        background: $surface;
        padding: 0 1;
        margin-bottom: 1;
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
        height: 1fr;
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
    )

    def __init__(
        self,
        item: LabItem,
        detail_loader: DetailLoader,
        *,
        include_logs: bool = False,
        frontend_url: str = "",
        workspace: Path | None = None,
    ) -> None:
        super().__init__()
        self._base_item = item
        self._detail_loader = detail_loader
        self._include_logs = include_logs
        self._frontend_url = frontend_url
        self._workspace = workspace.expanduser().resolve() if workspace is not None else Path.cwd()
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
        return lab_header(_training_title(item))

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
        if action == "edit_config":
            item = self._detail or self._base_item
            return _training_config_item(item, item.raw, workspace=self._workspace) is not None
        if action == "open_platform":
            item = self._detail or self._base_item
            return (
                _training_platform_url(self._frontend_url, item.raw, fallback_id=item.title)
                is not None
            )
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

    def action_edit_config(self) -> None:
        item = self._detail or self._base_item
        config_item = _training_config_item(item, item.raw, workspace=self._workspace)
        if config_item is not None:
            self.app.push_screen(ConfigRunScreen(config_item))

    def action_open_platform(self) -> None:
        item = self._detail or self._base_item
        url = _training_platform_url(self._frontend_url, item.raw, fallback_id=item.title)
        if url is None:
            self.notify("No platform URL is available for this training run.", severity="warning")
            return
        if not webbrowser.open(url):
            self.notify("Could not open the training run in a browser.", severity="warning")

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
        if had_detail and self._patch_active_tab_after_background_update(detail):
            pass
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
                frontend_url=self._frontend_url,
                workspace=self._workspace,
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

    def _patch_active_tab_after_background_update(self, detail: LabItem) -> bool:
        self.query_one("#run-title", Static).update(self._run_topbar_title(detail))
        self._sync_tabs()
        if self._active_tab == "overview":
            return self._update_chart_panel()
        if self._active_tab == "system":
            return self._update_log_panel()
        if self._active_tab == "data":
            return True
        return False

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
            plot = self.query(".run-chart").first(LabPlotWidget)
            if plot.chart_identity != _chart_identity(chart):
                parent = plot.parent
                if parent is None:
                    return False
                parent.mount(LabPlotWidget(chart), before=plot)
                plot.remove()
                return True
            plot.update_chart(chart)
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

    @on(Button.Pressed, "#run-edit-config")
    def _edit_config_pressed(self, _event: Button.Pressed) -> None:
        self.action_edit_config()

    @on(Button.Pressed, "#run-train")
    def _train_pressed(self, _event: Button.Pressed) -> None:
        self.action_edit_config()

    @on(Button.Pressed, "#run-platform")
    def _platform_pressed(self, _event: Button.Pressed) -> None:
        self.action_open_platform()


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
    combined_metrics = (
        _append_metric_rows(previous_metrics, incoming_metrics)
        if incoming_min_step is not None
        else _append_metric_rows(incoming_metrics, previous_metrics)
    )
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


def _current_metrics_limit(item: LabItem | None) -> int:
    if item is None:
        return _RUN_METRIC_PREVIEW_LIMIT
    return _int_value(item.raw.get("metrics_limit")) or _RUN_METRIC_PREVIEW_LIMIT
