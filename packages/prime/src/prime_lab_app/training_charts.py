"""Training metric chart models, plot widgets, and chart data shaping."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

from rich.table import Table
from rich.text import Text
from textual.css.query import NoMatches
from textual_hires_canvas import Canvas, TextAlign
from textual_plot import HiResMode, PlotWidget
from textual_plot.axis_formatter import CategoricalAxisFormatter

from .palette import GRID, NEUTRAL, PRIMARY
from .values import finite_float, first_number, format_number, list_value


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


class LabPlotWidget(PlotWidget, can_focus=False):
    """Plot widget that draws after mount so textual-plot internals are ready."""

    BINDINGS = []

    def __init__(self, chart: ChartSpec) -> None:
        super().__init__(classes="chart-plot run-chart", allow_pan_and_zoom=False)
        self._chart = chart
        self.chart_identity = chart_identity(chart)

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
        self.border_title = ""
        self.border_subtitle = ""
        try:
            draw_plot(self, self._chart)
        except Exception:
            self.clear()
            self.border_title = "Chart unavailable"
            self.border_subtitle = "Could not render chart"

    def update_chart(self, chart: ChartSpec) -> None:
        self._chart = chart
        self.chart_identity = chart_identity(chart)
        self._x_ticks = None
        self._y_ticks = None
        self._draw_chart()


def chart_specs_for_raw(raw: dict[str, Any], *, mode: ChartMode | None = None) -> list[ChartSpec]:
    metrics = list_value(raw.get("recent_metrics"))
    reward_distribution = raw.get("reward_distribution")
    reward_distribution = reward_distribution if isinstance(reward_distribution, dict) else {}
    if mode is None:
        return [
            *chart_specs(metrics, reward_distribution, mode="metrics"),
            *chart_specs(metrics, reward_distribution, mode="distribution"),
        ]
    return chart_specs(metrics, reward_distribution, mode=mode)


def chart_specs(
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


def chart_count(raw: dict[str, Any]) -> int:
    return len(chart_specs_for_raw(raw))


def chart_count_for_mode(raw: dict[str, Any], mode: ChartMode) -> int:
    return len(chart_specs_for_raw(raw, mode=mode))


def metric_chart_count(raw: dict[str, Any]) -> int:
    return len(chart_specs_for_raw(raw, mode="metrics"))


def distribution_chart_count(raw: dict[str, Any]) -> int:
    return len(chart_specs_for_raw(raw, mode="distribution"))


def selected_chart(
    raw: dict[str, Any],
    *,
    mode: ChartMode,
    metric_index: int,
    distribution_index: int,
) -> ChartSpec | None:
    charts = chart_specs_for_raw(raw, mode=mode)
    if not charts:
        return None
    selected_index = (metric_index if mode == "metrics" else distribution_index) % len(charts)
    return charts[selected_index]


def chart_heading(
    raw: dict[str, Any],
    *,
    mode: ChartMode,
    metric_index: int,
    distribution_index: int,
) -> Text:
    return chart_heading_from_specs(
        chart_specs_for_raw(raw, mode=mode),
        mode=mode,
        metric_index=metric_index,
        distribution_index=distribution_index,
    )


def chart_heading_from_specs(
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


def raw_with_visible_metric_rows(
    raw: dict[str, Any], visible_metric_rows: int | None
) -> dict[str, Any]:
    if visible_metric_rows is None:
        return raw
    metrics = list_value(raw.get("recent_metrics"))
    visible_metrics = visible_metric_rows_for(metrics, visible_metric_rows)
    if len(visible_metrics) == len(metrics):
        return raw
    return {**raw, "recent_metrics": visible_metrics}


def visible_metric_rows_for(metrics: list[Any], visible_metric_rows: int | None) -> list[Any]:
    if visible_metric_rows is None or visible_metric_rows >= len(metrics):
        return metrics
    if visible_metric_rows <= 0:
        return []
    return metrics[:visible_metric_rows]


def chart_fallback(chart: ChartSpec) -> Text:
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


def plot_widget(chart: ChartSpec) -> PlotWidget:
    return LabPlotWidget(chart)


def chart_identity(chart: ChartSpec) -> tuple[str, str, str, str]:
    return (chart.kind, chart.title, chart.xlabel, chart.ylabel)


def draw_plot(plot: PlotWidget, chart: ChartSpec) -> None:
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


def histogram_charts_from_raw(raw: dict[str, Any]) -> list[tuple[str, Table]]:
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


def reward_distribution_chart(distribution: dict[str, Any]) -> Table | None:
    return _distribution_chart(distribution, title="Reward Distribution")


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
            number = finite_float(value)
            if number is not None:
                series.setdefault(key, []).append((step, number))
    return series


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
        table.add_row(label, format_number(count), Text("#" * width, style=PRIMARY))
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
            count = first_number(
                raw_bin.get("count"),
                raw_bin.get("value"),
                raw_bin.get("frequency"),
                raw_bin.get("total"),
                raw_bin.get("percentage"),
            )
            label = _distribution_bin_label(raw_bin, idx)
        elif isinstance(raw_bin, list | tuple) and len(raw_bin) >= 2:
            label = str(raw_bin[0])
            count = first_number(raw_bin[1])
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
