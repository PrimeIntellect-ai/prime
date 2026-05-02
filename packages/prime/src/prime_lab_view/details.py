"""Selection detail renderers for Lab items."""

from __future__ import annotations

import json
from typing import Any

from rich.console import Group
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from .models import LabItem
from .palette import CODE_THEME
from .rows import item_badges_text
from .training_charts import histogram_charts_from_raw as _histogram_charts_from_raw
from .training_config import training_config_toml as _training_config_toml
from .values import format_number as _format_number
from .values import metadata_value as _metadata_value


def item_details(item: LabItem) -> Group:
    if item.section == "environments":
        return _environment_item_details(item)

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
                        Syntax(config, "toml", theme=CODE_THEME),
                    ]
                )
            return Group(*chunks)
        if item.section == "workspace":
            chunks.extend(_workspace_detail_chunks(item))
            return Group(*chunks)
        if item.section in {"evaluations", "local-evals"}:
            chunks.extend(evaluation_detail_chunks(item.raw))
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
            [Text(""), Text("Raw", style="bold dim"), Syntax(raw, "json", theme=CODE_THEME)]
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


def _environment_item_details(item: LabItem) -> Group:
    raw = item.raw
    chunks: list[Any] = [_environment_summary_text(item)]

    details = _environment_summary_table(item)
    if details.row_count:
        chunks.extend([Text(""), details])

    files = _environment_local_files(raw)
    if files.row_count:
        chunks.extend([Text(""), Text("Files", style="bold dim"), files])

    status = raw.get("status")
    if isinstance(status, dict):
        status_table = _environment_status_summary_table(status)
        if status_table.row_count:
            chunks.extend([Text(""), Text("Status", style="bold dim"), status_table])

    platform = _environment_platform_payload(raw)
    platform_table = _environment_platform_summary_table(raw, platform)
    if platform_table.row_count:
        chunks.extend([Text(""), Text("Platform", style="bold dim"), platform_table])

    return Group(*chunks)


def _environment_summary_text(item: LabItem) -> Text:
    text = Text()
    text.append(item.title, style="bold")
    text.append_text(item_badges_text(item))
    if item.subtitle:
        text.append(f"\n{item.subtitle}")
    return text


def _environment_summary_table(item: LabItem) -> Table:
    raw = item.raw
    platform = _environment_platform_payload(raw)
    local = raw.get("local") if isinstance(raw.get("local"), dict) else {}
    local_metadata = local.get("metadata") if isinstance(local, dict) else {}
    if not isinstance(local_metadata, dict):
        local_metadata = {}
    local_project = local.get("project") if isinstance(local, dict) else {}
    if not isinstance(local_project, dict):
        local_project = {}

    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold dim", no_wrap=True)
    table.add_column()
    for key, value in (
        ("Source", ", ".join(str(source) for source in raw.get("sources", []) or [])),
        ("Version", _environment_version(raw, platform, local_project, local_metadata)),
        ("Visibility", raw.get("visibility") or platform.get("visibility")),
        ("Stars", platform.get("stars")),
        (
            "Updated",
            _format_detail_time(
                platform.get("updated_at")
                or platform.get("updatedAt")
                or raw.get("updated_at")
                or raw.get("updatedAt")
            ),
        ),
        ("Path", local.get("relative_path") or local.get("path") if local else None),
        ("Environment ID", local_metadata.get("environment_id")),
        ("Install", _environment_install_command(raw)),
    ):
        text = _display_value(value)
        if text != "-":
            table.add_row(key, text)
    return table


def _environment_version(
    raw: dict[str, Any],
    platform: dict[str, Any],
    local_project: dict[str, Any],
    local_metadata: dict[str, Any],
) -> Any:
    return (
        raw.get("semantic_version")
        or raw.get("semanticVersion")
        or platform.get("semantic_version")
        or platform.get("semanticVersion")
        or raw.get("latest_version")
        or local_project.get("version")
        or local_metadata.get("version")
    )


def _environment_install_command(raw: dict[str, Any]) -> str:
    slug = str(raw.get("slug") or raw.get("id") or "")
    return f"prime env install {slug}" if slug else ""


def _environment_local_files(raw: dict[str, Any]) -> Table:
    table = Table.grid()
    local = raw.get("local") if isinstance(raw.get("local"), dict) else raw
    files = local.get("files") if isinstance(local, dict) else None
    if isinstance(files, list):
        for file_name in files:
            table.add_row(str(file_name))
    return table


def _environment_status_summary_table(status: dict[str, Any]) -> Table:
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold dim", no_wrap=True)
    table.add_column()
    latest = status.get("latest_version") or status.get("latestVersion")
    if isinstance(latest, dict):
        version = latest.get("semantic_version") or latest.get("version")
        action = latest.get("latest_ci_status") or latest.get("status")
        if version not in (None, "", "-"):
            table.add_row("Latest version", str(version))
        if action not in (None, "", "-"):
            table.add_row("Action", str(action))
    else:
        for key in ("latest_ci_status", "status", "action"):
            value = status.get(key)
            if value not in (None, "", "-"):
                table.add_row(_humanize_key(key), str(value))
    return table


def _environment_platform_summary_table(raw: dict[str, Any], platform: dict[str, Any]) -> Table:
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold dim", no_wrap=True)
    table.add_column()
    for key, value in (
        ("Python", platform.get("python_version") or platform.get("pythonVersion")),
        ("Tags", platform.get("tags")),
        ("Content hash", raw.get("content_hash") or platform.get("content_hash")),
        (
            "Created",
            _format_detail_time(
                raw.get("created_at") or raw.get("createdAt") or platform.get("created_at")
            ),
        ),
    ):
        text = _display_value(value)
        if text != "-":
            table.add_row(key, text)
    return table


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

    if kind in {"local_environment", "environment"}:
        files = _environment_local_files(raw)
        if files.row_count:
            chunks.extend([Text(""), Text("Files", style="bold dim"), files])
        status = raw.get("status")
        if isinstance(status, dict):
            status_table = _environment_status_summary_table(status)
            if status_table.row_count:
                chunks.extend([Text(""), Text("Status", style="bold dim"), status_table])
        platform = _environment_platform_payload(raw)
        platform_table = _environment_platform_summary_table(raw, platform)
        if platform_table.row_count:
            chunks.extend([Text(""), Text("Platform", style="bold dim"), platform_table])
        return chunks

    if kind == "setup_action":
        chunks.extend(
            [
                Text(""),
                Text("Setup", style="bold dim"),
                Text("Open this row to run prime lab setup in the active workspace."),
            ]
        )
        return chunks

    if kind == "doctor_action":
        chunks.extend(
            [
                Text(""),
                Text("Workspace Doctor", style="bold dim"),
                Text("Open this row to check setup, configs, output ignores, and agent assets."),
            ]
        )
        return chunks

    if kind == "add_workspace":
        chunks.extend(
            [
                Text(""),
                Text("Workspace Memory", style="bold dim"),
                Text("Open this row to remember another local workspace path."),
            ]
        )
        return chunks

    if kind == "agent_sync":
        chunks.extend(
            [
                Text(""),
                Text("Lab Asset Sync", style="bold dim"),
                Text("Open this row to refresh templates, skills, docs, and agent guidance."),
            ]
        )
        return chunks

    if kind == "config_file":
        toml_text = raw.get("toml")
        if isinstance(toml_text, str) and toml_text:
            chunks.extend(
                [
                    Text(""),
                    Text("Config", style="bold dim"),
                    Syntax(toml_text.rstrip(), "toml", theme=CODE_THEME),
                ]
            )
        return chunks

    return chunks


def _environment_display_name(env: Any) -> str:
    if not isinstance(env, dict):
        return str(env)
    return str(env.get("slug") or env.get("id") or env.get("name") or "-")


def evaluation_detail_chunks(raw: dict[str, Any]) -> list[Any]:
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


def _environment_platform_payload(raw: dict[str, Any]) -> dict[str, Any]:
    detail = raw.get("platform_detail")
    if isinstance(detail, dict):
        return detail
    platform = raw.get("platform")
    if isinstance(platform, dict):
        return platform
    return raw


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
