"""Config edit and launch screens for Lab workflows."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import toml
from rich.console import Group
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll
from textual.css.query import NoMatches
from textual.screen import Screen
from textual.widgets import Button, Footer, Input, Label, Static

from .launch_runner import ConfigLaunchRunner
from .models import LabItem
from .palette import BUTTON_CSS, CODE_THEME, PRIMARY, STATUS_ERROR, STATUS_SUCCESS, STATUS_WARNING
from .shell import lab_header
from .toml_format import format_toml_blocks
from .training_config import normalize_rl_config
from .widgets import ClearableInput


@dataclass(frozen=True)
class ValidationResult:
    parsed: dict[str, Any]
    error: str | None


@dataclass(frozen=True)
class ConfigBuildResult:
    parsed: dict[str, Any]
    toml_text: str
    errors: tuple[str, ...] = ()


def build_config_from_fields(
    config: dict[str, Any],
    config_kind: str,
    field_value: Any,
) -> ConfigBuildResult:
    """Build a normalized Lab config from form field values."""

    return _form_config(config, config_kind, field_value)


def initial_config_field_values(
    config: dict[str, Any],
    config_kind: str,
    *,
    fallback_name: str,
) -> dict[str, str]:
    """Return the standard editable field values for a Lab config."""

    return _initial_field_values(config, config_kind, fallback_name=fallback_name)


def launch_command_for_config(config_kind: str, rel_path: str) -> str:
    """Return the Lab CLI command for launching a generated config."""

    return _launch_command(config_kind, rel_path)


FIELD_INITIAL_KEYS = {
    "config-name": "name",
    "config-model": "model",
    "config-envs": "envs",
    "config-max-steps": "max_steps",
    "config-rollouts": "rollouts_per_example",
    "config-batch-size": "batch_size",
    "config-max-tokens": "max_tokens",
    "config-seq-len": "seq_len",
}


class ConfigRunScreen(Screen[None]):
    """Native TUI editor for a local Lab config."""

    BINDINGS = [
        Binding("escape", "back", "Back", key_display="Esc"),
        Binding("tab", "focus_next", "Next", key_display="Tab", show=False),
        Binding("shift+tab", "focus_previous", "Previous", key_display="Shift+Tab", show=False),
    ]

    CSS = (
        BUTTON_CSS
        + """
    ConfigRunScreen {
        background: $background;
        color: $foreground;
        layout: vertical;
    }

    #config-header {
        height: 3;
        padding: 0 1;
        border-bottom: solid $primary;
        background: $background;
    }

    #config-body {
        height: 1fr;
        padding: 1 0;
    }

    .config-pane {
        border: round $primary;
        background: $surface;
        padding: 0 1;
        margin-right: 1;
    }

    #config-source {
        width: 36;
        min-width: 30;
    }

    #config-fields {
        width: 2fr;
        min-width: 48;
    }

    #config-preview {
        width: 56;
        min-width: 42;
    }

    .field-label {
        color: $text-muted;
        height: 1;
        margin-top: 1;
    }

    Input {
        height: 3;
    }

    Input.read-only {
        color: $text-muted;
    }

    .config-actions {
        height: auto;
        margin-top: 1;
    }
    """
    )

    def __init__(self, item: LabItem) -> None:
        super().__init__()
        self._item = item
        self._original = str(item.raw.get("toml") or "")
        self._workspace = Path(str(item.raw.get("workspace") or ".")).expanduser().resolve()
        self._config_kind = str(item.raw.get("config_kind") or "rl")
        parsed_original = _validate_toml(self._original).parsed
        self._original_config = _normalize_config(parsed_original, self._config_kind)
        self._clone_path = self._default_clone_path()
        self._saved = False
        self._initial_fields = _initial_field_values(
            self._original_config,
            self._config_kind,
            fallback_name=self._clone_path.stem,
        )

    def compose(self) -> ComposeResult:
        yield Static(_config_header(self._item), id="config-header", markup=False)
        with Horizontal(id="config-body"):
            yield VerticalScroll(
                Static(_source_panel(self._item), markup=False),
                id="config-source",
                classes="config-pane",
            )
            with VerticalScroll(id="config-fields", classes="config-pane"):
                yield Static("Edit configuration", classes="pane-title")
                yield from _field_widgets(self._initial_fields)
                with Horizontal(classes="config-actions"):
                    yield Button("Save copy", id="config-save", variant="default")
                    yield Button("Launch", id="config-launch", variant="primary")
            yield VerticalScroll(
                Static(self._preview_renderable(), id="config-preview-body"),
                id="config-preview",
                classes="config-pane",
            )
        yield Footer()

    def action_back(self) -> None:
        self.app.pop_screen()

    def save_clone(self) -> bool:
        build = self._current_build()
        if build.errors:
            self._saved = False
            self._render_preview()
            return False
        self._clone_path.parent.mkdir(parents=True, exist_ok=True)
        self._clone_path.write_text(build.toml_text.rstrip() + "\n", encoding="utf-8")
        self._saved = True
        self._render_preview()
        return True

    def launch(self) -> None:
        if not self.save_clone():
            return
        command = _launch_command(
            self._config_kind,
            _relative_path(self._clone_path, self._workspace),
        )
        self.app.push_screen(
            ConfigLaunchScreen(
                command=command,
                workspace=self._workspace,
                config_path=self._clone_path,
                follow_training_logs=self._config_kind == "rl",
            )
        )

    @on(Button.Pressed, "#config-save")
    def _save_pressed(self, _event: Button.Pressed) -> None:
        self.save_clone()

    @on(Button.Pressed, "#config-launch")
    def _launch_pressed(self, _event: Button.Pressed) -> None:
        self.launch()

    @on(Input.Changed)
    def _input_changed(self, _event: Input.Changed) -> None:
        self._saved = False
        self._render_preview()

    def _render_preview(self) -> None:
        self.query_one("#config-preview-body", Static).update(self._preview_renderable())

    def _preview_renderable(self) -> Group:
        return _preview_panel(
            self._item,
            self._original,
            self._current_build(),
            self._clone_path,
            saved=self._saved,
        )

    def _current_build(self) -> ConfigBuildResult:
        return _form_config(self._original_config, self._config_kind, self._field_value)

    def _field_value(self, field_id: str) -> str:
        try:
            return self.query_one(f"#{field_id}", ClearableInput).value.strip()
        except NoMatches:
            return self._initial_fields.get(FIELD_INITIAL_KEYS.get(field_id, ""), "").strip()

    def _default_clone_path(self) -> Path:
        source_path = Path(str(self._item.raw.get("path") or self._item.title))
        stem = source_path.stem or self._item.title
        return self._workspace / ".prime" / "lab" / "configs" / self._config_kind / f"{stem}.toml"


class ConfigLaunchScreen(Screen[None]):
    """Native launch/follow screen for config-backed Lab commands."""

    BINDINGS = [
        Binding("escape", "back", "Back", key_display="Esc"),
        Binding("s", "stop", "Stop"),
    ]

    CSS = (
        BUTTON_CSS
        + """
    ConfigLaunchScreen {
        background: $background;
        color: $foreground;
        layout: vertical;
    }

    #launch-header {
        height: 4;
        padding: 0 1;
        border-bottom: solid $primary;
        background: $background;
    }

    #launch-body {
        height: 1fr;
        padding: 1;
    }

    #launch-output {
        height: 1fr;
        border: round $primary;
        background: $surface;
        padding: 0 1;
    }

    .launch-actions {
        height: auto;
        margin: 1 0;
    }

    .launch-action-button {
        width: 1fr;
        margin-right: 1;
    }
    """
    )

    def __init__(
        self,
        *,
        command: str,
        workspace: Path,
        config_path: Path | None = None,
        title: str = "Launching config",
        subtitle: str = "",
        follow_training_logs: bool = False,
    ) -> None:
        super().__init__()
        self._command = command
        self._workspace = workspace
        self._config_path = config_path
        self._title = title
        self._subtitle = subtitle
        self._follow_training_logs = follow_training_logs
        self._output = ""
        self._runner: ConfigLaunchRunner | None = None
        self._running = False

    def compose(self) -> ComposeResult:
        yield Static(
            _launch_header(
                self._command,
                self._config_path,
                title=self._title,
                subtitle=self._subtitle,
            ),
            id="launch-header",
        )
        with VerticalScroll(id="launch-body"):
            yield Static(_launch_summary(self._command, self._workspace, self._config_path))
            with Horizontal(classes="launch-actions"):
                yield Button("Stop", id="launch-stop", classes="launch-action-button")
            with VerticalScroll(id="launch-output"):
                yield Label("Starting launch ...", id="launch-status")
                yield Static("", id="launch-log", markup=False)
        yield Footer()

    def on_mount(self) -> None:
        self._running = True
        self._runner = ConfigLaunchRunner(
            command=self._command,
            workspace=self._workspace,
            follow_training_logs=self._follow_training_logs,
            append_output=lambda text: self.app.call_from_thread(self._append_output, text),
            update_status=lambda text, style: self.app.call_from_thread(
                self._update_status,
                text,
                style,
            ),
            finish=lambda kind, returncode: self.app.call_from_thread(
                self._finish_runner,
                kind,
                returncode,
            ),
        )
        self._run_launch_worker()

    def action_back(self) -> None:
        self.app.pop_screen()

    def action_stop(self) -> None:
        self._stop_process()

    @work(thread=True, exclusive=True)
    def _run_launch_worker(self) -> None:
        if self._runner is not None:
            self._runner.run()

    def _append_output(self, text: str) -> None:
        self._output = (self._output + text)[-100_000:]
        self.query_one("#launch-log", Static).update(Text(self._output))

    def _update_status(self, text: str, style: str) -> None:
        self.query_one("#launch-status", Label).update(Text(text, style=style))

    def _finish_runner(self, kind: str, returncode: int | None) -> None:
        if kind == "stopped":
            self._finish_stopped()
        elif kind == "logs":
            self._finish_log_follow()
        else:
            self._finish_launch(int(returncode or 0))

    def _finish_launch(self, returncode: int) -> None:
        self._running = False
        status = (
            Text("Launch command completed", style=STATUS_SUCCESS)
            if returncode == 0
            else Text(f"Launch command exited with {returncode}", style=STATUS_ERROR)
        )
        self.query_one("#launch-status", Label).update(status)
        try:
            self.query_one("#launch-stop", Button).disabled = True
        except NoMatches:
            pass

    def _finish_log_follow(self) -> None:
        self._running = False
        self.query_one("#launch-status", Label).update(
            Text("Live log stream completed", style=STATUS_SUCCESS)
        )
        try:
            self.query_one("#launch-stop", Button).disabled = True
        except NoMatches:
            pass

    def _finish_stopped(self) -> None:
        self._running = False
        self.query_one("#launch-status", Label).update(Text("Stopped", style=STATUS_WARNING))
        try:
            self.query_one("#launch-stop", Button).disabled = True
        except NoMatches:
            pass

    def _stop_process(self) -> None:
        if self._runner is None:
            self._finish_stopped()
            return
        self._runner.stop()

    @on(Button.Pressed, "#launch-stop")
    def _stop_pressed(self, _event: Button.Pressed) -> None:
        self.action_stop()


def _field_widgets(values: dict[str, str]) -> list[Any]:
    fields = [
        ("config-name", "Name", values.get("name", "")),
        ("config-model", "Model", values.get("model", "")),
        ("config-envs", "Environments", values.get("envs", "")),
        ("config-max-steps", "Max steps", values.get("max_steps", ""), "integer"),
        (
            "config-rollouts",
            "Rollouts per example",
            values.get("rollouts_per_example", ""),
            "integer",
        ),
        ("config-batch-size", "Batch size", values.get("batch_size", ""), "integer"),
        ("config-max-tokens", "Max tokens", values.get("max_tokens", ""), "integer"),
        ("config-seq-len", "Seq len", values.get("seq_len", ""), "integer", True),
    ]
    widgets: list[Any] = []
    for spec in fields:
        field_id, label, value, *rest = spec
        input_type = rest[0] if rest else "text"
        disabled = bool(rest[1]) if len(rest) > 1 else False
        widgets.append(Static(label, classes="field-label", markup=False))
        widgets.append(
            ClearableInput(
                str(value),
                id=field_id,
                type=input_type,
                disabled=disabled,
                classes="read-only" if disabled else "",
            )
        )
    return widgets


def _form_config(config: dict[str, Any], config_kind: str, field_value: Any) -> ConfigBuildResult:
    updated = _normalize_config(deepcopy(config), config_kind)
    errors: list[str] = []
    initial = _initial_field_values(updated, config_kind, fallback_name="")

    _set_if_present(updated, "name", field_value("config-name"))
    _set_if_present(updated, "model", field_value("config-model"))
    _set_int_field(updated, "max_steps", field_value("config-max-steps"), errors)
    _set_rollouts_field(updated, config_kind, field_value("config-rollouts"), errors)
    _set_int_field(updated, "batch_size", field_value("config-batch-size"), errors)
    _set_max_tokens_field(updated, config_kind, field_value("config-max-tokens"), errors)

    env_text = field_value("config-envs")
    if env_text != initial.get("envs", ""):
        _set_environment_field(updated, config_kind, env_text)

    filtered = _filter_empty_values(updated)
    toml_text = format_toml_blocks(toml.dumps(filtered)).rstrip()
    validation = _validate_toml(toml_text)
    if validation.error is not None:
        errors.append(validation.error)
    return ConfigBuildResult(
        validation.parsed if validation.error is None else updated,
        toml_text,
        tuple(errors),
    )


def _set_if_present(config: dict[str, Any], key: str, value: str) -> None:
    if value:
        config[key] = value
    else:
        config.pop(key, None)


def _set_int_field(
    config: dict[str, Any],
    key: str,
    value: str,
    errors: list[str],
) -> None:
    if not value:
        config.pop(key, None)
        return
    parsed = _parse_int_field(key.replace("_", " "), value, errors)
    if parsed is not None:
        config[key] = parsed


def _set_int_path(
    config: dict[str, Any],
    path: tuple[str, ...],
    value: str,
    errors: list[str],
) -> None:
    if len(path) == 1:
        _set_int_field(config, path[0], value, errors)
        return
    parsed = _parse_int_field(path[-1].replace("_", " "), value, errors)
    if parsed is None:
        return
    parent = _ensure_dict_path(config, path[:-1])
    parent[path[-1]] = parsed


def _set_rollouts_field(
    config: dict[str, Any],
    config_kind: str,
    value: str,
    errors: list[str],
) -> None:
    if config_kind == "eval" and _single_eval_config(config) is not None:
        eval_config = _single_eval_config(config)
        if eval_config is not None:
            _set_int_field(eval_config, "rollouts_per_example", value, errors)
            return
    _set_int_field(config, "rollouts_per_example", value, errors)


def _set_max_tokens_field(
    config: dict[str, Any],
    config_kind: str,
    value: str,
    errors: list[str],
) -> None:
    config.pop("max_tokens", None)
    if config_kind == "eval" and _single_eval_config(config) is not None:
        eval_config = _single_eval_config(config)
        if eval_config is not None:
            sampling_args = eval_config.get("sampling_args")
            if not isinstance(sampling_args, dict):
                sampling_args = {}
                eval_config["sampling_args"] = sampling_args
            _set_int_field(sampling_args, "max_tokens", value, errors)
            return
    if config_kind == "rl":
        if not value:
            sampling = config.get("sampling")
            if isinstance(sampling, dict):
                sampling.pop("max_tokens", None)
            return
        _set_int_path(config, ("sampling", "max_tokens"), value, errors)
        return
    if isinstance(config.get("sampling"), dict):
        _set_int_path(config, ("sampling", "max_tokens"), value, errors)
        return
    _set_int_field(config, "max_tokens", value, errors)


def _parse_int_field(label: str, value: str, errors: list[str]) -> int | None:
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        errors.append(f"{label} must be an integer")
        return None


def _ensure_dict_path(config: dict[str, Any], path: tuple[str, ...]) -> dict[str, Any]:
    current = config
    for key in path:
        child = current.get(key)
        if not isinstance(child, dict):
            child = {}
            current[key] = child
        current = child
    return current


def _field_text(config: dict[str, Any], key: str) -> str:
    value = config.get(key)
    return "" if value is None else str(value)


def _initial_field_values(
    config: dict[str, Any],
    config_kind: str,
    *,
    fallback_name: str,
) -> dict[str, str]:
    single_eval = _single_eval_config(config)
    sampling = _dict_or_empty(config.get("sampling"))
    eval_sampling = _dict_or_empty(single_eval.get("sampling_args") if single_eval else None)
    return {
        "name": _field_text(config, "name") or fallback_name,
        "model": _field_text(config, "model"),
        "envs": ", ".join(_environment_tokens(config, config_kind)),
        "max_steps": _field_text(config, "max_steps"),
        "rollouts_per_example": _first_non_empty(
            _field_text(single_eval or {}, "rollouts_per_example"),
            _field_text(config, "rollouts_per_example"),
        ),
        "batch_size": _field_text(config, "batch_size"),
        "max_tokens": _first_non_empty(
            _field_text(eval_sampling, "max_tokens"),
            _field_text(sampling, "max_tokens"),
            _field_text(config, "max_tokens"),
        ),
        "seq_len": _field_text(config, "seq_len"),
    }


def _environment_tokens(config: dict[str, Any], config_kind: str) -> list[str]:
    raw_envs = config.get("env")
    token_key = "id"
    if not isinstance(raw_envs, list):
        raw_envs = config.get("environments")
    if not isinstance(raw_envs, list) and config_kind == "eval":
        raw_eval = config.get("eval")
        if isinstance(raw_eval, list):
            raw_envs = raw_eval
            token_key = "env_id"
    if not isinstance(raw_envs, list):
        return []
    tokens = []
    for env in raw_envs:
        if isinstance(env, dict):
            env_id = str(env.get(token_key) or env.get("id") or env.get("name") or "")
            version = env.get("version")
            tokens.append(f"{env_id}@{version}" if env_id and version else env_id)
        else:
            tokens.append(str(env))
    return [token for token in tokens if token]


def _set_environment_field(config: dict[str, Any], config_kind: str, value: str) -> None:
    tokens = [part.strip() for part in value.split(",") if part.strip()]
    if config_kind == "eval":
        evals = config.get("eval")
        template = {}
        if isinstance(evals, list) and evals and isinstance(evals[0], dict):
            template = dict(evals[0])
        if not tokens:
            config["eval"] = []
            return
        config["eval"] = [
            _env_eval_dict(token, template if index == 0 else {})
            for index, token in enumerate(tokens)
        ]
        return
    if config_kind == "rl":
        config.pop("environments", None)
        config["env"] = [_env_dict(token) for token in tokens]
        return
    if "environments" in config and "env" not in config:
        config["environments"] = tokens
        return
    config["env"] = [_env_dict(token) for token in tokens]


def _env_dict(token: str) -> dict[str, str]:
    env_id, version = _split_environment_token(token)
    env = {"id": env_id}
    if version:
        env["version"] = version
    return env


def _env_eval_dict(token: str, template: dict[str, Any]) -> dict[str, Any]:
    env_id, version = _split_environment_token(token)
    env = dict(template)
    env["env_id"] = env_id
    if version:
        env["version"] = version
    return env


def _split_environment_token(token: str) -> tuple[str, str | None]:
    if "@" in token:
        env_id, version = token.rsplit("@", 1)
        return env_id.strip(), version.strip() or None
    if ":" in token and "/" in token:
        env_id, version = token.rsplit(":", 1)
        return env_id.strip(), version.strip() or None
    return token, None


def _single_eval_config(config: dict[str, Any]) -> dict[str, Any] | None:
    evals = config.get("eval")
    if not isinstance(evals, list) or len(evals) != 1 or not isinstance(evals[0], dict):
        return None
    return evals[0]


def _first_non_empty(*values: str) -> str:
    for value in values:
        if value:
            return value
    return ""


def _dict_or_empty(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _normalize_config(config: dict[str, Any], config_kind: str) -> dict[str, Any]:
    if config_kind == "rl":
        return normalize_rl_config(config)
    return config


def _config_header(item: LabItem) -> Group:
    text = Text()
    text.append(str(item.title), style="bold")
    text.append("  Edit and run", style=f"bold {PRIMARY}")
    text.append(f"\n{item.subtitle}", style="dim")
    return Group(lab_header(text))


def _source_panel(item: LabItem) -> Group:
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold dim", no_wrap=True)
    table.add_column()
    if item.status:
        table.add_row("Kind", item.status)
    for key, value in item.metadata:
        table.add_row(key, value)
    return Group(Text("Source config", style="bold"), table)


def _launch_header(
    command: str,
    config_path: Path | None,
    *,
    title: str,
    subtitle: str,
) -> Group:
    text = Text()
    text.append(title, style="bold")
    subject = subtitle or (config_path.name if config_path is not None else "")
    if subject:
        text.append(f"\n{subject}", style="dim")
    text.append("  ")
    text.append(command, style=PRIMARY)
    return Group(lab_header(text))


def _launch_summary(command: str, workspace: Path, config_path: Path | None) -> Group:
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold dim", no_wrap=True)
    table.add_column()
    table.add_row("Command", command)
    table.add_row("Workspace", str(workspace))
    if config_path is not None:
        table.add_row("Config", str(config_path))
    note = Text("Output follows below. New runs can take a moment to start.", style="dim")
    return Group(Text("Launch", style="bold"), table, Text(""), note)


def _preview_panel(
    item: LabItem,
    original: str,
    build: ConfigBuildResult,
    clone_path: Path,
    *,
    saved: bool = False,
) -> Group:
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold dim", no_wrap=True)
    table.add_column()
    status = (
        Text("Configuration is valid", style=STATUS_SUCCESS)
        if not build.errors
        else Text("\n".join(build.errors), style=STATUS_ERROR)
    )
    table.add_row("Validation", status)
    table.add_row("Clone", str(clone_path))
    workspace = Path(str(item.raw.get("workspace") or ".")).resolve()
    command = _launch_command(
        str(item.raw.get("config_kind") or "rl"),
        _relative_path(clone_path, workspace),
    )
    table.add_row("Command", command)
    if saved:
        table.add_row("Saved", Text("yes", style=STATUS_SUCCESS))
    if seq_len := build.parsed.get("seq_len"):
        table.add_row("Seq len", Text(f"{seq_len}  read-only", style=STATUS_WARNING))

    chunks: list[Any] = [Text("Validate & launch preview", style="bold"), table]
    diff = _diff_table(original, build.toml_text)
    if diff.row_count:
        chunks.extend([Text(""), Text("Field changes", style="bold"), diff])
    chunks.extend(
        [
            Text(""),
            Text("TOML", style="bold"),
            Syntax(build.toml_text.rstrip(), "toml", theme=CODE_THEME),
        ]
    )
    return Group(*chunks)


def _validate_toml(value: str) -> ValidationResult:
    try:
        parsed = toml.loads(value)
    except toml.TomlDecodeError as exc:
        return ValidationResult({}, str(exc))
    return ValidationResult(parsed if isinstance(parsed, dict) else {}, None)


def _diff_table(original: str, current: str) -> Table:
    table = Table(show_header=True, header_style="bold dim")
    table.add_column("Field")
    table.add_column("Original")
    table.add_column("New")
    before = _validate_toml(original).parsed
    after = _validate_toml(current).parsed
    before_flat = _flatten(before)
    after_flat = _flatten(after)
    for key in sorted(set(before_flat) | set(after_flat)):
        old = before_flat.get(key)
        new = after_flat.get(key)
        if old == new:
            continue
        table.add_row(key, _display(old), _display(new))
    return table


def _flatten(value: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, child in value.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(child, dict):
            flattened.update(_flatten(child, full_key))
        else:
            flattened[full_key] = child
    return flattened


def _display(value: Any) -> str:
    if value is None:
        return "-"
    return str(value)


def _launch_command(config_kind: str, rel_path: str) -> str:
    if config_kind == "rl":
        return f"prime rl run {rel_path}"
    if config_kind == "eval":
        return f"prime eval run {rel_path} --hosted"
    return f"prime gepa run {rel_path}"


def _filter_empty_values(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): child
            for key, child in ((key, _filter_empty_values(child)) for key, child in value.items())
            if child is not None and child != {} and child != []
        }
    if isinstance(value, list):
        return [
            child
            for child in (_filter_empty_values(child) for child in value)
            if child is not None and child != {} and child != []
        ]
    return value


def _relative_path(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)
