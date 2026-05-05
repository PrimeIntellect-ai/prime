"""Interactive agent control cards for Lab chat."""

from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from rich.console import Group
from rich.table import Table
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.css.query import NoMatches
from textual.widgets import Button, Select, Static

from .agent_runtime import AgentChatMessage
from .agent_widget_actions import (
    AgentWidgetLaunchPlan,
    agent_widget_choice_action,
    agent_widget_launch_action,
    build_agent_widget_config,
    prepare_agent_widget_launch,
)
from .agent_widget_model import (
    AgentWidgetModel,
    build_agent_widget_model,
    widget_payload,
)
from .config_screen import ConfigBuildResult
from .launch_runner import ConfigLaunchRunner
from .palette import PRIMARY, STATUS_ERROR, STATUS_WARNING, SUCCESS
from .widgets import ClearableInput

AgentActionRecorder = Callable[[dict[str, Any]], None]


class AgentWidgetCard(Vertical):
    """Interactive Lab control request embedded in the agent chat."""

    def __init__(
        self,
        message: AgentChatMessage,
        *,
        workspace: Path,
        record_action: AgentActionRecorder | None = None,
    ) -> None:
        super().__init__()
        self._message = message
        self._workspace = workspace
        self._record_action = record_action
        self._model = build_agent_widget_model(message, workspace)
        self._action = self._model.action
        self._config_context = self._model.config_context
        self._output = ""
        self._runner: ConfigLaunchRunner | None = None
        self._launch_running = False
        self._active_launch_action: dict[str, Any] | None = None
        self._closed = False

    def compose(self) -> ComposeResult:
        yield Static(
            _widget_card_heading(self._model),
            classes="agent-widget-heading",
            markup=False,
        )
        fields = self._model.fields
        if fields:
            with Vertical(classes="agent-widget-fields"):
                for field in fields:
                    row_classes = (
                        "agent-widget-field-row agent-widget-model-row"
                        if field.name == "model"
                        else "agent-widget-field-row"
                    )
                    with Vertical(classes=row_classes):
                        yield Static(
                            field.label,
                            classes="agent-widget-field-label",
                            markup=False,
                        )
                        if field.widget == "select":
                            yield Select(
                                field.options,
                                value=field.value,
                                allow_blank=False,
                                name=field.name,
                                classes="agent-widget-select",
                                compact=False,
                            )
                        else:
                            yield ClearableInput(
                                field.value,
                                name=field.name,
                                type=field.input_type,
                                disabled=field.disabled,
                                classes=(
                                    "agent-widget-field read-only"
                                    if field.disabled
                                    else "agent-widget-field"
                                ),
                            )
        else:
            yield Static(_widget_card_body(self._model), markup=False)
        actions = self._model.actions
        if actions:
            with Horizontal(classes="agent-widget-actions"):
                for action in actions:
                    yield Button(
                        action.label,
                        name=action.name,
                        variant=action.variant,
                        classes=f"agent-widget-action {_action_class(action.name)}",
                    )
        yield Static(Text("Ready", style="dim"), classes="agent-widget-status", markup=False)
        yield Static("", classes="agent-widget-log", markup=False)

    def on_mount(self) -> None:
        self._closed = False
        self._set_launch_buttons(running=self._launch_running)

    def on_unmount(self) -> None:
        self._closed = True
        if self._launch_running and self._runner is not None:
            self._runner.stop()
        self._launch_running = False

    def on_button_pressed(self, event: Button.Pressed) -> None:
        name = event.button.name or ""
        if name == "launch":
            event.stop()
            self._start_inline_launch()
            return
        if name == "stop":
            event.stop()
            self._stop_inline_launch()
            return
        if name.startswith("choice:"):
            event.stop()
            self._select_choice(name.removeprefix("choice:"))

    def _start_inline_launch(self) -> None:
        if self._launch_running:
            return
        plan = prepare_agent_widget_launch(
            self._model,
            workspace=self._workspace,
            field_values=self._widget_input_values(),
        )
        if plan.errors:
            self._set_widget_status("Fix config values before launching.", STATUS_ERROR)
            self._set_widget_log("\n".join(plan.errors), visible=True)
            return
        if not plan.command:
            self._set_widget_status("No launch command available.", STATUS_ERROR)
            return
        self._output = f"$ {plan.command}\n\n"
        self._launch_running = True
        launch_action = agent_widget_launch_action(
            self._model,
            command=plan.command,
            workspace=self._workspace,
            status="started",
        )
        self._active_launch_action = launch_action
        self._record_widget_action(launch_action)
        self._set_launch_buttons(running=True)
        self._set_widget_status("Launching ...", PRIMARY)
        self._set_widget_log(self._output, visible=True)
        self._runner = ConfigLaunchRunner(
            command=plan.command,
            workspace=self._workspace,
            follow_training_logs=plan.follow_training_logs,
            append_output=lambda text: self._call_from_launch_thread(
                self._append_widget_output,
                text,
            ),
            update_status=lambda text, style: self._call_from_launch_thread(
                self._set_widget_status,
                text,
                style,
            ),
            finish=lambda kind, returncode: self._call_from_launch_thread(
                self._finish_inline_runner,
                kind,
                returncode,
            ),
        )
        self.run_worker(self._runner.run, exclusive=True, thread=True)

    def _stop_inline_launch(self) -> None:
        if self._runner is None:
            self._finish_inline_stopped()
            return
        self._runner.stop()

    def _call_from_launch_thread(self, callback: Any, *args: Any) -> None:
        if self._closed:
            return
        if getattr(self.app, "_thread_id", None) == threading.get_ident():
            callback(*args)
            return
        try:
            self.app.call_from_thread(callback, *args)
        except RuntimeError:
            return

    def _append_widget_output(self, text: str) -> None:
        if self._closed:
            return
        self._output = (self._output + text)[-50_000:]
        self._set_widget_log(self._output, visible=True)

    def _finish_inline_launch(self, returncode: int) -> None:
        self._launch_running = False
        if self._closed:
            return
        if returncode == 0:
            self._set_widget_status("Completed", SUCCESS)
        else:
            self._set_widget_status(f"Exited with {returncode}", STATUS_ERROR)
        self._record_widget_action(
            agent_widget_launch_action(
                self._model,
                command=(
                    self._active_launch_action.get("command", "")
                    if self._active_launch_action
                    else ""
                ),
                workspace=self._workspace,
                status="completed" if returncode == 0 else "failed",
                returncode=returncode,
            )
        )
        self._set_launch_buttons(running=False)

    def _finish_inline_logs(self) -> None:
        self._launch_running = False
        if self._closed:
            return
        self._set_widget_status("Live log stream completed", SUCCESS)
        if self._active_launch_action is not None:
            self._record_widget_action(
                {
                    **self._active_launch_action,
                    "type": "agent_inline_launch",
                    "status": "logs_completed",
                }
            )
        self._set_launch_buttons(running=False)

    def _finish_inline_stopped(self) -> None:
        self._launch_running = False
        if self._closed:
            return
        self._set_widget_status("Stopped", STATUS_WARNING)
        if self._active_launch_action is not None:
            self._record_widget_action(
                {
                    **self._active_launch_action,
                    "type": "agent_inline_launch",
                    "status": "stopped",
                }
            )
        self._set_launch_buttons(running=False)

    def _finish_inline_runner(self, kind: str, returncode: int | None) -> None:
        if kind == "stopped":
            self._finish_inline_stopped()
        elif kind == "logs":
            self._finish_inline_logs()
        else:
            self._finish_inline_launch(int(returncode or 0))

    def _set_widget_status(self, text: str, style: str = "") -> None:
        if self._closed:
            return
        status = Text(text, style=style)
        try:
            self.query_one(".agent-widget-status", Static).update(status)
        except NoMatches:
            return

    def _set_widget_log(self, text: str, *, visible: bool) -> None:
        if self._closed:
            return
        try:
            log = self.query_one(".agent-widget-log", Static)
        except NoMatches:
            return
        log.set_class(visible, "visible")
        log.update(Text(text))

    def _set_launch_buttons(self, *, running: bool) -> None:
        for button in self.query(Button):
            if button.name == "launch":
                button.disabled = running
            elif button.name == "stop":
                button.disabled = not running

    def _record_widget_action(self, action: dict[str, Any]) -> None:
        if self._record_action is not None:
            self._record_action(action)

    def _select_choice(self, choice_id: str) -> None:
        label, action = agent_widget_choice_action(
            self._model,
            choice_id=choice_id,
            workspace=self._workspace,
        )
        self._set_widget_status(f"Selected {label}", SUCCESS)
        self._record_widget_action(action)

    def _current_config_build(self) -> ConfigBuildResult:
        return build_agent_widget_config(self._model, self._widget_input_values())

    def current_launch_plan(self) -> AgentWidgetLaunchPlan:
        """Return the launch plan represented by the current visible field values."""

        return prepare_agent_widget_launch(
            self._model,
            workspace=self._workspace,
            field_values=self._widget_input_values(),
        )

    def _widget_input_values(self) -> dict[str, str]:
        values: dict[str, str] = {}
        for input_widget in self.query(ClearableInput):
            if input_widget.name:
                values[input_widget.name] = input_widget.value.strip()
        for select_widget in self.query(Select):
            if select_widget.name:
                value = select_widget.value
                values[select_widget.name] = "" if value is Select.BLANK else str(value).strip()
        return values


def _widget_card_heading(model: AgentWidgetModel) -> Group:
    payload = model.payload
    title = model.title
    heading = Text()
    heading.append(title, style="bold")
    summary = (
        "" if model.config_context is not None else str(payload.get("description") or "").strip()
    )
    if summary:
        return Group(heading, Text(summary, style="dim"))
    return Group(heading)


def _widget_card_body(model: AgentWidgetModel) -> Group:
    payload = widget_payload(model.action)
    title = model.title
    description = str(payload.get("description") or "").strip()
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold dim", no_wrap=True)
    table.add_column()
    if config_path := str(payload.get("config_path") or "").strip():
        table.add_row("Path", _short_path(config_path))
    if description:
        table.add_row("Summary", description)
    if candidates := payload.get("candidates"):
        count = len(candidates) if isinstance(candidates, list) else 0
        if count:
            table.add_row("Choices", str(count))
    heading = Text.assemble((title, "bold"))
    return Group(heading, Text(""), table)


def _short_path(value: str) -> str:
    path = Path(value).expanduser()
    try:
        return f"~/{path.resolve().relative_to(Path.home())}"
    except ValueError:
        return value


def _action_class(name: str) -> str:
    safe = "".join(
        character if character.isalnum() or character == "-" else "-" for character in name
    )
    safe = safe.strip("-") or "action"
    return f"agent-widget-action-{safe}"
