"""Lab workspace setup, sync, and doctor screens."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from prime_cli.lab_setup import (
    LabDoctorOptions,
    LabDoctorResult,
    LabSetupOptions,
    LabSetupResult,
    LabSyncOptions,
    LabSyncResult,
    run_lab_doctor_service,
    run_lab_setup_service,
    run_lab_sync_service,
)
from rich.console import Group
from rich.table import Table
from rich.text import Text
from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, Footer, Select, Static

from .agent_capabilities import agent_select_options
from .environment_screen import EnvironmentScreen
from .models import LabItem
from .palette import STATUS_ERROR, STATUS_SUCCESS, STATUS_WARNING
from .shell import lab_header

SetupCompleteAction = Callable[[], None]


class SetupScreen(Screen[None]):
    """Guided setup entry point for an uninitialized Lab workspace."""

    BINDINGS = [
        Binding("escape", "back", "Back", key_display="Esc"),
        Binding("b", "back", "Back", key_display="B"),
    ]

    CSS = (
        EnvironmentScreen.CSS
        + """
    SetupScreen {
        layout: vertical;
    }

    .setup-shell {
        height: 1fr;
    }

    .setup-controls {
        width: 42;
        height: auto;
        margin-top: 1;
    }

    .setup-field-label {
        height: 1;
        color: $text-muted;
    }

    .setup-agent-select {
        width: 40;
        height: auto;
        background: $panel;
    }

    .setup-agent-select > SelectOverlay {
        height: auto;
        max-height: 20;
        border: tall $primary;
        background: $panel;
    }

    .setup-run-button {
        width: auto;
        min-width: 16;
        margin-top: 1;
    }

    .setup-log {
        height: 1fr;
        min-height: 12;
        border: round $primary;
        background: $surface;
        padding: 0 1;
        margin-top: 1;
    }
    """
    )

    def __init__(self, item: LabItem, on_complete: SetupCompleteAction | None = None) -> None:
        super().__init__()
        self._item = item
        self._on_complete = on_complete
        self._agent_options = agent_select_options("codex")
        self._default_agent = str(self._agent_options[0][1])
        self._command_running = False
        self._output = ""

    def compose(self) -> ComposeResult:
        yield Static(_setup_header(self._item), id="env-header", classes="page-header")
        with Vertical(classes="empty-panel setup-shell"):
            yield Static(_setup_body(self._item), markup=False)
            with Vertical(classes="setup-controls"):
                yield Static(
                    "Choose your coding agent",
                    id="setup-agent-label",
                    classes="setup-field-label",
                    markup=False,
                )
                yield Select(
                    self._agent_options,
                    value=self._default_agent,
                    allow_blank=False,
                    id="setup-agent",
                    classes="setup-agent-select",
                    compact=False,
                )
                yield Button(
                    "Run setup",
                    id="setup-run",
                    classes="setup-run-button",
                    variant="primary",
                )
            yield Static("Ready", id="setup-status", markup=False)
            with VerticalScroll(classes="setup-log"):
                yield Static("Setup output will appear here.", id="setup-output", markup=False)
        yield Footer()

    def action_back(self) -> None:
        self.app.pop_screen()

    def action_run_setup(self) -> None:
        if self._command_running:
            return
        workspace = Path(str(self._item.raw.get("workspace") or self._item.subtitle)).resolve()
        self._command_running = True
        self._output = ""
        self._set_setup_buttons_disabled(True)
        self.query_one("#setup-status", Static).update("Running setup ...")
        self.query_one("#setup-output", Static).update("")
        self._run_setup_worker(workspace, self._selected_agent())

    @work(thread=True, exclusive=True)
    def _run_setup_worker(self, workspace: Path, agent: str) -> None:
        result = run_lab_setup_service(
            LabSetupOptions(agents=(agent,)),
            workspace=workspace,
            emit=lambda text: self.app.call_from_thread(self._append_setup_output, text),
        )
        self.app.call_from_thread(self._finish_setup, result)

    def _append_setup_output(self, text: str) -> None:
        self._output = (self._output + text)[-50000:]
        self.query_one("#setup-output", Static).update(Text(self._output))

    def _finish_setup(self, result: LabSetupResult) -> None:
        self._command_running = False
        self._set_setup_buttons_disabled(False)
        if result.exit_code == 0:
            self.query_one("#setup-status", Static).update("Setup completed")
            if self._on_complete is not None:
                self._on_complete()
        else:
            self.query_one("#setup-status", Static).update("Setup failed")

    def _selected_agent(self) -> str:
        value = self.query_one("#setup-agent", Select).value
        return self._default_agent if value is Select.BLANK else str(value)

    def _set_setup_buttons_disabled(self, disabled: bool) -> None:
        self.query_one("#setup-agent", Select).disabled = disabled
        self.query_one("#setup-run", Button).disabled = disabled

    @on(Button.Pressed, "#setup-run")
    def _setup_pressed(self, _event: Button.Pressed) -> None:
        self.action_run_setup()


class AgentSyncScreen(Screen[None]):
    """Refresh Lab templates, skills, docs, and local agent guidance."""

    BINDINGS = [
        Binding("escape", "back", "Back", key_display="Esc"),
        Binding("b", "back", "Back", key_display="B"),
        Binding("enter", "run_sync", "Run sync", key_display="Enter"),
    ]

    CSS = (
        EnvironmentScreen.CSS
        + """
    AgentSyncScreen {
        layout: vertical;
    }

    .sync-shell {
        height: 1fr;
    }

    .sync-action-button {
        width: 1fr;
        margin-right: 1;
    }

    .sync-log {
        height: 1fr;
        min-height: 12;
        border: round $primary;
        background: $surface;
        padding: 0 1;
        margin-top: 1;
    }
    """
    )

    def __init__(self, item: LabItem, on_complete: SetupCompleteAction | None = None) -> None:
        super().__init__()
        self._item = item
        self._on_complete = on_complete
        self._command_running = False
        self._output = ""

    def compose(self) -> ComposeResult:
        yield Static(_agent_sync_header(self._item), id="env-header", classes="page-header")
        with Vertical(classes="empty-panel sync-shell"):
            yield Static(_agent_sync_body(self._item), markup=False)
            with Horizontal(classes="setup-actions"):
                yield Button("Run sync", id="sync-run", variant="primary")
            yield Static("Ready", id="sync-status", markup=False)
            with VerticalScroll(classes="sync-log"):
                yield Static("Sync output will appear here.", id="sync-output", markup=False)
        yield Footer()

    def action_back(self) -> None:
        self.app.pop_screen()

    def action_run_sync(self) -> None:
        if self._command_running:
            return
        workspace = Path(str(self._item.raw.get("workspace") or self._item.subtitle)).resolve()
        self._command_running = True
        self._output = ""
        self.query_one("#sync-run", Button).disabled = True
        self.query_one("#sync-status", Static).update("Running sync ...")
        self.query_one("#sync-output", Static).update("")
        agent = str(self._item.raw.get("agent") or "codex")
        self._run_sync_worker(workspace, agent)

    @work(thread=True, exclusive=True)
    def _run_sync_worker(self, workspace: Path, agent: str) -> None:
        result = run_lab_sync_service(
            LabSyncOptions(agents=(agent,)),
            workspace=workspace,
            emit=lambda text: self.app.call_from_thread(self._append_sync_output, text),
        )
        self.app.call_from_thread(self._finish_sync, result)

    def _append_sync_output(self, text: str) -> None:
        self._output = (self._output + text)[-50000:]
        self.query_one("#sync-output", Static).update(Text(self._output))

    def _finish_sync(self, result: LabSyncResult) -> None:
        self._command_running = False
        self.query_one("#sync-run", Button).disabled = False
        if result.exit_code == 0:
            self.query_one("#sync-status", Static).update("Sync completed")
            if self._on_complete is not None:
                self._on_complete()
        else:
            self.query_one("#sync-status", Static).update("Sync failed")

    @on(Button.Pressed, "#sync-run")
    def _sync_pressed(self, _event: Button.Pressed) -> None:
        self.action_run_sync()


class DoctorScreen(Screen[None]):
    """Run deterministic Lab workspace checks."""

    BINDINGS = [
        Binding("escape", "back", "Back", key_display="Esc"),
        Binding("b", "back", "Back", key_display="B"),
        Binding("enter", "run_check", "Run check", key_display="Enter"),
        Binding("f", "fix", "Apply fixes"),
    ]

    CSS = (
        EnvironmentScreen.CSS
        + """
    DoctorScreen {
        layout: vertical;
    }

    .doctor-shell {
        height: 1fr;
    }

    .doctor-action-button {
        width: 1fr;
        margin-right: 1;
    }

    .doctor-results {
        height: 1fr;
        min-height: 12;
        border: round $primary;
        background: $surface;
        padding: 0 1;
        margin-top: 1;
    }
    """
    )

    def __init__(self, item: LabItem, on_complete: SetupCompleteAction | None = None) -> None:
        super().__init__()
        self._item = item
        self._on_complete = on_complete
        self._command_running = False

    def compose(self) -> ComposeResult:
        yield Static(_doctor_header(self._item), id="env-header", classes="page-header")
        with Vertical(classes="empty-panel doctor-shell"):
            yield Static(_doctor_body(self._item), markup=False)
            with Horizontal(classes="setup-actions"):
                yield Button(
                    "Run check",
                    id="doctor-run",
                    classes="doctor-action-button",
                    variant="primary",
                )
                yield Button(
                    "Apply safe fixes",
                    id="doctor-fix",
                    classes="doctor-action-button",
                )
            yield Static("Ready", id="doctor-status", markup=False)
            with VerticalScroll(classes="doctor-results"):
                yield Static("Workspace checks will appear here.", id="doctor-output")
        yield Footer()

    def on_mount(self) -> None:
        self.action_run_check()

    def action_back(self) -> None:
        self.app.pop_screen()

    def action_run_check(self) -> None:
        self._run_doctor(fix=False)

    def action_fix(self) -> None:
        self._run_doctor(fix=True)

    def _run_doctor(self, *, fix: bool) -> None:
        if self._command_running:
            return
        workspace = Path(str(self._item.raw.get("workspace") or self._item.subtitle)).resolve()
        self._command_running = True
        self._set_doctor_buttons_disabled(True)
        self.query_one("#doctor-status", Static).update(
            "Applying safe fixes ..." if fix else "Checking workspace ..."
        )
        self._run_doctor_worker(workspace, fix)

    @work(thread=True, exclusive=True)
    def _run_doctor_worker(self, workspace: Path, fix: bool) -> None:
        result = run_lab_doctor_service(LabDoctorOptions(fix=fix), workspace=workspace)
        self.app.call_from_thread(self._finish_doctor, result, fix)

    def _finish_doctor(self, result: LabDoctorResult, fix: bool) -> None:
        self._command_running = False
        self._set_doctor_buttons_disabled(False)
        self.query_one("#doctor-status", Static).update(
            "Workspace passed" if result.exit_code == 0 else "Workspace needs attention"
        )
        self.query_one("#doctor-output", Static).update(_doctor_result_table(result))
        if fix and self._on_complete is not None:
            self._on_complete()

    def _set_doctor_buttons_disabled(self, disabled: bool) -> None:
        self.query_one("#doctor-run", Button).disabled = disabled
        self.query_one("#doctor-fix", Button).disabled = disabled

    @on(Button.Pressed, "#doctor-run")
    def _doctor_run_pressed(self, _event: Button.Pressed) -> None:
        self.action_run_check()

    @on(Button.Pressed, "#doctor-fix")
    def _doctor_fix_pressed(self, _event: Button.Pressed) -> None:
        self.action_fix()


def _setup_header(item: LabItem) -> Table:
    text = Text()
    text.append(item.title, style="bold")
    text.append(f"\n{item.subtitle}", style="dim")
    return lab_header(text)


def _agent_sync_header(item: LabItem) -> Table:
    text = Text()
    text.append(item.title, style="bold")
    text.append(f"\n{item.subtitle}", style="dim")
    return lab_header(text)


def _doctor_header(item: LabItem) -> Table:
    text = Text()
    text.append(item.title, style="bold")
    text.append(f"\n{item.subtitle}", style="dim")
    return lab_header(text)


def _setup_body(item: LabItem) -> Group:
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold dim", no_wrap=True)
    table.add_column()
    table.add_row("Workspace", str(item.raw.get("workspace") or item.subtitle))
    table.add_row("Command", str(item.raw.get("command") or "prime lab setup"))
    note = Text("Choose an agent, then run setup in this workspace.", style=STATUS_WARNING)
    return Group(Text("Setup", style="bold"), table, Text(""), note)


def _agent_sync_body(item: LabItem) -> Group:
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold dim", no_wrap=True)
    table.add_column()
    table.add_row("Workspace", str(item.raw.get("workspace") or item.subtitle))
    table.add_row("Agent", str(item.raw.get("agent") or "codex"))
    table.add_row("Command", str(item.raw.get("command") or "prime lab sync"))
    note = Text(
        "Refresh Prime-owned templates, skills, docs, and agent guidance for this workspace.",
        style="dim",
    )
    return Group(Text("Lab asset sync", style="bold"), table, Text(""), note)


def _doctor_body(item: LabItem) -> Group:
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold dim", no_wrap=True)
    table.add_column()
    table.add_row("Workspace", str(item.raw.get("workspace") or item.subtitle))
    table.add_row("Mode", "local deterministic checks")
    table.add_row("Fixes", "standard dirs and gitignore entries only")
    note = Text(
        "Use this before setup, launch, sync, or agent-assisted edits.",
        style="dim",
    )
    return Group(Text("Workspace doctor", style="bold"), table, Text(""), note)


def _doctor_result_table(result: LabDoctorResult) -> Table:
    table = Table.grid(padding=(0, 2))
    table.add_column(no_wrap=True)
    table.add_column(style="bold", no_wrap=True)
    table.add_column()
    table.add_column(style="dim")
    for check in result.checks:
        status_style = {
            "PASS": STATUS_SUCCESS,
            "WARN": STATUS_WARNING,
            "FAIL": STATUS_ERROR,
        }.get(check.status, "dim")
        table.add_row(
            Text(check.status, style=status_style),
            check.name,
            check.message,
            check.remediation,
        )
    return table
