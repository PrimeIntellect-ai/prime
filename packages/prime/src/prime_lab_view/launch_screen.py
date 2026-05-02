"""Dedicated launch screen for the Lab TUI."""

from __future__ import annotations

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.css.query import NoMatches
from textual.screen import Screen
from textual.widgets import Button, Static

from .palette import BUTTON_CSS, PRIMARY
from .widgets import HomeLaunchPanel, HomeLaunchState


class LaunchScreen(Screen[str | None]):
    """Full-screen launch surface shown before entering the Lab workspace."""

    BINDINGS = [
        Binding("enter", "enter_lab", "Enter Lab", key_display="Enter"),
        Binding("c", "agent", "Agent"),
        Binding("r", "refresh_lab", "Refresh"),
        Binding("q", "quit", "Quit"),
    ]

    CSS = (
        BUTTON_CSS
        + """
    LaunchScreen {
        background: $background;
        color: $foreground;
        layout: vertical;
    }

    #home-launch {
        height: 24;
        min-height: 22;
        background: $background;
        padding: 0;
        margin: 3 0 0 0;
    }

    #launch-actions {
        height: auto;
        align-horizontal: center;
        padding: 0 2;
        margin: 0 0 0 0;
    }

    .launch-action-row {
        height: 4;
        align-horizontal: center;
    }

    .launch-action-button {
        width: 40;
        height: 3;
        margin: 0 1 1 1;
    }

    #launch-hotkeys {
        height: 1;
        color: $text-muted;
        content-align: center middle;
        margin: 0 0 0 0;
    }
    """
    )

    def __init__(self) -> None:
        super().__init__()
        self._state = HomeLaunchState(
            workspace="-",
            auth_label="auth ?",
            team="-",
            agent_label="agent none",
            loading=True,
        )

    def compose(self) -> ComposeResult:
        yield HomeLaunchPanel(id="home-launch")
        with Vertical(id="launch-actions"):
            with Horizontal(classes="launch-action-row"):
                yield _launch_button(
                    "Explore Environments",
                    id="launch-explore",
                )
                yield _launch_button(
                    "Train Models",
                    id="launch-train",
                )
            with Horizontal(classes="launch-action-row"):
                yield _launch_button(
                    "Run Evaluations",
                    id="launch-evaluate",
                )
                yield _launch_button(
                    self._state.agent_action_label,
                    id="launch-agent",
                )
            yield Static(_launch_hotkeys(), id="launch-hotkeys")

    def on_mount(self) -> None:
        self.update_state(self._state)

    def update_state(self, state: HomeLaunchState) -> None:
        self._state = state
        try:
            self.query_one("#home-launch", HomeLaunchPanel).update_state(state)
            self.query_one("#launch-agent", Button).label = state.agent_action_label
        except NoMatches:
            return

    def action_enter_lab(self) -> None:
        self.dismiss("home")

    def action_refresh_lab(self) -> None:
        action_refresh = getattr(self.app, "action_refresh", None)
        if callable(action_refresh):
            action_refresh()

    def action_agent(self) -> None:
        self.dismiss("agent")

    def action_quit(self) -> None:
        self.app.exit()

    @on(Button.Pressed, ".launch-action-button")
    def _launch_action_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id or ""
        self.dismiss(button_id.removeprefix("launch-"))


def _launch_hotkeys() -> Text:
    return Text.assemble(
        ("Enter", f"bold {PRIMARY}"),
        (" Lab", "dim"),
        ("   ", "dim"),
        ("c", f"bold {PRIMARY}"),
        (" agent", "dim"),
        ("   ", "dim"),
        ("r", f"bold {PRIMARY}"),
        (" refresh", "dim"),
        ("   ", "dim"),
        ("q", f"bold {PRIMARY}"),
        (" quit", "dim"),
    )


def _launch_button(label: str, *, id: str) -> Button:
    button = Button(label, id=id, classes="launch-action-button -style-default")
    button.can_focus = False
    return button
