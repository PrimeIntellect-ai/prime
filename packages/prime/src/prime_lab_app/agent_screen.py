"""Coding-agent chat screen for Lab workspaces."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from rich.console import Group, RenderableType
from rich.table import Table
from rich.text import Text
from textual import events, on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.message import Message
from textual.screen import Screen
from textual.widgets import Static, TextArea

from .agent_adapters import agent_adapter
from .agent_cards import AgentWidgetCard
from .agent_runtime import AgentChatMessage, AgentConnectionState
from .agent_sessions import append_agent_prompt_history, load_agent_prompt_history
from .agent_widgets import lab_widget_diagnostic_prompt
from .chat_parts import chat_transcript as _render_chat_transcript
from .chat_parts import render_chat_turn
from .launch_backdrop import LaunchBackdrop
from .models import LabItem
from .palette import BUTTON_CSS
from .shell import action_hint_text, lab_header

AgentStateProvider = Callable[[], AgentConnectionState]
AgentMessagesProvider = Callable[[], tuple[AgentChatMessage, ...]]
AgentSelector = Callable[[Path, str], None]
AgentSender = Callable[[str], None]
AgentSessionStarter = Callable[[Path, str], None]
AgentActionRecorder = Callable[[dict[str, Any]], None]
StatusTextProvider = Callable[[], Text]
CommandMenuRow = tuple[str, str, str]

_LARGE_PASTE_LINE_THRESHOLD = 6
_LARGE_PASTE_CHAR_THRESHOLD = 800
_MAX_PROMPT_LINES = 8


class AgentPrompt(TextArea):
    """Multiline agent prompt where Enter submits and Shift+Enter inserts a newline."""

    class Submitted(Message):
        """Prompt submission event."""

        def __init__(self, prompt: AgentPrompt) -> None:
            self.prompt = prompt
            super().__init__()

    class CommandPrevious(Message):
        """Move command picker selection up."""

        def __init__(self, prompt: AgentPrompt) -> None:
            self.prompt = prompt
            super().__init__()

    class CommandNext(Message):
        """Move command picker selection down."""

        def __init__(self, prompt: AgentPrompt) -> None:
            self.prompt = prompt
            super().__init__()

    class HistoryPrevious(Message):
        """Move to the previous prompt history entry."""

        def __init__(self, prompt: AgentPrompt) -> None:
            self.prompt = prompt
            super().__init__()

    class HistoryNext(Message):
        """Move to the next prompt history entry."""

        def __init__(self, prompt: AgentPrompt) -> None:
            self.prompt = prompt
            super().__init__()

    BINDINGS = [
        Binding("enter", "submit_prompt", show=False, priority=True),
        Binding("shift+enter", "insert_newline", show=False, priority=True),
    ]

    def __init__(self, text: str = "", **kwargs: Any) -> None:
        super().__init__(text, **kwargs)
        self._large_paste_payload = ""
        self._large_paste_placeholder = ""
        self._history_mode = False

    @property
    def submitted_text(self) -> str:
        if self._large_paste_payload and self.text == self._large_paste_placeholder:
            return self._large_paste_payload
        return self.text

    def load_text(self, text: str) -> None:
        self._large_paste_payload = ""
        self._large_paste_placeholder = ""
        self._history_mode = False
        self.remove_class("large-paste")
        super().load_text(text)
        self._resize_for_text()

    def load_history_text(self, text: str) -> None:
        self.load_text(text)
        self._history_mode = True

    def clear_prompt(self) -> None:
        self._history_mode = False
        self.load_text("")

    def sync_after_change(self) -> None:
        if self._large_paste_payload and self.text != self._large_paste_placeholder:
            self._large_paste_payload = ""
            self._large_paste_placeholder = ""
            self.remove_class("large-paste")
        self._resize_for_text()

    def apply_large_paste(self, text: str) -> None:
        self._large_paste_payload = text
        self._large_paste_placeholder = _large_paste_placeholder(text)
        self.add_class("large-paste")
        super().load_text(self._large_paste_placeholder)
        self._resize_for_text()

    def action_submit_prompt(self) -> None:
        self.post_message(self.Submitted(self))

    def action_insert_newline(self) -> None:
        if self._large_paste_payload and self.text == self._large_paste_placeholder:
            self.load_text("")
        self.insert("\n")

    async def _on_key(self, event: events.Key) -> None:
        has_menu = _active_menu_prefix(self.text) is not None
        is_empty = not self.text.strip()
        if event.key == "ctrl+c" and not is_empty:
            event.prevent_default()
            event.stop()
            self.clear_prompt()
        elif event.key == "ctrl+c":
            event.prevent_default()
            event.stop()
            self.app.action_quit()
        elif event.key == "up" and has_menu:
            event.prevent_default()
            event.stop()
            self.post_message(self.CommandPrevious(self))
        elif event.key == "down" and has_menu:
            event.prevent_default()
            event.stop()
            self.post_message(self.CommandNext(self))
        elif event.key == "up" and (is_empty or self._history_mode):
            event.prevent_default()
            event.stop()
            self.post_message(self.HistoryPrevious(self))
        elif event.key == "down" and (is_empty or self._history_mode):
            event.prevent_default()
            event.stop()
            self.post_message(self.HistoryNext(self))

    async def _on_paste(self, event: events.Paste) -> None:
        if not _is_large_paste(event.text):
            return
        event.prevent_default()
        event.stop()
        self.apply_large_paste(event.text)

    def _resize_for_text(self) -> None:
        line_count = max(1, self.text.count("\n") + 1)
        self.styles.height = min(_MAX_PROMPT_LINES, line_count)


class AgentChatScreen(Screen[None]):
    """Server-backed coding-agent chat for a Lab workspace."""

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", show=False),
        Binding("escape", "back", "Back", key_display="Esc", show=False),
        Binding("ctrl+w", "show_welcome", "Welcome", key_display="Ctrl+W", show=False),
        Binding("tab", "focus_next", "Next", key_display="Tab", show=False),
        Binding("shift+tab", "focus_previous", "Previous", key_display="Shift+Tab", show=False),
    ]

    CSS = (
        BUTTON_CSS
        + """
    AgentChatScreen {
        background: $background;
        color: $foreground;
        layout: vertical;
    }

    #agent-header {
        height: 1;
        padding: 0 1;
        border-bottom: solid $primary;
        background: $background;
    }

    #agent-body {
        height: 1fr;
        padding: 0;
        align-horizontal: center;
    }

    #agent-stage {
        width: 88%;
        min-width: 72;
        max-width: 176;
        height: 1fr;
        background: $background;
        padding: 1 0;
    }

    #agent-atmosphere {
        height: 7;
        color: $text-muted;
        margin-bottom: 0;
    }

    #agent-chat {
        height: 1fr;
        background: $background;
        padding: 0 1;
        scrollbar-size-vertical: 1;
    }

    #agent-chat-body {
        height: auto;
    }

    .agent-turn {
        height: auto;
        margin-bottom: 0;
    }

    AgentWidgetCard {
        height: auto;
        background: $surface;
        border-left: solid $success;
        padding: 1 1;
        margin: 0 0 1 0;
    }

    .agent-widget-heading {
        height: auto;
        margin-bottom: 1;
    }

    .agent-widget-fields {
        height: auto;
        layout: grid;
        grid-size: 2;
        grid-columns: 1fr 1fr;
        grid-gutter: 1;
    }

    .agent-widget-field-row {
        height: 4;
    }

    .agent-widget-field-label {
        height: 1;
        color: $text-muted;
    }

    .agent-widget-field {
        height: 3;
        background: $panel;
    }

    .agent-widget-select {
        height: 3;
        background: $panel;
    }

    .agent-widget-field.read-only {
        color: $text-muted;
    }

    .agent-widget-actions {
        height: auto;
        margin-top: 1;
    }

    .agent-widget-actions Button {
        margin-right: 1;
        min-width: 16;
    }

    .agent-widget-status {
        height: auto;
        margin-top: 1;
    }

    .agent-widget-log {
        display: none;
        height: auto;
        max-height: 12;
        margin-top: 1;
        padding: 1;
        background: $panel;
        color: $foreground;
        overflow-y: auto;
    }

    .agent-widget-log.visible {
        display: block;
    }

    #agent-composer {
        height: auto;
        background: $background;
        padding: 0 1;
        margin-top: 1;
    }

    #agent-input-shell {
        height: auto;
        background: $panel;
        border-left: solid $primary;
        padding: 1 1;
    }

    #agent-command-menu {
        display: none;
        height: auto;
        background: $panel;
        color: $text-muted;
        margin-bottom: 1;
    }

    #agent-command-menu.visible {
        display: block;
    }

    #agent-prompt {
        height: 1;
        width: 1fr;
        background: $panel;
        border: none;
        scrollbar-size-vertical: 0;
    }

    #agent-prompt:focus {
        border: none;
    }

    #agent-prompt.large-paste {
        color: $primary;
        text-style: bold;
    }

    #agent-statusbar {
        height: 1;
        padding: 0 1;
        background: $background;
    }
    """
    )

    def __init__(
        self,
        item: LabItem,
        *,
        state_provider: AgentStateProvider,
        messages_provider: AgentMessagesProvider,
        select_agent: AgentSelector,
        send_prompt: AgentSender,
        start_new_session: AgentSessionStarter,
        record_action: AgentActionRecorder,
        status_text_provider: StatusTextProvider,
    ) -> None:
        super().__init__()
        self._item = item
        self._workspace = Path(str(item.raw.get("workspace") or ".")).expanduser().resolve()
        self._agent = str(item.raw.get("agent") or "codex")
        self._state_provider = state_provider
        self._messages_provider = messages_provider
        self._select_agent = select_agent
        self._send_prompt = send_prompt
        self._start_new_session = start_new_session
        self._record_action = record_action
        self._status_text_provider = status_text_provider
        self._template_prompts_by_id = _prompt_templates(item)
        self._references_by_value = _agent_references(item)
        self._frame = 0
        self._last_atmosphere_key: tuple[object, ...] | None = None
        self._last_chat_key: tuple[object, ...] | None = None
        self._last_chat_shape: tuple[tuple[object, ...], ...] = ()
        self._last_status_key = ""
        self._command_menu_rows: tuple[CommandMenuRow, ...] = ()
        self._command_menu_index = 0
        self._global_prompt_history = load_agent_prompt_history(limit=50)
        self._prompt_history: tuple[str, ...] = ()
        self._history_index = 0

    def compose(self) -> ComposeResult:
        yield Static(_agent_header(), id="agent-header", markup=False)
        with Vertical(id="agent-body"):
            with Vertical(id="agent-stage"):
                yield Static(
                    _agent_atmosphere(
                        self._item,
                        self._workspace,
                        self._agent,
                        self._state_provider(),
                        frame=self._frame,
                        width=108,
                    ),
                    id="agent-atmosphere",
                    markup=False,
                )
                yield VerticalScroll(
                    Vertical(id="agent-chat-body"),
                    id="agent-chat",
                )
                with Vertical(id="agent-composer"):
                    with Vertical(id="agent-input-shell"):
                        yield Static(
                            _agent_command_menu(
                                self._template_prompts_by_id,
                                self._references_by_value,
                            ),
                            id="agent-command-menu",
                            markup=False,
                        )
                        yield AgentPrompt(
                            "",
                            placeholder=_agent_prompt_placeholder(
                                self._state_provider(),
                                self._agent,
                            ),
                            id="agent-prompt",
                            show_line_numbers=False,
                            highlight_cursor_line=False,
                            compact=True,
                        )
        yield Static(
            _agent_statusbar(self._status_text_provider),
            id="agent-statusbar",
            markup=False,
        )

    def on_mount(self) -> None:
        self.set_interval(0.25, self._refresh_runtime_view)
        self._refresh_runtime_view()
        self.call_after_refresh(lambda: self.query_one("#agent-prompt", AgentPrompt).focus())

    def action_back(self) -> None:
        self.app.pop_screen()

    def action_quit(self) -> None:
        self.app.action_quit()

    def action_show_welcome(self) -> None:
        show_welcome = getattr(self.app, "action_show_welcome", None)
        if callable(show_welcome):
            show_welcome()

    @on(TextArea.Changed, "#agent-prompt")
    def _prompt_changed(self, event: TextArea.Changed) -> None:
        prompt = event.text_area
        if isinstance(prompt, AgentPrompt):
            prompt.sync_after_change()
        self._update_command_menu(prompt.text)

    @on(AgentPrompt.Submitted)
    def _prompt_submitted(self, event: AgentPrompt.Submitted) -> None:
        value = event.prompt.submitted_text
        prefix = _active_menu_prefix(value)
        if prefix == "/":
            self._handle_agent_command(self._submitted_command_value(value))
            return
        if prefix == "?":
            self._insert_selected_help_prompt()
            return
        if prefix == "@":
            self._insert_selected_reference()
            return
        self._send_current_prompt()

    @on(AgentPrompt.CommandPrevious)
    def _command_previous(self, _event: AgentPrompt.CommandPrevious) -> None:
        if not self._command_menu_rows:
            return
        self._command_menu_index = (self._command_menu_index - 1) % len(self._command_menu_rows)
        self._render_command_menu()

    @on(AgentPrompt.CommandNext)
    def _command_next(self, _event: AgentPrompt.CommandNext) -> None:
        if not self._command_menu_rows:
            return
        self._command_menu_index = (self._command_menu_index + 1) % len(self._command_menu_rows)
        self._render_command_menu()

    @on(AgentPrompt.HistoryPrevious)
    def _history_previous(self, _event: AgentPrompt.HistoryPrevious) -> None:
        if not self._prompt_history:
            return
        self._history_index = max(0, min(self._history_index, len(self._prompt_history)) - 1)
        self.query_one("#agent-prompt", AgentPrompt).load_history_text(
            self._prompt_history[self._history_index]
        )

    @on(AgentPrompt.HistoryNext)
    def _history_next(self, _event: AgentPrompt.HistoryNext) -> None:
        if not self._prompt_history:
            return
        self._history_index = min(len(self._prompt_history), self._history_index + 1)
        prompt = self.query_one("#agent-prompt", AgentPrompt)
        if self._history_index >= len(self._prompt_history):
            prompt.clear_prompt()
        else:
            prompt.load_history_text(self._prompt_history[self._history_index])

    def _send_current_prompt(self) -> None:
        state = self._state_provider()
        if not _chat_transport_ready(state):
            return
        prompt_widget = self.query_one("#agent-prompt", AgentPrompt)
        prompt = prompt_widget.submitted_text
        if not prompt.strip():
            return
        append_agent_prompt_history(self._workspace, self._agent, prompt)
        self._global_prompt_history = load_agent_prompt_history(limit=50)
        prompt_widget.clear_prompt()
        self._send_prompt(prompt)
        self._refresh_runtime_view()

    def _handle_agent_command(self, value: str) -> None:
        command, _, arg = value.strip().partition(" ")
        command = command.lower()
        arg = arg.strip()
        if command == "?":
            self._update_command_menu("?")
            return
        if command == "@":
            self._update_command_menu("@")
            return
        if command in {"/agent", "/agents"}:
            if not arg:
                self._update_command_menu("/agent")
                return
            adapter = agent_adapter(arg)
            self._agent = adapter.name
            self._select_agent(self._workspace, self._agent)
            self.query_one("#agent-prompt", AgentPrompt).clear_prompt()
            self._update_command_menu("")
            self._refresh_runtime_view()
            return
        if command in {"/template", "/templates"}:
            if arg.isdigit():
                template = self._template_prompts_by_id.get(f"agent-template-{int(arg) - 1}")
                if template is not None:
                    self.query_one("#agent-prompt", AgentPrompt).load_text(template["prompt"])
                    self._update_command_menu("")
                    return
            self._update_command_menu("/template")
            return
        if command == "/clear":
            self._start_new_session(self._workspace, self._agent)
            self._global_prompt_history = load_agent_prompt_history(limit=50)
            self.query_one("#agent-prompt", AgentPrompt).clear_prompt()
            self._update_command_menu("")
            self._refresh_runtime_view()
            return
        if command in {"/diagnose", "/diagnostic"}:
            state = self._state_provider()
            self.query_one("#agent-prompt", AgentPrompt).clear_prompt()
            self._update_command_menu("")
            if not _chat_transport_ready(state):
                self._refresh_runtime_view()
                return
            self._record_action(
                {
                    "type": "agent_tool_diagnostic_started",
                    "agent": self._agent,
                    "workspace": str(self._workspace),
                }
            )
            self._send_prompt(lab_widget_diagnostic_prompt())
            self._refresh_runtime_view()
            return
        if command in {"/help", "/"}:
            self._update_command_menu("/")
            return
        self._update_command_menu(value)

    def _submitted_command_value(self, value: str) -> str:
        stripped = value.strip()
        command = stripped.split(maxsplit=1)[0].lower() if stripped else ""
        if command in {"/diagnose", "/diagnostic"}:
            return stripped
        return self._selected_command_value() or value

    def _update_command_menu(self, value: str) -> None:
        menu = self.query_one("#agent-command-menu", Static)
        stripped = value.lstrip()
        visible = _active_menu_prefix(stripped) is not None
        menu.set_class(visible, "visible")
        if not visible:
            self._command_menu_rows = ()
            self._command_menu_index = 0
            return
        previous = self._selected_command_value()
        rows = _agent_command_rows(
            self._template_prompts_by_id,
            self._references_by_value,
            query=stripped,
        )
        self._command_menu_rows = rows
        if previous:
            commands = [value for value, _label, _detail in rows]
            self._command_menu_index = commands.index(previous) if previous in commands else 0
        else:
            self._command_menu_index = 0
        self._render_command_menu()

    def _render_command_menu(self) -> None:
        self.query_one("#agent-command-menu", Static).update(
            _agent_command_menu(
                self._template_prompts_by_id,
                self._references_by_value,
                rows=self._command_menu_rows,
                selected_index=self._command_menu_index,
            )
        )

    def _selected_command_value(self) -> str:
        if not self._command_menu_rows:
            return ""
        index = self._command_menu_index % len(self._command_menu_rows)
        return self._command_menu_rows[index][0]

    def _insert_selected_help_prompt(self) -> None:
        value = self._selected_command_value()
        template_id = value.removeprefix("?")
        template = self._template_prompts_by_id.get(template_id)
        prompt = self.query_one("#agent-prompt", AgentPrompt)
        if template is None:
            self._update_command_menu("?")
            return
        prompt.load_text(template["prompt"])
        self._update_command_menu("")

    def _insert_selected_reference(self) -> None:
        value = self._selected_command_value()
        reference = self._references_by_value.get(value)
        prompt = self.query_one("#agent-prompt", AgentPrompt)
        if reference is None:
            self._update_command_menu("@")
            return
        prompt.load_text(f"{reference['insert']} ")
        self._update_command_menu("")

    def _refresh_runtime_view(self) -> None:
        self._frame += 1
        state = self._state_provider()
        messages = self._messages_provider()
        self._sync_prompt_history(messages)
        atmosphere = self.query_one("#agent-atmosphere", Static)
        atmosphere.display = not bool(messages)
        atmosphere_key = (self._workspace, self._agent, state, self._frame // 2, bool(messages))
        if atmosphere.display and atmosphere_key != self._last_atmosphere_key:
            self._last_atmosphere_key = atmosphere_key
            atmosphere.update(
                _agent_atmosphere(
                    self._item,
                    self._workspace,
                    self._agent,
                    state,
                    frame=self._frame,
                    width=self._atmosphere_width(),
                )
            )
        chat_key = (messages, state.status, state.label, state.agent, state.message)
        if chat_key != self._last_chat_key:
            self._last_chat_key = chat_key
            self._render_chat(messages, state)
        ready = _chat_transport_ready(state)
        prompt = self.query_one("#agent-prompt", AgentPrompt)
        prompt.disabled = not ready
        prompt.placeholder = _agent_prompt_placeholder(state, self._agent)
        status_text = self._status_text_provider()
        status_key = status_text.plain
        if status_key != self._last_status_key:
            self._last_status_key = status_key
            self.query_one("#agent-statusbar", Static).update(_agent_statusbar(lambda: status_text))

    def _render_chat(
        self,
        messages: tuple[AgentChatMessage, ...],
        state: AgentConnectionState,
    ) -> None:
        body = self.query_one("#agent-chat-body", Vertical)
        if not messages:
            body.remove_children()
            self._last_chat_shape = ()
            body.mount(
                Static(_chat_transcript(messages, state), classes="agent-turn", markup=False)
            )
            return
        shape = _chat_shape(messages)
        if self._update_existing_chat_body(body, messages, shape):
            self._last_chat_shape = shape
            return
        body.remove_children()
        body.mount(*self._chat_widgets(messages, start_index=0))
        self._last_chat_shape = shape

    def _update_existing_chat_body(
        self,
        body: Vertical,
        messages: tuple[AgentChatMessage, ...],
        shape: tuple[tuple[object, ...], ...],
    ) -> bool:
        previous = self._last_chat_shape
        if not previous or previous != shape[: len(previous)]:
            return False
        children = list(body.children)
        expected_children = _chat_child_count(len(previous))
        if len(children) < expected_children:
            return False
        cursor = 0
        for index, message in enumerate(messages[: len(previous)]):
            if index:
                separator = children[cursor]
                if not isinstance(separator, Static):
                    return False
                separator.update(Text("\n· · ·\n", style="dim"))
                cursor += 1
            child = children[cursor]
            if message.status == "widget":
                if not isinstance(child, AgentWidgetCard):
                    return False
            else:
                if not isinstance(child, Static):
                    return False
                child.update(render_chat_turn(message))
            cursor += 1
        if len(messages) > len(previous):
            body.mount(*self._chat_widgets(messages[len(previous) :], start_index=len(previous)))
        return True

    def _chat_widgets(
        self,
        messages: tuple[AgentChatMessage, ...],
        *,
        start_index: int,
    ) -> list[Static | AgentWidgetCard]:
        widgets: list[Static | AgentWidgetCard] = []
        for offset, message in enumerate(messages):
            index = start_index + offset
            if index:
                widgets.append(Static(Text("\n· · ·\n", style="dim"), classes="agent-turn"))
            if message.status == "widget":
                widgets.append(
                    AgentWidgetCard(
                        message,
                        workspace=self._workspace,
                        record_action=self._record_action,
                    )
                )
            else:
                widgets.append(
                    Static(render_chat_turn(message), classes="agent-turn", markup=False)
                )
        return widgets

    def _atmosphere_width(self) -> int:
        width = self.query_one("#agent-stage").size.width or self.size.width
        return max(72, min(176, width or 108))

    def _sync_prompt_history(self, messages: tuple[AgentChatMessage, ...]) -> None:
        session_prompts = tuple(
            message.content
            for message in messages
            if message.role == "user" and message.content.strip()
        )
        prompts = _dedupe_prompt_history((*self._global_prompt_history, *session_prompts))[-50:]
        if prompts == self._prompt_history:
            return
        self._prompt_history = prompts
        self._history_index = len(prompts)


def _agent_header() -> Group:
    return Group(lab_header("Agent"))


def _agent_atmosphere(
    _item: LabItem,
    _workspace: Path,
    _agent: str,
    _state: AgentConnectionState,
    *,
    frame: int,
    width: int,
) -> Group:
    backdrop_width = max(72, min(176, width))
    return Group(LaunchBackdrop(frame=frame).render_text(backdrop_width, 6))


def _chat_transcript(
    messages: tuple[AgentChatMessage, ...],
    state: AgentConnectionState,
) -> RenderableType:
    return _render_chat_transcript(messages, state)


def _chat_shape(messages: tuple[AgentChatMessage, ...]) -> tuple[tuple[object, ...], ...]:
    shape: list[tuple[object, ...]] = []
    for message in messages:
        if message.status == "widget":
            metadata = message.metadata if isinstance(message.metadata, dict) else {}
            raw_payload = metadata.get("payload")
            payload = raw_payload if isinstance(raw_payload, dict) else {}
            shape.append(
                (
                    "widget",
                    metadata.get("widget_id") or metadata.get("id") or "",
                    metadata.get("kind") or payload.get("kind") or "",
                    metadata.get("title") or payload.get("title") or "",
                )
            )
        else:
            shape.append(("turn", message.role))
    return tuple(shape)


def _chat_child_count(message_count: int) -> int:
    if message_count <= 0:
        return 0
    return message_count * 2 - 1


def _chat_transport_ready(state: AgentConnectionState) -> bool:
    return state.status == "connected"


def _agent_prompt_placeholder(state: AgentConnectionState, fallback_agent: str) -> str:
    label = state.label or agent_adapter(fallback_agent).label
    if state.status == "unsupported":
        return f"{label} not yet supported in Lab"
    return f"Ask {label}  •  /  ?  @"


def _agent_statusbar(status_text_provider: StatusTextProvider) -> Table:
    bar = Table.grid(expand=True)
    bar.add_column(ratio=1)
    bar.add_column(justify="right", no_wrap=True)
    bar.add_row(status_text_provider(), action_hint_text(("Esc", "Back"), ("Ctrl+W", "Welcome")))
    return bar


def _agent_command_menu(
    templates: dict[str, dict[str, str]],
    references: dict[str, dict[str, str]],
    *,
    rows: tuple[CommandMenuRow, ...] | None = None,
    query: str = "/",
    selected_index: int = 0,
) -> Text:
    text = Text()
    rows = rows if rows is not None else _agent_command_rows(templates, references, query=query)
    selected_index = selected_index % len(rows) if rows else 0
    for index, (_value, label, detail) in enumerate(rows):
        if index:
            text.append("\n")
        style = "bold white on #2a2538" if index == selected_index else "white"
        text.append(label.ljust(24), style=style)
        text.append(detail, style="dim")
    return text


def _agent_command_rows(
    templates: dict[str, dict[str, str]],
    references: dict[str, dict[str, str]] | None = None,
    *,
    query: str = "/",
) -> tuple[CommandMenuRow, ...]:
    references = references or {}
    prefix = _active_menu_prefix(query) or "/"
    if prefix == "?":
        rows = [
            (f"?{key}", template["label"], "Insert starter prompt")
            for key, template in templates.items()
        ]
        return _filter_menu_rows(tuple(rows), query)
    if prefix == "@":
        rows = [
            (value, value, f"{data['kind']} · {data['detail']}")
            for value, data in references.items()
        ]
        if not rows:
            rows = [("@", "@", "No Lab references loaded")]
        return _filter_menu_rows(tuple(rows), query)

    from .agent_capabilities import agent_capability, known_agent_names

    rows = [
        (f"/agent {name}", f"/agent {name}", f"Switch to {agent_capability(name).label}")
        for name in known_agent_names()
    ]
    rows.extend(
        [
            ("/clear", "/clear", "Start a fresh session"),
            ("/help", "/help", "Show chat commands"),
        ]
    )
    if templates:
        rows.append(("?", "?", "Prompt starters"))
    if references:
        rows.append(("@", "@", "Reference Lab objects"))
    query = query.strip().lower()
    if query and query not in {"/", "/help"}:
        rows = [
            row
            for row in rows
            if row[0].startswith(query) or query in row[0] or query in row[1].lower()
        ]
        if not rows:
            prefix = query.split(maxsplit=1)[0]
            rows = [
                row
                for row in _agent_command_rows(templates, references)
                if row[0].startswith(prefix)
            ]
        if not rows:
            rows = [("/help", "/help", "Show chat commands")]

    return tuple(rows)


def _filter_menu_rows(
    rows: tuple[CommandMenuRow, ...],
    query: str,
) -> tuple[CommandMenuRow, ...]:
    needle = query.strip().lower()[1:].strip()
    if not needle:
        return rows
    filtered = [
        row
        for row in rows
        if needle in row[0].lower() or needle in row[1].lower() or needle in row[2].lower()
    ]
    return tuple(filtered or rows[:1])


def _is_large_paste(text: str) -> bool:
    return text.count("\n") + 1 >= _LARGE_PASTE_LINE_THRESHOLD or len(text) >= (
        _LARGE_PASTE_CHAR_THRESHOLD
    )


def _large_paste_placeholder(text: str) -> str:
    line_count = text.count("\n") + 1
    return f"[{line_count} lines pasted]"


def _dedupe_prompt_history(values: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in reversed(values):
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    unique.reverse()
    return tuple(unique)


def _prompt_templates(item: LabItem) -> dict[str, dict[str, str]]:
    templates = item.raw.get("prompt_templates")
    if not isinstance(templates, (tuple, list)):
        templates = _default_prompt_templates()
    result: dict[str, dict[str, str]] = {}
    for index, template in enumerate(templates):
        if not isinstance(template, dict):
            continue
        label = str(template.get("label") or "").strip()
        prompt = str(template.get("prompt") or "").strip()
        if not label or not prompt:
            continue
        result[f"agent-template-{index}"] = {"label": label, "prompt": prompt}
    return result


def _default_prompt_templates() -> tuple[dict[str, str], ...]:
    return (
        {
            "label": "Build an environment",
            "prompt": "\n".join(
                [
                    "Help me build a new verifiers environment in this Lab workspace.",
                    "",
                    "Task idea:",
                    "- ",
                    "",
                    "Inspect the existing environments first. Ask me to choose if there are",
                    "multiple plausible implementation paths.",
                ]
            ),
        },
        {
            "label": "Create an eval config",
            "prompt": (
                "Help me create a Lab evaluation config. Inspect the local environments and "
                "existing eval configs first, then draft the smallest runnable config."
            ),
        },
        {
            "label": "Modify a training config",
            "prompt": (
                "Help me modify a training config in this workspace. If multiple configs match, "
                "show me the options before editing."
            ),
        },
        {
            "label": "Debug workspace",
            "prompt": (
                "Help me debug the current Lab workspace. Check setup, configs, generated "
                "outputs, and recent errors before proposing a fix."
            ),
        },
    )


def _agent_references(item: LabItem) -> dict[str, dict[str, str]]:
    references = item.raw.get("references")
    if not isinstance(references, (tuple, list)):
        return {}
    result: dict[str, dict[str, str]] = {}
    for reference in references:
        if not isinstance(reference, dict):
            continue
        insert = str(reference.get("insert") or "").strip()
        if not insert.startswith("@"):
            continue
        result[insert] = {
            "insert": insert,
            "kind": str(reference.get("kind") or "reference"),
            "label": str(reference.get("label") or insert),
            "detail": str(reference.get("detail") or ""),
        }
    return result


def _active_menu_prefix(value: str) -> str | None:
    stripped = value.lstrip()
    if not stripped:
        return None
    prefix = stripped[0]
    return prefix if prefix in {"/", "?", "@"} else None
