"""Coding-agent chat screen for Lab workspaces."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from rich.console import Group
from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, Footer, Input, Select, Static

from .agent_adapters import agent_adapter, agent_select_options
from .agent_runtime import AgentChatMessage, AgentConnectionState
from .launch_backdrop import LaunchBackdrop
from .models import LabItem
from .palette import BUTTON_CSS, MUTED, PRIMARY, STATUS_ERROR, STATUS_SUCCESS, STATUS_WARNING
from .shell import compact_path, lab_header

AgentStateProvider = Callable[[], AgentConnectionState]
AgentMessagesProvider = Callable[[], tuple[AgentChatMessage, ...]]
AgentSelector = Callable[[Path, str], None]
AgentSender = Callable[[str], None]
StatusTextProvider = Callable[[], Text]


class AgentChatScreen(Screen[None]):
    """Server-backed coding-agent chat for a Lab workspace."""

    BINDINGS = [
        Binding("b,backspace", "back", "Back"),
        Binding("q", "quit", "Quit"),
        Binding("tab", "focus_next", "Next", key_display="Tab"),
        Binding("shift+tab", "focus_previous", "Previous", key_display="Shift+Tab"),
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
        height: 3;
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
        width: 82%;
        min-width: 72;
        max-width: 150;
        height: 1fr;
        background: $background;
        padding: 1 0;
    }

    #agent-atmosphere {
        height: 8;
        color: $text-muted;
        margin-bottom: 1;
    }

    #agent-chat {
        height: 1fr;
        background: $background;
        padding: 0 1;
        scrollbar-size-vertical: 1;
    }

    #agent-composer {
        height: auto;
        background: $surface;
        padding: 1;
        margin-top: 1;
    }

    #agent-template-row {
        height: 3;
        margin-bottom: 1;
    }

    #agent-control-row {
        height: 3;
    }

    #agent-prompt {
        height: 3;
        width: 1fr;
    }

    #agent-select {
        height: 3;
        width: 28;
        margin-right: 1;
    }

    .agent-action-button {
        width: 1fr;
    }

    .agent-template-button {
        width: 1fr;
        height: 3;
        margin-right: 1;
    }

    #agent-send {
        width: 12;
        margin-left: 1;
    }

    #agent-statusbar {
        height: 1;
        padding: 0 1;
        background: $surface;
        border-top: solid $primary;
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
        self._status_text_provider = status_text_provider
        self._template_prompts_by_id = _prompt_templates(item)
        self._frame = 0
        self._last_atmosphere_key: tuple[object, ...] | None = None
        self._last_chat_key: tuple[object, ...] | None = None
        self._last_status_key = ""

    def compose(self) -> ComposeResult:
        yield Static(_agent_header(self._item), id="agent-header", markup=False)
        with Vertical(id="agent-body"):
            with Vertical(id="agent-stage"):
                yield Static(
                    _agent_atmosphere(
                        self._item,
                        self._workspace,
                        self._agent,
                        self._state_provider(),
                        frame=self._frame,
                    ),
                    id="agent-atmosphere",
                    markup=False,
                )
                yield VerticalScroll(
                    Static(
                        _chat_transcript(self._messages_provider(), self._state_provider()),
                        id="agent-chat-body",
                        markup=False,
                    ),
                    id="agent-chat",
                )
                with Vertical(id="agent-composer"):
                    if self._template_prompts_by_id:
                        with Horizontal(id="agent-template-row"):
                            for template_id, template in self._template_prompts_by_id.items():
                                yield Button(
                                    template["label"],
                                    id=template_id,
                                    classes="agent-template-button -style-default",
                                )
                    with Horizontal(id="agent-control-row"):
                        yield Select(
                            agent_select_options(self._agent),
                            value=self._agent,
                            allow_blank=False,
                            id="agent-select",
                        )
                        yield Input(placeholder="Ask the coding agent...", id="agent-prompt")
                        yield Button(
                            "Send",
                            id="agent-send",
                            classes="agent-action-button",
                            variant="primary",
                        )
        yield Static(self._status_text_provider(), id="agent-statusbar", markup=False)
        yield Footer()

    def on_mount(self) -> None:
        self.set_interval(0.25, self._refresh_runtime_view)
        self._refresh_runtime_view()

    def action_back(self) -> None:
        self.app.pop_screen()

    @on(Button.Pressed, "#agent-send")
    def _send_pressed(self, _event: Button.Pressed) -> None:
        self._send_current_prompt()

    @on(Button.Pressed, ".agent-template-button")
    def _template_pressed(self, event: Button.Pressed) -> None:
        template = self._template_prompts_by_id.get(event.button.id or "")
        if template is None:
            return
        self.query_one("#agent-prompt", Input).value = template["prompt"]
        self.query_one("#agent-prompt", Input).focus()

    @on(Input.Submitted, "#agent-prompt")
    def _prompt_submitted(self, _event: Input.Submitted) -> None:
        self._send_current_prompt()

    @on(Select.Changed, "#agent-select")
    def _agent_selected(self, event: Select.Changed) -> None:
        if not isinstance(event.value, str):
            return
        self._agent = event.value
        self._select_agent(self._workspace, self._agent)
        self._refresh_runtime_view()

    def _send_current_prompt(self) -> None:
        state = self._state_provider()
        if not _chat_transport_ready(state):
            return
        prompt = self.query_one("#agent-prompt", Input).value
        if not prompt.strip():
            return
        self.query_one("#agent-prompt", Input).value = ""
        self._send_prompt(prompt)
        self._refresh_runtime_view()

    def _refresh_runtime_view(self) -> None:
        self._frame += 1
        state = self._state_provider()
        messages = self._messages_provider()
        atmosphere_key = (self._workspace, self._agent, state, self._frame // 2)
        if atmosphere_key != self._last_atmosphere_key:
            self._last_atmosphere_key = atmosphere_key
            self.query_one("#agent-atmosphere", Static).update(
                _agent_atmosphere(
                    self._item,
                    self._workspace,
                    self._agent,
                    state,
                    frame=self._frame,
                )
            )
        chat_key = (messages, state.status, state.label, state.agent, state.message)
        if chat_key != self._last_chat_key:
            self._last_chat_key = chat_key
            self.query_one("#agent-chat-body", Static).update(_chat_transcript(messages, state))
        ready = _chat_transport_ready(state)
        self.query_one("#agent-prompt", Input).disabled = not ready
        self.query_one("#agent-send", Button).disabled = not ready
        status_text = self._status_text_provider()
        status_key = status_text.plain
        if status_key != self._last_status_key:
            self._last_status_key = status_key
            self.query_one("#agent-statusbar", Static).update(status_text)


def _agent_header(item: LabItem) -> Group:
    title = Text()
    title.append(item.title, style="bold")
    title.append(f"\n{item.subtitle}", style="dim")
    return Group(lab_header(title))


def _agent_atmosphere(
    item: LabItem,
    workspace: Path,
    agent: str,
    state: AgentConnectionState,
    *,
    frame: int,
) -> Group:
    adapter = agent_adapter(agent)
    backdrop = LaunchBackdrop(frame=frame).render_text(108, 5)
    heading = Text()
    heading.append(_agent_prompt_heading(item, workspace), style="bold white")
    heading.append("\n")
    heading.append(adapter.label, style=PRIMARY)
    heading.append("  ")
    heading.append(_connection_label(state), style=_connection_style(state.status))
    heading.append("  ·  ", style="dim")
    heading.append(compact_path(workspace), style=MUTED)
    if state.transport:
        heading.append("  ·  ", style="dim")
        heading.append(state.transport, style="dim")
    return Group(backdrop, heading)


def _agent_prompt_heading(item: LabItem, workspace: Path) -> str:
    if item.raw.get("prompt_templates"):
        return f"What should we build in {workspace.name or 'this workspace'}?"
    return f"Ask {item.title or 'the coding agent'}"


def _chat_transcript(
    messages: tuple[AgentChatMessage, ...],
    state: AgentConnectionState,
) -> Text:
    text = Text()
    if not messages:
        text.append("Chat session\n", style="bold")
        if state.status == "none":
            text.append("No coding agent configured for this workspace.", style="dim")
        elif state.status == "connected":
            text.append("Connected. Send a prompt to start the session.", style="dim")
        else:
            text.append(_connection_label(state), style=_connection_style(state.status))
        return text

    for idx, message in enumerate(messages):
        if idx:
            text.append("\n\n")
        role_style = {
            "user": "bold",
            "assistant": STATUS_SUCCESS,
            "system": STATUS_WARNING if message.status != "error" else STATUS_ERROR,
        }.get(message.role, "bold")
        text.append(message.role.capitalize(), style=role_style)
        if message.status:
            text.append(f"  {message.status}", style="dim")
        text.append("\n")
        text.append(message.content)
    return text


def _connection_label(state: AgentConnectionState) -> str:
    if state.status == "none":
        return "no agent"
    if state.label:
        return f"{state.label} {state.status}"
    if state.agent:
        return f"{state.agent} {state.status}"
    return state.status


def _chat_transport_ready(state: AgentConnectionState) -> bool:
    return state.status == "connected"


def _prompt_templates(item: LabItem) -> dict[str, dict[str, str]]:
    templates = item.raw.get("prompt_templates")
    if not isinstance(templates, (tuple, list)):
        return {}
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


def _connection_style(status: str) -> str:
    if status == "connected":
        return STATUS_SUCCESS
    if status in {"starting", "stopped"}:
        return STATUS_WARNING
    if status == "error":
        return STATUS_ERROR
    return "dim"
