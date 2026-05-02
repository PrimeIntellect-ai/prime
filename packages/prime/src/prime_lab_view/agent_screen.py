"""Coding-agent chat screen for Lab workspaces."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from rich.console import Group, RenderableType
from rich.markdown import Markdown
from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import Footer, Input, Static

from .agent_adapters import agent_adapter
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
        width: 86%;
        min-width: 72;
        max-width: 176;
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
        background: $background;
        padding: 0 1 1 1;
        margin-top: 1;
    }

    #agent-input-shell {
        height: auto;
        background: $panel;
        border-left: solid $primary;
        padding: 1 1 0 1;
    }

    #agent-command-menu {
        display: none;
        height: auto;
        color: $text-muted;
        margin-bottom: 1;
    }

    #agent-command-menu.visible {
        display: block;
    }

    #agent-prompt {
        height: 3;
        width: 1fr;
        background: $panel;
        border: none;
    }

    #agent-prompt:focus {
        border: none;
    }

    #agent-composer-meta {
        height: 1;
        content-align: right middle;
        color: $text-muted;
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
        yield Static(_agent_header(self._workspace, self._agent), id="agent-header", markup=False)
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
                    Static(
                        _chat_transcript(self._messages_provider(), self._state_provider()),
                        id="agent-chat-body",
                        markup=False,
                    ),
                    id="agent-chat",
                )
                with Vertical(id="agent-composer"):
                    with Vertical(id="agent-input-shell"):
                        yield Static(
                            _agent_command_menu(self._template_prompts_by_id),
                            id="agent-command-menu",
                            markup=False,
                        )
                        yield Input(
                            placeholder="Ask Agent...  / commands",
                            id="agent-prompt",
                        )
                        yield Static(_agent_composer_meta(self._agent), id="agent-composer-meta")
        yield Static(self._status_text_provider(), id="agent-statusbar", markup=False)
        yield Footer()

    def on_mount(self) -> None:
        self.set_interval(0.25, self._refresh_runtime_view)
        self._refresh_runtime_view()
        self.call_after_refresh(lambda: self.query_one("#agent-prompt", Input).focus())

    def action_back(self) -> None:
        self.app.pop_screen()

    @on(Input.Changed, "#agent-prompt")
    def _prompt_changed(self, event: Input.Changed) -> None:
        self._update_command_menu(event.value)

    @on(Input.Submitted, "#agent-prompt")
    def _prompt_submitted(self, _event: Input.Submitted) -> None:
        value = self.query_one("#agent-prompt", Input).value
        if value.lstrip().startswith("/"):
            self._handle_agent_command(value)
            return
        self._send_current_prompt()

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

    def _handle_agent_command(self, value: str) -> None:
        command, _, arg = value.strip().partition(" ")
        command = command.lower()
        arg = arg.strip()
        if command in {"/agent", "/agents"}:
            if not arg:
                self._update_command_menu("/agent")
                return
            adapter = agent_adapter(arg)
            self._agent = adapter.name
            self._select_agent(self._workspace, self._agent)
            self.query_one("#agent-header", Static).update(
                _agent_header(self._workspace, self._agent)
            )
            self.query_one("#agent-composer-meta", Static).update(_agent_composer_meta(self._agent))
            self.query_one("#agent-prompt", Input).value = ""
            self._update_command_menu("")
            self._refresh_runtime_view()
            return
        if command in {"/template", "/templates"}:
            if arg.isdigit():
                template = self._template_prompts_by_id.get(f"agent-template-{int(arg) - 1}")
                if template is not None:
                    self.query_one("#agent-prompt", Input).value = template["prompt"]
                    self._update_command_menu("")
                    return
            self._update_command_menu("/template")
            return
        if command in {"/help", "/"}:
            self._update_command_menu("/")
            return
        self._update_command_menu(value)

    def _update_command_menu(self, value: str) -> None:
        menu = self.query_one("#agent-command-menu", Static)
        stripped = value.lstrip()
        visible = stripped.startswith("/")
        menu.set_class(visible, "visible")
        if visible:
            menu.update(_agent_command_menu(self._template_prompts_by_id, query=stripped))

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
                    width=self._atmosphere_width(),
                )
            )
        chat_key = (messages, state.status, state.label, state.agent, state.message)
        if chat_key != self._last_chat_key:
            self._last_chat_key = chat_key
            self.query_one("#agent-chat-body", Static).update(_chat_transcript(messages, state))
        ready = _chat_transport_ready(state)
        self.query_one("#agent-prompt", Input).disabled = not ready
        status_text = self._status_text_provider()
        status_key = status_text.plain
        if status_key != self._last_status_key:
            self._last_status_key = status_key
            self.query_one("#agent-statusbar", Static).update(status_text)

    def _atmosphere_width(self) -> int:
        width = self.query_one("#agent-stage").size.width or self.size.width
        return max(72, min(176, width or 108))


def _agent_header(workspace: Path, agent: str) -> Group:
    adapter = agent_adapter(agent)
    title = Text()
    title.append("Agent", style="bold")
    title.append("\n")
    title.append(adapter.name, style="dim")
    title.append(" · ", style="dim")
    title.append(compact_path(workspace), style="dim")
    return Group(lab_header(title))


def _agent_atmosphere(
    item: LabItem,
    workspace: Path,
    agent: str,
    state: AgentConnectionState,
    *,
    frame: int,
    width: int,
) -> Group:
    adapter = agent_adapter(agent)
    backdrop_width = max(72, min(176, width))
    backdrop = LaunchBackdrop(frame=frame).render_text(backdrop_width, 5)
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
    return "Agent"


def _chat_transcript(
    messages: tuple[AgentChatMessage, ...],
    state: AgentConnectionState,
) -> RenderableType:
    text = Text()
    if not messages:
        text.append("Chat session\n", style="bold")
        if state.status == "none":
            text.append("No coding agent configured for this workspace.", style="dim")
        elif state.status == "connected":
            text.append("Connected. Enter sends. Type / for commands.", style="dim")
        else:
            text.append(_connection_label(state), style=_connection_style(state.status))
        return text

    renderables: list[RenderableType] = []
    for idx, message in enumerate(messages):
        if idx:
            renderables.append(Text(""))
        role_style = {
            "user": "bold",
            "assistant": STATUS_SUCCESS,
            "system": STATUS_WARNING if message.status != "error" else STATUS_ERROR,
        }.get(message.role, "bold")
        header = Text()
        header.append(message.role.capitalize(), style=role_style)
        if message.status:
            header.append(f"  {message.status}", style="dim")
        renderables.append(header)
        renderables.append(_message_body(message))
    return Group(*renderables)


def _message_body(message: AgentChatMessage) -> RenderableType:
    if message.role == "assistant":
        return Markdown(
            message.content or " ",
            code_theme="nord-darker",
            hyperlinks=True,
        )
    return Text(message.content)


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


def _agent_composer_meta(agent: str) -> Text:
    adapter = agent_adapter(agent)
    return Text.assemble(
        (adapter.label, PRIMARY),
        ("  ·  Enter send", "dim"),
        ("  ·  / commands", "dim"),
    )


def _agent_command_menu(
    templates: dict[str, dict[str, str]],
    *,
    query: str = "/",
) -> Text:
    text = Text()
    rows = [
        ("/agent codex", "Switch to Codex"),
        ("/agent claude", "Switch to Claude Code one-shot"),
        ("/agent opencode", "Switch to OpenCode"),
        ("/agent pi", "Switch to Pi Coding Agent"),
        ("/agent hermes", "Switch to Hermes Agent"),
        ("/help", "Show chat commands"),
    ]
    if templates:
        rows.append(("/template N", "Load a prompt starter"))
    query = query.strip().lower()
    if query.startswith("/template") and templates:
        rows = [
            (
                f"/template {index + 1}",
                template["label"],
            )
            for index, template in enumerate(templates.values())
        ]
    elif query and query not in {"/", "/help"}:
        prefix = query.split(maxsplit=1)[0]
        rows = [row for row in rows if row[0].startswith(prefix) or prefix in row[0]]
        if not rows:
            rows = [("/help", "Show chat commands")]

    for index, (command, detail) in enumerate(rows):
        if index:
            text.append("\n")
        style = "bold white on #2a2538" if index == 0 else "white"
        text.append(command.ljust(18), style=style)
        text.append(detail, style="dim")
    return text


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
