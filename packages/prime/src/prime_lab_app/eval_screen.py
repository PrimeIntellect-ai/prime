"""Textual widgets for local eval rollout viewing."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal

from rich.markup import escape
from rich.text import Text
from textual import events, on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, VerticalScroll
from textual.css.query import NoMatches
from textual.dom import DOMNode
from textual.screen import ModalScreen, Screen
from textual.widget import Widget
from textual.widgets import (
    Collapsible,
    Footer,
    Input,
    Label,
    OptionList,
    Static,
    TabbedContent,
    TabPane,
    TextArea,
)
from textual.widgets._option_list import Option
from textual.widgets._tabbed_content import ContentTabs

from .detail_loader import DetailLoader
from .eval_markdown import MathMarkdown
from .eval_records import (
    HistorySectionData,
    LazyLogFile,
    LazyRunResults,
    LocalEvalRun,
    RolloutCopyItem,
    SearchHit,
    SearchResult,
    discover_log_files,
    log_tab_label,
    merge_log_files,
)
from .eval_render import (
    append_styled_log_line,
    build_metric_summary_table,
    build_reward_distribution_table,
    build_reward_text,
    build_rollout_prompt,
    build_run_metric_text,
    build_run_summary_text,
    build_score_text,
    build_state_text,
    build_task_text,
    build_usage_text,
    compute_run_overview_stats,
    format_message_preview,
    format_prompt_or_completion,
    format_reward_value,
    history_groups,
    indent_block,
    normalize_rollout_record,
    pretty_json_or_str,
    reward_style,
    stringify_message,
    stringify_message_content,
    stringify_message_reasoning,
    text_to_plain,
    tool_call_parts,
    tool_group_preview,
    tool_output_preview,
)
from .models import LabItem
from .palette import ROLLOUT_SUCCESS, ROLLOUT_WARNING, STATUS_ERROR, TOOL_CALL
from .widgets import ClearableInput


class EvalPanel(Container):
    """Reusable rounded panel container for eval viewer panes."""


class TabbedScrollPane(VerticalScroll):
    """Scroll pane that switches sibling detail tabs with left/right."""

    BINDINGS = [
        Binding("left", "prev_tab", "Prev tab", show=False),
        Binding("right", "next_tab", "Next tab", show=False),
    ]

    def _get_tabbed_content(self) -> TabbedContent | None:
        node = self.parent
        while node is not None:
            if isinstance(node, TabbedContent):
                return node
            node = node.parent
        return None

    def action_prev_tab(self) -> None:
        tabs = self._get_tabbed_content()
        if tabs is not None:
            tabs.query_one(ContentTabs).action_previous_tab()

    def action_next_tab(self) -> None:
        tabs = self._get_tabbed_content()
        if tabs is not None:
            tabs.query_one(ContentTabs).action_next_tab()


class LogScrollPane(VerticalScroll):
    """Log scroll pane that switches log file tabs with left/right."""

    BINDINGS = [
        Binding("left", "prev_log_tab", "Prev log", show=False),
        Binding("right", "next_log_tab", "Next log", show=False),
    ]

    def action_prev_log_tab(self) -> None:
        screen = self.screen
        if isinstance(screen, LocalEvalRunScreen):
            screen.cycle_log_tab(-1)

    def action_next_log_tab(self) -> None:
        screen = self.screen
        if isinstance(screen, LocalEvalRunScreen):
            screen.cycle_log_tab(1)


class EvalSearchScreen(ModalScreen[SearchResult | None]):
    """Modal search over prompt/completion or log text."""

    BINDINGS = [
        Binding("escape", "close", "Close", key_display="Esc"),
        Binding("enter", "select", "Select", key_display="Enter"),
    ]

    CSS = """
    EvalSearchScreen {
        align: center middle;
    }

    EvalSearchScreen > Container {
        width: 110;
        height: 34;
        border: round $primary;
        padding: 1 2;
        background: $surface;
    }

    .modal-columns {
        height: 1fr;
    }

    .modal-panel {
        width: 1fr;
        border: round $primary;
        padding: 0 1;
    }
    """

    def __init__(
        self,
        prompt_lines: list[tuple[int, int, str]],
        completion_lines: list[tuple[int, int, str]],
    ) -> None:
        super().__init__()
        self._tagged_lines = {
            "prompt": prompt_lines,
            "completion": completion_lines,
        }
        self._hits: dict[str, list[SearchHit]] = {"prompt": [], "completion": []}
        self._cursors: dict[str, int | None] = {"prompt": None, "completion": None}
        self._active_column: str | None = None

    def compose(self) -> ComposeResult:
        with Container():
            yield Label(Text("Search", style="bold"))
            yield ClearableInput(placeholder="regex, case-insensitive", id="search-input")
            yield Label("", id="search-error", classes="subtitle")
            with Horizontal(classes="modal-columns"):
                with EvalPanel(classes="modal-panel"):
                    yield Label(Text("Prompt results", style="bold"), id="prompt-count")
                    yield OptionList(id="prompt-results")
                with EvalPanel(classes="modal-panel"):
                    yield Label(
                        Text("Completion results", style="bold"),
                        id="completion-count",
                    )
                    yield OptionList(id="completion-results")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#search-input", ClearableInput).focus()
        self._update_results("")

    def on_key(self, event: events.Key) -> None:
        if event.key == "left":
            self._switch_column("prompt")
        elif event.key == "right":
            self._switch_column("completion")
        elif event.key == "up":
            self._move_selection(-1)
        elif event.key == "down":
            self._move_selection(1)
        else:
            return
        event.prevent_default()
        event.stop()

    @on(Input.Changed, "#search-input")
    def on_input_changed(self, event: Input.Changed) -> None:
        self._update_results(event.value)

    @on(Input.Submitted, "#search-input")
    def on_input_submitted(self, _event: Input.Submitted) -> None:
        self.action_select()

    @on(OptionList.OptionHighlighted, "#prompt-results")
    def on_prompt_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        self._set_active_hit("prompt", event.option_id)

    @on(OptionList.OptionHighlighted, "#completion-results")
    def on_completion_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        self._set_active_hit("completion", event.option_id)

    @on(OptionList.OptionSelected, "#prompt-results")
    def on_prompt_selected(self, event: OptionList.OptionSelected) -> None:
        self._set_active_hit("prompt", event.option_id, select=True)

    @on(OptionList.OptionSelected, "#completion-results")
    def on_completion_selected(self, event: OptionList.OptionSelected) -> None:
        self._set_active_hit("completion", event.option_id, select=True)

    def action_close(self) -> None:
        self.dismiss(None)

    def action_select(self) -> None:
        selection = self._current_selection()
        if selection is None:
            return
        pattern = self.query_one("#search-input", ClearableInput).value
        self.dismiss(
            SearchResult(
                column=selection.column,
                pattern=pattern,
                section_index=selection.section_index,
                nested_index=selection.nested_index,
            )
        )

    def _set_active_hit(self, column: str, option_id: str | None, *, select: bool = False) -> None:
        if option_id is None:
            return
        self._active_column = column
        self._cursors[column] = int(option_id)
        self._sync_highlights()
        if select:
            self.action_select()

    def _update_results(self, pattern: str) -> None:
        option_lists = {
            "prompt": self.query_one("#prompt-results", OptionList),
            "completion": self.query_one("#completion-results", OptionList),
        }
        labels = {
            "prompt": self.query_one("#prompt-count", Label),
            "completion": self.query_one("#completion-count", Label),
        }
        error_label = self.query_one("#search-error", Label)

        for column, option_list in option_lists.items():
            option_list.clear_options()
            self._hits[column] = []
            self._cursors[column] = None

        if not pattern:
            error_label.update("")
            labels["prompt"].update(Text("Prompt results", style="bold"))
            labels["completion"].update(Text("Completion results", style="bold"))
            self._active_column = None
            return

        try:
            compiled = re.compile(pattern, re.IGNORECASE)
        except re.error as exc:
            error_label.update(Text(f"Invalid regex: {exc}", style=STATUS_ERROR))
            self._active_column = None
            return

        error_label.update("")
        for column, tagged_lines in self._tagged_lines.items():
            hits: list[SearchHit] = []
            for line_index, (section_index, nested_index, line) in enumerate(tagged_lines):
                if not compiled.search(line):
                    continue
                hits.append(
                    SearchHit(
                        column=column,
                        line_index=line_index,
                        line_text=line,
                        section_index=section_index,
                        nested_index=nested_index,
                    )
                )
                content = Text(line)
                stylize_matches(content, compiled, "reverse")
                option_lists[column].add_option(
                    Option(
                        Text(f"{line_index + 1:>5} | ", style="dim") + content,
                        id=str(len(hits) - 1),
                    )
                )
            self._hits[column] = hits
            labels[column].update(
                Text(f"{column.capitalize()} results ({len(hits)})", style="bold")
            )

        if self._hits["completion"]:
            self._active_column = "completion"
            self._cursors["completion"] = 0
        elif self._hits["prompt"]:
            self._active_column = "prompt"
            self._cursors["prompt"] = 0
        else:
            self._active_column = None
        self._sync_highlights()

    def _sync_highlights(self) -> None:
        for column, option_list in (
            ("prompt", self.query_one("#prompt-results", OptionList)),
            ("completion", self.query_one("#completion-results", OptionList)),
        ):
            if self._active_column == column and self._cursors[column] is not None:
                option_list.highlighted = self._cursors[column]
                option_list.scroll_to_highlight()
            else:
                option_list.highlighted = None

    def _switch_column(self, target: str) -> None:
        if self._hits[target]:
            self._active_column = target
            if self._cursors[target] is None:
                self._cursors[target] = 0
        self._sync_highlights()

    def _move_selection(self, delta: int) -> None:
        if self._active_column is None:
            return
        hits = self._hits[self._active_column]
        cursor = self._cursors[self._active_column]
        if not hits:
            return
        self._cursors[self._active_column] = (
            0 if cursor is None else max(0, min(len(hits) - 1, cursor + delta))
        )
        self._sync_highlights()

    def _current_selection(self) -> SearchHit | None:
        if self._active_column is None:
            return None
        cursor = self._cursors[self._active_column]
        hits = self._hits[self._active_column]
        if cursor is None or not hits:
            return None
        return hits[cursor]


class RolloutCopyScreen(ModalScreen[None]):
    """Modal screen for copying rollout viewer sections."""

    BINDINGS = [
        Binding("escape", "close", "Back", key_display="Esc"),
        Binding("b", "close", "Back", key_display="B"),
        Binding("c", "copy", "Copy"),
        Binding("y", "copy", "Copy"),
    ]

    CSS = """
    RolloutCopyScreen {
        align: center middle;
    }

    RolloutCopyScreen > Container {
        width: 120;
        height: 38;
        border: round $primary;
        padding: 1 2;
        background: $surface;
    }

    .copy-columns {
        height: 1fr;
    }

    .copy-target-panel {
        width: 34;
        border: round $primary;
        padding: 0 1;
    }

    .copy-preview-panel {
        width: 1fr;
        border: round $primary;
        padding: 0 1;
    }

    #rollout-copy-preview {
        height: 1fr;
    }
    """

    def __init__(
        self,
        items: list[RolloutCopyItem],
        *,
        start_key: str | None = None,
        title: str = "Copy Rollout",
    ) -> None:
        super().__init__()
        self._items = items
        self._title = title
        self._current_idx = 0
        if start_key:
            for idx, item in enumerate(items):
                if item.key == start_key:
                    self._current_idx = idx
                    break
        self._last_copied_selection = ""

    def compose(self) -> ComposeResult:
        with Container():
            yield Label(Text(self._title, style="bold"))
            yield Label("", id="rollout-copy-status", classes="subtitle")
            with Horizontal(classes="copy-columns"):
                with EvalPanel(classes="copy-target-panel"):
                    yield Label(Text("Copy targets", style="bold"))
                    yield OptionList(id="rollout-copy-targets")
                with EvalPanel(classes="copy-preview-panel"):
                    yield Label(Text("Preview", style="bold"), id="rollout-copy-preview-label")
                    preview = TextArea("", id="rollout-copy-preview")
                    preview.read_only = True
                    yield preview
        yield Footer()

    def on_mount(self) -> None:
        option_list = self.query_one("#rollout-copy-targets", OptionList)
        for item in self._items:
            option_list.add_option(Option(Text(item.label), id=item.key))
        option_list.highlighted = self._current_idx
        self._sync_preview()
        self.query_one("#rollout-copy-preview", TextArea).focus()

    @on(OptionList.OptionHighlighted, "#rollout-copy-targets")
    def _on_target_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        if event.option_index is not None and event.option_index != self._current_idx:
            self._current_idx = event.option_index
            self._sync_preview()

    @on(OptionList.OptionSelected, "#rollout-copy-targets")
    def _on_target_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_index is not None:
            self._current_idx = event.option_index
            self._sync_preview()
        self.query_one("#rollout-copy-preview", TextArea).focus()

    @on(TextArea.SelectionChanged)
    def _on_selection_changed(self, event: TextArea.SelectionChanged) -> None:
        if event.text_area.id != "rollout-copy-preview":
            return
        selected = event.text_area.selected_text or ""
        if selected and selected != self._last_copied_selection:
            self.app.copy_to_clipboard(selected)
            self._last_copied_selection = selected
            self.query_one("#rollout-copy-status", Label).update(
                Text(f"Copied selection ({len(selected):,} chars).", style="dim")
            )

    def action_close(self) -> None:
        self.dismiss(None)

    def action_copy(self) -> None:
        if not self._items:
            return
        item = self._items[self._current_idx]
        preview = self.query_one("#rollout-copy-preview", TextArea)
        selected = preview.selected_text or ""
        copied_text = selected or item.body
        if not copied_text:
            self.query_one("#rollout-copy-status", Label).update(
                Text("Nothing to copy.", style="dim")
            )
            return
        self.app.copy_to_clipboard(copied_text)
        self._last_copied_selection = copied_text
        label = "selection" if selected else item.label.lower()
        self.query_one("#rollout-copy-status", Label).update(
            Text(f"Copied {label} ({len(copied_text):,} chars).", style="dim")
        )

    def _sync_preview(self) -> None:
        if not self._items:
            return
        item = self._items[self._current_idx]
        self.query_one("#rollout-copy-preview-label", Label).update(
            Text(f"{item.label}  ({len(item.body):,} chars)", style="bold")
        )
        self.query_one("#rollout-copy-preview", TextArea).load_text(item.body)


class _RolloutRecordSequence(Sequence[dict[str, Any]]):
    def __init__(self, records: Sequence[dict[str, Any]] | LazyRunResults) -> None:
        self._lazy_records = records if isinstance(records, LazyRunResults) else None
        self._cache: dict[int, dict[str, Any]] = {}
        if self._lazy_records is None:
            self._records = [
                normalize_rollout_record(record) for record in records if isinstance(record, dict)
            ]
            self._count = len(self._records)
        else:
            self._records = []
            self._count = self._lazy_records.count_hint()

    def __len__(self) -> int:
        if self._count is None and self._lazy_records is not None:
            self._count = len(self._lazy_records)
        return self._count

    def __bool__(self) -> bool:
        if self._count is None and self._lazy_records is not None:
            return bool(self._lazy_records)
        return self._count > 0

    def __getitem__(self, index: int | slice) -> dict[str, Any] | list[dict[str, Any]]:
        count = len(self)
        if isinstance(index, slice):
            return [self[idx] for idx in range(*index.indices(count))]
        if index < 0:
            index += count
        if index < 0 or index >= count:
            raise IndexError(index)
        if self._lazy_records is None:
            return self._records[index]
        cached = self._cache.get(index)
        if cached is not None:
            return cached
        raw = self._lazy_records[index]
        record = normalize_rollout_record(raw if isinstance(raw, dict) else {})
        self._cache[index] = record
        return record

    def loaded(self, index: int) -> dict[str, Any] | None:
        if self._lazy_records is None:
            return self._records[index] if 0 <= index < self._count else None
        return self._cache.get(index)


class RolloutViewer(Container):
    """Reusable rollout transcript viewer for evals and training samples."""

    BINDINGS = [
        Binding("p", "prev_record", "Prev rollout"),
        Binding("n", "next_record", "Next rollout"),
        Binding("pageup", "history_page_up", show=False),
        Binding("pagedown", "history_page_down", show=False),
        Binding("home", "history_home", show=False),
        Binding("end", "history_end", show=False),
        Binding("e", "expand_all", "Expand all", show=False),
        Binding("x", "collapse_all", "Collapse all", show=False),
        Binding("/", "search", "Search"),
        Binding("m", "toggle_markdown_math", "Toggle markdown", show=False),
        Binding("c", "copy", "Copy", show=False),
        Binding("y", "copy", "Copy"),
    ]

    DEFAULT_CSS = (
        """
    RolloutViewer {
        layout: horizontal;
        height: 1fr;
        min-height: 24;
    }

    RolloutViewer EvalPanel {
        border: round $primary;
        padding: 0 1;
        background: $surface;
    }

    RolloutViewer .rollouts-panel {
        width: 34;
        min-width: 30;
    }

    RolloutViewer .history-panel {
        width: 2fr;
        min-width: 50;
    }

    RolloutViewer .details-panel {
        width: 34;
        min-width: 30;
    }

    RolloutViewer .column-header {
        height: 1;
    }

    RolloutViewer .subtitle {
        height: auto;
        color: $text-muted;
    }

    RolloutViewer #viewer-rollout-list,
    RolloutViewer #viewer-completion-scroll,
    RolloutViewer .details-scroll {
        height: 1fr;
        background: $surface;
        scrollbar-size-vertical: 2;
        scrollbar-color: $primary 40%;
        scrollbar-color-hover: $primary 70%;
        scrollbar-color-active: $accent;
        scrollbar-background: $surface;
        scrollbar-background-hover: $surface;
        scrollbar-background-active: $surface;
        scrollbar-corner-color: $panel;
    }

    RolloutViewer .history-section {
        margin: 0 0 1 0;
        background: $surface;
        border: round $secondary;
    }

    RolloutViewer .history-section:focus-within {
        background-tint: $foreground 4%;
    }

    RolloutViewer .history-section > CollapsibleTitle {
        text-style: bold;
        padding: 0 1;
    }

    RolloutViewer .history-section > CollapsibleTitle:hover {
        background: $secondary 12%;
        color: $text;
    }

    RolloutViewer .history-section > CollapsibleTitle:focus {
        background: $secondary 24%;
        color: $text;
    }

    RolloutViewer .assistant-section {
        background: $rollout-success 2%;
        border: round $rollout-success 80%;
    }

    RolloutViewer .assistant-section > CollapsibleTitle {
        color: $rollout-success;
    }

    RolloutViewer .assistant-section > CollapsibleTitle:hover {
        background: $rollout-success 7%;
    }

    RolloutViewer .assistant-section > CollapsibleTitle:focus {
        background: $rollout-success 11%;
    }

    RolloutViewer .tool-section {
        background: $rollout-warning 2%;
        border: round $rollout-warning 80%;
    }

    RolloutViewer .tool-section > CollapsibleTitle {
        color: $rollout-warning;
    }

    RolloutViewer .tool-section > CollapsibleTitle:hover {
        background: $rollout-warning 7%;
    }

    RolloutViewer .tool-section > CollapsibleTitle:focus {
        background: $rollout-warning 11%;
    }

    RolloutViewer .prompt-section {
        background: $secondary 3%;
        border: round $secondary;
    }

    RolloutViewer .prompt-section > CollapsibleTitle {
        color: $secondary;
    }

    RolloutViewer .prompt-section > CollapsibleTitle:hover {
        background: $secondary 8%;
    }

    RolloutViewer .prompt-section > CollapsibleTitle:focus {
        background: $secondary 14%;
    }

    RolloutViewer .prompt-section .section-body {
        color: $text-muted;
    }

    RolloutViewer .tool-call-section {
        background: $tool-call 4%;
        border: round $tool-call 85%;
    }

    RolloutViewer .tool-call-section > CollapsibleTitle {
        color: $tool-call;
    }

    RolloutViewer .tool-call-section > CollapsibleTitle:hover {
        background: $tool-call 8%;
    }

    RolloutViewer .tool-call-section > CollapsibleTitle:focus {
        background: $tool-call 14%;
    }

    RolloutViewer .reasoning-section {
        background: $primary 3%;
        border: round $primary;
    }

    RolloutViewer .reasoning-section > CollapsibleTitle {
        color: $primary;
    }

    RolloutViewer .reasoning-section > CollapsibleTitle:hover {
        background: $primary 8%;
    }

    RolloutViewer .reasoning-section > CollapsibleTitle:focus {
        background: $primary 14%;
    }

    RolloutViewer .assistant-section .nested-section > CollapsibleTitle:hover {
        background: $rollout-success 9%;
    }

    RolloutViewer .assistant-section .nested-section > CollapsibleTitle:focus {
        background: $rollout-success 16%;
    }

    RolloutViewer .tool-section .nested-section > CollapsibleTitle:hover {
        background: $rollout-warning 9%;
    }

    RolloutViewer .tool-section .nested-section > CollapsibleTitle:focus {
        background: $rollout-warning 16%;
    }

    RolloutViewer .assistant-section .tool-call-section > CollapsibleTitle:hover {
        background: $tool-call 8%;
    }

    RolloutViewer .assistant-section .tool-call-section > CollapsibleTitle:focus {
        background: $tool-call 14%;
    }

    RolloutViewer .prompt-section .nested-section > CollapsibleTitle:hover {
        background: $secondary 12%;
    }

    RolloutViewer .prompt-section .nested-section > CollapsibleTitle:focus {
        background: $secondary 24%;
    }

    RolloutViewer .nested-section {
        margin: 0 0 0 1;
    }

    RolloutViewer .section-body {
        padding: 0 1 0 1;
        color: $text;
    }
    """.replace("$rollout-success", ROLLOUT_SUCCESS)
        .replace("$rollout-warning", ROLLOUT_WARNING)
        .replace("$tool-call", TOOL_CALL)
    )

    def __init__(
        self,
        records: Sequence[dict[str, Any]] | LazyRunResults,
        *,
        metadata: dict[str, Any] | None = None,
        title: str = "Rollouts",
        on_record_changed: Callable[[int, dict[str, Any]], None] | None = None,
        classes: str | None = None,
        id: str | None = None,
    ) -> None:
        super().__init__(id=id, classes=classes)
        self.records: _RolloutRecordSequence = (
            records
            if isinstance(records, _RolloutRecordSequence)
            else _RolloutRecordSequence(records)
        )
        self.metadata = metadata or {}
        self.title = title
        self._on_record_changed = on_record_changed
        self.current_record_idx = 0
        self._prompt_text = ""
        self._completion_text = ""
        self._highlight_regex: re.Pattern[str] | None = None
        self._highlight_column: str | None = None
        self._highlight_timer: Any = None
        self._highlight_section_index = 0
        self._highlight_nested_index = -1
        self._render_markdown_math = True
        if self.records:
            self._set_record_text_state(self.records[0])

    def compose(self) -> ComposeResult:
        if not self.records:
            yield Static(Text("No rollout samples loaded.", style="dim"))
            return
        record = self.records[self.current_record_idx]
        with EvalPanel(classes="rollouts-panel"):
            yield Label(Text(self.title, style="bold"), classes="column-header")
            yield Label("", id="viewer-rollout-summary", classes="subtitle")
            yield OptionList(id="viewer-rollout-list")
        with EvalPanel(classes="history-panel"):
            yield Label(Text("Completion History", style="bold"), classes="column-header")
            yield Static("", id="viewer-history-summary", classes="subtitle", markup=False)
            yield VerticalScroll(
                *self._completion_sections(record),
                id="viewer-completion-scroll",
            )
        with EvalPanel(classes="details-panel"):
            yield Label(Text("Details", style="bold"), classes="column-header")
            with TabbedContent(initial="viewer-details-task", id="viewer-details-tabs"):
                with TabPane("Task", id="viewer-details-task"):
                    yield TabbedScrollPane(
                        Static("", id="viewer-task-content", markup=False),
                        classes="details-scroll",
                    )
                with TabPane("Score", id="viewer-details-score"):
                    yield TabbedScrollPane(
                        Static("", id="viewer-score-content", markup=False),
                        classes="details-scroll",
                    )
                with TabPane("Usage", id="viewer-details-usage"):
                    yield TabbedScrollPane(
                        Static("", id="viewer-usage-content", markup=False),
                        classes="details-scroll",
                    )
                with TabPane("State", id="viewer-details-state"):
                    yield TabbedScrollPane(
                        Static("", id="viewer-state-content", markup=False),
                        classes="details-scroll",
                    )

    def on_mount(self) -> None:
        if not self.records:
            return
        self._populate_rollout_list()
        if self._on_record_changed is not None:
            self._on_record_changed(self.current_record_idx, self.records[self.current_record_idx])
        self.call_after_refresh(self.update_display)
        self.call_after_refresh(self._focus_primary_content)

    def action_prev_record(self) -> None:
        self._move_record_cursor(-1)

    def action_next_record(self) -> None:
        self._move_record_cursor(1)

    def action_expand_all(self) -> None:
        if not self.records:
            return
        container = self._completion_scroll()
        if container is None:
            return
        for section in container.query(Collapsible):
            section.collapsed = False
        self._focus_primary_content()

    def action_collapse_all(self) -> None:
        if not self.records:
            return
        container = self._completion_scroll()
        if container is None:
            return
        for section in container.query(Collapsible):
            section.collapsed = True
        self._focus_primary_content(prefer_expanded=False)

    def action_history_page_up(self) -> None:
        if self.records and (container := self._completion_scroll()) is not None:
            container.scroll_page_up(animate=False)

    def action_history_page_down(self) -> None:
        if self.records and (container := self._completion_scroll()) is not None:
            container.scroll_page_down(animate=False)

    def action_history_home(self) -> None:
        if self.records and (container := self._completion_scroll()) is not None:
            container.scroll_home(animate=False)

    def action_history_end(self) -> None:
        if self.records and (container := self._completion_scroll()) is not None:
            container.scroll_end(animate=False)

    def action_toggle_markdown_math(self) -> None:
        self._render_markdown_math = not self._render_markdown_math
        self._swap_section_bodies()

    def action_search(self) -> None:
        if not self.records:
            return
        record = self.records[self.current_record_idx]
        prompt_lines, completion_lines = self._build_search_lines(record)
        self.app.push_screen(
            EvalSearchScreen(prompt_lines, completion_lines),
            self._handle_search_result,
        )

    def action_copy(self) -> None:
        if not self.records:
            return
        record = self.records[self.current_record_idx]
        self.app.push_screen(
            RolloutCopyScreen(
                self._build_rollout_copy_items(record),
                start_key="snapshot",
                title=f"Copy Rollout #{self.current_record_idx}",
            )
        )

    def _record_progress_label(self) -> str:
        return f"{self.current_record_idx + 1}/{len(self.records)}"

    def _populate_rollout_list(self) -> None:
        rollout_list = self._rollout_list()
        if rollout_list is None:
            return
        rollout_list.clear_options()
        for idx in range(len(self.records)):
            rollout_list.add_option(Option(self._rollout_option_label(idx), id=str(idx)))
        rollout_list.highlighted = self.current_record_idx
        rollout_list.scroll_to_highlight()

    def _rollout_option_label(self, idx: int) -> Text:
        record = self.records.loaded(idx)
        if record is not None:
            return build_rollout_prompt(idx, record)
        return Text(f"Rollout {idx + 1}")

    def _move_record_cursor(self, delta: int) -> None:
        if not self.records:
            return
        new_index = (self.current_record_idx + delta) % len(self.records)
        rollout_list = self._rollout_list()
        if rollout_list is None:
            return
        rollout_list.highlighted = new_index
        rollout_list.scroll_to_highlight()
        self._set_current_record(new_index)

    def _set_current_record(self, index: int, *, focus_history: bool = False) -> None:
        if not (0 <= index < len(self.records)):
            return
        self.current_record_idx = index
        self._set_highlight(None, repaint=False)
        if self._on_record_changed is not None:
            self._on_record_changed(index, self.records[index])
        self.update_display(focus_history=focus_history)
        if container := self._completion_scroll():
            container.scroll_y = 0
        for scroll in self.query(".details-scroll"):
            if isinstance(scroll, VerticalScroll):
                scroll.scroll_y = 0

    def _set_record_text_state(self, record: dict[str, Any]) -> None:
        prompt_text = format_prompt_or_completion(record.get("prompt", ""))
        completion_text = format_prompt_or_completion(record.get("completion", ""))
        error = record.get("error")
        if error is not None:
            completion_text.append("\n\n")
            completion_text.append("error: ", style=STATUS_ERROR)
            completion_text.append(str(error), style=STATUS_ERROR)
        self._prompt_text = prompt_text.plain
        self._completion_text = completion_text.plain

    def update_display(self, *, focus_history: bool = False) -> None:
        if not self.records:
            return
        record = self.records[self.current_record_idx]
        self._set_record_text_state(record)
        self._update_static(
            "#viewer-history-summary",
            self._build_history_summary_text(record),
        )
        self._update_static("#viewer-task-content", build_task_text(record, self.metadata))
        self._update_static("#viewer-score-content", build_score_text(record))
        self._update_static("#viewer-usage-content", build_usage_text(record))
        self._update_static(
            "#viewer-state-content",
            build_state_text(record, self.metadata),
        )
        self._update_label("#viewer-rollout-summary", self._build_rollout_summary_text(record))
        self._rebuild_completion_sections(record, focus_history)

    @on(TabbedContent.TabActivated, "#viewer-details-tabs")
    def _details_tab_activated(self, _event: TabbedContent.TabActivated) -> None:
        self.call_after_refresh(self.update_display)

    def _update_static(self, selector: str, renderable: object) -> None:
        try:
            self.query_one(selector, Static).update(renderable)
        except NoMatches:
            return

    def _update_label(self, selector: str, renderable: object) -> None:
        try:
            self.query_one(selector, Label).update(renderable)
        except NoMatches:
            return

    @on(OptionList.OptionHighlighted, "#viewer-rollout-list")
    def on_rollout_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        if event.option_id is None:
            return
        idx = int(event.option_id)
        if idx != self.current_record_idx:
            self._set_current_record(idx)

    @on(OptionList.OptionSelected, "#viewer-rollout-list")
    def on_rollout_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_id is None:
            return
        self._set_current_record(int(event.option_id), focus_history=True)

    def _completion_sections(self, record: dict[str, Any]) -> list[Collapsible]:
        return [self._make_section(section) for section in self._history_section_data(record)]

    def _rebuild_completion_sections(
        self, record: dict[str, Any], focus_history: bool = False
    ) -> None:
        if not self.is_mounted:
            return
        container = self._completion_scroll()
        if container is None:
            return
        container.remove_children()
        container.mount(*self._completion_sections(record))
        if focus_history:
            self.call_after_refresh(self._focus_primary_content)

    def _history_section_data(self, record: dict[str, Any]) -> list[HistorySectionData]:
        sections = [self._initial_prompt_section_data(record)]
        completion = record.get("completion")
        if not isinstance(completion, list) or not completion:
            sections.append(
                HistorySectionData(
                    title="Completion",
                    body=self._completion_text,
                    column="completion",
                    collapsed=False,
                    classes="history-section assistant-section",
                )
            )
            return sections

        for idx, group in enumerate(history_groups(completion), start=1):
            message = group["message"]
            if group["kind"] != "assistant-tools":
                role = str(message.get("role", "message"))
                title = f"{idx}. {role}"
                preview = format_message_preview(message)
                if preview:
                    title += f"  {preview}"
                reasoning_sections = self._reasoning_section_data(message)
                sections.append(
                    HistorySectionData(
                        title=title,
                        body=stringify_message_content(message.get("content", "")),
                        column="completion",
                        collapsed=True,
                        classes=(
                            "history-section tool-section"
                            if role == "tool"
                            else (
                                "history-section prompt-section"
                                if role not in ("assistant", "tool")
                                else "history-section assistant-section"
                            )
                        ),
                        nested_sections=reasoning_sections,
                        body_first=not reasoning_sections,
                    )
                )
                continue

            tool_calls = group["tool_calls"]
            tool_outputs = group["tool_outputs"]
            preview = tool_group_preview(message, tool_outputs) or format_message_preview(message)
            title = f"{idx}. assistant"
            if preview:
                title += f"  {preview}"
            body = stringify_message_content(message.get("content", ""))
            collapsed = True
            if self._highlight_regex and self._highlight_column == "completion":
                collapsed = not (body and self._highlight_regex.search(body))
                if collapsed:
                    for tool_call in tool_calls:
                        name, arguments, _ = tool_call_parts(tool_call)
                        if self._highlight_regex.search(name) or self._highlight_regex.search(
                            arguments
                        ):
                            collapsed = False
                            break
                if collapsed:
                    for output in tool_outputs:
                        output_text = (
                            stringify_message(output) if isinstance(output, dict) else str(output)
                        )
                        if self._highlight_regex.search(output_text):
                            collapsed = False
                            break
            nested_sections: list[HistorySectionData] = list(self._reasoning_section_data(message))
            used_output_indexes: set[int] = set()
            for tool_idx, tool_call in enumerate(tool_calls, start=1):
                name, arguments, call_id = tool_call_parts(tool_call)
                matched_output = None
                if call_id is not None:
                    for output_idx, candidate in enumerate(tool_outputs):
                        if isinstance(candidate, dict) and candidate.get("tool_call_id") == call_id:
                            matched_output = candidate
                            used_output_indexes.add(output_idx)
                            break
                if matched_output is None:
                    for output_idx, candidate in enumerate(tool_outputs):
                        if output_idx not in used_output_indexes:
                            matched_output = candidate
                            used_output_indexes.add(output_idx)
                            break
                output_text = (
                    stringify_message(matched_output)
                    if isinstance(matched_output, dict)
                    else (str(matched_output) if matched_output is not None else "")
                )
                nested_sections.append(
                    HistorySectionData(
                        title=f"tool {tool_idx}  {name}  ... {tool_output_preview(matched_output)}",
                        body="\n".join(["Call", arguments, "", "Output", output_text]),
                        column="completion",
                        collapsed=collapsed or tool_idx > 1,
                        classes="history-section tool-call-section nested-section",
                    )
                )

            for output_idx, output_message in enumerate(tool_outputs):
                if output_idx in used_output_indexes:
                    continue
                output_text = (
                    stringify_message(output_message)
                    if isinstance(output_message, dict)
                    else str(output_message)
                )
                nested_sections.append(
                    HistorySectionData(
                        title=(
                            f"tool output {len(nested_sections) + 1}  "
                            f"{tool_output_preview(output_message)}"
                        ),
                        body=output_text,
                        column="completion",
                        collapsed=True,
                        classes="history-section tool-section nested-section",
                    )
                )

            sections.append(
                HistorySectionData(
                    title=title,
                    body=body,
                    column="completion",
                    collapsed=collapsed,
                    classes="history-section assistant-section",
                    nested_sections=tuple(nested_sections),
                    body_first=False if nested_sections else True,
                )
            )
        return sections

    def _initial_prompt_section_data(self, record: dict[str, Any]) -> HistorySectionData:
        prompt = record.get("prompt")
        if not isinstance(prompt, list) or not prompt:
            return HistorySectionData(
                title="Initial Prompt",
                body=self._prompt_text,
                column="prompt",
                collapsed=True,
                classes="history-section prompt-section",
            )

        nested_sections: list[HistorySectionData] = []
        for idx, message in enumerate(prompt, start=1):
            if not isinstance(message, dict):
                nested_sections.append(
                    HistorySectionData(
                        title=f"{idx}. prompt",
                        body=str(message),
                        column="prompt",
                        collapsed=False,
                        classes="history-section prompt-section nested-section",
                    )
                )
                continue
            role = str(message.get("role") or "message")
            preview = format_message_preview(message)
            title = f"{idx}. {role}"
            if preview:
                title += f"  {preview}"
            nested_sections.append(
                HistorySectionData(
                    title=title,
                    body=stringify_message_content(message.get("content", "")),
                    column="prompt",
                    collapsed=False,
                    classes="history-section prompt-section nested-section",
                    nested_sections=self._reasoning_section_data(message),
                    body_first=True,
                )
            )

        return HistorySectionData(
            title="Initial Prompt",
            body="",
            column="prompt",
            collapsed=True,
            classes="history-section prompt-section",
            nested_sections=tuple(nested_sections),
            body_first=False,
        )

    def _reasoning_section_data(self, message: dict[str, Any]) -> tuple[HistorySectionData, ...]:
        reasoning = stringify_message_reasoning(message)
        if not reasoning:
            return ()
        return (
            HistorySectionData(
                title="Reasoning",
                body=reasoning,
                column="completion",
                collapsed=True,
                classes="history-section reasoning-section nested-section",
            ),
        )

    def _section_matches_highlight(self, section: HistorySectionData) -> bool:
        if not (self._highlight_regex and self._highlight_column == section.column):
            return False
        if self._highlight_regex.search(section.title) or self._highlight_regex.search(
            section.body
        ):
            return True
        return any(self._section_matches_highlight(child) for child in section.nested_sections)

    def _make_body_widget(self, body: str, column: str) -> Widget:
        if self._render_markdown_math and not (
            self._highlight_regex and self._highlight_column == column
        ):
            return MathMarkdown(body, classes="section-body")
        text = Text(body)
        if self._highlight_regex and self._highlight_column == column:
            stylize_matches(text, self._highlight_regex, "reverse")
        return Static(text, classes="section-body", markup=False)

    def _make_section(self, section: HistorySectionData) -> Collapsible:
        collapsed = section.collapsed
        if self._section_matches_highlight(section):
            collapsed = False
        body_children: list[Widget] = []
        if section.body or not section.nested_sections:
            body_children.append(self._make_body_widget(section.body, section.column))
        nested_children = [self._make_section(child) for child in section.nested_sections]
        children = (
            [*body_children, *nested_children]
            if section.body_first
            else [*nested_children, *body_children]
        )
        return Collapsible(
            *children,
            title=escape(section.title),
            collapsed=collapsed,
            classes=section.classes,
        )

    def _collect_section_bodies(self, sections: list[HistorySectionData]) -> list[tuple[str, str]]:
        result: list[tuple[str, str]] = []
        for section in sections:
            parent = (
                [(section.body, section.column)]
                if section.body or not section.nested_sections
                else []
            )
            nested = [
                (nested_section.body, nested_section.column)
                for nested_section in section.nested_sections
                if nested_section.body or not nested_section.nested_sections
            ]
            if section.body_first:
                result.extend(parent)
                result.extend(nested)
            else:
                result.extend(nested)
                result.extend(parent)
        return result

    def _swap_section_bodies(self) -> None:
        if not (self.records and self.is_mounted):
            return
        record = self.records[self.current_record_idx]
        body_entries = self._collect_section_bodies(self._history_section_data(record))
        container = self._completion_scroll()
        if container is None:
            return
        body_widgets = list(container.query(".section-body"))
        for idx, body_widget in enumerate(body_widgets):
            parent = body_widget.parent
            if not isinstance(parent, Widget) or idx >= len(body_entries):
                continue
            body, column = body_entries[idx]
            replacement = self._make_body_widget(body, column)
            parent.mount(replacement, after=body_widget)
            body_widget.remove()

    def _build_history_summary_text(self, record: dict[str, Any]) -> Text:
        completion = record.get("completion")
        if not isinstance(completion, list) or not completion:
            return Text()
        groups = history_groups(completion)
        tool_groups = sum(1 for group in groups if group.get("kind") == "assistant-tools")
        user_messages = sum(
            1
            for group in groups
            if isinstance(group.get("message"), dict) and group["message"].get("role") == "user"
        )
        return Text.assemble(
            (f"{len(groups)} events", "bold"),
            ("  ", ""),
            (f"{tool_groups} tool exchanges", "dim"),
            ("  ", ""),
            (f"{user_messages} user turns", "dim"),
        )

    def _build_rollout_summary_text(self, record: dict[str, Any]) -> Text:
        reward = record.get("reward")
        return Text.assemble(
            (self._record_progress_label(), "bold"),
            ("  reward ", "dim"),
            (format_reward_value(reward), reward_style(reward, subdued=True)),
        )

    def _build_search_lines(
        self, record: dict[str, Any]
    ) -> tuple[list[tuple[int, int, str]], list[tuple[int, int, str]]]:
        sections = self._history_section_data(record)
        prompt_lines: list[tuple[int, int, str]] = []
        completion_lines: list[tuple[int, int, str]] = []
        for idx, section in enumerate(sections):
            target = prompt_lines if section.column == "prompt" else completion_lines

            def append_body(lines: list[tuple[int, int, str]], body: str) -> None:
                for line in body.splitlines():
                    lines.append((idx, -1, line))

            def append_nested() -> None:
                for nested_idx, nested in enumerate(section.nested_sections):
                    nested_target = prompt_lines if nested.column == "prompt" else completion_lines
                    for line in nested.body.splitlines():
                        nested_target.append((idx, nested_idx, line))

            if section.body_first:
                append_body(target, section.body)
                append_nested()
            else:
                append_nested()
                append_body(target, section.body)
        return prompt_lines, completion_lines

    def _handle_search_result(self, result: SearchResult | None) -> None:
        if result is not None:
            self._set_highlight(result)

    def _set_highlight(self, result: SearchResult | None, *, repaint: bool = True) -> None:
        if self._highlight_timer is not None:
            self._highlight_timer.stop()
            self._highlight_timer = None

        had_highlight = self._highlight_regex is not None
        self._highlight_regex = None
        self._highlight_column = None
        self._highlight_section_index = 0
        self._highlight_nested_index = -1

        if result is not None:
            try:
                self._highlight_regex = re.compile(result.pattern, re.IGNORECASE)
            except re.error:
                return
            self._highlight_column = result.column
            self._highlight_section_index = result.section_index
            self._highlight_nested_index = result.nested_index
            self._highlight_timer = self.set_timer(3.0, lambda: self._set_highlight(None))

        if repaint and self.is_mounted and (had_highlight or result is not None):
            self._swap_section_bodies()
            if result is not None:
                container = self._completion_scroll()
                if container is None:
                    return
                self._expand_and_scroll_to_match(container)

    def _expand_and_scroll_to_match(self, container: VerticalScroll) -> None:
        sections = [child for child in container.children if isinstance(child, Collapsible)]
        idx = self._highlight_section_index
        if not (0 <= idx < len(sections)):
            return
        parent = sections[idx]
        parent.collapsed = False
        scroll_target: Collapsible = parent
        nested_idx = self._highlight_nested_index
        if nested_idx >= 0:
            nested_collapsibles = [
                child for child in parent.query(Collapsible) if child is not parent
            ]
            if 0 <= nested_idx < len(nested_collapsibles):
                nested = nested_collapsibles[nested_idx]
                nested.collapsed = False
                scroll_target = nested
        self.call_after_refresh(lambda target=scroll_target: target.scroll_visible(animate=False))

    def _focus_primary_content(self, *, prefer_expanded: bool = True) -> None:
        container = self._completion_scroll()
        if container is None:
            if rollout_list := self._rollout_list():
                rollout_list.focus()
            return
        sections = [child for child in container.children if isinstance(child, Collapsible)]
        if not sections:
            if rollout_list := self._rollout_list():
                rollout_list.focus()
            return
        target = sections[0]
        if prefer_expanded:
            target = next((section for section in sections if not section.collapsed), target)
        title_widget = next(iter(target.children), None)
        if title_widget is not None and getattr(title_widget, "can_focus", False):
            title_widget.focus()

    def _rollout_list(self) -> OptionList | None:
        try:
            return self.query_one("#viewer-rollout-list", OptionList)
        except NoMatches:
            return None

    def _completion_scroll(self) -> VerticalScroll | None:
        try:
            return self.query_one("#viewer-completion-scroll", VerticalScroll)
        except NoMatches:
            return None

    def _render_history_section_copy_text(
        self, section: HistorySectionData, *, depth: int = 0
    ) -> str:
        heading = f"{'#' * (depth + 2)} {section.title}"
        parts = [heading]
        body = [indent_block(section.body, "  ")] if section.body else []
        nested = [
            self._render_history_section_copy_text(child, depth=depth + 1)
            for child in section.nested_sections
        ]
        if section.body_first:
            parts.extend(body)
            parts.extend(nested)
        else:
            parts.extend(nested)
            parts.extend(body)
        return "\n\n".join(part for part in parts if part)

    def _render_history_copy_text(self, sections: list[HistorySectionData]) -> str:
        return "\n\n".join(self._render_history_section_copy_text(section) for section in sections)

    def _detail_copy_sections(self, record: dict[str, Any]) -> list[tuple[str, str, str]]:
        sections = [
            (
                "viewer-details-task",
                "Task",
                text_to_plain(build_task_text(record, self.metadata)),
            ),
            ("viewer-details-score", "Score", text_to_plain(build_score_text(record))),
            ("viewer-details-usage", "Usage", text_to_plain(build_usage_text(record))),
            (
                "viewer-details-state",
                "State",
                text_to_plain(build_state_text(record, self.metadata)),
            ),
        ]
        return [section for section in sections if section[2]]

    def _render_detail_copy_text(self, sections: list[tuple[str, str, str]]) -> str:
        return "\n\n".join(f"{label}\n{body}" for _, label, body in sections if body)

    def _append_history_copy_items(
        self,
        items: list[RolloutCopyItem],
        sections: list[HistorySectionData],
        *,
        depth: int = 0,
        prefix: str = "history",
    ) -> None:
        for idx, section in enumerate(sections, start=1):
            key = f"{prefix}:{idx}"
            indent = "  " * depth
            items.append(
                RolloutCopyItem(
                    key=key,
                    label=f"History: {indent}{section.title}",
                    body=self._render_history_section_copy_text(section),
                )
            )
            self._append_history_copy_items(
                items,
                list(section.nested_sections),
                depth=depth + 1,
                prefix=key,
            )

    def _build_rollout_snapshot_text(
        self,
        record: dict[str, Any],
        history_sections: list[HistorySectionData],
        detail_sections: list[tuple[str, str, str]],
    ) -> str:
        history_summary = text_to_plain(self._build_history_summary_text(record))
        history_text = self._render_history_copy_text(history_sections)
        history_parts = ["Completion History"]
        if history_summary:
            history_parts.append(history_summary)
        if history_text:
            history_parts.append(history_text)

        detail_text = self._render_detail_copy_text(detail_sections)
        blocks = [
            f"Current Rollout\n{build_rollout_prompt(self.current_record_idx, record).plain}",
            "\n\n".join(history_parts),
        ]
        if detail_text:
            blocks.append(f"Details\n\n{detail_text}")
        return "\n\n".join(block for block in blocks if block)

    def _build_rollout_copy_items(self, record: dict[str, Any]) -> list[RolloutCopyItem]:
        history_sections = self._history_section_data(record)
        detail_sections = self._detail_copy_sections(record)
        raw_text = pretty_json_or_str(record)
        history_text = self._render_history_copy_text(history_sections)
        items = [
            RolloutCopyItem(
                key="snapshot",
                label="Full rollout snapshot",
                body=self._build_rollout_snapshot_text(
                    record,
                    history_sections,
                    detail_sections,
                ),
            ),
            RolloutCopyItem(
                key="rollout",
                label="Rollout card",
                body=build_rollout_prompt(self.current_record_idx, record).plain,
            ),
        ]
        if history_text:
            items.append(
                RolloutCopyItem(
                    key="history",
                    label="Completion history",
                    body=history_text,
                )
            )
        detail_text = self._render_detail_copy_text(detail_sections)
        if detail_text:
            items.append(
                RolloutCopyItem(
                    key="details",
                    label="Details panel",
                    body=detail_text,
                )
            )
        for detail_id, label, body in detail_sections:
            items.append(
                RolloutCopyItem(
                    key=f"details:{detail_id}",
                    label=f"Details: {label}",
                    body=body,
                )
            )
        self._append_history_copy_items(items, history_sections)
        if raw_text:
            items.append(RolloutCopyItem(key="raw", label="Raw JSON", body=raw_text))
        return items


class HostedEvalSamplesScreen(Screen[None]):
    """Full-page hosted eval sample viewer."""

    BINDINGS = [
        Binding("escape", "back", "Back", key_display="Esc"),
        Binding("b", "back", "Back", key_display="B"),
    ]

    CSS = """
    HostedEvalSamplesScreen {
        background: $background;
        color: $foreground;
        layout: vertical;
    }

    #hosted-eval-samples-container {
        height: 100%;
        layout: vertical;
        padding: 0 1;
    }

    #hosted-eval-samples-summary {
        height: auto;
        min-height: 3;
        max-height: 5;
        padding: 0 1;
        border-bottom: solid $primary;
    }

    #hosted-eval-samples-body {
        height: 1fr;
    }
    """

    def __init__(
        self,
        item: LabItem,
        detail_loader: DetailLoader | None = None,
    ) -> None:
        super().__init__()
        self.item = item
        self._detail_loader = detail_loader
        self._load_error = ""

    def compose(self) -> ComposeResult:
        with Container(id="hosted-eval-samples-container"):
            yield Static("", id="hosted-eval-samples-summary", markup=False)
            with Container(id="hosted-eval-samples-body"):
                yield Static(Text("Loading samples ...", style="dim"), markup=False)
        yield Footer()

    def on_mount(self) -> None:
        if self._sample_records():
            self._render_samples()
            return
        if self._detail_loader is None:
            self._render_samples()
            return
        self._load_detail_worker()

    def action_back(self) -> None:
        self.app.pop_screen()

    @work(thread=True, exclusive=True)
    def _load_detail_worker(self) -> None:
        if self._detail_loader is None:
            return
        try:
            item = self._detail_loader(self.item, False, 50, 10, None)
        except Exception as exc:
            self.app.call_from_thread(self._set_load_error, str(exc))
            return
        self.app.call_from_thread(self._set_loaded_item, item)

    def _set_loaded_item(self, item: LabItem) -> None:
        self.item = item
        self._render_samples()

    def _set_load_error(self, message: str) -> None:
        self._load_error = message
        self._render_samples()

    def _sample_payload(self) -> dict[str, Any]:
        payload = self.item.raw.get("samples_preview")
        return payload if isinstance(payload, dict) else {}

    def _sample_records(self) -> list[dict[str, Any]]:
        samples = self._sample_payload().get("samples")
        if not isinstance(samples, list):
            return []
        return [sample for sample in samples if isinstance(sample, dict)]

    def _render_samples(self) -> None:
        self.query_one("#hosted-eval-samples-summary", Static).update(self._summary_text())
        body = self.query_one("#hosted-eval-samples-body", Container)
        body.remove_children()

        records = self._sample_records()
        if records:
            body.mount(
                RolloutViewer(
                    records,
                    metadata=self.item.raw.get("metadata")
                    if isinstance(self.item.raw.get("metadata"), dict)
                    else {},
                    title="Samples",
                    id="hosted-eval-rollout-viewer",
                )
            )
            self.call_after_refresh(lambda: self.query_one(RolloutViewer)._focus_primary_content())
            return

        error = self._load_error or str(self._sample_payload().get("error") or "")
        message = f"Failed to load samples: {error}" if error else "No samples found."
        body.mount(Static(Text(message, style=STATUS_ERROR if error else "dim"), markup=False))

    def _summary_text(self) -> Text:
        payload = self._sample_payload()
        total = payload.get("total")
        page = payload.get("page")
        limit = payload.get("limit")

        text = Text()
        text.append("Evaluation Samples\n", style="bold")
        text.append(self.item.title, style="bold")
        if self.item.subtitle:
            text.append("  ")
            text.append(self.item.subtitle, style="dim")
        text.append("\n")
        text.append(f"loaded {len(self._sample_records())}", style="dim")
        if total is not None:
            text.append(f" / {total}", style="dim")
        if page is not None and limit is not None:
            text.append(f"  page {page}  limit {limit}", style="dim")
        return text


class LocalEvalRunScreen(Screen[None]):
    """Full-page local eval run viewer with rollout and log panes."""

    COMPACT_LAYOUT_WIDTH = 150

    BINDINGS = [
        Binding("escape", "back", "Back", key_display="Esc"),
        Binding("b", "back", "Back", key_display="B"),
        Binding("p", "prev_record", "Prev rollout"),
        Binding("n", "next_record", "Next rollout"),
        Binding("l", "show_logs", "Logs"),
        Binding("r", "show_rollouts", "Rollouts"),
        Binding("pageup", "history_page_up", show=False),
        Binding("pagedown", "history_page_down", show=False),
        Binding("home", "history_home", show=False),
        Binding("end", "history_end", show=False),
        Binding("tab", "focus_next_pane", "Next pane", show=False),
        Binding("shift+tab", "focus_prev_pane", show=False),
        Binding("e", "expand_all", "Expand all", show=False),
        Binding("x", "collapse_all", "Collapse all", show=False),
        Binding("/", "search", "Search"),
        Binding("c", "copy", "Copy", show=False),
        Binding("y", "copy", "Copy"),
    ]

    CSS = """
    LocalEvalRunScreen {
        background: $background;
        color: $foreground;
        layout: vertical;
    }

    #eval-view-container {
        layout: vertical;
        height: 100%;
    }

    EvalPanel {
        border: round $primary;
        padding: 0 1;
        background: $surface;
    }

    .metadata-panel {
        height: auto;
        min-height: 6;
        max-height: 9;
    }

    .metadata-layout {
        height: auto;
        width: 100%;
    }

    .metadata-layout > Static {
        width: 1fr;
    }

    .view-columns {
        height: 1fr;
    }

    .rollouts-panel {
        width: 28;
        min-width: 24;
    }

    .history-panel,
    .logs-panel {
        width: 2fr;
        min-width: 50;
    }

    .details-panel {
        width: 1fr;
        min-width: 36;
    }

    .column-header {
        height: 1;
    }

    .subtitle {
        height: auto;
        color: $text-muted;
    }

    #rollout-list {
        height: 1fr;
    }

    #completion-scroll,
    #logs-scroll,
    .details-scroll {
        height: 1fr;
        background: $surface;
    }

    .history-section {
        margin-bottom: 1;
    }

    .nested-section {
        margin-left: 2;
    }

    .section-body,
    .log-content {
        padding: 0 1 1 1;
    }

    .logs-panel {
        display: none;
    }
    """

    def __init__(self, run: LocalEvalRun) -> None:
        super().__init__()
        self.run = run
        self.records = LazyRunResults(run)
        self._record_count = self.records.count_hint()
        self.current_record_idx = 0
        self._rollout_records = _RolloutRecordSequence(self.records)
        self._prompt_text = ""
        self._completion_text = ""
        self._highlight_regex: re.Pattern[str] | None = None
        self._highlight_column: str | None = None
        self._highlight_timer: Any = None
        self._highlight_section_index = 0
        self._highlight_nested_index = -1
        self._log_files: list[Path] = discover_log_files(run.path)
        self._log_loaders: dict[int, LazyLogFile] = {}
        self._merged_log_lines: list[str] | None = None
        self._active_log_tab = 0
        self._view_mode: Literal["rollouts", "logs"] = "rollouts"
        self._log_highlight_regex: re.Pattern[str] | None = None
        self._log_highlight_timer: Any = None
        if self._rollout_records:
            self._set_record_text_state(self._rollout_records[self.current_record_idx])

    def compose(self) -> ComposeResult:
        with Container(id="eval-view-container"):
            with EvalPanel(classes="metadata-panel"):
                with Horizontal(classes="metadata-layout"):
                    yield Static("", id="metadata-summary", markup=False)
                    yield Static("", id="metadata-metrics", markup=False)
                    yield Static("", id="metadata-reward", markup=False)
            with Horizontal(classes="view-columns"):
                yield RolloutViewer(
                    self._rollout_records,
                    metadata=self.run.load_metadata(),
                    title="Rollouts",
                    on_record_changed=self._handle_rollout_changed,
                    id="local-rollout-viewer",
                )
                with EvalPanel(id="logs-panel", classes="logs-panel"):
                    yield Label(
                        Text("Logs", style="bold"),
                        id="logs-header",
                        classes="column-header",
                    )
                    yield Static("", id="logs-tab-bar", classes="subtitle", markup=False)
                    yield LogScrollPane(id="logs-scroll")
        yield Footer()

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        if self._view_mode == "logs" and action in {
            "expand_all",
            "collapse_all",
            "show_logs",
            "toggle_markdown_math",
        }:
            return False
        if self._view_mode == "rollouts" and action == "show_rollouts":
            return False
        return True

    def on_mount(self) -> None:
        self.update_display()
        self.call_after_refresh(lambda: self._rollout_viewer()._focus_primary_content())

    def on_resize(self, event: events.Resize) -> None:
        self._update_responsive_layout(event.size.width)

    def on_unmount(self) -> None:
        self.records.close()
        for loader in self._log_loaders.values():
            loader.close()

    def action_back(self) -> None:
        self.app.pop_screen()

    def action_prev_record(self) -> None:
        if self._view_mode == "rollouts":
            self._rollout_viewer().action_prev_record()

    def action_next_record(self) -> None:
        if self._view_mode == "rollouts":
            self._rollout_viewer().action_next_record()

    def action_show_logs(self) -> None:
        if not self._log_files:
            self.notify("No log files available for this run", severity="warning")
            return
        if self._view_mode == "logs":
            return
        self._view_mode = "logs"
        self._rollout_viewer().display = False
        self.query_one("#logs-panel", EvalPanel).display = True
        self._populate_logs_view()
        self.query_one("#logs-scroll", LogScrollPane).focus()
        self.refresh_bindings()

    def action_show_rollouts(self) -> None:
        if self._view_mode == "rollouts":
            return
        self._view_mode = "rollouts"
        self.query_one("#logs-panel", EvalPanel).display = False
        self._rollout_viewer().display = True
        self._rollout_viewer()._focus_primary_content()
        self.refresh_bindings()

    def action_expand_all(self) -> None:
        if self._view_mode == "rollouts":
            self._rollout_viewer().action_expand_all()

    def action_collapse_all(self) -> None:
        if self._view_mode == "rollouts":
            self._rollout_viewer().action_collapse_all()

    def action_history_page_up(self) -> None:
        if target := self._center_scroll_target():
            target.scroll_page_up(animate=False)

    def action_history_page_down(self) -> None:
        if target := self._center_scroll_target():
            target.scroll_page_down(animate=False)

    def action_history_home(self) -> None:
        if target := self._center_scroll_target():
            target.scroll_home(animate=False)

    def action_history_end(self) -> None:
        if target := self._center_scroll_target():
            target.scroll_end(animate=False)

    def _should_skip_focus(self, widget: Widget) -> bool:
        if widget.id == "viewer-completion-scroll":
            return True
        if isinstance(widget, ContentTabs):
            return True
        node: DOMNode | None = widget.parent
        while node is not None:
            if isinstance(node, Widget) and not node.display:
                return True
            node = node.parent
        return False

    def action_focus_next_pane(self) -> None:
        starting = self.focused
        self.focus_next()
        first_candidate = self.focused
        while self.focused is not None and self.focused is not starting:
            if not self._should_skip_focus(self.focused):
                break
            self.focus_next()
            if self.focused is first_candidate:
                break

    def action_focus_prev_pane(self) -> None:
        starting = self.focused
        self.focus_previous()
        first_candidate = self.focused
        while self.focused is not None and self.focused is not starting:
            if not self._should_skip_focus(self.focused):
                break
            self.focus_previous()
            if self.focused is first_candidate:
                break

    def action_search(self) -> None:
        if self._view_mode == "logs":
            self._search_logs()
            return
        self._rollout_viewer().action_search()

    def action_copy(self) -> None:
        if self._view_mode == "logs":
            self._copy_logs()
            return
        self._rollout_viewer().action_copy()

    def action_toggle_markdown_math(self) -> None:
        if self._view_mode == "rollouts":
            self._rollout_viewer().action_toggle_markdown_math()

    def _rollout_viewer(self) -> RolloutViewer:
        return self.query_one("#local-rollout-viewer", RolloutViewer)

    def _handle_rollout_changed(self, index: int, record: dict[str, Any]) -> None:
        self.current_record_idx = index
        self._set_record_text_state(record)
        if self.is_mounted:
            self.update_display()

    def cycle_log_tab(self, delta: int) -> None:
        num_tabs = self._log_tab_count()
        if num_tabs < 2:
            return
        self._active_log_tab = (self._active_log_tab + delta) % num_tabs
        self._populate_logs_view()

    def _record_progress_label(self) -> str:
        total = "?" if self._record_count is None else str(self._record_count)
        return f"{self.current_record_idx + 1}/{total}"

    def _set_record_text_state(self, record: dict[str, Any]) -> None:
        prompt_text = format_prompt_or_completion(record.get("prompt", ""))
        completion_text = format_prompt_or_completion(record.get("completion", ""))
        error = record.get("error")
        if error is not None:
            completion_text.append("\n\n")
            completion_text.append("error: ", style=STATUS_ERROR)
            completion_text.append(str(error), style=STATUS_ERROR)
        self._prompt_text = prompt_text.plain
        self._completion_text = completion_text.plain

    def update_display(self, *, focus_history: bool = False) -> None:
        if not self._rollout_records:
            return
        record = self._rollout_records[self.current_record_idx]
        self._set_record_text_state(record)
        self.query_one("#metadata-summary", Static).update(
            build_run_summary_text(self.run, record_progress_label=self._record_progress_label())
        )
        self.query_one("#metadata-metrics", Static).update(build_run_metric_text(self.run))
        self.query_one("#metadata-reward", Static).update(
            build_reward_text(record, heading="Current Reward", multiline=False, limit=3)
        )

    def _center_scroll_target(self) -> VerticalScroll | None:
        if self._view_mode == "logs":
            return self.query_one("#logs-scroll", LogScrollPane)
        return self._rollout_viewer()._completion_scroll()

    def _update_responsive_layout(self, width: int) -> None:
        _ = width

    def _log_tab_count(self) -> int:
        if len(self._log_files) >= 2:
            return len(self._log_files) + 1
        return len(self._log_files)

    def _build_log_tab_bar(self) -> Text:
        num_tabs = self._log_tab_count()
        if num_tabs <= 1:
            return Text()
        text = Text()
        labels = ["all"] + [log_tab_label(path) for path in self._log_files]
        for idx, label in enumerate(labels):
            if idx:
                text.append("  ")
            text.append(f"[{label}]" if idx == self._active_log_tab else f" {label} ")
            if idx != self._active_log_tab:
                text.stylize("dim", len(text.plain) - len(label) - 2, len(text.plain))
        text.append("  Left/Right to switch", style="dim")
        return text

    def _get_active_log_lines(self) -> tuple[list[str], str]:
        is_merged = len(self._log_files) >= 2 and self._active_log_tab == 0
        if is_merged:
            if self._merged_log_lines is None:
                self._merged_log_lines = merge_log_files(self._log_files)
            return self._merged_log_lines, "all"
        file_idx = self._active_log_tab - 1 if len(self._log_files) >= 2 else self._active_log_tab
        if file_idx not in self._log_loaders:
            self._log_loaders[file_idx] = LazyLogFile(self._log_files[file_idx])
        loader = self._log_loaders[file_idx]
        line_count = len(loader)
        lines = [loader.get_line(i) for i in range(line_count)]
        return lines, log_tab_label(self._log_files[file_idx])

    def _populate_logs_view(self) -> None:
        if not self._log_files:
            self.query_one("#logs-tab-bar", Static).update(
                Text("No log files available", style="dim")
            )
            return
        self.query_one("#logs-tab-bar", Static).update(self._build_log_tab_bar())
        lines, log_name = self._get_active_log_lines()
        line_count = len(lines)
        self.query_one("#logs-header", Label).update(
            Text.assemble(
                ("Logs", "bold"),
                (f"  {log_name}", "dim"),
                (f"  ({line_count:,} lines)", "dim"),
            )
        )

        container = self.query_one("#logs-scroll", LogScrollPane)
        container.remove_children()
        if line_count == 0:
            container.mount(Static(Text("(empty log file)", style="dim"), markup=False))
            return

        text = Text()
        start = max(0, line_count - LazyLogFile.MAX_DISPLAY_LINES)
        if start > 0:
            text.append(f"... {start:,} earlier lines not shown ...\n\n", style="dim italic")
        for idx in range(start, line_count):
            if idx > start:
                text.append("\n")
            line = lines[idx]
            if self._log_highlight_regex:
                append_styled_log_line(text, line)
                offset = len(text.plain) - len(line)
                for match in self._log_highlight_regex.finditer(line):
                    text.stylize("reverse", offset + match.start(), offset + match.end())
            else:
                append_styled_log_line(text, line)
        container.mount(Static(text, markup=False, classes="log-content"))

    def _search_logs(self) -> None:
        if not self._log_files:
            return
        lines, _ = self._get_active_log_lines()
        line_count = len(lines)
        start = max(0, line_count - LazyLogFile.MAX_DISPLAY_LINES)
        log_lines = [(0, -1, lines[idx]) for idx in range(start, line_count)]
        self.app.push_screen(EvalSearchScreen([], log_lines), self._handle_log_search_result)

    def _handle_log_search_result(self, result: SearchResult | None) -> None:
        if self._log_highlight_timer is not None:
            self._log_highlight_timer.stop()
            self._log_highlight_timer = None
        self._log_highlight_regex = None
        if result is not None:
            try:
                self._log_highlight_regex = re.compile(result.pattern, re.IGNORECASE)
            except re.error:
                return
            self._log_highlight_timer = self.set_timer(3.0, self._clear_log_highlight)
        self._populate_logs_view()
        if result is not None and self._log_highlight_regex is not None:
            self._scroll_to_first_log_match()

    def _scroll_to_first_log_match(self) -> None:
        if not self._log_highlight_regex or not self._log_files:
            return
        lines, _ = self._get_active_log_lines()
        line_count = len(lines)
        start = max(0, line_count - LazyLogFile.MAX_DISPLAY_LINES)
        first_match_display_idx: int | None = None
        for idx in range(start, line_count):
            if self._log_highlight_regex.search(lines[idx]):
                first_match_display_idx = idx - start
                break
        if first_match_display_idx is None:
            return

        offset_lines = 2 if start > 0 else 0
        target_line = first_match_display_idx + offset_lines
        container = self.query_one("#logs-scroll", LogScrollPane)

        def do_scroll() -> None:
            content_height = container.virtual_size.height
            visible_height = container.size.height
            total_lines = (line_count - start) + offset_lines
            if total_lines <= 0 or content_height <= visible_height:
                return
            fraction = target_line / total_lines
            target_y = int(fraction * content_height)
            container.scroll_to(y=max(0, target_y - visible_height // 2), animate=False)

        self.call_after_refresh(do_scroll)

    def _clear_log_highlight(self) -> None:
        self._log_highlight_regex = None
        self._log_highlight_timer = None
        if self._view_mode == "logs":
            self._populate_logs_view()

    def _copy_logs(self) -> None:
        if not self._log_files:
            return
        items: list[RolloutCopyItem] = []
        has_merged = len(self._log_files) >= 2
        if has_merged:
            if self._merged_log_lines is None:
                self._merged_log_lines = merge_log_files(self._log_files)
            items.append(
                RolloutCopyItem(
                    key="log-all",
                    label="Log: all (merged)",
                    body="\n".join(self._merged_log_lines),
                )
            )
        for idx, path in enumerate(self._log_files):
            items.append(
                RolloutCopyItem(
                    key=f"log-file-{idx}",
                    label=f"Log: {log_tab_label(path)}",
                    body=path.read_text(encoding="utf-8", errors="replace"),
                )
            )
        if has_merged and self._active_log_tab == 0:
            current_key = "log-all"
        else:
            file_idx = self._active_log_tab - 1 if has_merged else self._active_log_tab
            current_key = f"log-file-{file_idx}"
        self.app.push_screen(RolloutCopyScreen(items, start_key=current_key, title="Copy Logs"))

    def overview_renderables(self) -> list[Any]:
        """Return reusable overview renderables for this run."""

        stats = compute_run_overview_stats(self.run)
        return [
            build_reward_distribution_table(stats.rewards, "Reward distribution"),
            build_metric_summary_table(stats.metric_summaries),
        ]


def stylize_matches(text: Text, pattern: re.Pattern[str], style: str) -> Text:
    for match in pattern.finditer(text.plain):
        text.stylize(style, match.start(), match.end())
    return text
