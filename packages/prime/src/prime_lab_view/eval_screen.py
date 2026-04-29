"""Textual widgets for local eval rollout viewing."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal

from rich.text import Text
from textual import events, on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, VerticalScroll
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
    build_info_text,
    build_metric_summary_table,
    build_reward_distribution_table,
    build_reward_text,
    build_rollout_prompt,
    build_run_metric_text,
    build_run_summary_text,
    build_score_text,
    build_task_text,
    build_usage_text,
    compute_run_overview_stats,
    format_message_preview,
    format_prompt_or_completion,
    history_groups,
    indent_block,
    stringify_message,
    stringify_message_content,
    stringify_message_reasoning,
    text_to_plain,
    tool_call_parts,
    tool_group_preview,
    tool_output_preview,
)


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
            yield Input(placeholder="regex, case-insensitive", id="search-input")
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
        self.query_one("#search-input", Input).focus()
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
        pattern = self.query_one("#search-input", Input).value
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
            error_label.update(Text(f"Invalid regex: {exc}", style="red"))
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
        Binding("b,backspace", "close", show=False),
        Binding("c", "copy", "Copy"),
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


class RolloutViewer(Container):
    """Reusable rollout transcript viewer for evals and training samples."""

    BINDINGS = [
        Binding("p", "prev_record", "Prev rollout"),
        Binding("n", "next_record", "Next rollout"),
        Binding("pageup", "history_page_up", show=False),
        Binding("pagedown", "history_page_down", show=False),
        Binding("home", "history_home", show=False),
        Binding("end", "history_end", show=False),
        Binding("e", "expand_all", "Expand all"),
        Binding("x", "collapse_all", "Collapse all"),
        Binding("/", "search", "Search"),
        Binding("c", "copy", "Copy"),
    ]

    DEFAULT_CSS = """
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
        width: 28;
        min-width: 24;
    }

    RolloutViewer .history-panel {
        width: 2fr;
        min-width: 50;
    }

    RolloutViewer .details-panel {
        width: 1fr;
        min-width: 36;
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
    }

    RolloutViewer .history-section {
        margin-bottom: 1;
    }

    RolloutViewer .nested-section {
        margin-left: 2;
    }

    RolloutViewer .section-body {
        padding: 0 1 1 1;
    }
    """

    def __init__(
        self,
        records: Sequence[dict[str, Any]],
        *,
        metadata: dict[str, Any] | None = None,
        title: str = "Rollouts",
        on_record_changed: Callable[[int, dict[str, Any]], None] | None = None,
        classes: str | None = None,
        id: str | None = None,
    ) -> None:
        super().__init__(id=id, classes=classes)
        self.records = [record for record in records if isinstance(record, dict)]
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
                with TabPane("Info", id="viewer-details-info"):
                    yield TabbedScrollPane(
                        Static("", id="viewer-info-content", markup=False),
                        classes="details-scroll",
                    )

    def on_mount(self) -> None:
        if not self.records:
            return
        self._populate_rollout_list()
        if self._on_record_changed is not None:
            self._on_record_changed(self.current_record_idx, self.records[self.current_record_idx])
        self.update_display()
        self.call_after_refresh(self._focus_primary_content)

    def action_prev_record(self) -> None:
        self._move_record_cursor(-1)

    def action_next_record(self) -> None:
        self._move_record_cursor(1)

    def action_expand_all(self) -> None:
        if not self.records:
            return
        container = self.query_one("#viewer-completion-scroll", VerticalScroll)
        for section in container.query(Collapsible):
            section.collapsed = False
        self._focus_primary_content()

    def action_collapse_all(self) -> None:
        if not self.records:
            return
        container = self.query_one("#viewer-completion-scroll", VerticalScroll)
        for section in container.query(Collapsible):
            section.collapsed = True
        self._focus_primary_content(prefer_expanded=False)

    def action_history_page_up(self) -> None:
        if self.records:
            self.query_one("#viewer-completion-scroll", VerticalScroll).scroll_page_up(
                animate=False
            )

    def action_history_page_down(self) -> None:
        if self.records:
            self.query_one("#viewer-completion-scroll", VerticalScroll).scroll_page_down(
                animate=False
            )

    def action_history_home(self) -> None:
        if self.records:
            self.query_one("#viewer-completion-scroll", VerticalScroll).scroll_home(animate=False)

    def action_history_end(self) -> None:
        if self.records:
            self.query_one("#viewer-completion-scroll", VerticalScroll).scroll_end(animate=False)

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
        rollout_list = self.query_one("#viewer-rollout-list", OptionList)
        rollout_list.clear_options()
        for idx, record in enumerate(self.records):
            rollout_list.add_option(Option(build_rollout_prompt(idx, record), id=str(idx)))
        rollout_list.highlighted = self.current_record_idx
        rollout_list.scroll_to_highlight()

    def _move_record_cursor(self, delta: int) -> None:
        if not self.records:
            return
        new_index = (self.current_record_idx + delta) % len(self.records)
        rollout_list = self.query_one("#viewer-rollout-list", OptionList)
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
        self.query_one("#viewer-completion-scroll", VerticalScroll).scroll_y = 0
        for scroll in self.query(".details-scroll"):
            if isinstance(scroll, VerticalScroll):
                scroll.scroll_y = 0

    def _set_record_text_state(self, record: dict[str, Any]) -> None:
        prompt_text = format_prompt_or_completion(record.get("prompt", ""))
        completion_text = format_prompt_or_completion(record.get("completion", ""))
        error = record.get("error")
        if error is not None:
            completion_text.append("\n\n")
            completion_text.append("error: ", style="bold red")
            completion_text.append(str(error), style="red")
        self._prompt_text = prompt_text.plain
        self._completion_text = completion_text.plain

    def update_display(self, *, focus_history: bool = False) -> None:
        if not self.records:
            return
        record = self.records[self.current_record_idx]
        self._set_record_text_state(record)
        self.query_one("#viewer-history-summary", Static).update(
            self._build_history_summary_text(record)
        )
        self.query_one("#viewer-task-content", Static).update(build_task_text(record))
        self.query_one("#viewer-score-content", Static).update(build_score_text(record))
        self.query_one("#viewer-usage-content", Static).update(build_usage_text(record))
        self.query_one("#viewer-info-content", Static).update(
            build_info_text(record, self.metadata)
        )
        self.query_one("#viewer-rollout-summary", Label).update(
            self._build_rollout_summary_text(record)
        )
        self._rebuild_completion_sections(record, focus_history)

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
        container = self.query_one("#viewer-completion-scroll", VerticalScroll)
        container.remove_children()
        container.mount(*self._completion_sections(record))
        if focus_history:
            self.call_after_refresh(self._focus_primary_content)

    def _history_section_data(self, record: dict[str, Any]) -> list[HistorySectionData]:
        sections = [
            HistorySectionData(
                title="Initial Prompt",
                body=self._prompt_text,
                column="prompt",
                collapsed=True,
                classes="history-section prompt-section",
            )
        ]
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
                        body=stringify_message_content(message.get("content", "")).strip(),
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
            body = stringify_message_content(message.get("content", "")).strip()
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
                        collapsed=tool_idx > 1,
                        classes="history-section tool-call-section nested-section",
                    )
                )
            sections.append(
                HistorySectionData(
                    title=title,
                    body=body,
                    column="completion",
                    collapsed=True,
                    classes="history-section assistant-section",
                    nested_sections=tuple(nested_sections),
                    body_first=False if nested_sections else True,
                )
            )
        return sections

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

    def _make_section(self, section: HistorySectionData) -> Collapsible:
        collapsed = section.collapsed
        if self._section_matches_highlight(section):
            collapsed = False
        body_children: list[Widget] = []
        if section.body or not section.nested_sections:
            text = Text(section.body)
            if self._highlight_regex and self._highlight_column == section.column:
                stylize_matches(text, self._highlight_regex, "reverse")
            body_children.append(Static(text, classes="section-body", markup=False))
        nested_children = [self._make_section(child) for child in section.nested_sections]
        children = (
            [*body_children, *nested_children]
            if section.body_first
            else [*nested_children, *body_children]
        )
        return Collapsible(
            *children,
            title=section.title,
            collapsed=collapsed,
            classes=section.classes,
        )

    def _section_matches_highlight(self, section: HistorySectionData) -> bool:
        if not (self._highlight_regex and self._highlight_column == section.column):
            return False
        if self._highlight_regex.search(section.title) or self._highlight_regex.search(
            section.body
        ):
            return True
        return any(self._section_matches_highlight(child) for child in section.nested_sections)

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
            (str(reward), "bold"),
        )

    def _build_search_lines(
        self, record: dict[str, Any]
    ) -> tuple[list[tuple[int, int, str]], list[tuple[int, int, str]]]:
        sections = self._history_section_data(record)
        prompt_lines: list[tuple[int, int, str]] = []
        completion_lines: list[tuple[int, int, str]] = []
        for idx, section in enumerate(sections):
            target = prompt_lines if section.column == "prompt" else completion_lines
            for line in section.body.splitlines():
                target.append((idx, -1, line))
            for nested_idx, nested in enumerate(section.nested_sections):
                nested_target = prompt_lines if nested.column == "prompt" else completion_lines
                for line in nested.body.splitlines():
                    nested_target.append((idx, nested_idx, line))
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
                container = self.query_one("#viewer-completion-scroll", VerticalScroll)
                self._expand_and_scroll_to_match(container)

    def _swap_section_bodies(self) -> None:
        if not (self.records and self.is_mounted):
            return
        record = self.records[self.current_record_idx]
        self._rebuild_completion_sections(record)

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
        container = self.query_one("#viewer-completion-scroll", VerticalScroll)
        sections = [child for child in container.children if isinstance(child, Collapsible)]
        if not sections:
            self.query_one("#viewer-rollout-list", OptionList).focus()
            return
        target = sections[0]
        if prefer_expanded:
            target = next((section for section in sections if not section.collapsed), target)
        title_widget = next(iter(target.children), None)
        if title_widget is not None and getattr(title_widget, "can_focus", False):
            title_widget.focus()

    def _render_history_section_copy_text(
        self, section: HistorySectionData, *, depth: int = 0
    ) -> str:
        heading = f"{'#' * (depth + 2)} {section.title}"
        parts = [heading]
        if section.body:
            parts.append(indent_block(section.body, "  "))
        parts.extend(
            self._render_history_section_copy_text(child, depth=depth + 1)
            for child in section.nested_sections
        )
        return "\n\n".join(part for part in parts if part)

    def _render_history_copy_text(self, sections: list[HistorySectionData]) -> str:
        return "\n\n".join(self._render_history_section_copy_text(section) for section in sections)

    def _build_rollout_copy_items(self, record: dict[str, Any]) -> list[RolloutCopyItem]:
        history_sections = self._history_section_data(record)
        raw_text = text_to_plain(format_prompt_or_completion(record))
        history_text = self._render_history_copy_text(history_sections)
        snapshot_parts = [
            f"Current Rollout\n{build_rollout_prompt(self.current_record_idx, record).plain}",
            f"Completion History\n\n{history_text}" if history_text else "",
            f"Raw\n\n{raw_text}" if raw_text else "",
        ]
        items = [
            RolloutCopyItem(
                key="snapshot",
                label="Full rollout snapshot",
                body="\n\n".join(part for part in snapshot_parts if part).strip(),
            ),
            RolloutCopyItem(
                key="rollout",
                label="Rollout row",
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
        if raw_text:
            items.append(RolloutCopyItem(key="raw", label="Raw JSON", body=raw_text))
        return items


class LocalEvalRunScreen(Screen[None]):
    """Full-page local eval run viewer with rollout and log panes."""

    COMPACT_LAYOUT_WIDTH = 150

    BINDINGS = [
        Binding("b,backspace", "back", "Back"),
        Binding("q", "quit", "Quit"),
        Binding("p", "prev_record", "Prev rollout"),
        Binding("n", "next_record", "Next rollout"),
        Binding("l", "show_logs", "Logs"),
        Binding("r", "show_rollouts", "Rollouts"),
        Binding("pageup", "history_page_up", show=False),
        Binding("pagedown", "history_page_down", show=False),
        Binding("home", "history_home", show=False),
        Binding("end", "history_end", show=False),
        Binding("tab", "focus_next_pane", "Next pane"),
        Binding("shift+tab", "focus_prev_pane", show=False),
        Binding("e", "expand_all", "Expand all"),
        Binding("x", "collapse_all", "Collapse all"),
        Binding("/", "search", "Search"),
        Binding("c", "copy", "Copy"),
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
        self._rollout_records = (
            [self.records[idx] for idx in range(len(self.records))] if self.records else []
        )
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
        self._center_scroll_target().scroll_page_up(animate=False)

    def action_history_page_down(self) -> None:
        self._center_scroll_target().scroll_page_down(animate=False)

    def action_history_home(self) -> None:
        self._center_scroll_target().scroll_home(animate=False)

    def action_history_end(self) -> None:
        self._center_scroll_target().scroll_end(animate=False)

    def action_focus_next_pane(self) -> None:
        self.focus_next()

    def action_focus_prev_pane(self) -> None:
        self.focus_previous()

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
            completion_text.append("error: ", style="bold red")
            completion_text.append(str(error), style="red")
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

    def _center_scroll_target(self) -> VerticalScroll:
        if self._view_mode == "logs":
            return self.query_one("#logs-scroll", LogScrollPane)
        return self.query_one("#viewer-completion-scroll", VerticalScroll)

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
