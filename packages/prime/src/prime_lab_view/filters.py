"""Reusable filter and picker primitives for Lab TUI screens."""

from __future__ import annotations

from dataclasses import dataclass

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container
from textual.screen import ModalScreen
from textual.widgets import Input, Label, OptionList, Static
from textual.widgets._option_list import Option


@dataclass(frozen=True)
class FilterChoice:
    key: str
    label: str | Text
    search_text: str
    value: str


class FilterScreen(ModalScreen[str | None]):
    """Modal filter over a pre-populated option set."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
    ]

    CSS = """
    FilterScreen {
        align: center middle;
    }

    #filter-dialog {
        border: round $primary;
        padding: 1 2;
        background: $surface;
    }

    #filter-query {
        height: 3;
    }

    #filter-results {
        height: 1fr;
        background: $surface;
    }

    #filter-actions {
        height: 1;
        color: $text-muted;
    }
    """

    def __init__(
        self,
        choices: list[FilterChoice],
        *,
        title: str = "Filter",
        placeholder: str = "Filter rows",
        initial: str = "",
        actions_label: str = "/  Type filter    Enter  Apply    Esc  Cancel",
        width: int = 72,
        height: int = 24,
    ) -> None:
        super().__init__()
        self._choices = choices
        self._title = title
        self._placeholder = placeholder
        self._query = initial
        self._actions_label = actions_label
        self._width = width
        self._height = height

    def compose(self) -> ComposeResult:
        with Container(id="filter-dialog"):
            yield Label(self._title)
            yield Input(
                value=self._query,
                placeholder=self._placeholder,
                id="filter-query",
            )
            yield OptionList(id="filter-results")
            yield Static(self._actions_label, id="filter-actions", markup=False)

    def on_mount(self) -> None:
        dialog = self.query_one("#filter-dialog", Container)
        dialog.styles.width = self._width
        dialog.styles.height = self._height
        self._sync_results()
        query = self.query_one("#filter-query", Input)
        query.focus()
        query.cursor_position = len(query.value)

    @on(Input.Changed, "#filter-query")
    def _query_changed(self, event: Input.Changed) -> None:
        self._query = event.value.strip().lower()
        self._sync_results()

    @on(Input.Submitted, "#filter-query")
    def _submit(self, event: Input.Submitted) -> None:
        self.dismiss(event.value.strip() or None)

    @on(OptionList.OptionSelected, "#filter-results")
    def _result_selected(self, event: OptionList.OptionSelected) -> None:
        key = event.option.id
        if key is None:
            return
        choice = next((choice for choice in self._choices if choice.key == str(key)), None)
        if choice is not None:
            self.dismiss(choice.value)

    def _sync_results(self) -> None:
        option_list = self.query_one("#filter-results", OptionList)
        option_list.clear_options()
        visible = filter_choices(self._choices, self._query)
        if not visible:
            option_list.add_option(Option("No matching rows", disabled=True))
            option_list.highlighted = 0
            return
        for choice in visible:
            option_list.add_option(Option(choice.label, id=choice.key))
        option_list.highlighted = 0

    def action_cancel(self) -> None:
        self.dismiss(None)


def filter_choices(choices: list[FilterChoice], query: str) -> list[FilterChoice]:
    if not query:
        return choices
    return [choice for choice in choices if query in choice.search_text]
