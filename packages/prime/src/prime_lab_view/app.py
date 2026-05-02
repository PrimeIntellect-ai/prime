"""Textual application for `prime lab`."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from rich.console import Group
from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.css.query import NoMatches
from textual.widgets import Button, Footer, Label, OptionList, Static, Tree
from textual.widgets._option_list import Option

from .agent_runtime import AgentChatMessage, AgentConnectionState, AgentRuntime
from .agent_screen import AgentChatScreen
from .config_screen import ConfigLaunchScreen, ConfigRunScreen
from .data import LabLoadOptions
from .details import (
    evaluation_detail_chunks as _evaluation_detail_chunks,
)
from .details import (
    item_details as _item_details,
)
from .environment_screen import (
    AddWorkspaceScreen,
    EnvironmentScreen,
    WorkspaceScreen,
)
from .eval_records import LocalEvalRun, RunOverviewStats
from .eval_render import (
    compute_run_overview_stats,
    format_reward_value,
    numeric_reward,
    reward_style,
)
from .eval_screen import LocalEvalRunScreen
from .evaluation_browser import (
    EVALUATION_VIEWS,
    evaluation_env_tree_label,
    evaluation_group_selection_details,
    evaluation_index,
    evaluation_model_tree_label,
    evaluation_reward,
    evaluation_run_id,
    evaluation_run_selection_details,
    evaluation_run_tree_label,
    local_eval_stats_key,
    sorted_evaluation_runs,
)
from .filters import FilterChoice, FilterScreen
from .home import (
    home_action_label,
    workspace_action_items,
    workspace_content_items,
    workspace_home_groups,
)
from .launch_screen import LaunchScreen
from .logs import (
    parse_log_records,
)
from .models import LabItem, LabSection, LabSnapshot
from .palette import (
    BUTTON_CSS,
    LAB_THEME,
    STATUS_ERROR,
)
from .quickstart import (
    build_environment_item,
    evaluation_config_item,
    training_config_item,
)
from .rows import filter_choice_for_item, item_label, item_search_text
from .setup_screens import AgentSyncScreen, DoctorScreen, SetupScreen
from .shell import (
    agent_status_label,
    compact_path,
    configured_workspace_agent,
    lab_header,
    statusbar_text,
    write_workspace_agent_choice,
)
from .snapshots import merge_snapshot_rows
from .training_screen import (
    _LOG_TAIL_STEPS,
    _RUN_METRIC_PREVIEW_LIMIT,
    DetailLoader,
    TrainingRunScreen,
    _has_training_detail,
    _merge_training_detail,
    _with_metric_load_seconds,
)
from .widgets import (
    EvaluationNodeData,
    EvaluationTree,
    EvaluationViewToggle,
    HomeGroupToggle,
    HomeLaunchState,
    LabInspector,
    LabOptionList,
)

WorkspaceSwitcher = Callable[[Path], None]


class PrimeLabView(App[None]):
    """Lab terminal viewer."""

    ENABLE_COMMAND_PALETTE = False
    PRIME_THEME = LAB_THEME

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("r", "refresh", "Refresh"),
        Binding("w", "show_welcome", "Welcome"),
        Binding("enter", "load_detail", "Open", key_display="Enter"),
        Binding("g", "load_more_rows", "More rows"),
        Binding("/", "search", "Filter"),
        Binding("left", "previous_pane", "Prev pane", key_display="Left"),
        Binding("right", "next_pane", "Next pane", key_display="Right"),
        Binding("escape", "clear_filter", "Clear filter", key_display="Esc"),
        Binding("tab", "focus_next", "Next pane", key_display="Tab"),
        Binding("shift+tab", "focus_previous", "Prev pane", key_display="Shift+Tab"),
    ]

    CSS = (
        BUTTON_CSS
        + """
    Screen {
        background: $background;
        color: $foreground;
        layout: vertical;
    }

    #topbar {
        height: 2;
        padding: 0 1;
        border-bottom: solid $primary;
        background: $background;
    }

    #body {
        height: 1fr;
    }

    .pane {
        border: round $primary;
        background: $surface;
        padding: 0 1;
    }

    #nav-pane {
        width: 28;
        min-width: 24;
    }

    #list-pane {
        width: 2fr;
        min-width: 42;
    }

    #inspector-pane {
        width: 1.35fr;
        min-width: 36;
    }

    .pane-title {
        text-style: bold;
        color: $foreground;
        height: 1;
    }

    .pane-subtitle {
        color: $text-muted;
        height: 1;
    }

    Tree {
        background: $surface;
        height: 1fr;
    }

    OptionList {
        background: $surface;
        height: 1fr;
    }

    #home-toggle {
        display: none;
        height: 1;
        color: $foreground;
    }

    #home-actions {
        display: none;
        height: auto;
        margin: 1 0;
    }

    .home-action-button {
        width: 1fr;
        margin-right: 1;
    }

    #evaluation-toggle {
        display: none;
        height: 1;
        color: $foreground;
    }

    #evaluation-tree {
        display: none;
        background: $surface;
        height: 1fr;
    }

    OptionList > .option-list--option-highlighted {
        background: $primary 20%;
    }

    #inspector {
        height: 1fr;
        overflow-y: auto;
    }

    #statusbar {
        height: 1;
        padding: 0 1;
        background: $surface;
        border-top: solid $primary;
    }
    """
    )

    def __init__(
        self,
        loader: Callable[[], LabSnapshot],
        detail_loader: DetailLoader | None = None,
        initial_loader: Callable[[], LabSnapshot] | None = None,
        ladder_loader: Callable[[int], LabSnapshot] | None = None,
        ladder_limits: tuple[int, ...] = (),
        workspace_switcher: WorkspaceSwitcher | None = None,
    ):
        super().__init__()
        self._loader = loader
        self._detail_loader = detail_loader
        self._initial_loader = initial_loader
        self._ladder_loader = ladder_loader
        self._ladder_limits = ladder_limits
        self._workspace_switcher = workspace_switcher
        self._loaded_section_limit = 0
        self._requested_section_limit = max(ladder_limits, default=0)
        self._snapshot: LabSnapshot | None = None
        self._active_section_key = "workspace"
        self._filter = ""
        self._visible_items: list[LabItem] = []
        self._items_by_key: dict[str, LabItem] = {}
        self._selected_item: LabItem | None = None
        self._detail_cache: dict[str, LabItem] = {}
        self._prefetching_detail_key: str | None = None
        self._expand_ready_run_key: str | None = None
        self._home_group = "workspaces"
        self._launch_screen: LaunchScreen | None = None
        self._evaluation_view = "runs"
        self._evaluation_tree_index: dict[str, dict[str, list[LabItem]]] = {}
        self._evaluation_click_selected_node: object | None = None
        self._local_eval_stats_cache: dict[str, RunOverviewStats] = {}
        self._loading_local_eval_stats: set[str] = set()
        self._home_action_items_by_id: dict[str, LabItem] = {}
        self._home_action_render_id = 0
        self._agent_runtime = AgentRuntime(
            on_state=lambda state: self._dispatch_agent_callback(self._set_agent_state, state),
            on_messages=lambda messages: self._dispatch_agent_callback(
                self._set_agent_messages, messages
            ),
        )
        self._agent_state = AgentConnectionState()
        self._agent_messages: tuple[AgentChatMessage, ...] = ()

    def compose(self) -> ComposeResult:
        yield Static("Loading Lab ...", id="topbar", markup=False)
        with Horizontal(id="body"):
            with Vertical(id="nav-pane", classes="pane"):
                yield Tree("Sections", id="section-tree")
            with Vertical(id="list-pane", classes="pane"):
                yield Label("Loading", id="section-title", classes="pane-title")
                yield Static("", id="section-subtitle", classes="pane-subtitle", markup=False)
                yield HomeGroupToggle("", id="home-toggle", markup=False)
                yield Horizontal(id="home-actions", classes="home-actions")
                yield EvaluationViewToggle("", id="evaluation-toggle", markup=False)
                yield LabOptionList(id="item-list")
                yield EvaluationTree("Evaluations", id="evaluation-tree")
            with Vertical(id="inspector-pane", classes="pane"):
                yield Label("Inspector", id="inspector-title", classes="pane-title")
                yield LabInspector("", id="inspector", markup=False)
        yield Static("", id="statusbar", markup=False)
        yield Footer()

    def on_mount(self) -> None:
        self.register_theme(self.PRIME_THEME)
        self.theme = "prime-lab"
        self.query_one("#section-tree", Tree).show_root = False
        evaluation_tree = self.query_one("#evaluation-tree", EvaluationTree)
        evaluation_tree.show_root = False
        evaluation_tree.auto_expand = False
        evaluation_tree.guide_depth = 2
        self._open_launch_screen()
        self._show_initial_snapshot()
        self._reload()

    def on_unmount(self) -> None:
        self._agent_runtime.stop()

    def _open_launch_screen(self) -> None:
        screen = LaunchScreen()
        self._launch_screen = screen
        self.push_screen(screen, self._handle_launch_result)
        self._sync_launch_screen()

    def _handle_launch_result(self, target: str | None) -> None:
        self._launch_screen = None
        if target in {None, "home"}:
            self._focus_main_pane(prefer_rows=False)
            return
        self._open_launch_target(target)

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        if self._launch_screen_active() and action in {
            "load_more_rows",
            "search",
            "previous_pane",
            "next_pane",
            "clear_filter",
        }:
            return False
        if isinstance(
            self.screen,
            EnvironmentScreen
            | WorkspaceScreen
            | AddWorkspaceScreen
            | AgentSyncScreen
            | DoctorScreen
            | SetupScreen
            | ConfigRunScreen
            | AgentChatScreen
            | TrainingRunScreen
            | FilterScreen
            | LocalEvalRunScreen,
        ) and action in {
            "refresh",
            "load_detail",
            "load_logs",
            "load_more_rows",
            "search",
            "previous_pane",
            "next_pane",
            "clear_filter",
        }:
            return False
        if isinstance(self.focused, LabInspector) and action in {
            "refresh",
            "load_detail",
            "load_more_rows",
            "search",
            "clear_filter",
        }:
            return False
        if action in {"search", "load_more_rows"} and isinstance(
            self.screen, TrainingRunScreen | FilterScreen | LocalEvalRunScreen
        ):
            return False
        if action == "load_detail":
            return self._main_open_action_available()
        if action == "clear_filter":
            return bool(self._filter)
        if action == "load_more_rows":
            return self._ladder_loader is not None and self._requested_section_limit > 0
        return True

    def action_refresh(self) -> None:
        self._loaded_section_limit = 0
        self._show_initial_snapshot()
        self._reload()

    def action_show_welcome(self) -> None:
        if self._launch_screen_active():
            return
        if isinstance(self.screen, ConfigLaunchScreen):
            return
        if _is_lab_child_screen(self.screen):
            self.pop_screen()
            self.call_after_refresh(self._open_launch_screen)
            return
        self._open_launch_screen()

    def action_clear_filter(self) -> None:
        if not self._filter:
            return
        self._filter = ""
        self._render_active_section()

    def action_previous_pane(self) -> None:
        if self._dismiss_launch_screen():
            return
        focused = self.focused
        if isinstance(focused, LabInspector):
            self._focus_main_pane(prefer_rows=True)
            return
        self.focus_nav_pane()

    def action_next_pane(self) -> None:
        if self._dismiss_launch_screen():
            self._focus_main_pane(prefer_rows=False)
            return
        focused = self.focused
        if isinstance(focused, Tree):
            self._focus_main_pane(prefer_rows=False)
            return
        if isinstance(focused, LabOptionList | HomeGroupToggle | EvaluationViewToggle | Button):
            self._focus_inspector_pane()
            return
        self._focus_main_pane(prefer_rows=False)

    def action_back_from_list(self) -> None:
        if self._dismiss_launch_screen():
            return
        if self._active_section_key == "workspace":
            toggle = self.query_one("#home-toggle", HomeGroupToggle)
            if toggle.display:
                toggle.focus()
                return
        if self._active_section_key == "evaluations":
            toggle = self.query_one("#evaluation-toggle", EvaluationViewToggle)
            if toggle.display:
                toggle.focus()
                return
        self.focus_nav_pane()

    def focus_nav_pane(self) -> None:
        self.query_one("#section-tree", Tree).focus()

    def focus_home_rows(self) -> None:
        self._focus_main_pane(prefer_rows=True)

    def focus_evaluation_rows(self) -> None:
        self._focus_main_pane(prefer_rows=True)

    def action_previous_home_group(self) -> None:
        self._dismiss_launch_screen()
        self._switch_home_group(-1)

    def action_next_home_group(self) -> None:
        self._dismiss_launch_screen()
        self._switch_home_group(1)

    def _switch_home_group(self, direction: int) -> None:
        groups = workspace_home_groups(self._workspace_items())
        if not groups:
            return
        keys = [key for key, _, items in groups if items]
        if not keys:
            return
        try:
            index = keys.index(self._home_group)
        except ValueError:
            index = 0
        self._home_group = keys[(index + direction) % len(keys)]
        self._render_active_section()

    def set_home_group(self, group: str) -> None:
        groups = workspace_home_groups(self._workspace_items())
        if group not in {key for key, _, items in groups if items}:
            return
        self._dismiss_launch_screen()
        if group == self._home_group:
            return
        self._home_group = group
        self._render_active_section()

    def set_segmented_toggle(self, toggle_id: str, key: str) -> None:
        if toggle_id == "home-toggle":
            self.set_home_group(key)
            self.focus_home_rows()
        elif toggle_id == "evaluation-toggle":
            self.set_evaluation_view(key)
            self.focus_evaluation_rows()

    def action_previous_evaluation_view(self) -> None:
        self._switch_evaluation_view(-1)

    def action_next_evaluation_view(self) -> None:
        self._switch_evaluation_view(1)

    def _switch_evaluation_view(self, direction: int) -> None:
        keys = [key for key, _ in EVALUATION_VIEWS]
        try:
            index = keys.index(self._evaluation_view)
        except ValueError:
            index = 0
        self._evaluation_view = keys[(index + direction) % len(keys)]
        self._evaluation_click_selected_node = None
        self._render_active_section()

    def set_evaluation_view(self, view: str) -> None:
        if view not in {key for key, _ in EVALUATION_VIEWS}:
            return
        if view == self._evaluation_view:
            return
        self._evaluation_view = view
        self._evaluation_click_selected_node = None
        self._render_active_section()

    def _focus_main_pane(self, *, prefer_rows: bool) -> None:
        if self._active_section_key == "workspace" and not prefer_rows:
            toggle = self.query_one("#home-toggle", HomeGroupToggle)
            if toggle.display:
                toggle.focus()
                return
        if self._active_section_key == "evaluations":
            if not prefer_rows:
                toggle = self.query_one("#evaluation-toggle", EvaluationViewToggle)
                if toggle.display:
                    toggle.focus()
                    return
            if self._evaluation_view == "env":
                tree = self.query_one("#evaluation-tree", EvaluationTree)
                if tree.display:
                    tree.focus()
                    return
        self.query_one("#item-list", OptionList).focus()

    def _focus_inspector_pane(self) -> None:
        self.query_one("#inspector", LabInspector).focus()

    def _workspace_items(self) -> list[LabItem]:
        if self._snapshot is None:
            return []
        section = self._snapshot.section("workspace")
        if section is None:
            return []
        return [
            self._detail_cache.get(item.key, item)
            for item in section.items
            if not self._filter or self._matches_filter(item)
        ]

    def action_search(self) -> None:
        def apply_filter(value: str | None) -> None:
            if value is not None:
                self._filter = value.lower()
                self._render_active_section()

        self.push_screen(
            FilterScreen(
                self._filter_choices_for_active_section(),
                title="Filter",
                placeholder="environment, model, run, status",
                initial=self._filter,
            ),
            apply_filter,
        )

    def action_load_detail(self) -> None:
        if self._launch_screen_active():
            self._dismiss_launch_screen()
            self._focus_main_pane(prefer_rows=False)
            return
        if self._handle_evaluation_tree_enter():
            return
        self._load_selected_detail(include_logs=False)

    def action_load_logs(self) -> None:
        self._load_selected_detail(include_logs=True)

    def action_load_more_rows(self) -> None:
        if self._ladder_loader is None or self._requested_section_limit <= 0:
            return
        current_limit = max(self._loaded_section_limit, self._requested_section_limit)
        next_limit = current_limit * 2
        self._requested_section_limit = next_limit
        self._reload_limits((next_limit,))

    def _show_initial_snapshot(self) -> None:
        if self._initial_loader is None:
            return
        try:
            self._set_snapshot(self._initial_loader())
        except Exception as exc:
            text = Text()
            text.append("Loading Lab ...", style="bold")
            text.append(f"\nInitial workspace view unavailable: {exc}", style="dim")
            self.query_one("#topbar", Static).update(text)

    @work(thread=True, exclusive=True)
    def _reload(self) -> None:
        if self._ladder_loader is not None and self._ladder_limits:
            target_limit = max(
                self._requested_section_limit,
                max(self._ladder_limits, default=0),
            )
            self._load_snapshots_for_limits(_ladder_limits(target_limit))
            return
        snapshot = self._loader()
        self.call_from_thread(self._set_snapshot, snapshot)

    @work(thread=True, exclusive=True)
    def _reload_limits(self, limits: tuple[int, ...]) -> None:
        self._load_snapshots_for_limits(limits)

    def _load_snapshots_for_limits(self, limits: tuple[int, ...]) -> None:
        if self._ladder_loader is None:
            return
        for limit in limits:
            snapshot = self._ladder_loader(limit)
            self.call_from_thread(self._set_ladder_snapshot, snapshot, limit)

    def _set_ladder_snapshot(self, snapshot: LabSnapshot, limit: int) -> None:
        self._loaded_section_limit = max(self._loaded_section_limit, limit)
        self._set_snapshot(snapshot)

    def _set_snapshot(self, snapshot: LabSnapshot) -> None:
        snapshot = merge_snapshot_rows(self._snapshot, snapshot)
        self._snapshot = snapshot
        if snapshot.section(self._active_section_key) is None and snapshot.sections:
            self._active_section_key = snapshot.sections[0].key
        self._render_topbar()
        self._render_statusbar()
        self._render_tree()
        self._render_active_section()
        self._sync_launch_screen()
        self._sync_agent_runtime(snapshot)

    @work(thread=True, exclusive=True, group="detail-load")
    def _load_detail_worker(self, item: LabItem, include_logs: bool) -> None:
        if self._detail_loader is None:
            return
        started_at = time.perf_counter()
        try:
            detailed = self._detail_loader(
                item,
                include_logs,
                _LOG_TAIL_STEPS[0],
                _RUN_METRIC_PREVIEW_LIMIT,
                None,
            )
            detailed = _with_metric_load_seconds(detailed, time.perf_counter() - started_at)
        except Exception as exc:
            detailed = LabItem(
                key=item.key,
                section=item.section,
                title=item.title,
                subtitle=str(exc),
                status="error",
                status_style=STATUS_ERROR,
                metadata=item.metadata,
                raw=item.raw,
            )
        self.call_from_thread(self._replace_selected_item, detailed)

    @work(thread=True, exclusive=True, group="detail-prefetch")
    def _prefetch_training_detail_worker(self, item: LabItem) -> None:
        if self._detail_loader is None:
            return
        started_at = time.perf_counter()
        try:
            detailed = self._detail_loader(
                item,
                False,
                _LOG_TAIL_STEPS[0],
                _RUN_METRIC_PREVIEW_LIMIT,
                None,
            )
            detailed = _with_metric_load_seconds(detailed, time.perf_counter() - started_at)
        except Exception:
            self.call_from_thread(self._finish_training_prefetch, item.key, None)
            return
        self.call_from_thread(self._finish_training_prefetch, item.key, detailed)

    def _render_topbar(self) -> None:
        if self._snapshot is None:
            return
        self.query_one("#topbar", Static).update(lab_header())

    def _render_statusbar(self) -> None:
        try:
            statusbar = self.query_one("#statusbar", Static)
        except NoMatches:
            return
        statusbar.update(self._statusbar_text())

    def _statusbar_text(self) -> Text:
        return statusbar_text(self._snapshot, self._agent_state)

    def _set_agent_state(self, state: AgentConnectionState) -> None:
        self._agent_state = state
        self._render_statusbar()
        self._sync_launch_screen()

    def _set_agent_messages(self, messages: tuple[AgentChatMessage, ...]) -> None:
        self._agent_messages = messages

    def _dispatch_agent_callback(self, callback: Callable[..., None], *args: Any) -> None:
        if getattr(self, "_thread_id", None) == threading.get_ident():
            callback(*args)
            return
        try:
            self.call_from_thread(callback, *args)
        except RuntimeError:
            return

    def _sync_agent_runtime(self, snapshot: LabSnapshot) -> None:
        agent = configured_workspace_agent(snapshot)
        if not agent:
            self._agent_runtime.start(snapshot.workspace, "")
            return
        self._agent_runtime.start(snapshot.workspace, agent)

    def _render_tree(self) -> None:
        if self._snapshot is None:
            return
        tree = self.query_one("#section-tree", Tree)
        tree.clear()
        root = tree.root
        root.expand()
        selected_node = None
        for section in self._snapshot.sections:
            label = Text()
            label.append(section.title, style="bold")
            label.append(f"  {section.status}", style=section.status_style)
            node = root.add(label, data=section.key, allow_expand=False)
            if section.key == self._active_section_key:
                selected_node = node
        if selected_node is not None:
            self.call_after_refresh(lambda: tree.move_cursor(selected_node))

    def _render_active_section(self) -> None:
        if self._snapshot is None:
            return
        section = self._snapshot.section(self._active_section_key)
        if section is None:
            return
        selected_key = (
            self._selected_item.key
            if self._selected_item is not None and self._selected_item.section == section.key
            else None
        )

        title = section.title
        if self._filter:
            title = f"{title} / {self._filter}"
        self.query_one("#section-title", Label).update(title)
        self.query_one("#section-subtitle", Static).update(section.description)

        self.query_one("#home-toggle", Static).display = False
        self.query_one("#home-actions", Horizontal).display = False
        self._home_action_items_by_id = {}

        if section.key == "workspace":
            self.query_one("#evaluation-toggle", Static).display = False
            self.query_one("#evaluation-tree", EvaluationTree).display = False
            self._render_workspace_home(section, selected_key)
            return

        if section.key == "evaluations":
            self._render_evaluations_section(section, selected_key)
            return

        self.query_one("#evaluation-toggle", Static).display = False
        self.query_one("#evaluation-tree", EvaluationTree).display = False
        option_list = self.query_one("#item-list", OptionList)
        option_list.display = True
        option_list.clear_options()
        self._items_by_key = {}
        self._visible_items = []
        for item in section.items:
            item = self._detail_cache.get(item.key, item)
            if not self._filter or self._matches_filter(item):
                self._visible_items.append(item)

        if not self._visible_items:
            empty = LabItem(
                key=f"{section.key}:empty",
                section=section.key,
                title="No rows",
                subtitle="Clear the filter or refresh the view.",
                status="empty",
                status_style="dim",
            )
            self._visible_items = [empty]

        for item in self._visible_items:
            self._items_by_key[item.key] = item
            option_list.add_option(Option(item_label(item), id=item.key))

        self.call_after_refresh(lambda: self._highlight_item(option_list, selected_key))

    def _render_evaluations_section(self, section: LabSection, selected_key: str | None) -> None:
        toggle = self.query_one("#evaluation-toggle", EvaluationViewToggle)
        toggle.display = True
        toggle.update_views(EVALUATION_VIEWS, self._evaluation_view)

        self._items_by_key = {}
        self._visible_items = []
        for item in section.items:
            item = self._detail_cache.get(item.key, item)
            if not self._filter or self._matches_filter(item):
                self._visible_items.append(item)
                self._items_by_key[item.key] = item
        self._evaluation_tree_index = evaluation_index(self._visible_items)

        option_list = self.query_one("#item-list", OptionList)
        evaluation_tree = self.query_one("#evaluation-tree", EvaluationTree)
        if self._evaluation_view == "env":
            option_list.display = False
            option_list.clear_options()
            evaluation_tree.display = True
            self._populate_evaluation_tree(evaluation_tree, selected_key)
            return

        evaluation_tree.display = False
        evaluation_tree.clear()
        option_list.display = True
        option_list.clear_options()

        if not self._visible_items:
            empty = LabItem(
                key="evaluations:empty",
                section="evaluations",
                title="No rows",
                subtitle="Clear the filter or refresh the view.",
                status="empty",
                status_style="dim",
            )
            self._visible_items = [empty]
            self._items_by_key[empty.key] = empty

        for item in self._visible_items:
            self._items_by_key[item.key] = item
            option_list.add_option(Option(item_label(item), id=item.key))

        self.call_after_refresh(lambda: self._highlight_item(option_list, selected_key))

    def _populate_evaluation_tree(self, tree: EvaluationTree, selected_key: str | None) -> None:
        tree.clear()
        root = tree.root
        root.expand()
        first_run_node = None
        selected_node = None
        if not self._evaluation_tree_index:
            root.add("No evaluation runs", allow_expand=False)
            self.query_one("#inspector-title", Label).update("Selection Details")
            self.query_one("#inspector", Static).update(Text("No evaluation runs", style="dim"))
            return

        for env_index, env_id in enumerate(sorted(self._evaluation_tree_index)):
            models = self._evaluation_tree_index[env_id]
            total_runs = sum(len(runs) for runs in models.values())
            env_node = root.add(
                evaluation_env_tree_label(env_id, models, total_runs),
                data=EvaluationNodeData(
                    kind="env",
                    env_id=env_id,
                    tree_name=env_id,
                    tree_suffix=(
                        ("  ", ""),
                        (f"{len(models)} models", "dim"),
                        ("  ", ""),
                        (f"{total_runs} runs", "dim"),
                    ),
                ),
                expand=env_index == 0,
            )
            for model_index, model in enumerate(sorted(models)):
                runs = sorted_evaluation_runs(models[model])
                model_node = env_node.add(
                    evaluation_model_tree_label(model, runs),
                    data=EvaluationNodeData(
                        kind="model",
                        env_id=env_id,
                        model=model,
                        tree_name=model,
                        tree_suffix=(("  ", ""), (f"{len(runs)} runs", "dim")),
                    ),
                    expand=env_index == 0 and model_index == 0,
                )
                for item in runs:
                    self._items_by_key[item.key] = item
                    tree_suffix: tuple[tuple[str, str], ...] = ()
                    reward = evaluation_reward(item)
                    if reward is not None:
                        tree_suffix = (
                            ("  ", ""),
                            (format_reward_value(reward), reward_style(reward)),
                        )
                    elif item.status and numeric_reward(item.status) is None:
                        tree_suffix = (("  ", ""), (item.status, item.status_style))
                    run_node = model_node.add(
                        evaluation_run_tree_label(item),
                        data=EvaluationNodeData(
                            kind="run",
                            env_id=env_id,
                            model=model,
                            item_key=item.key,
                            tree_name=evaluation_run_id(item),
                            tree_suffix=tree_suffix,
                        ),
                        allow_expand=False,
                    )
                    if first_run_node is None:
                        first_run_node = run_node
                    if selected_key == item.key:
                        selected_node = run_node

        target_node = selected_node or first_run_node
        if target_node is not None:
            self.call_after_refresh(lambda: tree.move_cursor(target_node))

    def _render_workspace_home(self, section: LabSection, selected_key: str | None) -> None:
        toggle = self.query_one("#home-toggle", HomeGroupToggle)
        self._items_by_key = {}
        workspace_items: list[LabItem] = []
        for item in section.items:
            item = self._detail_cache.get(item.key, item)
            if not self._filter or self._matches_filter(item):
                workspace_items.append(item)
                self._items_by_key[item.key] = item

        self._render_home_actions(workspace_action_items(workspace_items))

        groups = workspace_home_groups(workspace_content_items(workspace_items))
        available_keys = [key for key, _, items in groups if items]
        if self._home_group not in available_keys and available_keys:
            self._home_group = available_keys[0]
        active_items = next(
            (items for key, _, items in groups if key == self._home_group),
            [],
        )
        self._visible_items = active_items

        option_list = self.query_one("#item-list", OptionList)

        toggle.display = True
        toggle.update_groups(groups, self._home_group)

        option_list.display = True
        option_list.clear_options()
        if not self._visible_items:
            empty = LabItem(
                key="workspace:empty",
                section="workspace",
                title="No rows",
                subtitle="Switch category or refresh the view.",
                status="empty",
                status_style="dim",
            )
            self._visible_items = [empty]
            self._items_by_key[empty.key] = empty
        for item in self._visible_items:
            self._items_by_key[item.key] = item
            option_list.add_option(Option(item_label(item), id=item.key))
        self.call_after_refresh(lambda: self._highlight_item(option_list, selected_key))

    def _dismiss_launch_screen(self) -> bool:
        if not self._launch_screen_active():
            return False
        if self._launch_screen is not None:
            self._launch_screen.dismiss("home")
        return True

    def _launch_screen_active(self) -> bool:
        return self._launch_screen is not None

    def _open_launch_target(self, target: str) -> None:
        if target == "build":
            self._open_quickstart_agent_flow()
            return
        if target == "evaluate":
            self._open_quickstart_config("eval")
            return
        if target == "train":
            self._open_quickstart_config("rl")
            return
        target_section = {
            "explore": "environments",
        }.get(target, "workspace")
        if self._snapshot is not None and self._snapshot.section(target_section) is None:
            target_section = "workspace"
        self._active_section_key = target_section
        self._filter = ""
        self._render_active_section()

    def _open_quickstart_agent_flow(self) -> None:
        workspace = self._snapshot.workspace if self._snapshot is not None else Path.cwd()
        agent = (
            configured_workspace_agent(self._snapshot)
            if self._snapshot is not None
            else self._agent_state.agent
        )
        item = build_environment_item(workspace, agent=agent or self._agent_state.agent or "codex")
        self.push_screen(
            AgentChatScreen(
                item,
                state_provider=lambda: self._agent_state,
                messages_provider=lambda: self._agent_messages,
                select_agent=self._select_workspace_agent,
                send_prompt=self._send_agent_prompt,
                status_text_provider=self._statusbar_text,
            )
        )

    def _open_quickstart_config(self, config_kind: str) -> None:
        workspace = self._snapshot.workspace if self._snapshot is not None else Path.cwd()
        item = (
            training_config_item(workspace)
            if config_kind == "rl"
            else evaluation_config_item(workspace)
        )
        self.push_screen(ConfigRunScreen(item))

    def _sync_launch_screen(self) -> None:
        if self._launch_screen is None:
            return
        snapshot = self._snapshot
        loading = any(
            item.raw.get("loading") is True
            for section in (snapshot.sections if snapshot is not None else ())
            for item in section.items
        )
        workspace = compact_path(snapshot.workspace) if snapshot is not None else "~"
        auth_label = "auth ok" if snapshot is not None and snapshot.authenticated else "auth x"
        team = (snapshot.team or "personal") if snapshot is not None else "-"
        workspace_items: list[LabItem] = []
        if snapshot is not None:
            section = snapshot.section("workspace")
            if section is not None:
                workspace_items = [self._detail_cache.get(item.key, item) for item in section.items]
        groups = workspace_home_groups(workspace_content_items(workspace_items))
        counts = tuple((label.lower(), len(items)) for _, label, items in groups)
        self._launch_screen.update_state(
            HomeLaunchState(
                workspace=workspace,
                auth_label=auth_label,
                team=team,
                agent_label=agent_status_label(self._agent_state),
                loading=loading,
                counts=counts,
            )
        )

    def _render_home_actions(self, items: list[LabItem]) -> None:
        actions = self.query_one("#home-actions", Horizontal)
        actions.remove_children()
        self._home_action_items_by_id = {}
        if not items:
            actions.display = False
            return
        actions.display = True
        self._home_action_render_id += 1
        for index, item in enumerate(items):
            button_id = f"home-action-{self._home_action_render_id}-{index}"
            self._home_action_items_by_id[button_id] = item
            actions.mount(
                Button(
                    home_action_label(item),
                    id=button_id,
                    classes="home-action-button",
                    variant="primary" if item.raw.get("type") == "setup_action" else "default",
                )
            )

    def _highlight_item(self, option_list: OptionList, key: str | None) -> None:
        if not option_list.option_count:
            return
        index = 0
        if key is not None:
            for item_index, item in enumerate(self._visible_items):
                if item.key == key:
                    index = item_index
                    break
        option_list.highlighted = index
        self._show_item(self._visible_items[index])

    def _matches_filter(self, item: LabItem) -> bool:
        return self._filter in item_search_text(item)

    def _show_item(self, item: LabItem) -> None:
        item = self._detail_cache.get(item.key, item)
        self._selected_item = item
        if item.section != "training" or self._expand_ready_run_key != item.key:
            self._expand_ready_run_key = None
        if item.section == "evaluations":
            self.query_one("#inspector-title", Label).update("Selection Details")
            self.query_one("#inspector", Static).update(self._evaluation_run_details(item))
        else:
            self.query_one("#inspector-title", Label).update(item.title)
            self.query_one("#inspector", Static).update(_item_details(item))
        self._prefetch_training_detail(item)

    def is_training_item_key(self, key: str) -> bool:
        item = self._items_by_key.get(key)
        return item is not None and item.section == "training"

    def is_guarded_option_key(self, key: str) -> bool:
        return self.is_training_item_key(key)

    def is_run_expand_ready(self, key: str) -> bool:
        return self._expand_ready_run_key == key

    def is_option_expand_ready(self, key: str) -> bool:
        return self.is_run_expand_ready(key)

    def arm_training_run_from_mouse(self, key: str) -> None:
        item = self._items_by_key.get(key)
        if item is None or item.section != "training":
            return
        self._show_item(item)
        self._expand_ready_run_key = item.key

    def arm_option_from_mouse(self, key: str) -> None:
        self.arm_training_run_from_mouse(key)

    def _evaluation_run_details(self, item: LabItem) -> Group:
        stats: RunOverviewStats | None = None
        if item.raw.get("type") == "local_eval":
            cache_key = local_eval_stats_key(item)
            stats = self._local_eval_stats_cache.get(cache_key)
            if stats is None and cache_key not in self._loading_local_eval_stats:
                self._loading_local_eval_stats.add(cache_key)
                self._load_local_eval_stats_worker(item)
        return evaluation_run_selection_details(
            item,
            stats,
            platform_detail_chunks=_evaluation_detail_chunks,
        )

    @work(thread=True, exclusive=False, group="local-eval-stats")
    def _load_local_eval_stats_worker(self, item: LabItem) -> None:
        cache_key = local_eval_stats_key(item)
        stats = compute_run_overview_stats(LocalEvalRun.from_item(item))
        self.call_from_thread(self._finish_local_eval_stats, item.key, cache_key, stats)

    def _finish_local_eval_stats(
        self, item_key: str, cache_key: str, stats: RunOverviewStats
    ) -> None:
        self._loading_local_eval_stats.discard(cache_key)
        self._local_eval_stats_cache[cache_key] = stats
        if self._selected_item is not None and self._selected_item.key == item_key:
            item = self._detail_cache.get(item_key, self._selected_item)
            self.query_one("#inspector-title", Label).update("Selection Details")
            self.query_one("#inspector", Static).update(
                evaluation_run_selection_details(
                    item,
                    stats,
                    platform_detail_chunks=_evaluation_detail_chunks,
                )
            )

    def _prefetch_training_detail(self, item: LabItem) -> None:
        if (
            item.section != "training"
            or self._detail_loader is None
            or item.raw.get("loading")
            or _has_training_detail(item)
            or (cached := self._detail_cache.get(item.key)) is not None
            and _has_training_detail(cached)
            or self._prefetching_detail_key == item.key
        ):
            return
        self._prefetching_detail_key = item.key
        self._prefetch_training_detail_worker(item)

    def _load_selected_detail(self, *, include_logs: bool) -> None:
        if self._selected_item is None:
            return
        item = self._detail_cache.get(self._selected_item.key, self._selected_item)
        if item.raw.get("loading"):
            return
        if item.raw.get("type") == "setup_action":
            self.push_screen(
                SetupScreen(item, on_complete=self._refresh_after_workspace_memory_change)
            )
            return
        if item.raw.get("type") == "workspace_context":
            self.push_screen(
                WorkspaceScreen(
                    item,
                    self._switch_workspace,
                    self._forget_workspace,
                )
            )
            return
        if item.raw.get("type") == "add_workspace":
            self.push_screen(AddWorkspaceScreen(item, self._add_workspace))
            return
        if item.raw.get("type") == "agent_sync":
            self.push_screen(
                AgentSyncScreen(item, on_complete=self._refresh_after_workspace_memory_change)
            )
            return
        if item.raw.get("type") == "doctor_action":
            self.push_screen(
                DoctorScreen(item, on_complete=self._refresh_after_workspace_memory_change)
            )
            return
        if item.raw.get("type") == "config_file":
            self.push_screen(ConfigRunScreen(item))
            return
        if item.raw.get("type") == "agent_chat":
            self.push_screen(
                AgentChatScreen(
                    item,
                    state_provider=lambda: self._agent_state,
                    messages_provider=lambda: self._agent_messages,
                    select_agent=self._select_workspace_agent,
                    send_prompt=self._send_agent_prompt,
                    status_text_provider=self._statusbar_text,
                )
            )
            return
        if item.section == "evaluations" and item.raw.get("type") == "local_eval":
            self.push_screen(LocalEvalRunScreen(LocalEvalRun.from_item(item)))
            return
        if self._detail_loader is None:
            return
        if item.section == "training":
            frontend_url = self._snapshot.frontend_url if self._snapshot is not None else ""
            workspace = self._snapshot.workspace if self._snapshot is not None else Path.cwd()
            self.push_screen(
                TrainingRunScreen(
                    item,
                    self._detail_loader,
                    include_logs=include_logs,
                    frontend_url=frontend_url,
                    workspace=workspace,
                )
            )
            return
        if item.section == "environments":
            frontend_url = self._snapshot.frontend_url if self._snapshot is not None else ""
            workspace = self._snapshot.workspace if self._snapshot is not None else Path.cwd()
            self.push_screen(
                EnvironmentScreen(
                    item,
                    self._detail_loader,
                    frontend_url=frontend_url,
                    workspace=workspace,
                )
            )
            return
        if item.section not in {"training", "environments", "evaluations"}:
            return
        self.query_one("#inspector-title", Label).update(item.title)
        self.query_one("#inspector", Static).update(
            Text("Loading logs ..." if include_logs else "Loading details ...", style="dim")
        )
        self._load_detail_worker(item, include_logs)

    def _replace_selected_item(self, item: LabItem) -> None:
        if item.section == "training":
            self._detail_cache[item.key] = _merge_training_detail(
                self._detail_cache.get(item.key), item
            )
            item = self._detail_cache[item.key]
        elif item.section in {"environments", "evaluations"}:
            self._detail_cache[item.key] = item
        self._items_by_key[item.key] = item
        self._visible_items = [
            item if visible.key == item.key else visible for visible in self._visible_items
        ]
        self._show_item(item)

    def _switch_workspace(self, workspace: Path) -> None:
        if self._workspace_switcher is None:
            return
        self._workspace_switcher(workspace)
        self._refresh_after_workspace_memory_change()

    def _add_workspace(self, workspace: Path) -> None:
        from .cache import record_recent_workspace

        record_recent_workspace(workspace)
        self._refresh_after_workspace_memory_change()

    def _forget_workspace(self, workspace: Path) -> None:
        from .cache import forget_recent_workspace

        forget_recent_workspace(workspace)
        self._refresh_after_workspace_memory_change()

    def _select_workspace_agent(self, workspace: Path, agent: str) -> None:
        write_workspace_agent_choice(workspace, agent)
        self._agent_runtime.start(workspace, agent)
        self._render_statusbar()

    def _send_agent_prompt(self, prompt: str) -> None:
        self._send_agent_prompt_worker(prompt)

    @work(thread=True, exclusive=True, group="agent-prompt")
    def _send_agent_prompt_worker(self, prompt: str) -> None:
        self._agent_runtime.send_prompt(prompt)

    def _refresh_after_workspace_memory_change(self) -> None:
        self._detail_cache.clear()
        self._selected_item = None
        self._active_section_key = "workspace"
        self._filter = ""
        self._loaded_section_limit = 0
        self._show_initial_snapshot()
        self._reload()

    def _finish_training_prefetch(self, key: str, item: LabItem | None) -> None:
        if self._prefetching_detail_key == key:
            self._prefetching_detail_key = None
        if item is None:
            return
        self._detail_cache[key] = _merge_training_detail(self._detail_cache.get(key), item)
        item = self._detail_cache[key]
        self._items_by_key[key] = item
        self._visible_items = [
            item if visible.key == key else visible for visible in self._visible_items
        ]
        if self._selected_item is not None and self._selected_item.key == key:
            self._show_item(item)

    @on(Tree.NodeSelected, "#section-tree")
    def _section_selected(self, event: Tree.NodeSelected) -> None:
        key = getattr(event.node, "data", None)
        if isinstance(key, str):
            self._active_section_key = key
            self._filter = ""
            self._render_active_section()

    @on(Tree.NodeHighlighted, "#section-tree")
    def _section_highlighted(self, event: Tree.NodeHighlighted) -> None:
        key = getattr(event.node, "data", None)
        if isinstance(key, str):
            self._active_section_key = key
            self._render_active_section()

    @on(Tree.NodeHighlighted, "#evaluation-tree")
    def _evaluation_tree_highlighted(self, event: Tree.NodeHighlighted) -> None:
        self._show_evaluation_tree_selection(getattr(event.node, "data", None))

    @on(Tree.NodeSelected, "#evaluation-tree")
    def _evaluation_tree_selected(self, event: Tree.NodeSelected) -> None:
        payload = getattr(event.node, "data", None)
        if not isinstance(payload, EvaluationNodeData):
            return
        self._show_evaluation_tree_selection(payload)
        if payload.kind == "run":
            if self._evaluation_click_selected_node is event.node:
                self._open_evaluation_tree_run(payload)
            else:
                self._evaluation_click_selected_node = event.node
            return
        self._evaluation_click_selected_node = None
        if event.node.allow_expand:
            event.node.toggle()

    @on(OptionList.OptionHighlighted, "#item-list")
    def _item_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        item = self._items_by_key.get(str(event.option.id))
        if item is not None:
            self._show_item(item)

    @on(OptionList.OptionSelected, "#item-list")
    def _item_selected(self, event: OptionList.OptionSelected) -> None:
        item = self._items_by_key.get(str(event.option.id))
        if item is not None:
            self._show_item(item)
            option_list = event.option_list
            mouse_option_id: str | None = None
            mouse_was_armed = False
            if isinstance(option_list, LabOptionList):
                mouse_option_id, mouse_was_armed = option_list.consume_mouse_selection()
            is_mouse_activation = mouse_option_id == item.key
            if is_mouse_activation and item.section == "training":
                if not mouse_was_armed:
                    self._prefetch_training_detail(item)
                    return
            self._load_selected_detail(include_logs=False)

    @on(Button.Pressed, ".home-action-button")
    def _home_action_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id
        if button_id is None:
            return
        item = self._home_action_items_by_id.get(button_id)
        if item is None:
            return
        self._show_item(item)
        self._load_selected_detail(include_logs=False)

    def _show_evaluation_tree_selection(self, payload: Any) -> None:
        if not isinstance(payload, EvaluationNodeData):
            return
        self.query_one("#inspector-title", Label).update("Selection Details")
        if payload.kind == "run":
            item = self._items_by_key.get(payload.item_key)
            if item is None:
                return
            self._selected_item = item
            self.query_one("#inspector", Static).update(self._evaluation_run_details(item))
            return
        self._selected_item = None
        self._expand_ready_run_key = None
        self.query_one("#inspector", Static).update(
            evaluation_group_selection_details(payload, self._evaluation_tree_index)
        )

    def _open_evaluation_tree_run(self, payload: EvaluationNodeData) -> None:
        item = self._items_by_key.get(payload.item_key)
        if item is None:
            return
        self._show_item(item)
        self._load_selected_detail(include_logs=False)

    def _handle_evaluation_tree_enter(self) -> bool:
        if self._active_section_key != "evaluations" or self._evaluation_view != "env":
            return False
        if not isinstance(self.focused, EvaluationTree):
            return False
        tree = self.query_one("#evaluation-tree", EvaluationTree)
        node = tree.cursor_node
        if node is None:
            return False
        payload = getattr(node, "data", None)
        if not isinstance(payload, EvaluationNodeData):
            return False
        if payload.kind == "run":
            self._open_evaluation_tree_run(payload)
            return True
        if node.allow_expand:
            node.toggle()
            return True
        return False

    def _main_open_action_available(self) -> bool:
        if self._active_section_key == "evaluations" and self._evaluation_view == "env":
            return isinstance(self.focused, EvaluationTree)
        return self._selected_item is not None and not self._selected_item.raw.get("loading")

    def _filter_choices_for_active_section(self) -> list[FilterChoice]:
        if self._snapshot is None:
            return []
        section = self._snapshot.section(self._active_section_key)
        if section is None:
            return []
        if section.key == "workspace":
            items = self._workspace_items()
        else:
            items = [self._detail_cache.get(item.key, item) for item in section.items]
        return [filter_choice_for_item(item) for item in items]


def _is_lab_child_screen(screen: object) -> bool:
    return isinstance(
        screen,
        EnvironmentScreen
        | WorkspaceScreen
        | AddWorkspaceScreen
        | AgentSyncScreen
        | DoctorScreen
        | SetupScreen
        | ConfigRunScreen
        | AgentChatScreen
        | TrainingRunScreen
        | FilterScreen
        | LocalEvalRunScreen,
    )


def run_lab_view(
    *,
    limit: int = 1000,
    env_dir: str = "./environments",
    outputs_dir: str = "./outputs",
    workspace: Path | None = None,
) -> None:
    current_workspace = {"path": (workspace or Path.cwd()).resolve()}

    def options_for(current_limit: int) -> LabLoadOptions:
        return LabLoadOptions(
            limit=current_limit,
            workspace=current_workspace["path"],
            env_dir=env_dir,
            outputs_dir=outputs_dir,
        )

    from .cache import record_recent_workspace
    from .data import LabDataSource

    def switch_workspace(next_workspace: Path) -> None:
        current_workspace["path"] = next_workspace.resolve()
        record_recent_workspace(current_workspace["path"])

    data_source = LabDataSource()
    record_recent_workspace(current_workspace["path"])
    PrimeLabView(
        lambda: data_source.load(options_for(limit)),
        lambda item, include_logs, log_tail_lines, metrics_limit, metrics_min_step: (
            data_source.load_item_detail(
                item,
                include_logs=include_logs,
                log_tail_lines=log_tail_lines,
                metrics_limit=metrics_limit,
                metrics_min_step=metrics_min_step,
            )
        ),
        lambda: data_source.load_initial(options_for(limit)),
        lambda current_limit: data_source.load(options_for(current_limit)),
        _ladder_limits(limit),
        switch_workspace,
    ).run()


def _ladder_limits(max_limit: int, *, start: int = 5) -> tuple[int, ...]:
    if max_limit <= start:
        return (max_limit,)
    limits = []
    current = start
    while current < max_limit:
        limits.append(current)
        current *= 2
    limits.append(max_limit)
    return tuple(limits)


def _parse_log_records(value: str) -> list[dict[str, Any]]:
    return parse_log_records(value)
