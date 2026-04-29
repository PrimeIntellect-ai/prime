"""Reusable Textual widgets for Lab TUI compositions."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

from rich.text import Text
from textual import events
from textual.binding import Binding
from textual.style import Style
from textual.widgets import OptionList, Static, Tree
from textual.widgets._tree import TreeNode

from .palette import PRIMARY

TreeBinding = Binding | tuple[str, str] | tuple[str, str, str]


def _binding_key(binding: TreeBinding) -> str:
    if isinstance(binding, Binding):
        return binding.key
    return binding[0]


class LabOptionList(OptionList):
    """Option list where mouse clicks can be guarded by the host app."""

    BINDINGS = [
        Binding("space", "back", "Back", key_display="Space"),
    ]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._mouse_selected_option_id: str | None = None
        self._mouse_selected_was_armed = False

    async def _on_click(self, event: events.Click) -> None:
        clicked_option = event.style.meta.get("option")
        if not isinstance(clicked_option, int):
            return
        if self._options[clicked_option].disabled:
            event.stop()
            return
        self.focus()
        option_id = str(self._options[clicked_option].id)
        is_guarded = _call_bool(self.app, "is_guarded_option_key", option_id)
        was_armed = _call_bool(self.app, "is_option_expand_ready", option_id)
        self._mouse_selected_option_id = option_id
        self._mouse_selected_was_armed = was_armed
        self.highlighted = clicked_option
        if is_guarded and not was_armed:
            _call_app(self.app, "arm_option_from_mouse", option_id)
            event.prevent_default()
            event.stop()
            return
        self.action_select()
        event.prevent_default()
        event.stop()

    def consume_mouse_selection(self) -> tuple[str | None, bool]:
        option_id = self._mouse_selected_option_id
        was_armed = self._mouse_selected_was_armed
        self._mouse_selected_option_id = None
        self._mouse_selected_was_armed = False
        return option_id, was_armed

    def action_back(self) -> None:
        _call_app(self.app, "action_back_from_list")


@dataclass(frozen=True)
class EvaluationNodeData:
    kind: Literal["env", "model", "run"]
    env_id: str = ""
    model: str = ""
    item_key: str = ""
    tree_name: str = ""
    tree_suffix: tuple[tuple[str, str], ...] = ()


class EvaluationTree(Tree[EvaluationNodeData]):
    """Folder-organized evaluation selector."""

    BINDINGS = [
        *(
            binding
            for binding in Tree.BINDINGS
            if _binding_key(binding) not in {"enter", "space", "left", "right"}
        ),
        Binding("left", "cursor_parent", "Parent", key_display="Left"),
        Binding("right", "cursor_right", "Expand/next", key_display="Right"),
        Binding("enter", "enter_cursor", "Open/toggle", key_display="Enter"),
        Binding("space", "toggle_node", "Toggle", key_display="Space"),
    ]

    def _visible_depth(self, node: TreeNode[Any]) -> int:
        depth = 0
        parent = node.parent
        while parent is not None and (self.show_root or not parent.is_root):
            depth += 1
            parent = parent.parent
        return depth

    def _render_browser_label(
        self, payload: EvaluationNodeData, style: Style, max_width: int
    ) -> Text:
        label = Text()
        label.append(payload.tree_name or "", style="bold")
        for text, segment_style in payload.tree_suffix:
            label.append(text, style=segment_style or None)

        if max_width <= 0:
            label.truncate(1, overflow="ellipsis")
            label.stylize(cast(Any, style))
            return label

        suffix = Text()
        for text, segment_style in payload.tree_suffix:
            suffix.append(text, style=segment_style or None)

        if suffix.cell_len < max_width:
            name = Text(payload.tree_name or "", style="bold")
            name.truncate(max_width - suffix.cell_len, overflow="ellipsis")
            label = Text.assemble(name, suffix)
        else:
            label.truncate(max_width, overflow="ellipsis")

        label.stylize(cast(Any, style))
        return label

    def render_label(
        self,
        node: TreeNode[Any],
        base_style: Style,
        style: Style,
    ) -> Text:
        payload = node.data
        available_width = self.size.width - (self._visible_depth(node) * self.guide_depth)
        prefix_text = (
            self.ICON_NODE_EXPANDED
            if node.allow_expand and node.is_expanded
            else self.ICON_NODE
            if node.allow_expand
            else ""
        )
        content_width = max(1, available_width - len(prefix_text))

        if isinstance(payload, EvaluationNodeData) and payload.tree_name:
            label = self._render_browser_label(payload, style, content_width)
        else:
            label = node._label.copy()
            label.stylize(cast(Any, style))
            label.truncate(content_width, overflow="ellipsis")

        return Text.assemble((prefix_text, cast(Any, base_style)), label)

    def action_cursor_parent(self) -> None:
        cursor_node = self.cursor_node
        if cursor_node is None:
            return
        parent = cursor_node.parent
        if parent is None or (not self.show_root and parent.parent is None):
            return
        self.move_cursor(parent, animate=True)

    def action_cursor_right(self) -> None:
        cursor_node = self.cursor_node
        if cursor_node is None:
            return
        if cursor_node.allow_expand:
            if cursor_node.is_collapsed:
                cursor_node.expand()
                return
            if cursor_node.children:
                self.move_cursor(cursor_node.children[0], animate=True)
                return

        node = cursor_node.parent if not cursor_node.allow_expand else cursor_node
        while node is not None:
            next_sibling = node.next_sibling
            if next_sibling is not None:
                self.move_cursor(next_sibling, animate=True)
                return
            node = node.parent
            if node is not None and not self.show_root and node.is_root:
                return

    def action_toggle_node(self) -> None:
        node = self.cursor_node
        while node is not None and not node.allow_expand:
            node = node.parent
        if node is None or (not self.show_root and node.parent is None):
            return
        if node is not self.cursor_node:
            self.move_cursor(node, animate=False)
        self._toggle_node(node)

    def action_enter_cursor(self) -> None:
        _call_app(self.app, "action_load_detail")


class SegmentedToggle(Static, can_focus=True):
    """Focusable segmented control for switching related views."""

    BINDINGS = [
        Binding("up", "previous_segment", "Prev", key_display="Up"),
        Binding("down", "next_segment", "Next", key_display="Down"),
        Binding("enter", "select_segment", "Select", key_display="Enter"),
        Binding("space", "back", "Back", key_display="Space"),
    ]

    previous_action = ""
    next_action = ""
    select_action = ""
    back_action = "focus_nav_pane"

    def __init__(self, *args: Any, active_style: str | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._segments: list[tuple[str, int, int]] = []
        self._active_style = active_style or f"bold white on {PRIMARY}"

    async def _on_click(self, event: events.Click) -> None:
        for key, start, end in self._segments:
            if start <= event.x < end:
                self.focus()
                self._activate_segment(key)
                event.stop()
                return

    def action_previous_segment(self) -> None:
        _call_app(self.app, self.previous_action)

    def action_next_segment(self) -> None:
        _call_app(self.app, self.next_action)

    def action_select_segment(self) -> None:
        _call_app(self.app, self.select_action)

    def action_back(self) -> None:
        _call_app(self.app, self.back_action)

    def update_segments(
        self,
        segments: Sequence[tuple[str, str, int | None]],
        active_key: str,
    ) -> None:
        text = Text()
        self._segments = []
        cursor = 0
        for index, (key, label, count) in enumerate(segments):
            if index:
                text.append("  ")
                cursor += 2
            count_text = f" {count}" if count is not None else ""
            segment = f" {label}{count_text} "
            self._segments.append((key, cursor, cursor + len(segment)))
            style = self._active_style if key == active_key else "dim"
            text.append(segment, style=style)
            cursor += len(segment)
        self.update(text)

    def _activate_segment(self, key: str) -> None:
        _call_app(self.app, "set_segmented_toggle", self.id or "", key)


class HomeGroupToggle(SegmentedToggle):
    """Segmented control for Home item groups."""

    previous_action = "action_previous_home_group"
    next_action = "action_next_home_group"
    select_action = "focus_home_rows"

    def update_groups(
        self,
        groups: Sequence[tuple[str, str, Sequence[Any]]],
        active_group: str,
    ) -> None:
        self.update_segments(
            [(key, label, len(items)) for key, label, items in groups],
            active_group,
        )


class EvaluationViewToggle(SegmentedToggle):
    """Segmented control for evaluation grouping."""

    previous_action = "action_previous_evaluation_view"
    next_action = "action_next_evaluation_view"
    select_action = "focus_evaluation_rows"

    def update_views(self, views: Sequence[tuple[str, str]], active_view: str) -> None:
        self.update_segments([(key, label, None) for key, label in views], active_view)


class LabInspector(Static, can_focus=True):
    """Focusable inspector pane so left/right can move across all panes."""


class LoadingMessage(Static):
    """Small loading row with a terminal-friendly spinner."""

    _FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")

    def __init__(self, message: str, *, classes: str | None = None) -> None:
        super().__init__("", markup=False, classes=classes)
        self._message = message
        self._frame_index = 0

    def on_mount(self) -> None:
        self.set_interval(0.12, self._tick)
        self._tick()

    def _tick(self) -> None:
        frame = self._FRAMES[self._frame_index % len(self._FRAMES)]
        self._frame_index += 1
        text = Text()
        text.append(frame, style=PRIMARY)
        text.append(f" {self._message}", style="dim")
        self.update(text)
        self.refresh()


class LoadingChart(LoadingMessage):
    """Chart-shaped loading placeholder."""

    def __init__(self, message: str) -> None:
        super().__init__(message, classes="chart-loading")


def _call_app(app: Any, method: str, *args: Any) -> None:
    if not method:
        return
    handler = getattr(app, method, None)
    if callable(handler):
        handler(*args)


def _call_bool(app: Any, method: str, *args: Any) -> bool:
    handler = getattr(app, method, None)
    if not callable(handler):
        return False
    return bool(handler(*args))
