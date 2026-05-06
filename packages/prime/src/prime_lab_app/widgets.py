"""Reusable Textual widgets for Lab TUI compositions."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

from rich.text import Text
from textual import events
from textual.binding import Binding
from textual.style import Style
from textual.widgets import Input, OptionList, Static, Tree
from textual.widgets._tree import TreeNode

from .launch_backdrop import LaunchBackdrop
from .palette import MUTED, PRIMARY, STATUS_SUCCESS

TreeBinding = Binding | tuple[str, str] | tuple[str, str, str]


def _binding_key(binding: TreeBinding) -> str:
    if isinstance(binding, Binding):
        return binding.key
    return binding[0]


class LabOptionList(OptionList):
    """Option list where mouse clicks can be guarded by the host app."""

    BINDINGS = [
        Binding("escape", "back", "Back", key_display="Esc", show=False),
        Binding("b", "back", "Back", key_display="B", show=False),
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


class ClearableInput(Input):
    """Input field where Ctrl+C clears text instead of closing Lab."""

    BINDINGS = [
        Binding("ctrl+c", "clear_field", show=False, priority=True),
    ]

    def action_clear_field(self) -> None:
        self.value = ""


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
        Binding("left", "cursor_parent", "Parent", key_display="Left", show=False),
        Binding("right", "cursor_right", "Expand/next", key_display="Right", show=False),
        Binding("enter", "enter_cursor", "Open/toggle", key_display="Enter"),
        Binding("space", "toggle_node", "Toggle", key_display="Space", show=False),
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
        Binding("left", "previous_segment", "Previous", key_display="Left", show=False),
        Binding("right", "next_segment", "Next", key_display="Right", show=False),
        Binding("up", "previous_segment", "Previous", key_display="Up", show=False),
        Binding("down", "next_segment", "Next", key_display="Down", show=False),
        Binding("enter", "select_segment", "Rows", key_display="Enter"),
        Binding("escape", "back", "Back", key_display="Esc", show=False),
        Binding("b", "back", "Back", key_display="B", show=False),
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
    """Segmented control for Settings item groups."""

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


class ScopeToggle(SegmentedToggle):
    """Segmented control for source/account scoping."""

    previous_action = "action_previous_scope_view"
    next_action = "action_next_scope_view"
    select_action = "focus_after_scope_toggle"

    def update_scopes(
        self,
        scopes: Sequence[tuple[str, str, int]],
        active_scope: str,
    ) -> None:
        self.update_segments(scopes, active_scope)


@dataclass(frozen=True)
class HomeLaunchState:
    """Display state for the Welcome launch visual."""

    workspace: str
    auth_label: str
    team: str
    agent_label: str
    loading: bool
    agent_action_label: str = "Configure Agent"
    counts: tuple[tuple[str, int], ...] = ()


class HomeLaunchPanel(Static):
    """Animated launch panel for the Welcome screen."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__("", *args, markup=False, **kwargs)
        self._frame = 0
        self._state = HomeLaunchState(
            workspace="-",
            auth_label="?",
            team="-",
            agent_label="none",
            loading=True,
        )

    def on_mount(self) -> None:
        self.set_interval(0.18, self._tick)
        self._tick()

    def update_state(self, state: HomeLaunchState) -> None:
        self._state = state
        self._refresh_panel()

    def _tick(self) -> None:
        self._frame += 1
        self._refresh_panel()

    def _refresh_panel(self) -> None:
        text = Text()
        width = max(48, self.size.width or 96)
        height = max(18, self.size.height or 28)
        content_width = min(190, max(64, width - 4))
        left_margin = max(1, (width - content_width) // 2)
        lines = self._launch_lines(content_width)
        top_pad = max(0, (height - len(lines)) // 2)

        text.append("\n" * top_pad)
        for line in lines:
            self._append_launch_line(text, line, left_margin=left_margin)

        self.update(text)
        self.refresh()

    def _launch_lines(self, width: int) -> list[list[tuple[str, str]]]:
        wide = width >= 94
        rows = 22 if wide else 18
        lines = LaunchBackdrop(frame=self._frame).render_parts(width, rows)
        self._overlay_brand(lines, width=width, wide=wide)
        self._overlay_status(lines, width=width, wide=wide)
        return lines

    def _overlay_brand(
        self,
        lines: list[list[tuple[str, str]]],
        *,
        width: int,
        wide: bool,
    ) -> None:
        if wide:
            self._overlay_parts(
                lines,
                1,
                2,
                [("PRIME", "bold white"), (" Intellect", "italic white")],
            )
            self._overlay_parts(
                lines,
                4,
                2,
                [
                    ("L A B", f"bold {PRIMARY}"),
                    (" / ", "#5f5f68"),
                    ("research control plane", "#5f5f68"),
                ],
            )
            self._overlay_parts(
                lines,
                6,
                2,
                [
                    (self._truncate("Create. Evaluate. Train. Deploy.", width - 4), "bold white"),
                ],
            )
            return

        self._overlay_parts(
            lines,
            1,
            2,
            [("PRIME", "bold white"), (" Intellect", "italic white")],
        )
        self._overlay_parts(lines, 3, 2, [("LAB", f"bold {PRIMARY}"), (".", "bold white")])
        self._overlay_parts(
            lines,
            4,
            2,
            [(self._truncate("Create. Evaluate. Train. Deploy.", width - 4), "bold white")],
        )

    def _overlay_status(
        self,
        lines: list[list[tuple[str, str]]],
        *,
        width: int,
        wide: bool,
    ) -> None:
        state = self._state
        status = "loading workspace" if state.loading else "ready"
        if state.loading:
            status = f"{status}{_pulse(self._frame)}"
            status_style = MUTED
        else:
            status_style = STATUS_SUCCESS

        counts = self._count_summary(width)
        row = len(lines) - (4 if wide else 3)
        self._overlay_parts(lines, row, 2, [(self._truncate(status, width - 4), status_style)])
        self._overlay_parts(
            lines,
            row + 1,
            2,
            [(self._truncate(self._context_line(state, width), width - 4), MUTED)],
        )
        if counts:
            self._overlay_parts(
                lines,
                row + 2,
                2,
                [(self._truncate(counts, width - 4), "dim")],
            )

    def _append_launch_line(
        self,
        text: Text,
        parts: list[tuple[str, str]],
        *,
        left_margin: int,
    ) -> None:
        text.append(" " * left_margin)
        for value, style in parts:
            text.append(value, style=style or None)
        text.append("\n")

    def _overlay_parts(
        self,
        lines: list[list[tuple[str, str]]],
        row: int,
        col: int,
        parts: list[tuple[str, str]],
    ) -> None:
        if row < 0 or row >= len(lines):
            return
        cells = self._expand_parts(lines[row])
        cursor = col
        for value, style in parts:
            for char in value:
                if 0 <= cursor < len(cells):
                    cells[cursor] = (char, style)
                cursor += 1
        lines[row] = self._compress_parts(cells)

    def _context_line(self, state: HomeLaunchState, width: int) -> str:
        parts = [state.workspace]
        identity = ""
        if state.auth_label and state.team and state.team != "-":
            identity = f"{state.auth_label} {state.team}"
        elif state.auth_label:
            identity = state.auth_label
        elif state.team and state.team != "-":
            identity = state.team
        if identity:
            parts.append(identity)
        if state.agent_label and state.agent_label not in {"agent none", "none"}:
            parts.append(state.agent_label)
        return self._truncate("  ·  ".join(parts), width)

    def _count_summary(self, width: int) -> str:
        if not self._state.counts:
            return ""
        parts = [
            f"{label} {count}"
            for label, count in self._state.counts
            if count > 0 and label in {"workspaces", "profiles", "environments", "configs"}
        ]
        return self._truncate("  /  ".join(parts[:4]), width)

    def _pad_parts(
        self,
        parts: list[tuple[str, str]],
        width: int,
    ) -> list[tuple[str, str]]:
        visible = sum(len(value) for value, _ in parts)
        if visible >= width:
            style = parts[0][1] if parts else ""
            return [(self._truncate("".join(value for value, _ in parts), width), style)]
        return parts + [(" " * (width - visible), "")]

    def _truncate(self, value: str, width: int) -> str:
        if len(value) <= width:
            return value
        return value[: max(1, width - 1)].rstrip() + "…"

    def _expand_parts(self, parts: list[tuple[str, str]]) -> list[tuple[str, str]]:
        cells: list[tuple[str, str]] = []
        for value, style in parts:
            cells.extend((char, style) for char in value)
        return cells

    def _compress_parts(self, cells: list[tuple[str, str]]) -> list[tuple[str, str]]:
        if not cells:
            return []
        compressed: list[tuple[str, str]] = []
        current_style = cells[0][1]
        current = [cells[0][0]]
        for char, style in cells[1:]:
            if style == current_style:
                current.append(char)
                continue
            compressed.append(("".join(current), current_style))
            current = [char]
            current_style = style
        compressed.append(("".join(current), current_style))
        return compressed


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


def _pulse(frame: int) -> str:
    return "." * ((frame % 3) + 1)


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
