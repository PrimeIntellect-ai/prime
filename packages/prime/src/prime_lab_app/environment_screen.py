"""Environment and workspace source browsing screens."""

from __future__ import annotations

import shlex
import webbrowser
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import toml
from rich.console import Group
from rich.table import Table
from rich.text import Text
from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, Footer, Input, Markdown, OptionList, Static
from textual.widgets._option_list import Option

from .cache import CachedEnvironmentSource, cached_environment_source, ensure_environment_source
from .config_screen import ConfigLaunchScreen, ConfigRunScreen
from .environment_records import environment_platform_url
from .models import LabItem
from .palette import BUTTON_CSS, PRIMARY, STATUS_ERROR
from .readme import readme_markdown as _readme_markdown
from .rows import item_badges_text
from .shell import lab_header
from .source_browser import SourceEntry, format_size, readme_path, source_entries, source_preview
from .toml_format import format_toml_blocks
from .widgets import ClearableInput, LoadingChart

DetailLoader = Callable[[LabItem, bool, int, int, int | None], LabItem]
WorkspaceSwitcher = Callable[[Path], None]
WorkspaceMemoryAction = Callable[[Path], None]
SetupCompleteAction = Callable[[], None]


@dataclass(frozen=True)
class EnvironmentAction:
    key: str
    label: str
    detail: str


class EnvironmentScreen(Screen[None]):
    """Hub-like terminal page for one Lab environment."""

    BINDINGS = [
        Binding("escape", "back", "Back", key_display="Esc"),
        Binding("enter", "open_entry", "Open", key_display="Enter"),
        Binding("space", "parent_dir", "Parent", key_display="Space"),
    ]

    CSS = (
        BUTTON_CSS
        + """
    EnvironmentScreen {
        background: $background;
        color: $foreground;
        layout: vertical;
    }

    .page-header {
        height: 3;
        padding: 0 1;
        background: $background;
    }

    .page-body {
        height: 1fr;
        padding: 0;
    }

    .source-list {
        height: 1fr;
        min-height: 10;
        border: round $primary;
        background: $panel;
    }

    .source-column {
        width: 36;
        min-width: 34;
        margin-right: 1;
    }

    .source-extras {
        height: auto;
        min-height: 0;
        padding: 0;
        background: $background;
        margin-top: 1;
    }

    .external-link-button {
        width: 1fr;
        margin: 1 0 0 0;
    }

    .env-action-button {
        width: 1fr;
        margin: 0 0 1 0;
    }

    .source-preview {
        width: 1fr;
        border: round $primary;
        background: $panel;
        padding: 0 1;
    }

    .source-preview-toolbar {
        height: 1;
        align-horizontal: right;
    }

    .source-preview-toolbar Button {
        width: auto;
        height: auto;
    }

    .source-file-markdown {
        background: $panel;
    }

    .about-sidebar {
        width: 46;
        min-width: 34;
        border: round $primary;
        background: $surface;
        padding: 0 1;
    }

    .empty-panel {
        border: round $primary;
        background: $surface;
        padding: 1 2;
        height: auto;
        margin: 0 1;
    }

    .setup-actions {
        height: auto;
        margin-top: 1;
    }
    """
    )

    def __init__(
        self,
        item: LabItem,
        detail_loader: DetailLoader | None,
        *,
        frontend_url: str,
        workspace: Path | None = None,
    ) -> None:
        super().__init__()
        self._item = item
        self._detail_loader = detail_loader
        self._frontend_url = frontend_url
        self._workspace = Path(workspace or ".").expanduser().resolve()
        self._source: CachedEnvironmentSource | None = None
        self._root: Path | None = None
        self._selected_version = str(item.raw.get("selected_version") or "latest")
        self._current_dir = ""
        self._entry_by_id: dict[str, SourceEntry] = {}
        self._action_by_id: dict[str, EnvironmentAction] = {}
        self._link_by_id: dict[str, str] = {}
        self._selected_entry_id: str | None = None
        self._preview_container: VerticalScroll | None = None
        self._preview_path: Path | None = None
        self._preview_raw = False

    def compose(self) -> ComposeResult:
        yield Static(_environment_header(self._item), id="env-header", classes="page-header")
        yield Horizontal(id="env-body", classes="page-body")
        yield Footer()

    def on_mount(self) -> None:
        self._source = cached_environment_source(
            {**self._item.raw, "selected_version": self._selected_version}
        )
        self._root = (
            self._source.root if self._source is not None else _local_environment_root(self._item)
        )
        if self._root is not None and self._root.is_dir():
            self._render_tab()
        else:
            self._render_loading()
        self._load_environment_worker()

    def action_back(self) -> None:
        self.app.pop_screen()

    def action_next_version(self) -> None:
        versions = _environment_versions(self._item.raw)
        if not versions:
            return
        refs = [_version_ref(version) for version in versions]
        refs = [ref for ref in refs if ref]
        if not refs:
            return
        try:
            idx = refs.index(self._selected_version)
        except ValueError:
            idx = 0
        self._selected_version = refs[(idx + 1) % len(refs)]
        self._source = None
        self._root = None
        self._current_dir = ""
        self._selected_entry_id = None
        self._render_loading()
        self._load_environment_worker()

    def action_open_entry(self) -> None:
        if not self._selected_entry_id:
            return
        entry = self._entry_by_id.get(self._selected_entry_id)
        if entry is None:
            return
        if entry.is_dir:
            self._current_dir = entry.relative_path
            self._selected_entry_id = None
            self._render_code()
            return
        self._render_file(entry.path)

    def action_parent_dir(self) -> None:
        if self._root is None:
            return
        if not self._current_dir:
            return
        self._current_dir = str(Path(self._current_dir).parent)
        if self._current_dir == ".":
            self._current_dir = ""
        self._selected_entry_id = None
        self._render_code()

    @work(thread=True, exclusive=True)
    def _load_environment_worker(self) -> None:
        item = LabItem(
            key=self._item.key,
            section=self._item.section,
            title=self._item.title,
            subtitle=self._item.subtitle,
            status=self._item.status,
            status_style=self._item.status_style,
            metadata=self._item.metadata,
            raw={**self._item.raw, "selected_version": self._selected_version},
        )
        if self._detail_loader is not None:
            try:
                item = self._detail_loader(item, False, 50, 10, None)
            except Exception:
                item = self._item
        source: CachedEnvironmentSource | None = None
        try:
            source = ensure_environment_source(item.raw)
        except Exception as exc:
            raw = {**item.raw, "source_error": str(exc)}
            item = LabItem(
                key=item.key,
                section=item.section,
                title=item.title,
                subtitle=item.subtitle,
                status="error",
                status_style=STATUS_ERROR,
                metadata=item.metadata,
                raw=raw,
            )
        self.app.call_from_thread(self._set_environment, item, source)

    def _set_environment(self, item: LabItem, source: CachedEnvironmentSource | None) -> None:
        self._item = item
        if source is not None:
            self._source = source
            self._root = source.root
        elif self._source is None:
            self._root = _local_environment_root(item)
        self._selected_version = _selected_version(item.raw)
        self.query_one("#env-header", Static).update(_environment_header(item))
        self._render_tab()

    @on(OptionList.OptionHighlighted, ".env-files")
    def _file_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        option_id = str(event.option.id)
        self._selected_entry_id = option_id
        entry = self._entry_by_id.get(option_id)
        if entry is not None and not entry.is_dir:
            self._render_file(entry.path)

    @on(OptionList.OptionSelected, ".env-files")
    def _file_selected(self, event: OptionList.OptionSelected) -> None:
        self._selected_entry_id = str(event.option.id)
        self.action_open_entry()

    @on(Button.Pressed, ".external-link-button")
    def _external_link_pressed(self, event: Button.Pressed) -> None:
        self._open_link_button(event.button)

    def _open_link_button(self, button: Button) -> None:
        button_id = button.id
        if button_id is None:
            return
        url = self._link_by_id.get(button_id)
        if url:
            webbrowser.open(url)

    @on(Button.Pressed, "#source-toggle-raw")
    def _source_toggle_raw_pressed(self, _event: Button.Pressed) -> None:
        self._preview_raw = not self._preview_raw
        if self._preview_path is not None:
            self._render_file(self._preview_path)

    @on(Button.Pressed, ".env-action-button")
    def _env_action_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id
        if button_id is None:
            return
        action = self._action_by_id.get(button_id)
        if action is not None:
            self._run_environment_action(action)

    def _render_loading(self) -> None:
        self._preview_container = None
        self._preview_path = None
        body = self.query_one("#env-body", Horizontal)
        body.remove_children()
        panel = Vertical(classes="empty-panel")
        body.mount(panel)
        panel.mount(LoadingChart("Loading environment source ..."))

    def _render_tab(self) -> None:
        self._render_code()

    def _render_code(self) -> None:
        body = self.query_one("#env-body", Horizontal)
        body.remove_children()
        root = self._root
        files = OptionList(classes="source-list env-files")
        preview = VerticalScroll(classes="source-preview")
        self._preview_container = preview
        self._preview_path = None
        self._entry_by_id = {}
        self._link_by_id = {}
        source_column = Vertical(classes="source-column")
        body.mount(source_column)
        source_column.mount(files)
        extras = Vertical(classes="source-extras")
        source_column.mount(extras)
        self._mount_extras(extras)
        body.mount(preview)

        if root is None or not root.is_dir():
            preview.mount(Static(_missing_source(self._item), markup=False))
            return

        entries = source_entries(root, self._current_dir)
        for index, entry in enumerate(entries):
            option_id = f"entry:{index}"
            self._entry_by_id[option_id] = entry
            files.add_option(Option(_source_entry_label(entry), id=option_id))

        _set_source_preview_message(preview, "Select a file to preview it here.")
        if entries:
            self._selected_entry_id = "entry:0"
            files.highlighted = 0
        else:
            _set_source_preview_message(preview, "No files found")

    def _mount_extras(self, container: Vertical) -> None:
        self._action_by_id = {}
        actions = [
            EnvironmentAction(
                "train",
                "Train",
                "Create a training config for this environment.",
            ),
            EnvironmentAction(
                "evaluate",
                "Evaluate",
                "Create an evaluation config for this environment.",
            ),
            *_environment_platform_action_specs(self._item, self._frontend_url),
        ]
        for index, action in enumerate(actions):
            option_id = f"env-action-{index}"
            self._action_by_id[option_id] = action
            container.mount(
                Button(
                    action.label,
                    id=option_id,
                    classes="env-action-button",
                    compact=True,
                    variant="primary" if action.key in {"train", "evaluate"} else "default",
                )
            )

    def _render_file(self, path: Path) -> None:
        if self._preview_container is None:
            return
        if self._preview_path != path:
            self._preview_raw = False
        self._preview_path = path
        try:
            _mount_source_file_preview(
                self._preview_container,
                path,
                raw_markdown=self._preview_raw,
            )
        except Exception as exc:
            _set_source_preview_message(self._preview_container, str(exc), style=STATUS_ERROR)

    def _run_environment_action(self, action: EnvironmentAction) -> None:
        slug = str(self._item.raw.get("slug") or self._item.title)
        if action.key == "platform":
            url = action.detail
            if url:
                webbrowser.open(url)
            return
        if action.key == "sync":
            self.app.push_screen(
                ConfigLaunchScreen(
                    command=action.detail,
                    workspace=self._workspace,
                    title="Syncing environment",
                    subtitle=slug,
                )
            )
            return
        if action.key in {"train", "evaluate"}:
            config_kind = "rl" if action.key == "train" else "eval"
            self.app.push_screen(
                ConfigRunScreen(
                    _environment_config_item(
                        self._item,
                        config_kind=config_kind,
                        workspace=self._workspace,
                    )
                )
            )


class WorkspaceScreen(Screen[None]):
    """Source browser for a Lab workspace."""

    BINDINGS = [
        Binding("escape", "back", "Back", key_display="Esc"),
        Binding("enter", "open_entry", "Open", key_display="Enter"),
        Binding("space", "parent_dir", "Parent", key_display="Space"),
        Binding("s", "switch_workspace", "Switch active"),
        Binding("f", "forget_workspace", "Forget"),
    ]

    CSS = EnvironmentScreen.CSS

    def __init__(
        self,
        item: LabItem,
        switcher: WorkspaceSwitcher | None,
        forgetter: WorkspaceMemoryAction | None = None,
    ) -> None:
        super().__init__()
        self._item = item
        self._switcher = switcher
        self._forgetter = forgetter
        self._root = Path(str(item.raw.get("workspace") or item.subtitle)).expanduser().resolve()
        self._current_dir = ""
        self._entry_by_id: dict[str, SourceEntry] = {}
        self._selected_entry_id: str | None = None
        self._preview_container: VerticalScroll | None = None
        self._preview_path: Path | None = None
        self._preview_raw = False

    def compose(self) -> ComposeResult:
        yield Static(_workspace_header(self._item), id="env-header", classes="page-header")
        yield Horizontal(id="env-body", classes="page-body")
        yield Footer()

    def on_mount(self) -> None:
        self._render_code()

    def action_back(self) -> None:
        self.app.pop_screen()

    def action_switch_workspace(self) -> None:
        if self._switcher is not None and self._root.is_dir():
            self._switcher(self._root)
            self.app.pop_screen()

    def action_forget_workspace(self) -> None:
        if self._forgetter is None or self._item.raw.get("active") is True:
            return
        self._forgetter(self._root)
        self.app.pop_screen()

    def action_open_entry(self) -> None:
        if not self._selected_entry_id:
            return
        entry = self._entry_by_id.get(self._selected_entry_id)
        if entry is None:
            return
        if entry.is_dir:
            self._current_dir = entry.relative_path
            self._selected_entry_id = None
            self._render_code()
            return
        self._render_file(entry.path)

    def action_parent_dir(self) -> None:
        if not self._current_dir:
            return
        self._current_dir = str(Path(self._current_dir).parent)
        if self._current_dir == ".":
            self._current_dir = ""
        self._selected_entry_id = None
        self._render_code()

    @on(OptionList.OptionHighlighted, ".workspace-files")
    def _file_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        option_id = str(event.option.id)
        self._selected_entry_id = option_id
        entry = self._entry_by_id.get(option_id)
        if entry is not None and not entry.is_dir:
            self._render_file(entry.path)

    @on(OptionList.OptionSelected, ".workspace-files")
    def _file_selected(self, event: OptionList.OptionSelected) -> None:
        self._selected_entry_id = str(event.option.id)
        self.action_open_entry()

    def _render_code(self) -> None:
        body = self.query_one("#env-body", Horizontal)
        body.remove_children()
        files = OptionList(classes="source-list workspace-files")
        preview = VerticalScroll(classes="source-preview")
        self._preview_container = preview
        self._preview_path = None
        sidebar = VerticalScroll(
            Static(_workspace_about(self._item), markup=False),
            Button("Switch active", id="workspace-switch", variant="primary"),
            *(
                [Button("Forget workspace", id="workspace-forget", variant="default")]
                if self._item.raw.get("active") is not True
                else []
            ),
            classes="about-sidebar",
        )
        body.mount(files)
        body.mount(preview)
        body.mount(sidebar)

        self._entry_by_id = {}
        if not self._root.is_dir():
            _set_source_preview_message(preview, "Workspace path is not available")
            return

        entries = source_entries(self._root, self._current_dir)
        for index, entry in enumerate(entries):
            option_id = f"entry:{index}"
            self._entry_by_id[option_id] = entry
            files.add_option(Option(_source_entry_label(entry), id=option_id))
        initial = _initial_preview_path(self._root, entries)
        if initial is not None:
            self._render_file(initial)
        elif entries:
            files.highlighted = 0
            self._selected_entry_id = "entry:0"
            self._render_file(entries[0].path)
        else:
            _set_source_preview_message(preview, "No files found")

    def _render_file(self, path: Path) -> None:
        if self._preview_container is None:
            return
        if self._preview_path != path:
            self._preview_raw = False
        self._preview_path = path
        try:
            _mount_source_file_preview(
                self._preview_container,
                path,
                raw_markdown=self._preview_raw,
            )
        except Exception as exc:
            _set_source_preview_message(self._preview_container, str(exc), style=STATUS_ERROR)

    @on(Button.Pressed, "#workspace-switch")
    def _switch_pressed(self, _event: Button.Pressed) -> None:
        self.action_switch_workspace()

    @on(Button.Pressed, "#workspace-forget")
    def _forget_pressed(self, _event: Button.Pressed) -> None:
        self.action_forget_workspace()

    @on(Button.Pressed, "#source-toggle-raw")
    def _source_toggle_raw_pressed(self, _event: Button.Pressed) -> None:
        self._preview_raw = not self._preview_raw
        if self._preview_path is not None:
            self._render_file(self._preview_path)


class AddWorkspaceScreen(Screen[None]):
    """Add a remembered local workspace path."""

    BINDINGS = [
        Binding("escape", "back", "Back", key_display="Esc"),
        Binding("enter", "add_workspace", "Add", key_display="Enter"),
    ]

    CSS = (
        EnvironmentScreen.CSS
        + """
    AddWorkspaceScreen Input {
        height: 3;
        margin: 1 0;
    }

    .workspace-add-actions {
        height: auto;
    }
    """
    )

    def __init__(self, item: LabItem, recorder: WorkspaceMemoryAction | None) -> None:
        super().__init__()
        self._item = item
        self._recorder = recorder

    def compose(self) -> ComposeResult:
        yield Static(_add_workspace_header(self._item), id="env-header", classes="page-header")
        with Vertical(classes="empty-panel"):
            yield Static("Path", classes="field-label", markup=False)
            yield ClearableInput(
                str(Path.home()),
                placeholder="/path/to/workspace",
                id="workspace-path",
            )
            with Horizontal(classes="workspace-add-actions"):
                yield Button("Add workspace", id="workspace-add", variant="primary")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#workspace-path", ClearableInput).focus()

    def action_back(self) -> None:
        self.app.pop_screen()

    def action_add_workspace(self) -> None:
        path = Path(self.query_one("#workspace-path", ClearableInput).value).expanduser().resolve()
        if not path.is_dir():
            self.notify("Workspace path does not exist", severity="warning")
            return
        if self._recorder is not None:
            self._recorder(path)
        self.app.pop_screen()

    @on(Button.Pressed, "#workspace-add")
    def _add_pressed(self, _event: Button.Pressed) -> None:
        self.action_add_workspace()

    @on(Input.Submitted, "#workspace-path")
    def _path_submitted(self, _event: Input.Submitted) -> None:
        self.action_add_workspace()


def _environment_header(item: LabItem) -> Group:
    title = Text()
    title.append(item.title, style="bold")
    title.append_text(item_badges_text(item))
    title.append(f"\nEnvironment · {_selected_version(item.raw)}", style="dim")
    return Group(lab_header(title))


def _workspace_header(item: LabItem) -> Group:
    text = Text()
    text.append(item.title, style="bold")
    text.append(f"  {item.status}", style=item.status_style)
    text.append(f"\n{item.subtitle}", style="dim")
    return Group(lab_header(text))


def _add_workspace_header(item: LabItem) -> Table:
    text = Text()
    text.append(item.title, style="bold")
    text.append(f"\n{item.subtitle}", style="dim")
    return lab_header(text)


def _source_entry_label(entry: SourceEntry) -> Text:
    label = Text()
    icon = "▸" if entry.is_dir else " "
    label.append(f"{icon} {entry.name}", style="bold" if entry.is_dir else "")
    size = format_size(entry.size)
    if size:
        label.append(f"  {size}", style="dim")
    return label


def _mount_source_file_preview(
    container: VerticalScroll, path: Path, *, raw_markdown: bool
) -> None:
    container.remove_children()
    if _is_markdown_file(path):
        label = "Rendered" if raw_markdown else "Raw"
        container.mount(
            Horizontal(
                Button(label, id="source-toggle-raw", compact=True),
                classes="source-preview-toolbar",
            )
        )
        if raw_markdown:
            container.mount(Static(source_preview(path)))
            return
        container.mount(
            Markdown(
                _readme_markdown(_read_source_text(path, limit=80_000), path.stem),
                classes="source-file-markdown",
                open_links=True,
            )
        )
        return
    container.mount(Static(source_preview(path)))


def _set_source_preview_message(
    container: VerticalScroll, message: str, *, style: str = "dim"
) -> None:
    container.remove_children()
    container.mount(Static(Text(message, style=style), markup=False))


def _is_markdown_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() == ".md"


def _read_source_text(path: Path, *, limit: int) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    if len(text) > limit:
        return text[:limit].rstrip() + "\n..."
    return text


def _environment_about(item: LabItem) -> Group:
    raw = item.raw
    local_value = raw.get("local")
    local: dict[str, Any] = local_value if isinstance(local_value, dict) else {}
    platform = _platform(raw)
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold dim", no_wrap=True)
    table.add_column()
    for key, value in (
        ("About", raw.get("description") or platform.get("description")),
        ("Version", _version(raw)),
        ("Visibility", raw.get("visibility") or platform.get("visibility")),
        ("Python", platform.get("python_version") or platform.get("pythonVersion")),
        ("Dependencies", _dependencies(platform)),
        ("Local path", local.get("relative_path") or local.get("path")),
    ):
        if value:
            table.add_row(key, str(value))
    chunks: list[Any] = [Text("About", style="bold"), table]
    versions = _versions_table(raw)
    if versions.row_count:
        chunks.extend([Text(""), Text("Versions", style="bold"), versions])
    return Group(*chunks)


def _workspace_about(item: LabItem) -> Group:
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold dim", no_wrap=True)
    table.add_column()
    if item.status:
        table.add_row("Status", Text(item.status, style=item.status_style))
    for key, value in item.metadata:
        table.add_row(key, value)
    return Group(Text("Workspace", style="bold"), table)


def _environment_platform_action_specs(
    item: LabItem,
    frontend_url: str,
) -> list[EnvironmentAction]:
    slug = str(item.raw.get("slug") or item.title)
    actions = [
        EnvironmentAction("sync", "Sync", _environment_sync_command(item)),
        EnvironmentAction(
            "platform", "Platform", environment_platform_url(frontend_url, slug) or ""
        ),
    ]
    return [action for action in actions if action.detail]


def _environment_sync_command(item: LabItem) -> str:
    raw = item.raw
    slug = str(raw.get("slug") or item.title)
    local = raw.get("local") if isinstance(raw.get("local"), dict) else None
    if local is not None and local.get("path"):
        return f"prime env push --path {shlex.quote(str(local['path']))}"
    if "/" in slug:
        return f"prime env pull {shlex.quote(slug)}"
    return "prime env push"


def _environment_config_item(item: LabItem, *, config_kind: str, workspace: Path) -> LabItem:
    slug = str(item.raw.get("slug") or item.title)
    name = slug.split("/", 1)[-1]
    selected_version = _selected_version(item.raw)
    toml_text = _environment_config_toml(
        slug,
        config_kind=config_kind,
        version=selected_version if selected_version != "-" else "",
    )
    path = workspace / ".prime" / "lab" / "configs" / config_kind / f"{name}.toml"
    return LabItem(
        key=f"workspace:config:new:{config_kind}:{slug}",
        section="workspace",
        title=name,
        subtitle=f"{config_kind} · {slug}",
        status=config_kind,
        status_style=PRIMARY,
        metadata=(
            ("Kind", "Training config" if config_kind == "rl" else "Evaluation config"),
            ("Path", str(path)),
            ("Environment", slug),
        ),
        raw={
            "type": "config_file",
            "config_kind": config_kind,
            "workspace": str(workspace),
            "path": str(path),
            "relative_path": str(path),
            "toml": toml_text,
            "parsed": _safe_toml_loads(toml_text),
        },
    )


def _environment_config_toml(slug: str, *, config_kind: str, version: str = "") -> str:
    if config_kind == "eval":
        eval_config: dict[str, Any] = {
            "env_id": slug,
            "num_examples": -1,
            "rollouts_per_example": 1,
        }
        if version:
            eval_config["version"] = version
        config: dict[str, Any] = {
            "model": "",
            "save_results": True,
            "eval": [eval_config],
        }
        return format_toml_blocks(toml.dumps(config)).rstrip()

    env: dict[str, Any] = {"id": slug}
    if version:
        env["version"] = version
    config = {
        "model": "",
        "max_steps": 100,
        "batch_size": 256,
        "rollouts_per_example": 8,
        "sampling": {"max_tokens": 512},
        "env": [env],
    }
    return format_toml_blocks(toml.dumps(config)).rstrip()


def _safe_toml_loads(value: str) -> dict[str, Any]:
    try:
        parsed = toml.loads(value)
    except toml.TomlDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _missing_source(item: LabItem) -> Group:
    text = Text()
    text.append("Source is not cached yet.", style="bold dim")
    if error := item.raw.get("source_error"):
        text.append(f"\n{error}", style=STATUS_ERROR)
    return Group(text)


def _local_environment_root(item: LabItem) -> Path | None:
    local = item.raw.get("local")
    if isinstance(local, dict):
        path = local.get("path")
        if isinstance(path, str) and Path(path).is_dir():
            return Path(path)
    return None


def _initial_preview_path(root: Path, entries: list[SourceEntry]) -> Path | None:
    if readme := readme_path(root):
        return readme
    for entry in entries:
        if not entry.is_dir:
            return entry.path
    return entries[0].path if entries else None


def _platform(raw: dict[str, Any]) -> dict[str, Any]:
    detail = raw.get("platform_detail")
    if isinstance(detail, dict):
        return detail
    platform = raw.get("platform")
    return platform if isinstance(platform, dict) else raw


def _version(raw: dict[str, Any]) -> str:
    platform = _platform(raw)
    return str(
        raw.get("latest_version")
        or raw.get("semantic_version")
        or raw.get("semanticVersion")
        or platform.get("semantic_version")
        or platform.get("semanticVersion")
        or "-"
    )


def _selected_version(raw: dict[str, Any]) -> str:
    selected = str(raw.get("selected_version") or "")
    if selected and selected != "latest":
        return selected
    return _version(raw)


def _environment_versions(raw: dict[str, Any]) -> list[dict[str, Any]]:
    versions = raw.get("versions")
    if not isinstance(versions, list):
        return []
    return [version for version in versions if isinstance(version, dict)]


def _version_ref(version: dict[str, Any]) -> str:
    return str(
        version.get("semantic_version")
        or version.get("semanticVersion")
        or version.get("version")
        or version.get("id")
        or ""
    )


def _versions_table(raw: dict[str, Any]) -> Table:
    selected = _selected_version(raw)
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold dim", no_wrap=True)
    table.add_column()
    for version in _environment_versions(raw)[:8]:
        ref = _version_ref(version)
        if not ref:
            continue
        label = ref
        if ref == selected:
            label = f"{ref} *"
        table.add_row(label, _version_detail(version))
    return table


def _version_detail(version: dict[str, Any]) -> str:
    parts = []
    if created := version.get("created_at") or version.get("createdAt"):
        parts.append(str(created))
    if sha := version.get("sha256"):
        parts.append(str(sha)[:12])
    return " · ".join(parts)


def _dependencies(platform: dict[str, Any]) -> str:
    deps = platform.get("dependencies")
    if isinstance(deps, list):
        return "\n".join(str(dep) for dep in deps[:12])
    metadata = platform.get("metadata")
    if isinstance(metadata, dict):
        metadata_deps = metadata.get("dependencies")
        if isinstance(metadata_deps, list):
            return "\n".join(str(dep) for dep in metadata_deps[:12])
    return ""
