"""Reusable source tree browsing helpers for Lab screens."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rich.syntax import Syntax
from rich.text import Text

from .palette import CODE_THEME

TEXT_EXTENSIONS = {
    "",
    ".cfg",
    ".css",
    ".env",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".rst",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}

IGNORED_DIR_NAMES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "outputs",
}

IGNORED_FILE_NAMES = {
    ".DS_Store",
}

IGNORED_SUFFIXES = {
    ".egg-info",
    ".pyc",
    ".pyo",
    ".so",
}


@dataclass(frozen=True)
class SourceEntry:
    """One row in a source browser."""

    name: str
    path: Path
    relative_path: str
    is_dir: bool
    size: int | None = None


def source_entries(root: Path, relative_dir: str = "") -> list[SourceEntry]:
    """List a safe directory under a source root."""

    root = root.resolve()
    directory = safe_source_path(root, relative_dir)
    if not directory.is_dir():
        return []

    entries: list[SourceEntry] = []
    if directory != root:
        parent = directory.parent
        entries.append(
            SourceEntry(
                name="..",
                path=parent,
                relative_path=_relative_path(parent, root),
                is_dir=True,
            )
        )
    for path in sorted(directory.iterdir(), key=lambda child: (not child.is_dir(), child.name)):
        if _is_ignored_source_path(path):
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        entries.append(
            SourceEntry(
                name=path.name,
                path=path,
                relative_path=_relative_path(path, root),
                is_dir=path.is_dir(),
                size=None if path.is_dir() else stat.st_size,
            )
        )
    return entries


def _is_ignored_source_path(path: Path) -> bool:
    if path.is_dir() and path.name in IGNORED_DIR_NAMES:
        return True
    if path.name in IGNORED_FILE_NAMES:
        return True
    if path.name.startswith(".") and path.name != ".env":
        return True
    return path.suffix.lower() in IGNORED_SUFFIXES


def safe_source_path(root: Path, relative_path: str) -> Path:
    """Resolve a path inside root, rejecting traversal."""

    root = root.resolve()
    candidate = (root / relative_path).resolve() if relative_path else root
    if candidate != root and root not in candidate.parents:
        raise ValueError(f"Path escapes source root: {relative_path}")
    return candidate


def readme_path(root: Path) -> Path | None:
    """Find a README-like file at the source root."""

    for name in ("README.md", "README.rst", "README.txt", "README"):
        path = root / name
        if path.is_file():
            return path
    return None


def source_preview(path: Path) -> Text | Syntax:
    """Render a directory or text file preview."""

    if path.is_dir():
        lines = [f"{entry.name}/" if entry.is_dir else entry.name for entry in source_entries(path)]
        return Text("\n".join(lines) if lines else "Empty directory", style="dim")

    suffix = path.suffix.lower()
    if suffix not in TEXT_EXTENSIONS:
        return Text(f"Binary or unsupported file: {path.name}", style="dim")
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return Text(f"Unable to read {path.name}: {exc}", style="red")
    if len(text) > 80_000:
        text = text[:80_000].rstrip() + "\n..."
    return Syntax(text.rstrip() or " ", _lexer_for(path), theme=CODE_THEME, word_wrap=True)


def format_size(size: int | None) -> str:
    if size is None:
        return ""
    units = ("B", "KB", "MB", "GB")
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} B"
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{int(size)} B"


def _relative_path(path: Path, root: Path) -> str:
    try:
        rel = path.resolve().relative_to(root.resolve())
    except ValueError:
        return str(path)
    return "" if str(rel) == "." else str(rel)


def _lexer_for(path: Path) -> str:
    suffix = path.suffix.lower()
    return {
        ".json": "json",
        ".md": "markdown",
        ".py": "python",
        ".rst": "rst",
        ".toml": "toml",
        ".yaml": "yaml",
        ".yml": "yaml",
    }.get(suffix, "text")
