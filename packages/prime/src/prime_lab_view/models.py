"""Display models for the Lab TUI."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LabItem:
    """A normalized row shown in a Lab TUI section."""

    key: str
    section: str
    title: str
    subtitle: str = ""
    status: str = ""
    status_style: str = "dim"
    metadata: tuple[tuple[str, str], ...] = ()
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LabSection:
    """A navigable collection of Lab items."""

    key: str
    title: str
    description: str
    items: tuple[LabItem, ...] = ()
    status: str = ""
    status_style: str = "dim"


@dataclass(frozen=True)
class LabSnapshot:
    """All data needed to render one Lab TUI state."""

    workspace: Path
    base_url: str
    frontend_url: str
    authenticated: bool
    team: str | None
    sections: tuple[LabSection, ...]
    warnings: tuple[str, ...] = ()

    def section(self, key: str) -> LabSection | None:
        for section in self.sections:
            if section.key == key:
                return section
        return None
