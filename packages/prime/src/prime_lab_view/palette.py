"""Shared Lab terminal color palette."""

from __future__ import annotations

from textual.theme import Theme

BACKGROUND = "#030303"
SURFACE = "#09090b"
PANEL = "#111014"
FOREGROUND = "#f4f4f5"
MUTED = "#a1a1aa"

PRIMARY = "#8b5cf6"
PRIMARY_SOFT = "#c4b5fd"
SUCCESS = "#84cc16"
WARNING = "#f59e0b"
ERROR = "#ff2d55"
INFO = "#2dd4bf"
NEUTRAL = "#d4d4d8"
GRID = "#3f3f46"

STATUS_SUCCESS = f"bold {SUCCESS}"
STATUS_WARNING = f"bold {WARNING}"
STATUS_ERROR = f"bold {ERROR}"
STATUS_INFO = INFO
STATUS_DIM = "dim"

LAB_THEME = Theme(
    name="prime-lab",
    primary=PRIMARY,
    secondary=WARNING,
    accent=ERROR,
    warning=WARNING,
    error=ERROR,
    success=SUCCESS,
    background=BACKGROUND,
    surface=SURFACE,
    panel=PANEL,
    foreground=FOREGROUND,
    dark=True,
)
