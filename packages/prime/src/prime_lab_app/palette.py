"""Shared Lab terminal color palette."""

from __future__ import annotations

from pygments.style import Style
from pygments.token import (
    Comment,
    Error,
    Generic,
    Keyword,
    Name,
    Number,
    Operator,
    Punctuation,
    String,
    Token,
)
from rich.syntax import PygmentsSyntaxTheme
from textual.theme import Theme

BACKGROUND = "#050506"
SURFACE = "#0d0d10"
PANEL = "#151518"
FOREGROUND = "#f4f4f5"
MUTED = "#a1a1aa"

PRIMARY = "#7f70c7"
PRIMARY_SOFT = "#aa9be0"
SUCCESS = "#84cc16"
WARNING = "#f59e0b"
ERROR = "#ff2d55"
INFO = "#38bdf8"
LOCAL = INFO
NEUTRAL = "#d4d4d8"
GRID = "#52525b"
CODE_BACKGROUND = PANEL

LAUNCH_NOISE = "#18181d"
LAUNCH_CONTOUR = "#303039"
LAUNCH_SCAN = "#2f8b8d"
LAUNCH_SCAN_DIM = "#18383b"
LAUNCH_TRACE = "#d4d4d8"
LAUNCH_TRACE_DIM = "#85858f"

ROLLOUT_SUCCESS = "#6f9f25"
ROLLOUT_WARNING = "#c1842b"
TOOL_CALL = "#5f8d65"


class LabCodeStyle(Style):
    """Pygments style aligned with the Lab palette."""

    background_color = CODE_BACKGROUND
    default_style = FOREGROUND
    styles = {
        Token: FOREGROUND,
        Punctuation: NEUTRAL,
        Comment: f"italic {MUTED}",
        Keyword: PRIMARY_SOFT,
        Keyword.Constant: WARNING,
        Keyword.Type: WARNING,
        Operator: INFO,
        Name: NEUTRAL,
        Name.Attribute: INFO,
        Name.Builtin: WARNING,
        Name.Class: WARNING,
        Name.Decorator: PRIMARY_SOFT,
        Name.Exception: ERROR,
        Name.Function: f"bold {INFO}",
        Name.Function.Magic: f"bold {PRIMARY_SOFT}",
        Name.Tag: INFO,
        Name.Variable: NEUTRAL,
        Name.Variable.Instance: NEUTRAL,
        Number: WARNING,
        String: "#a3be8c",
        String.Doc: f"italic {MUTED}",
        String.Escape: PRIMARY_SOFT,
        String.Interpol: PRIMARY_SOFT,
        Error: ERROR,
        Generic.Deleted: ERROR,
        Generic.Error: ERROR,
        Generic.Inserted: SUCCESS,
        Generic.Output: MUTED,
        Generic.Prompt: INFO,
    }


CODE_THEME = PygmentsSyntaxTheme(LabCodeStyle)

STATUS_SUCCESS = f"bold {SUCCESS}"
STATUS_WARNING = f"bold {WARNING}"
STATUS_ERROR = f"bold {ERROR}"
STATUS_INFO = INFO
STATUS_LOCAL = f"bold {LOCAL}"
STATUS_DIM = "dim"
STATUS_ROLLOUT_SUCCESS = f"bold {ROLLOUT_SUCCESS}"
STATUS_ROLLOUT_WARNING = f"bold {ROLLOUT_WARNING}"

BUTTON_CSS = """
Button {
    border: none;
    background-tint: $background 0%;
    tint: $background 0%;
}

Button:focus {
    border: none;
    background-tint: $background 0%;
    tint: $background 0%;
}

Button.-style-default {
    color: $foreground;
    background: $surface;
    border: none;
    text-style: bold;
}

Button.-style-default:hover {
    color: $foreground;
    background: $panel;
    border: none;
}

Button.-style-default:focus {
    color: $foreground;
    background: $primary 24%;
    background-tint: $background 0%;
    border: none;
    text-style: bold;
}

Button.-style-default.-active {
    background: $primary 20%;
    background-tint: $background 0%;
    border: none;
    tint: $background 0%;
}

Button.-style-default.-primary {
    color: $foreground;
    background: $primary 72%;
    border: none;
}

Button.-style-default.-primary:hover,
Button.-style-default.-primary:focus {
    color: $foreground;
    background: $primary;
    background-tint: $background 0%;
    border: none;
}

Button.-style-default.-primary.-active {
    background: $primary 82%;
    background-tint: $background 0%;
    border: none;
    tint: $background 0%;
}

Button.-style-default:disabled {
    text-opacity: 0.45;
}
"""

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
