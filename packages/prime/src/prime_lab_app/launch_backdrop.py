"""Atmospheric renderer for the Lab launch surface."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, hypot, sin
from typing import ClassVar

from rich.text import Text

from .palette import (
    LAUNCH_CONTOUR,
    LAUNCH_NOISE,
    LAUNCH_SCAN,
    LAUNCH_SCAN_DIM,
    LAUNCH_TRACE,
    LAUNCH_TRACE_DIM,
    PRIMARY,
    WARNING,
)

StyledCell = tuple[str, str]


@dataclass(frozen=True)
class LaunchBackdrop:
    """Deterministic terminal art for the first Lab screen."""

    frame: int = 0

    _BRAILLE: ClassVar[tuple[str, ...]] = ("⠁", "⠂", "⠄", "⠈", "⠐", "⠠", "⢀", "⡀")

    def render_parts(self, width: int, rows: int) -> list[list[StyledCell]]:
        width = max(36, width)
        rows = max(5, rows)
        chars = [[" " for _ in range(width)] for _ in range(rows)]
        styles = [["" for _ in range(width)] for _ in range(rows)]
        priority = [[0 for _ in range(width)] for _ in range(rows)]

        self._paint_braille_noise(chars, styles, priority)
        self._paint_scan_bars(chars, styles, priority)
        self._paint_edge_anchor(chars, styles, priority)
        self._paint_contours(chars, styles, priority)
        self._paint_metric_traces(chars, styles, priority)
        self._paint_horizon(chars, styles, priority)

        return [
            _compress_row([(chars[y][x], styles[y][x]) for x in range(width)]) for y in range(rows)
        ]

    def render_text(self, width: int, rows: int) -> Text:
        text = Text()
        for row in self.render_parts(width, rows):
            for value, style in row:
                text.append(value, style=style or None)
            text.append("\n")
        return text

    def _paint_braille_noise(
        self,
        chars: list[list[str]],
        styles: list[list[str]],
        priority: list[list[int]],
    ) -> None:
        rows = len(chars)
        width = len(chars[0])
        for y in range(rows):
            for x in range(width):
                hashed = (x * 1_103_515_245 ^ y * 2_654_435_761 ^ self.frame * 97) & 255
                if hashed < 5:
                    char = self._BRAILLE[(hashed + x + y) % len(self._BRAILLE)]
                    _set_cell(chars, styles, priority, x, y, char, LAUNCH_NOISE, 1)

    def _paint_scan_bars(
        self,
        chars: list[list[str]],
        styles: list[list[str]],
        priority: list[list[int]],
    ) -> None:
        rows = len(chars)
        width = len(chars[0])
        start = int(width * 0.50)
        spacing = 4 if width < 78 else 5
        for index, x in enumerate(range(start, width - 3, spacing)):
            top = int(1 + sin(index * 0.9 + self.frame * 0.06) * 2)
            bottom = int(rows - 2 - cos(index * 0.55 + self.frame * 0.04) * 3)
            for y in range(max(0, top), min(rows, bottom)):
                if (y + index + self.frame) % 6 == 0:
                    continue
                char = "┃" if (index + y) % 4 == 0 else "╎"
                style = LAUNCH_SCAN if (index + y + self.frame) % 9 == 0 else LAUNCH_SCAN_DIM
                _set_cell(chars, styles, priority, x, y, char, style, 2)

    def _paint_edge_anchor(
        self,
        chars: list[list[str]],
        styles: list[list[str]],
        priority: list[list[int]],
    ) -> None:
        rows = len(chars)
        width = len(chars[0])
        if width < 82 or rows < 8:
            return
        x = width - 4
        top = max(1, int(rows * 0.18))
        bottom = min(rows - 2, int(rows * 0.76))
        for y in range(top, bottom):
            if (y + self.frame) % 7 == 0:
                continue
            style = LAUNCH_SCAN if (y + self.frame) % 11 == 0 else LAUNCH_SCAN_DIM
            _set_cell(chars, styles, priority, x, y, "┃", style, 2)

    def _paint_contours(
        self,
        chars: list[list[str]],
        styles: list[list[str]],
        priority: list[list[int]],
    ) -> None:
        rows = len(chars)
        width = len(chars[0])
        center_x = width * 0.36
        center_y = rows * 0.54
        max_x = int(width * 0.82)
        for y in range(1, rows - 1):
            for x in range(2, max_x):
                radius = hypot((x - center_x) / 10.0, (y - center_y) / 2.8)
                wave = radius * 2.8 + sin(x * 0.09) * 0.35 - self.frame * 0.055
                if abs((wave % 1) - 0.5) < 0.022:
                    char = "╌" if (x + y) % 5 == 0 else "·"
                    _set_cell(chars, styles, priority, x, y, char, LAUNCH_CONTOUR, 2)

    def _paint_metric_traces(
        self,
        chars: list[list[str]],
        styles: list[list[str]],
        priority: list[list[int]],
    ) -> None:
        rows = len(chars)
        width = len(chars[0])
        bases = (int(rows * 0.30), int(rows * 0.49), int(rows * 0.72))
        for trace_index, base in enumerate(bases):
            phase = trace_index * 0.8 + self.frame * 0.075
            for x in range(4, width - 6):
                y = int(
                    base + sin(x * 0.10 + phase) * 1.8 + sin(x * 0.025 + self.frame * 0.04) * 1.1
                )
                if not 0 <= y < rows:
                    continue
                if (x + self.frame + trace_index * 13) % 41 == 0:
                    _set_cell(chars, styles, priority, x, y, "◆", WARNING, 5)
                elif (x + self.frame) % 12 == 0:
                    _set_cell(chars, styles, priority, x, y, "•", PRIMARY, 4)
                else:
                    _set_cell(chars, styles, priority, x, y, "·", LAUNCH_TRACE_DIM, 3)

    def _paint_horizon(
        self,
        chars: list[list[str]],
        styles: list[list[str]],
        priority: list[list[int]],
    ) -> None:
        rows = len(chars)
        width = len(chars[0])
        y = max(2, min(rows - 3, int(rows * 0.58)))
        for x in range(0, width, 2):
            if (x + self.frame) % 13 in {0, 1}:
                style = LAUNCH_TRACE if x > width * 0.60 else LAUNCH_CONTOUR
                _set_cell(chars, styles, priority, x, y, "─", style, 2)


def _set_cell(
    chars: list[list[str]],
    styles: list[list[str]],
    priority: list[list[int]],
    x: int,
    y: int,
    char: str,
    style: str,
    weight: int,
) -> None:
    if not (0 <= y < len(chars) and 0 <= x < len(chars[0])):
        return
    if priority[y][x] > weight:
        return
    chars[y][x] = char
    styles[y][x] = style
    priority[y][x] = weight


def _compress_row(cells: list[StyledCell]) -> list[StyledCell]:
    if not cells:
        return []
    compressed: list[StyledCell] = []
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
