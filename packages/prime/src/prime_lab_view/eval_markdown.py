"""Markdown rendering helpers for eval rollout bodies."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from markdown_it import MarkdownIt
from mdit_py_plugins.amsmath import amsmath_plugin
from mdit_py_plugins.dollarmath import dollarmath_plugin
from textual.content import Content, Span
from textual.style import Style
from textual.widgets._markdown import (
    Markdown as BaseMarkdown,
)
from textual.widgets._markdown import (
    MarkdownBlock,
    MarkdownH1,
    MarkdownH2,
    MarkdownH3,
    MarkdownH4,
    MarkdownH5,
    MarkdownH6,
    MarkdownParagraph,
    MarkdownTD,
    MarkdownTH,
)

_LATEX_BEGIN_END_RE = re.compile(r"\\(?:begin|end)\{[^}]+\}")
_LATEX_BRACED_SCRIPT_RE = re.compile(r"([_^])\{([^{}]+)\}")
_LATEX_WRAPPER_RE = re.compile(
    r"\\(?:mathrm|mathbf|mathit|mathsf|mathtt|operatorname|text)\{([^{}]+)\}"
)
_LATEX_FRACTION_RE = re.compile(r"\\(?:d|t)?frac\{([^{}]+)\}\{([^{}]+)\}")
_LATEX_SQRT_RE = re.compile(r"\\sqrt\{([^{}]+)\}")
_LATEX_COMMAND_RE = re.compile(r"\\([A-Za-z]+|.)")
_LATEX_COMMAND_REPLACEMENTS = {
    "alpha": "alpha",
    "beta": "beta",
    "gamma": "gamma",
    "delta": "delta",
    "epsilon": "epsilon",
    "theta": "theta",
    "lambda": "lambda",
    "mu": "mu",
    "pi": "pi",
    "sigma": "sigma",
    "phi": "phi",
    "psi": "psi",
    "omega": "omega",
    "Gamma": "Gamma",
    "Delta": "Delta",
    "Theta": "Theta",
    "Lambda": "Lambda",
    "Pi": "Pi",
    "Sigma": "Sigma",
    "Phi": "Phi",
    "Psi": "Psi",
    "Omega": "Omega",
    "cdot": "*",
    "times": "*",
    "pm": "+/-",
    "neq": "!=",
    "leq": "<=",
    "geq": ">=",
    "approx": "~",
    "to": "->",
    "rightarrow": "->",
    "leftarrow": "<-",
    "infty": "inf",
    "ldots": "...",
    "cdots": "...",
    "sum": "sum",
    "prod": "prod",
    "log": "log",
    "ln": "ln",
    "exp": "exp",
    "sin": "sin",
    "cos": "cos",
    "tan": "tan",
    "|": "||",
    ",": " ",
    ";": " ",
    "!": "",
}


def _replace_latex_groups(
    text: str,
    pattern: re.Pattern[str],
    replacement: str | Callable[[re.Match[str]], str],
) -> str:
    while True:
        updated = pattern.sub(replacement, text)
        if updated == text:
            return updated
        text = updated


def _replace_latex_fraction(match: re.Match[str]) -> str:
    numerator, denominator = (part.strip() for part in match.groups())
    if re.search(r"\s|[+\-*/]", numerator):
        numerator = f"({numerator})"
    if re.search(r"\s|[+\-*/]", denominator):
        denominator = f"({denominator})"
    return f"{numerator}/{denominator}"


def _replace_latex_command(match: re.Match[str]) -> str:
    command = match.group(1)
    if command in _LATEX_COMMAND_REPLACEMENTS:
        return _LATEX_COMMAND_REPLACEMENTS[command]
    if len(command) == 1 and not command.isalpha():
        return command
    return command


def _latex_to_text(latex: str, *, preserve_newlines: bool) -> str:
    text = _LATEX_BEGIN_END_RE.sub("", latex)
    text = text.replace("&", " ")
    text = text.replace("\\\\", "\n" if preserve_newlines else " ")
    text = _replace_latex_groups(text, _LATEX_WRAPPER_RE, r"\1")
    text = _replace_latex_groups(text, _LATEX_BRACED_SCRIPT_RE, r"\1\2")
    text = _replace_latex_groups(text, _LATEX_FRACTION_RE, _replace_latex_fraction)
    text = _replace_latex_groups(text, _LATEX_SQRT_RE, r"sqrt(\1)")
    text = _LATEX_COMMAND_RE.sub(_replace_latex_command, text)
    text = text.replace("{", "").replace("}", "").replace("~", " ")
    if preserve_newlines:
        lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
        return "\n".join(line for line in lines if line)
    return " ".join(text.split())


def render_inline_math(latex: str) -> str:
    return " ".join(_latex_to_text(latex, preserve_newlines=False).split())


def render_block_math(latex: str) -> str:
    return _latex_to_text(latex, preserve_newlines=True).strip()


def make_math_parser() -> MarkdownIt:
    parser = MarkdownIt("gfm-like")
    parser.use(dollarmath_plugin, allow_space=False, allow_digits=False)
    parser.use(amsmath_plugin)
    return parser


class MathInlineMixin:
    """Render markdown inline math tokens as terminal-readable text."""

    def _token_to_content(self, token: Any) -> Content:
        if token.children is None:
            return Content("")

        parts: list[str] = []
        spans: list[Span] = []
        style_stack: list[tuple[Style | str, int]] = []
        position = 0

        def add_text(text: str) -> None:
            nonlocal position
            parts.append(text)
            position += len(text)

        def push_style(style: Style | str) -> None:
            style_stack.append((style, position))

        def pop_style() -> None:
            if not style_stack:
                return
            style, start = style_stack.pop()
            spans.append(Span(start, position, style))

        for child in token.children:
            child_type = child.type
            attrs = child.attrs or {}

            if child_type == "text":
                add_text(re.sub(r"\s+", " ", child.content))
            elif child_type == "hardbreak":
                add_text("\n")
            elif child_type == "softbreak":
                add_text(" ")
            elif child_type == "code_inline":
                push_style(".code_inline")
                add_text(child.content)
                pop_style()
            elif child_type in {"math_inline", "math_inline_double"}:
                push_style("italic")
                add_text(render_inline_math(child.content))
                pop_style()
            elif child_type == "em_open":
                push_style(".em")
            elif child_type == "strong_open":
                push_style(".strong")
            elif child_type == "s_open":
                push_style(".s")
            elif child_type == "link_open":
                href = attrs.get("href", "")
                push_style(Style.from_meta({"@click": f"link({href!r})"}))
            elif child_type == "image":
                href = attrs.get("src", "")
                alt = attrs.get("alt", "")
                push_style(Style.from_meta({"@click": f"link({href!r})"}))
                add_text(" ")
                if alt:
                    add_text(f"({alt})")
                if child.children is not None:
                    for grandchild in child.children:
                        add_text(grandchild.content)
                pop_style()
            elif child_type.endswith("_close"):
                pop_style()

        return Content("".join(parts), spans=spans)


class MathParagraph(MathInlineMixin, MarkdownParagraph):
    pass


class MathH1(MathInlineMixin, MarkdownH1):
    pass


class MathH2(MathInlineMixin, MarkdownH2):
    pass


class MathH3(MathInlineMixin, MarkdownH3):
    pass


class MathH4(MathInlineMixin, MarkdownH4):
    pass


class MathH5(MathInlineMixin, MarkdownH5):
    pass


class MathH6(MathInlineMixin, MarkdownH6):
    pass


class MathTH(MathInlineMixin, MarkdownTH):
    pass


class MathTD(MathInlineMixin, MarkdownTD):
    pass


class MathDisplayBlock(MarkdownBlock):
    DEFAULT_CSS = """
    MathDisplayBlock {
        width: 1fr;
        height: auto;
        margin: 0 0 1 0;
        padding: 0 1;
        background: $boost;
        border-left: outer $primary 60%;
    }
    """

    def __init__(self, markdown: "MathMarkdown", token: Any):
        super().__init__(markdown, token)
        text = render_block_math(token.content)
        if token.type == "math_block_label" and getattr(token, "info", ""):
            text = f"[{token.info}]\n{text}"
        self.set_content(Content(text))


class MathMarkdown(BaseMarkdown):
    BLOCKS = BaseMarkdown.BLOCKS | {
        "paragraph_open": MathParagraph,
        "h1": MathH1,
        "h2": MathH2,
        "h3": MathH3,
        "h4": MathH4,
        "h5": MathH5,
        "h6": MathH6,
        "th_open": MathTH,
        "td_open": MathTD,
    }

    def __init__(self, markdown: str | None = None, **kwargs: Any) -> None:
        super().__init__(markdown, parser_factory=make_math_parser, **kwargs)

    def unhandled_token(self, token: Any) -> MarkdownBlock | None:
        if token.type in {"math_block", "math_block_label", "amsmath"}:
            return MathDisplayBlock(self, token)
        return None
