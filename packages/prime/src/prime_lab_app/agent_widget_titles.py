"""Shared title normalization for Lab agent widgets."""

from __future__ import annotations


def clean_widget_title(value: str) -> str:
    title = value.strip() or "Action"
    lowered = title.lower()
    for prefix in ("eval:", "evaluation:", "train:", "training:", "run:"):
        if lowered.startswith(prefix):
            return title[len(prefix) :].strip() or title
    return title
