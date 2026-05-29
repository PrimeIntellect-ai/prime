"""Shared row labels and filter text for selector views."""

from __future__ import annotations

from rich.text import Text

from .filters import FilterChoice
from .models import LabItem


def filter_choice_for_item(item: LabItem) -> FilterChoice:
    return FilterChoice(
        key=item.key,
        label=item_label(item),
        search_text=item_search_text(item),
        value=item.title,
    )


def item_search_text(item: LabItem) -> str:
    return " ".join(
        [
            item.title,
            item.subtitle,
            item.status,
            *(value for _, value in item.metadata),
        ]
    ).lower()


def item_label(item: LabItem) -> Text:
    text = Text()
    text.append(item.title, style="bold")
    if row_date := str(item.raw.get("row_date") or "").strip():
        text.append("  ")
        text.append(row_date, style="dim")
    text.append_text(item_badges_text(item))
    if item.subtitle:
        text.append("\n")
        text.append(item.subtitle, style="dim")
    return text


def item_badges_text(item: LabItem) -> Text:
    text = Text()
    badges = item.raw.get("badges")
    if isinstance(badges, list) and badges:
        text.append("  ")
        for idx, badge in enumerate(badges):
            if idx:
                text.append(" ")
            if isinstance(badge, dict):
                text.append(str(badge.get("label") or ""), style=str(badge.get("style") or ""))
            else:
                text.append(str(badge), style=item.status_style)
    elif item.status:
        text.append("  ")
        text.append(item.status, style=item.status_style)
    return text
