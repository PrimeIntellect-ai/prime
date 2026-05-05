"""Snapshot merge helpers for progressive Lab loading."""

from __future__ import annotations

from .models import LabItem, LabSection, LabSnapshot


def merge_snapshot_rows(previous: LabSnapshot | None, incoming: LabSnapshot) -> LabSnapshot:
    if previous is None or previous.workspace != incoming.workspace:
        return incoming
    previous_sections = {section.key: section for section in previous.sections}
    merged_sections: list[LabSection] = []
    for section in incoming.sections:
        previous_section = previous_sections.get(section.key)
        if previous_section is None or section.key not in {
            "environments",
            "training",
            "evaluations",
        }:
            merged_sections.append(section)
            continue
        merged_items = _merge_section_rows(previous_section.items, section.items)
        refreshed_at = _newer_iso(previous_section.refreshed_at, section.refreshed_at)
        row_data_origin = _merged_origin(previous_section.row_data_origin, section.row_data_origin)
        if merged_items == section.items:
            merged_sections.append(
                LabSection(
                    key=section.key,
                    title=section.title,
                    description=section.description,
                    items=section.items,
                    status=section.status,
                    status_style=section.status_style,
                    refreshed_at=refreshed_at,
                    row_data_origin=row_data_origin,
                )
            )
            continue
        merged_sections.append(
            LabSection(
                key=section.key,
                title=section.title,
                description=section.description,
                items=merged_items,
                status=f"{len(merged_items)} shown",
                status_style=section.status_style,
                refreshed_at=refreshed_at,
                row_data_origin=row_data_origin,
            )
        )
    return LabSnapshot(
        workspace=incoming.workspace,
        base_url=incoming.base_url,
        frontend_url=incoming.frontend_url,
        authenticated=incoming.authenticated,
        team=incoming.team,
        sections=tuple(merged_sections),
        warnings=incoming.warnings,
    )


def _merge_section_rows(
    previous_items: tuple[LabItem, ...],
    incoming_items: tuple[LabItem, ...],
) -> tuple[LabItem, ...]:
    if any(_is_placeholder_lab_item(item) for item in incoming_items):
        incoming_items = tuple(
            item for item in incoming_items if not _is_placeholder_lab_item(item)
        )
        if not incoming_items:
            return previous_items
    if not previous_items or len(incoming_items) >= len(previous_items):
        return incoming_items
    incoming_keys = {item.key for item in incoming_items}
    preserved = [item for item in previous_items if item.key not in incoming_keys]
    if not preserved:
        return incoming_items
    return (*incoming_items, *preserved)


def _is_placeholder_lab_item(item: LabItem) -> bool:
    return (
        item.raw.get("loading") is True
        or item.key.endswith(":error")
        or item.key.endswith(":auth-required")
        or item.title in {"Unavailable", "Sign in required"}
    )


def _newer_iso(left: str | None, right: str | None) -> str | None:
    if not left:
        return right
    if not right:
        return left
    return max(left, right)


def _merged_origin(left: str | None, right: str | None) -> str | None:
    origins = {origin for origin in (left, right) if origin}
    if not origins:
        return None
    if origins == {"live"}:
        return "live"
    if origins == {"disk"}:
        return "disk"
    return "mixed"
