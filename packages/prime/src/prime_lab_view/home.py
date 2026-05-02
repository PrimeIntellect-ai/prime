"""Home/workspace grouping and action helpers for the Lab TUI."""

from __future__ import annotations

from .models import LabItem

WorkspaceHomeGroup = tuple[str, str, list[LabItem]]


def workspace_home_groups(items: list[LabItem]) -> list[WorkspaceHomeGroup]:
    group_order = [
        ("workspaces", "Workspaces"),
        ("profiles", "Profiles"),
        ("environments", "Environments"),
        ("configs", "Configs"),
    ]
    groups = {key: [] for key, _ in group_order}
    for item in items:
        group = workspace_home_group(item)
        groups.setdefault(group, []).append(item)
    return [(key, label, groups[key]) for key, label in group_order if groups.get(key)]


def workspace_home_group(item: LabItem) -> str:
    item_type = item.raw.get("type")
    if item_type == "auth_profile":
        return "profiles"
    if item_type in {"local_environment", "environment"}:
        return "environments"
    if item_type == "config_file":
        return "configs"
    return "workspaces"


def workspace_action_items(items: list[LabItem]) -> list[LabItem]:
    return [item for item in items if is_workspace_action_item(item)]


def workspace_content_items(items: list[LabItem]) -> list[LabItem]:
    return [item for item in items if not is_workspace_action_item(item)]


def is_workspace_action_item(item: LabItem) -> bool:
    return item.raw.get("type") in {
        "setup_action",
        "doctor_action",
        "agent_chat",
        "agent_sync",
        "add_workspace",
    }


def home_action_label(item: LabItem) -> str:
    item_type = item.raw.get("type")
    if item_type == "setup_action":
        return "Setup Lab workspace"
    if item_type == "doctor_action":
        return "Check workspace"
    if item_type == "agent_chat":
        agent = item.raw.get("agent")
        return f"Chat with {agent}" if agent else "Configure coding agent"
    if item_type == "agent_sync":
        return "Sync Lab assets"
    if item_type == "add_workspace":
        return "Add workspace"
    return item.title
