"""Shared display helpers for logical-image update commands."""

from typing import Any

from prime_sandboxes import PersonalImageOwner, PlatformImageOwner, TeamImageOwner


def format_image_owner(owner: Any) -> str:
    """Return a consistent human-readable label for an image owner."""
    if isinstance(owner, TeamImageOwner):
        return f"team {owner.team_id}"
    if isinstance(owner, PersonalImageOwner):
        return "personal"
    if isinstance(owner, PlatformImageOwner):
        return "platform"
    return "unknown"


def format_image_coordinate(state: Any, *, missing: str = "—") -> str:
    """Render an image coordinate and visibility for update output."""
    if state is None:
        return missing
    return f"{state.name}:{state.tag} [{format_image_owner(state.owner)}, {state.visibility.value}]"
