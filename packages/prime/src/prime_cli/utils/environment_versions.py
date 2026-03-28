from __future__ import annotations

from typing import Mapping, Optional, cast

from ..client import APIClient, APIError


def extract_env_version_summary(version_data: Mapping[str, object] | None) -> dict[str, str]:
    if version_data is None:
        return {}

    version_label = version_data.get("semantic_version") or version_data.get("version")
    semantic_version = (
        version_label.removesuffix(" (latest)") if isinstance(version_label, str) else ""
    )

    content_hash = version_data.get("content_hash") or version_data.get("sha256")

    summary: dict[str, str] = {}
    if semantic_version:
        summary["semantic_version"] = semantic_version
    if isinstance(content_hash, str) and content_hash:
        summary["content_hash"] = content_hash
    return summary


def fetch_latest_env_version_summary(
    client: APIClient,
    owner_slug: str,
    env_name: str,
) -> Optional[dict[str, str]]:
    try:
        response = client.get(f"/environmentshub/{owner_slug}/{env_name}/versions")
        versions = response["data"]["versions"]
        latest_version = cast(Mapping[str, object], versions[0])
    except (APIError, KeyError, IndexError, TypeError):
        return None

    summary = extract_env_version_summary(latest_version)
    return summary or None
