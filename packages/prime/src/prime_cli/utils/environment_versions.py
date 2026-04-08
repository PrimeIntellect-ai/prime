from __future__ import annotations

from typing import Mapping, Optional

from ..client import APIClient, APIError


def extract_env_version_summary(version_data: Mapping[str, object] | None) -> dict[str, str]:
    if version_data is None:
        return {}

    semantic_version = version_data.get("semantic_version")
    content_hash = version_data.get("content_hash")

    summary: dict[str, str] = {}
    if isinstance(semantic_version, str) and semantic_version:
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
        latest_version = response["data"]["versions"][0]
    except (APIError, KeyError, IndexError, TypeError):
        return None

    if not isinstance(latest_version, Mapping):
        return None

    summary = extract_env_version_summary(latest_version)
    return summary or None
