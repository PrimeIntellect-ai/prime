from __future__ import annotations

from typing import Mapping, Optional

from ..client import APIClient, APIError


def extract_env_version_summary(version_data: Mapping[str, object]) -> dict[str, str]:
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
    except APIError:
        return None

    versions_data = response.get("data", response)
    if not isinstance(versions_data, dict):
        return None

    versions = versions_data.get("versions")
    if not isinstance(versions, list) or not versions:
        return None

    latest_version = versions[0]
    if not isinstance(latest_version, Mapping):
        return None

    summary = extract_env_version_summary(latest_version)
    return summary or None
