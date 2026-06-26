"""Disk cache helpers for Lab TUI source browsing and workspace recents."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from prime_cli.utils.env_metadata import compute_content_hash

from .models import LabItem, LabSection

ROW_CACHE_SECTION_KEYS = {"environments", "training", "evaluations"}
MAX_CACHED_SECTION_ITEMS = 1000


@dataclass(frozen=True)
class CachedEnvironmentSource:
    """A cached source tree for an Environment Hub version."""

    root: Path
    manifest_path: Path
    manifest: dict[str, Any]


def lab_cache_root() -> Path:
    root = Path.home() / ".prime" / "lab" / "cache"
    root.mkdir(parents=True, exist_ok=True)
    return root


def workspace_state_path() -> Path:
    path = Path.home() / ".prime" / "lab" / "workspaces.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def lab_row_cache_key(
    *,
    workspace: Path,
    base_url: str,
    profile: str,
    team: str | None,
) -> str:
    """Stable key for list rows scoped to a workspace and Prime account context."""

    payload = json.dumps(
        {
            "base_url": base_url,
            "profile": profile,
            "team": team or "",
            "workspace": str(workspace.resolve()),
        },
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def lab_account_cache_key(
    *,
    base_url: str,
    profile: str,
    team: str | None,
) -> str:
    """Stable key for detail payloads scoped to a Prime account context."""

    payload = json.dumps(
        {
            "base_url": base_url,
            "profile": profile,
            "team": team or "",
        },
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def lab_row_cache_path(cache_key: str) -> Path:
    _validate_path_component(cache_key)
    return lab_cache_root() / "rows" / f"{cache_key}.json"


def lab_item_detail_cache_path(cache_key: str, item_key: str) -> Path:
    _validate_path_component(cache_key)
    item_hash = hashlib.sha1(item_key.encode("utf-8")).hexdigest()
    return lab_cache_root() / "details" / cache_key / f"{item_hash}.json"


def load_cached_lab_sections(cache_key: str, *, limit: int) -> dict[str, LabSection]:
    """Load cached platform list sections for first-paint hydration."""

    payload = _read_json(lab_row_cache_path(cache_key))
    raw_sections = payload.get("sections")
    if not isinstance(raw_sections, dict):
        return {}
    fallback_updated_at = str(payload.get("updated_at") or "") or None

    sections: dict[str, LabSection] = {}
    for key in ROW_CACHE_SECTION_KEYS:
        raw_section = raw_sections.get(key)
        if not isinstance(raw_section, dict):
            continue
        items_data = raw_section.get("items")
        if not isinstance(items_data, list):
            continue
        items = tuple(
            item
            for item in (_deserialize_lab_item(value) for value in items_data[:limit])
            if item is not None
        )
        if not items:
            continue
        sections[key] = LabSection(
            key=key,
            title=str(raw_section.get("title") or key.title()),
            description=str(raw_section.get("description") or ""),
            items=items,
            status=f"{len(items)} cached",
            status_style="dim",
            refreshed_at=str(raw_section.get("refreshed_at") or "") or fallback_updated_at,
            row_data_origin="disk",
        )
    return sections


def load_cached_lab_item_detail(cache_key: str, item_key: str) -> LabItem | None:
    payload = _read_json(lab_item_detail_cache_path(cache_key, item_key))
    item = _deserialize_lab_item(payload.get("item"))
    if item is None or item.key != item_key:
        return None
    return item


def write_cached_lab_item_detail(cache_key: str, item: LabItem) -> None:
    if not _is_cacheable_lab_item(item):
        return
    path = lab_item_detail_cache_path(cache_key, item.key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "item": _serialize_lab_item(item),
            },
            indent=2,
            sort_keys=True,
            default=str,
        ),
        encoding="utf-8",
    )


def write_cached_lab_sections(cache_key: str, sections: Iterable[LabSection]) -> None:
    """Persist successful list rows without clobbering good cache entries on failures."""

    path = lab_row_cache_path(cache_key)
    payload = _read_json(path)
    raw_sections = payload.get("sections")
    cached_sections: dict[str, Any] = raw_sections if isinstance(raw_sections, dict) else {}
    changed = False
    updated_at = datetime.now(timezone.utc).isoformat()

    for section in sections:
        if section.key not in ROW_CACHE_SECTION_KEYS:
            continue
        incoming_items = [item for item in section.items if _is_cacheable_lab_item(item)]
        existing_section = cached_sections.get(section.key)
        existing_items = _cached_section_items(existing_section)
        items = [
            _serialize_lab_item(item)
            for item in _merge_cached_section_items(incoming_items, existing_items)
        ]
        if not items:
            continue
        cached_sections[section.key] = {
            "title": section.title,
            "description": section.description,
            "status": section.status,
            "status_style": section.status_style,
            "refreshed_at": section.refreshed_at or updated_at,
            "row_data_origin": section.row_data_origin or "live",
            "items": items,
        }
        changed = True

    if not changed:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "updated_at": updated_at,
                "sections": cached_sections,
            },
            indent=2,
            sort_keys=True,
            default=str,
        ),
        encoding="utf-8",
    )


def record_recent_workspace(workspace: Path) -> None:
    """Remember a workspace path for Settings' inactive workspace list."""

    path = workspace_state_path()
    state = _read_json(path)
    recent = state.get("recent_workspaces") if isinstance(state, dict) else None
    values = [str(workspace.resolve())]
    if isinstance(recent, list):
        values.extend(str(value) for value in recent if str(value) not in values)
    path.write_text(
        json.dumps(
            {
                "recent_workspaces": values[:50],
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def forget_recent_workspace(workspace: Path) -> None:
    """Remove a workspace path from Settings' inactive workspace list."""

    path = workspace_state_path()
    state = _read_json(path)
    recent = state.get("recent_workspaces") if isinstance(state, dict) else None
    if not isinstance(recent, list):
        return
    try:
        target = str(workspace.expanduser().resolve())
    except OSError:
        target = str(workspace.expanduser())
    values = [str(value) for value in recent if str(value) != target]
    path.write_text(
        json.dumps(
            {
                "recent_workspaces": values[:50],
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def recent_workspaces() -> list[Path]:
    state = _read_json(workspace_state_path())
    recent = state.get("recent_workspaces") if isinstance(state, dict) else None
    if not isinstance(recent, list):
        return []
    paths: list[Path] = []
    for value in recent:
        try:
            path = Path(str(value)).expanduser().resolve()
        except OSError:
            continue
        if path.is_dir():
            paths.append(path)
    return paths


def environment_source_cache_path(owner: str, name: str, version: str) -> Path:
    for component in (owner, name, version):
        _validate_path_component(component)
    return lab_cache_root() / "environments" / owner / name / version


def environment_source_blob_cache_path(content_hash: str) -> Path:
    _validate_content_hash(content_hash)
    return lab_cache_root() / "sources" / content_hash / "files"


def environment_source_blob_manifest_path(content_hash: str) -> Path:
    _validate_content_hash(content_hash)
    return lab_cache_root() / "sources" / content_hash / "manifest.json"


def cached_environment_source(raw: dict[str, Any]) -> CachedEnvironmentSource | None:
    """Return a local or already-cached source tree without network access."""

    local_source = _local_environment_source(raw)
    if local_source is not None:
        return local_source

    slug = str(raw.get("slug") or raw.get("id") or "")
    if "/" not in slug:
        return None
    owner, name = slug.split("/", 1)

    for version in _environment_version_candidates(raw):
        try:
            source = _cached_environment_source_path(
                environment_source_cache_path(owner, name, version)
            )
        except ValueError:
            continue
        if source is not None:
            return source

    return _latest_cached_environment_source(owner, name)


def ensure_environment_source(raw: dict[str, Any]) -> CachedEnvironmentSource | None:
    """Return a local source tree for an environment, downloading it when needed."""
    cached = cached_environment_source(raw)
    if cached is not None:
        return cached

    from verifiers.utils.install_utils import (
        download_environment_source,
        environment_package_url,
    )

    slug = str(raw.get("slug") or raw.get("id") or "")
    if "/" not in slug:
        return None
    owner, name = slug.split("/", 1)

    detail = _platform_detail(raw)
    package_url = environment_package_url(detail)
    version = (
        detail.get("semanticVersion")
        or detail.get("semantic_version")
        or detail.get("version_id")
        or detail.get("sha256")
        or raw.get("latest_version")
        or "latest"
    )
    if not isinstance(package_url, str) or not package_url:
        return None

    cache_path = environment_source_cache_path(owner, name, str(version))
    manifest_path = cache_path / ".prime" / "lab-cache.json"

    download_environment_source({"package_url": package_url}, cache_path)
    content_hash = compute_content_hash(cache_path)
    manifest = {
        "cache_version": 1,
        "slug": slug,
        "version": str(version),
        "source_version": str(version),
        "content_hash": content_hash,
        "cached_at": datetime.now(timezone.utc).isoformat(),
        "package_url": package_url,
    }
    blob_root = _ensure_source_blob(cache_path, manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    blob_manifest_path = environment_source_blob_manifest_path(content_hash)
    blob_manifest = _read_json(blob_manifest_path) or manifest
    return CachedEnvironmentSource(blob_root, blob_manifest_path, blob_manifest)


def _local_environment_source(raw: dict[str, Any]) -> CachedEnvironmentSource | None:
    local = raw.get("local")
    if isinstance(local, dict):
        path = local.get("path")
        if isinstance(path, str) and Path(path).is_dir():
            root = Path(path)
            return CachedEnvironmentSource(root, root / ".prime" / "lab-cache.json", {})
    return None


def _cached_environment_source_path(path: Path) -> CachedEnvironmentSource | None:
    manifest_path = path / ".prime" / "lab-cache.json"
    manifest = _read_json(manifest_path)
    if not path.is_dir() or not manifest:
        return None
    content_hash = manifest.get("content_hash")
    if isinstance(content_hash, str) and content_hash:
        blob = _cached_environment_source_blob(content_hash)
        if blob is not None:
            return blob
    if "version" in manifest and "source_version" not in manifest:
        manifest = {**manifest, "source_version": manifest.get("version")}
    if path.is_dir() and manifest:
        return CachedEnvironmentSource(path, manifest_path, manifest)
    return None


def _cached_environment_source_blob(content_hash: str) -> CachedEnvironmentSource | None:
    try:
        root = environment_source_blob_cache_path(content_hash)
        manifest_path = environment_source_blob_manifest_path(content_hash)
    except ValueError:
        return None
    manifest = _read_json(manifest_path)
    if root.is_dir() and manifest:
        return CachedEnvironmentSource(root, manifest_path, manifest)
    return None


def _latest_cached_environment_source(owner: str, name: str) -> CachedEnvironmentSource | None:
    try:
        _validate_path_component(owner)
        _validate_path_component(name)
        root = lab_cache_root() / "environments" / owner / name
    except ValueError:
        return None
    if not root.is_dir():
        return None
    candidates: list[CachedEnvironmentSource] = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        source = _cached_environment_source_path(child)
        if source is not None:
            candidates.append(source)
    if not candidates:
        return None
    return max(candidates, key=_cached_source_sort_key)


def _cached_source_sort_key(source: CachedEnvironmentSource) -> tuple[str, float]:
    cached_at = source.manifest.get("cached_at")
    timestamp = source.root.stat().st_mtime
    try:
        timestamp = source.manifest_path.stat().st_mtime
    except OSError:
        pass
    return str(cached_at or ""), timestamp


def _environment_version_candidates(raw: dict[str, Any]) -> list[str]:
    detail = _platform_detail(raw)
    candidates: list[Any] = [
        raw.get("selected_version"),
        raw.get("semanticVersion"),
        raw.get("semantic_version"),
        detail.get("semanticVersion"),
        detail.get("semantic_version"),
        detail.get("version"),
        detail.get("version_id"),
        detail.get("sha256"),
        raw.get("latest_version"),
    ]
    latest = raw.get("latestVersion")
    if latest is not None:
        candidates.append(latest)
    versions = raw.get("versions")
    if isinstance(versions, list):
        for version in versions:
            if isinstance(version, dict):
                candidates.extend(
                    [
                        version.get("semanticVersion"),
                        version.get("semantic_version"),
                        version.get("version"),
                        version.get("version_id"),
                        version.get("sha256"),
                    ]
                )
            else:
                candidates.append(version)
    values: list[str] = []
    for candidate in candidates:
        if candidate in (None, "", "latest"):
            continue
        value = str(candidate)
        if value not in values:
            values.append(value)
    return values


def _ensure_source_blob(source_path: Path, manifest: dict[str, Any]) -> Path:
    content_hash = str(manifest["content_hash"])
    blob_root = environment_source_blob_cache_path(content_hash)
    blob_manifest_path = environment_source_blob_manifest_path(content_hash)
    if not blob_root.is_dir():
        blob_root.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(
            source_path,
            blob_root,
            ignore=shutil.ignore_patterns(".prime"),
        )
    blob_manifest = {
        **manifest,
        "cache_version": 1,
        "cached_at": manifest.get("cached_at") or datetime.now(timezone.utc).isoformat(),
    }
    blob_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    blob_manifest_path.write_text(
        json.dumps(blob_manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return blob_root


def _platform_detail(raw: dict[str, Any]) -> dict[str, Any]:
    detail = raw.get("platform_detail")
    if isinstance(detail, dict):
        return detail
    platform = raw.get("platform")
    if isinstance(platform, dict):
        return platform
    return raw


def _serialize_lab_item(item: LabItem) -> dict[str, Any]:
    return {
        "key": item.key,
        "section": item.section,
        "title": item.title,
        "subtitle": item.subtitle,
        "status": item.status,
        "status_style": item.status_style,
        "metadata": [list(row) for row in item.metadata],
        "raw": item.raw,
    }


def _deserialize_lab_item(value: Any) -> LabItem | None:
    if not isinstance(value, dict):
        return None
    metadata: list[tuple[str, str]] = []
    for row in value.get("metadata") or []:
        if isinstance(row, list | tuple) and len(row) == 2:
            metadata.append((str(row[0]), str(row[1])))
    raw = value.get("raw")
    return LabItem(
        key=str(value.get("key") or ""),
        section=str(value.get("section") or ""),
        title=str(value.get("title") or ""),
        subtitle=str(value.get("subtitle") or ""),
        status=str(value.get("status") or ""),
        status_style=str(value.get("status_style") or "dim"),
        metadata=tuple(metadata),
        raw=raw if isinstance(raw, dict) else {},
    )


def _cached_section_items(value: Any) -> list[LabItem]:
    if not isinstance(value, dict):
        return []
    raw_items = value.get("items")
    if not isinstance(raw_items, list):
        return []
    return [
        item
        for item in (_deserialize_lab_item(raw_item) for raw_item in raw_items)
        if item is not None and _is_cacheable_lab_item(item)
    ]


def _merge_cached_section_items(
    incoming_items: list[LabItem],
    existing_items: list[LabItem],
) -> list[LabItem]:
    merged: list[LabItem] = []
    seen: set[str] = set()
    for item in (*incoming_items, *existing_items):
        identity = _cached_item_identity(item)
        if identity in seen:
            continue
        seen.add(identity)
        merged.append(item)
        if len(merged) >= MAX_CACHED_SECTION_ITEMS:
            break
    return merged


def _cached_item_identity(item: LabItem) -> str:
    if item.section == "environments":
        return f"environment:{item.raw.get('slug') or item.title}"
    if item.section == "evaluations" and item.raw.get("type") == "local_eval":
        return f"local-eval:{item.raw.get('path') or item.title}"
    return item.key or item.title


def _is_cacheable_lab_item(item: LabItem) -> bool:
    if item.raw.get("loading"):
        return False
    if item.key.endswith(":error") or item.key.endswith(":auth-required"):
        return False
    if item.title in {"Unavailable", "Sign in required"}:
        return False
    if item.section == "evaluations" and item.raw.get("type") == "local_eval":
        return False
    return bool(item.key and item.title)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _validate_path_component(value: str) -> None:
    if not value or ".." in value or "/" in value or "\\" in value or "\x00" in value:
        raise ValueError(f"Unsafe cache path component: {value!r}")


def _validate_content_hash(value: str) -> None:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value.lower()):
        raise ValueError(f"Unsafe source content hash: {value!r}")
