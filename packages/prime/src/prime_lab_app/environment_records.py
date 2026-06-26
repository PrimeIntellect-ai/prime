"""Canonical environment records for Lab TUI local/platform views."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import toml
from prime_cli.utils.env_metadata import compute_content_hash, get_environment_metadata
from prime_cli.utils.time_utils import format_time_ago

from .models import LabItem
from .palette import STATUS_ERROR, STATUS_INFO, STATUS_LOCAL, STATUS_SUCCESS, STATUS_WARNING


@dataclass(frozen=True)
class LocalEnvironmentRecord:
    """Local environment metadata discovered in a Lab workspace."""

    name: str
    path: Path
    relative_path: str
    project: dict[str, Any] = field(default_factory=dict)
    hub_metadata: dict[str, Any] = field(default_factory=dict)
    readme_path: str | None = None
    readme_preview: str = ""
    files: tuple[str, ...] = ()
    content_hash: str = ""

    @property
    def owner(self) -> str | None:
        owner = self.hub_metadata.get("owner")
        return str(owner) if owner else None

    @property
    def hub_name(self) -> str:
        return str(self.hub_metadata.get("name") or self.project.get("name") or self.name)

    @property
    def slug(self) -> str | None:
        if self.owner:
            return f"{self.owner}/{self.hub_name}"
        return None

    @property
    def version(self) -> str | None:
        version = self.project.get("version") or self.hub_metadata.get("version")
        return str(version) if version else None

    @property
    def description(self) -> str:
        return str(self.project.get("description") or "")


@dataclass
class EnvironmentRecord:
    """Merged local/platform environment row."""

    key: str
    slug: str
    name: str
    owner: str | None = None
    local: LocalEnvironmentRecord | None = None
    platform: dict[str, Any] | None = None
    scopes: set[str] = field(default_factory=set)

    def merge_platform(self, env: dict[str, Any], *, scope: str) -> None:
        if self.platform is None or scope == "mine":
            self.platform = env
        self.scopes.add(scope)


def local_environment_items(
    workspace: Path,
    env_dir: str,
    limit: int,
    *,
    section: str,
) -> list[LabItem]:
    """Build Lab items for local-only environment rows."""

    return [
        _environment_item_from_record(record, idx, section=section)
        for idx, record in enumerate(_local_environment_records(workspace, env_dir, limit))
    ]


def merged_environment_items(
    workspace: Path,
    env_dir: str,
    limit: int,
    platform_entries: list[tuple[dict[str, Any], str]],
    *,
    section: str,
) -> list[LabItem]:
    """Merge local and platform environment records into deduplicated Lab items."""

    records: list[EnvironmentRecord] = []
    by_key: dict[str, EnvironmentRecord] = {}
    local_by_name: dict[str, list[EnvironmentRecord]] = {}
    local_by_platform_id: dict[str, EnvironmentRecord] = {}

    for local in _local_environment_records(workspace, env_dir, limit):
        slug = local.slug or local.hub_name
        key = _record_key(slug)
        record = EnvironmentRecord(
            key=key,
            slug=slug,
            name=local.hub_name,
            owner=local.owner,
            local=local,
            scopes={"local"},
        )
        records.append(record)
        by_key[key] = record
        local_by_name.setdefault(_record_key(local.hub_name), []).append(record)
        if environment_id := local.hub_metadata.get("environment_id"):
            local_by_platform_id[str(environment_id)] = record

    for env, scope in platform_entries:
        slug = _platform_slug(env)
        if not slug:
            continue
        key = _record_key(slug)
        record = by_key.get(key)
        if record is None:
            env_id = str(env.get("id") or "")
            record = local_by_platform_id.get(env_id)
        if record is None:
            name_key = _record_key(str(env.get("name") or env.get("slug") or ""))
            local_matches = local_by_name.get(name_key, [])
            if len(local_matches) == 1:
                record = local_matches[0]
        if record is None:
            owner, name = slug.split("/", 1)
            record = EnvironmentRecord(
                key=key,
                slug=slug,
                name=name,
                owner=owner,
                scopes=set(),
            )
            records.append(record)
            by_key[key] = record
        record.merge_platform(env, scope=scope)

    return [
        _environment_item_from_record(record, idx, section=section)
        for idx, record in enumerate(records[:limit])
    ]


def _local_environment_records(
    workspace: Path,
    env_dir: str,
    limit: int,
) -> list[LocalEnvironmentRecord]:
    env_root = _workspace_path(workspace, env_dir)
    try:
        if not env_root.is_dir():
            return []
    except OSError:
        return []

    records: list[LocalEnvironmentRecord] = []
    for path in _safe_environment_dirs(env_root):
        if path.name.startswith("."):
            continue
        project = _read_pyproject(path)
        try:
            hub_metadata = get_environment_metadata(path) or {}
        except OSError:
            hub_metadata = {}
        readme_path, readme_preview = _read_readme_preview(path)
        records.append(
            LocalEnvironmentRecord(
                name=path.name,
                path=path,
                relative_path=_relative_path(path, workspace),
                project=project,
                hub_metadata=hub_metadata,
                readme_path=readme_path,
                readme_preview=readme_preview,
                files=tuple(_interesting_child_files(path)),
                content_hash=_local_content_hash(path),
            )
        )
        if len(records) >= limit:
            break
    return records


def _safe_environment_dirs(env_root: Path) -> list[Path]:
    try:
        children = sorted(env_root.iterdir(), key=lambda child: child.name)
    except OSError:
        return []
    dirs: list[Path] = []
    for child in children:
        try:
            if child.is_dir():
                dirs.append(child)
        except OSError:
            continue
    return dirs


def _environment_item_from_record(
    record: EnvironmentRecord | LocalEnvironmentRecord,
    idx: int,
    *,
    section: str,
) -> LabItem:
    if isinstance(record, LocalEnvironmentRecord):
        record = EnvironmentRecord(
            key=_record_key(record.slug or record.hub_name),
            slug=record.slug or record.hub_name,
            name=record.hub_name,
            owner=record.owner,
            local=record,
            scopes={"local"},
        )

    local = record.local
    platform = record.platform or {}
    visibility = str(
        platform.get("visibility") or (local.hub_metadata.get("visibility") if local else "") or ""
    )
    version = (
        platform.get("latest_version")
        or platform.get("latestVersion")
        or platform.get("semantic_version")
        or platform.get("semanticVersion")
        or (local.version if local else None)
        or "-"
    )
    description = str(platform.get("description") or (local.description if local else "") or "")
    updated = _format_optional_time(platform.get("updated_at"))
    stars = str(platform.get("stars", 0)) if platform else "-"
    badges = _environment_badges(record, visibility)
    status = " ".join(badge["label"] for badge in badges)
    status_style = badges[0]["style"] if badges else STATUS_INFO
    subtitle = description or (local.relative_path if local else "")

    metadata: list[tuple[str, str]] = [
        ("Source", ", ".join(_source_labels(record))),
        ("Version", str(version)),
    ]
    if visibility:
        metadata.append(("Visibility", visibility))
    if platform:
        metadata.extend([("Stars", stars), ("Updated", updated)])
    if local:
        metadata.append(("Path", local.relative_path))
        if local.content_hash:
            metadata.append(("Source hash", local.content_hash[:12]))
    if local and (environment_id := local.hub_metadata.get("environment_id")):
        metadata.append(("Environment ID", str(environment_id)))

    return LabItem(
        key=f"environment:{record.key}:{idx}",
        section=section,
        title=record.slug,
        subtitle=subtitle,
        status=status,
        status_style=status_style,
        metadata=tuple(metadata),
        raw={
            "type": "environment",
            "slug": record.slug,
            "owner": record.owner,
            "name": record.name,
            "row_date": "" if updated == "-" else updated,
            "badges": badges,
            "sources": _source_labels(record),
            "local": _local_raw(local) if local else None,
            "platform": platform or None,
            "scopes": sorted(record.scopes),
            "visibility": visibility or None,
            "latest_version": version,
            "description": description,
        },
    )


def _environment_badges(record: EnvironmentRecord, visibility: str) -> list[dict[str, str]]:
    badges: list[dict[str, str]] = []
    if record.local is not None:
        badges.append({"label": "LOCAL", "style": STATUS_LOCAL})
    if visibility.upper() == "PUBLIC":
        badges.append({"label": "PUBLIC", "style": STATUS_SUCCESS})
    elif visibility.upper() == "PRIVATE":
        badges.append({"label": "PRIVATE", "style": STATUS_WARNING})
    elif record.platform:
        status = str(record.platform.get("latest_ci_status") or visibility or "PLATFORM")
        badges.append({"label": status, "style": _status_style(status)})
    return badges


def _source_labels(record: EnvironmentRecord) -> list[str]:
    labels: list[str] = []
    if record.local is not None:
        labels.append("local")
    if record.platform is not None:
        labels.append("platform")
    return labels or ["unknown"]


def _local_raw(local: LocalEnvironmentRecord | None) -> dict[str, Any] | None:
    if local is None:
        return None
    return {
        "name": local.name,
        "path": str(local.path),
        "relative_path": local.relative_path,
        "project": local.project,
        "metadata": local.hub_metadata,
        "readme_path": local.readme_path,
        "readme_preview": local.readme_preview,
        "files": list(local.files),
        "content_hash": local.content_hash,
    }


def _platform_slug(env: dict[str, Any]) -> str:
    owner = env.get("owner") or {}
    owner_name = owner.get("name") or owner.get("slug") or env.get("owner_name") or ""
    name = env.get("name") or env.get("slug") or ""
    if not owner_name:
        return str(name)
    return f"{owner_name}/{name}"


def _read_pyproject(path: Path) -> dict[str, Any]:
    pyproject_path = path / "pyproject.toml"
    try:
        if not pyproject_path.is_file():
            return {}
        parsed = toml.loads(pyproject_path.read_text(encoding="utf-8"))
    except (OSError, toml.TomlDecodeError):
        return {}
    project = parsed.get("project")
    return project if isinstance(project, dict) else {}


def _local_content_hash(path: Path) -> str:
    try:
        return compute_content_hash(path)
    except OSError:
        return ""


def _read_readme_preview(path: Path) -> tuple[str | None, str]:
    for name in ("README.md", "README.rst", "README.txt", "README"):
        readme = path / name
        try:
            if not readme.is_file():
                continue
            text = readme.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        preview = " ".join(text.strip().split())
        return str(readme), preview[:500]
    return None, ""


def _workspace_path(workspace: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = workspace / path
    return path.resolve()


def _relative_path(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _interesting_child_files(path: Path) -> list[str]:
    names = []
    try:
        children = sorted(path.iterdir())
    except OSError:
        return names
    for child in children:
        if child.name.startswith("."):
            continue
        try:
            if child.is_file() and child.suffix in {".py", ".toml", ".md", ".json"}:
                names.append(child.name)
        except OSError:
            continue
    return names[:12]


def _record_key(value: str) -> str:
    return value.strip().lower().replace("_", "-")


def _format_optional_time(value: Any) -> str:
    if not value:
        return "-"
    try:
        return format_time_ago(str(value))
    except Exception:
        return str(value)


def _status_style(status: str) -> str:
    upper = status.upper()
    if upper in {"COMPLETED", "SUCCESS", "PUBLIC"}:
        return STATUS_SUCCESS
    if upper in {"RUNNING", "PENDING"}:
        return STATUS_WARNING
    if upper in {"FAILED", "ERROR"}:
        return STATUS_ERROR
    if upper == "LOCAL":
        return STATUS_LOCAL
    return STATUS_INFO


def environment_platform_url(frontend_url: str, slug: str) -> str | None:
    if "/" not in slug:
        return None
    owner, name = slug.split("/", 1)
    return f"{frontend_url.rstrip('/')}/dashboard/environments/{owner}/{name}"


def environment_readme_text(raw: dict[str, Any]) -> str:
    local = raw.get("local")
    if isinstance(local, dict):
        readme_path = local.get("readme_path")
        if isinstance(readme_path, str) and readme_path:
            try:
                return Path(readme_path).read_text(encoding="utf-8", errors="replace")
            except OSError:
                return ""
    return ""


def pretty_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, default=str)
