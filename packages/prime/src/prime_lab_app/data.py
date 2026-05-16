"""Data loading for the Lab TUI."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import toml
from prime_cli.api.rl import RLClient
from prime_cli.client import APIClient, APIError
from prime_cli.core import Config
from prime_cli.utils.time_utils import format_time_ago
from prime_evals import EvalsClient

from .cache import (
    lab_account_cache_key,
    lab_row_cache_key,
    load_cached_lab_item_detail,
    load_cached_lab_sections,
    recent_workspaces,
    record_recent_workspace,
    write_cached_lab_item_detail,
    write_cached_lab_sections,
)
from .environment_records import local_environment_items, merged_environment_items
from .models import LabItem, LabSection, LabSnapshot
from .palette import (
    PRIMARY,
    STATUS_DIM,
    STATUS_ERROR,
    STATUS_INFO,
    STATUS_LOCAL,
    STATUS_SUCCESS,
    STATUS_WARNING,
)


@dataclass(frozen=True)
class LabLoadOptions:
    """Options for a single Lab TUI data refresh."""

    limit: int = 30
    workspace: Path = Path.cwd()
    env_dir: str = "./environments"
    outputs_dir: str = "./outputs"


class LabDataSource:
    """Read-only Lab data source."""

    def __init__(
        self,
        *,
        api_client_factory: Callable[..., Any] = APIClient,
        evals_client_factory: Callable[[Any], Any] = EvalsClient,
        rl_client_factory: Callable[[Any], Any] = RLClient,
        config_factory: Callable[[], Any] = Config,
    ) -> None:
        self._api_client_factory = api_client_factory
        self._evals_client_factory = evals_client_factory
        self._rl_client_factory = rl_client_factory
        self._config_factory = config_factory

    def load(self, options: LabLoadOptions) -> LabSnapshot:
        warnings: list[str] = []
        config = self._config_factory()
        authenticated = bool(config.api_key)
        team = config.team_name or config.team_id

        cache_key = _row_cache_key(options, config, team)
        detail_cache_key = _account_cache_key(config, team)
        cached_sections = load_cached_lab_sections(cache_key, limit=options.limit)

        sections = [
            self._workspace_section(options, config, authenticated, team),
            self._environment_section(options, authenticated, warnings),
            self._rl_section(options, config, authenticated, warnings),
            self._evaluation_section(options, config, authenticated, warnings),
        ]
        sections = _hydrate_platform_sections(detail_cache_key, sections, cached_sections)
        sections = _mark_live_sections(sections, refreshed_at=_utc_now_iso())

        snapshot = LabSnapshot(
            workspace=options.workspace.resolve(),
            base_url=config.base_url,
            frontend_url=config.frontend_url,
            authenticated=authenticated,
            team=team,
            sections=tuple(sections),
            warnings=tuple(warnings),
        )
        write_cached_lab_sections(
            cache_key,
            snapshot.sections,
        )
        return snapshot

    def load_initial(self, options: LabLoadOptions) -> LabSnapshot:
        """Load local Lab context while platform requests are still in flight."""
        config = self._config_factory()
        authenticated = bool(config.api_key)
        team = config.team_name or config.team_id
        cache_key = _row_cache_key(options, config, team)
        detail_cache_key = _account_cache_key(config, team)
        cached_sections = load_cached_lab_sections(cache_key, limit=options.limit)
        local_environment_items = tuple(
            _workspace_environment_items(
                options.workspace.resolve(),
                options.env_dir,
                options.limit,
                section="environments",
            )
        )
        local_eval_items = tuple(_local_eval_items(options, section="evaluations"))

        sections = [
            self._workspace_section(options, config, authenticated, team),
            _cached_or_loading_section(
                "environments",
                "Environments",
                "Local and platform environments.",
                authenticated=True,
                local_items=local_environment_items,
                cached_section=cached_sections.get("environments"),
            ),
            _cached_or_loading_section(
                "training",
                "Training",
                "Training runs for the active account or team.",
                authenticated=authenticated,
                cached_section=cached_sections.get("training"),
            ),
            _cached_or_loading_section(
                "evaluations",
                "Evaluations",
                "Local and platform evaluation runs.",
                authenticated=authenticated,
                local_items=local_eval_items,
                cached_section=cached_sections.get("evaluations"),
            ),
        ]
        sections = _hydrate_platform_sections(detail_cache_key, sections, cached_sections)

        return LabSnapshot(
            workspace=options.workspace.resolve(),
            base_url=config.base_url,
            frontend_url=config.frontend_url,
            authenticated=authenticated,
            team=team,
            sections=tuple(sections),
        )

    def load_evaluations(self, options: LabLoadOptions) -> LabSnapshot:
        """Load only the Evaluations section for the standalone eval TUI."""
        warnings: list[str] = []
        config = self._config_factory()
        authenticated = bool(config.api_key)
        team = config.team_name or config.team_id
        cache_key = _row_cache_key(options, config, team)
        detail_cache_key = _account_cache_key(config, team)
        cached_sections = load_cached_lab_sections(cache_key, limit=options.limit)

        sections = [
            self._evaluation_section(options, config, authenticated, warnings),
        ]
        sections = _hydrate_platform_sections(detail_cache_key, sections, cached_sections)
        sections = _mark_live_sections(sections, refreshed_at=_utc_now_iso())

        return LabSnapshot(
            workspace=options.workspace.resolve(),
            base_url=config.base_url,
            frontend_url=config.frontend_url,
            authenticated=authenticated,
            team=team,
            sections=tuple(sections),
            warnings=tuple(warnings),
        )

    def load_evaluations_initial(self, options: LabLoadOptions) -> LabSnapshot:
        """Load local/cached evaluation rows while the live eval request runs."""
        config = self._config_factory()
        authenticated = bool(config.api_key)
        team = config.team_name or config.team_id
        cache_key = _row_cache_key(options, config, team)
        detail_cache_key = _account_cache_key(config, team)
        cached_sections = load_cached_lab_sections(cache_key, limit=options.limit)
        local_eval_items = tuple(_local_eval_items(options, section="evaluations"))

        sections = [
            _cached_or_loading_section(
                "evaluations",
                "Evaluations",
                "Local and platform evaluation runs.",
                authenticated=authenticated,
                local_items=local_eval_items,
                cached_section=cached_sections.get("evaluations"),
            ),
        ]
        sections = _hydrate_platform_sections(detail_cache_key, sections, cached_sections)

        return LabSnapshot(
            workspace=options.workspace.resolve(),
            base_url=config.base_url,
            frontend_url=config.frontend_url,
            authenticated=authenticated,
            team=team,
            sections=tuple(sections),
        )

    def load_item_detail(
        self,
        item: LabItem,
        *,
        include_logs: bool = False,
        log_tail_lines: int = 50,
        metrics_limit: int = 10,
        metrics_min_step: int | None = None,
    ) -> LabItem:
        """Load expanded read-only data for a selected item."""
        if item.section == "training":
            result = self._load_rl_detail(
                item,
                include_logs=include_logs,
                log_tail_lines=log_tail_lines,
                metrics_limit=metrics_limit,
                metrics_min_step=metrics_min_step,
            )
        elif item.section == "evaluations" and item.title != "Sign in required":
            if item.raw.get("type") == "local_eval":
                result = item
            else:
                result = self._load_evaluation_detail(item)
        elif item.section == "environments":
            result = self._load_environment_detail(item)
        else:
            result = item

        self._write_cached_item_detail(result)
        return result

    def _write_cached_item_detail(self, item: LabItem) -> None:
        if item.section not in {"training", "environments", "evaluations"}:
            return
        config = self._config_factory()
        team = config.team_name or config.team_id
        write_cached_lab_item_detail(
            lab_account_cache_key(
                base_url=str(config.base_url),
                profile=_config_current_profile(config),
                team=team,
            ),
            item,
        )

    def _workspace_section(
        self,
        options: LabLoadOptions,
        config: Config,
        authenticated: bool,
        team: str | None,
    ) -> LabSection:
        auth_status = "authenticated" if authenticated else "public"
        workspace = options.workspace.resolve()
        profiles = _config_profiles(config)
        current_profile = _config_current_profile(config)
        items = [
            *_workspace_context_items(
                workspace,
                config,
                auth_status=auth_status,
                team=team,
                profiles=profiles,
                current_profile=current_profile,
                limit=options.limit,
            ),
            *_workspace_profile_items(profiles, current_profile),
            *_workspace_environment_items(
                workspace,
                options.env_dir,
                options.limit,
                section="workspace",
            ),
            *_workspace_config_items(workspace, options.limit),
        ]
        return LabSection(
            key="workspace",
            title="Settings",
            description="Workspaces, profiles, local assets, and setup.",
            items=tuple(items),
            status="",
            status_style=STATUS_SUCCESS if authenticated else STATUS_WARNING,
        )

    def _environment_section(
        self, options: LabLoadOptions, authenticated: bool, warnings: list[str]
    ) -> LabSection:
        platform_entries: list[tuple[dict[str, Any], str]] = []

        if authenticated:
            try:
                client = self._api_client_factory()
                mine_params: dict[str, Any] = {
                    "include_teams": True,
                    "limit": options.limit,
                    "offset": 0,
                    "sort_by": "updated_at",
                    "sort_order": "desc",
                    "mine_only": True,
                }
                response = client.get("/environmentshub/", params=mine_params)
                environments = response.get("data", response.get("environments", []))
                if not isinstance(environments, list):
                    raise APIError("environment list response did not contain a list")
                platform_entries.extend((env, "mine") for env in environments)
            except Exception as exc:
                warnings.append(f"Authenticated environments unavailable: {exc}")

        try:
            public_client = self._api_client_factory(require_auth=False)
            public_params: dict[str, Any] = {
                "include_teams": True,
                "limit": options.limit,
                "offset": 0,
                "sort_by": "stars",
                "sort_order": "desc",
                "visibility": "PUBLIC",
            }
            response = public_client.get("/environmentshub/", params=public_params)
            environments = response.get("data", response.get("environments", []))
            if not isinstance(environments, list):
                raise APIError("public environment response did not contain a list")
            platform_entries.extend((env, "public") for env in environments)
        except Exception as exc:
            warnings.append(f"Public environments unavailable: {exc}")
            if not platform_entries:
                return _error_section(
                    "environments",
                    "Environments",
                    "Environments Hub entries.",
                    exc,
                )

        items = merged_environment_items(
            options.workspace.resolve(),
            options.env_dir,
            options.limit,
            platform_entries,
            section="environments",
        )
        return LabSection(
            key="environments",
            title="Environments",
            description="Local and platform environments.",
            items=tuple(items),
            status=f"{len(items)} shown",
            status_style=STATUS_INFO,
        )

    def _evaluation_section(
        self,
        options: LabLoadOptions,
        config: Config,
        authenticated: bool,
        warnings: list[str],
    ) -> LabSection:
        local_items = _local_eval_items(options, section="evaluations")
        if not authenticated:
            auth_section = _auth_required_section(
                "evaluations", "Evaluations", "Local and platform evaluation runs."
            )
            items = tuple([*local_items[: max(options.limit - 1, 0)], *auth_section.items])[
                : options.limit
            ]
            return LabSection(
                key="evaluations",
                title="Evaluations",
                description="Local and platform evaluation runs.",
                items=items,
                status=f"{len(local_items)} local",
                status_style=STATUS_INFO if local_items else STATUS_WARNING,
            )

        try:
            api_client = self._api_client_factory()
            client = self._evals_client_factory(api_client)
            response = client.list_evaluations(skip=0, limit=options.limit, team_id=config.team_id)
            evaluations = response.get("evaluations", [])
            if not isinstance(evaluations, list):
                raise APIError("evaluation list response did not contain a list")
            platform_items = tuple(
                _evaluation_item(eval_data, idx) for idx, eval_data in enumerate(evaluations)
            )
            items = _combined_evaluation_items(
                platform_items=platform_items,
                local_items=tuple(local_items),
                limit=options.limit,
            )
            return LabSection(
                key="evaluations",
                title="Evaluations",
                description="Local and platform evaluation runs.",
                items=items,
                status=f"{len(items)} shown",
                status_style=STATUS_INFO,
            )
        except Exception as exc:
            warnings.append(f"Evaluations unavailable: {exc}")
            if local_items:
                error = _error_section(
                    "evaluations",
                    "Evaluations",
                    "Local and platform evaluation runs.",
                    exc,
                )
                return LabSection(
                    key="evaluations",
                    title="Evaluations",
                    description="Local and platform evaluation runs.",
                    items=tuple([*local_items[: max(options.limit - 1, 0)], *error.items])[
                        : options.limit
                    ],
                    status=f"{len(local_items)} local",
                    status_style=STATUS_WARNING,
                )
            return _error_section(
                "evaluations",
                "Evaluations",
                "Local and platform evaluation runs.",
                exc,
            )

    def _rl_section(
        self,
        options: LabLoadOptions,
        config: Config,
        authenticated: bool,
        warnings: list[str],
    ) -> LabSection:
        if not authenticated:
            return _auth_required_section(
                "training",
                "Training",
                "Hosted training runs for the active account or team.",
            )

        try:
            api_client = self._api_client_factory()
            client = self._rl_client_factory(api_client)
            runs = client.list_runs(team_id=config.team_id)
            runs = sorted(runs, key=lambda run: run.created_at, reverse=True)[: options.limit]
            items = tuple(_rl_run_item(run, idx) for idx, run in enumerate(runs))
            return LabSection(
                key="training",
                title="Training",
                description="Training runs for the active account or team.",
                items=items,
                status=f"{len(items)} shown",
                status_style=STATUS_INFO,
            )
        except Exception as exc:
            warnings.append(f"Training runs unavailable: {exc}")
            return _error_section(
                "training",
                "Training",
                "Training runs for the active account or team.",
                exc,
            )

    def _load_rl_detail(
        self,
        item: LabItem,
        *,
        include_logs: bool,
        log_tail_lines: int,
        metrics_limit: int,
        metrics_min_step: int | None,
    ) -> LabItem:
        run_id = item.title
        api_client = self._api_client_factory()
        client = self._rl_client_factory(api_client)

        run = client.get_run(run_id)
        run_data = run.model_dump(mode="json") if hasattr(run, "model_dump") else dict(run)

        progress: dict[str, Any] = {}
        metrics: list[dict[str, Any]] = []
        metrics_loaded = False
        environment_statuses: list[dict[str, Any]] = []
        reward_distribution: dict[str, Any] = {}
        reward_distribution_loaded = False
        rollout_samples: dict[str, Any] = {}
        rollout_samples_loaded = False
        rollout_samples_step: int | None = None
        logs = ""
        logs_loaded = False

        try:
            progress = client.get_progress(run_id)
        except Exception as exc:
            progress = {"error": str(exc)}

        try:
            metrics = client.get_metrics(
                run_id,
                min_step=metrics_min_step,
                limit=metrics_limit,
            )
            metrics_loaded = True
        except Exception as exc:
            metrics = [{"error": str(exc)}]

        try:
            reward_distribution = client.get_distributions(run_id, distribution_type="rewards")
            reward_distribution_loaded = True
        except Exception as exc:
            reward_distribution = {"error": str(exc)}

        sample_steps = progress.get("steps_with_samples") or progress.get("stepsWithSamples")
        rollout_samples_step = _latest_int(sample_steps)
        if rollout_samples_step is not None:
            try:
                rollout_samples = client.get_rollouts(run_id, step=rollout_samples_step, limit=50)
                rollout_samples_loaded = True
            except Exception as exc:
                rollout_samples = {"error": str(exc), "samples": []}
                rollout_samples_loaded = True
        else:
            rollout_samples = {"samples": [], "total": 0}
            rollout_samples_loaded = True

        if include_logs:
            try:
                logs = client.get_logs(run_id, tail_lines=log_tail_lines)
                logs_loaded = True
            except Exception as exc:
                logs = f"Failed to load logs: {exc}"
                logs_loaded = True

        for env in run_data.get("environments") or []:
            slug = _environment_slug(env)
            if slug is None:
                continue
            owner, name = slug.split("/", 1)
            try:
                environment_statuses.append(client.get_environment_status(owner, name))
            except Exception as exc:
                environment_statuses.append({"environment": slug, "error": str(exc)})

        detail_item = _rl_run_item(run, 0)
        raw = {
            **detail_item.raw,
            **run_data,
            "progress": progress,
            "recent_metrics": metrics,
            "metrics_loaded": metrics_loaded,
            "metrics_limit": metrics_limit,
            "metrics_page_limit": metrics_limit,
            "metrics_page_count": len(metrics),
            "metrics_min_step": metrics_min_step,
            "environment_statuses": environment_statuses,
            "reward_distribution": reward_distribution,
            "reward_distribution_loaded": reward_distribution_loaded,
            "rollout_samples": rollout_samples,
            "rollout_samples_loaded": rollout_samples_loaded,
            "rollout_samples_step": rollout_samples_step,
        }
        if include_logs:
            raw["logs_tail"] = logs
            raw["logs_loaded"] = logs_loaded
            raw["log_tail_lines"] = log_tail_lines

        metadata = list(detail_item.metadata)
        latest_step = progress.get("latest_step")
        steps_with_samples = progress.get("steps_with_samples")
        if latest_step is not None:
            metadata.append(("Latest step", str(latest_step)))
        if isinstance(steps_with_samples, list):
            metadata.append(("Sample steps", str(len(steps_with_samples))))
        if include_logs:
            metadata.append(("Logs", f"{len(logs.splitlines())} tail lines"))

        return LabItem(
            key=item.key,
            section=item.section,
            title=detail_item.title,
            subtitle=detail_item.subtitle,
            status=detail_item.status,
            status_style=detail_item.status_style,
            metadata=tuple(metadata),
            raw=raw,
        )

    def _load_evaluation_detail(self, item: LabItem) -> LabItem:
        eval_id = item.title
        api_client = self._api_client_factory()
        client = self._evals_client_factory(api_client)
        detail = client.get_evaluation(eval_id)
        samples: dict[str, Any] = {}
        try:
            samples = client.get_samples(eval_id, page=1, limit=20)
        except Exception as exc:
            samples = {"error": str(exc)}

        raw = {**detail, "samples_preview": samples}
        detail_item = _evaluation_item(raw, 0)
        raw = detail_item.raw
        metadata = list(detail_item.metadata)
        total = samples.get("total") if isinstance(samples, dict) else None
        if total is not None:
            metadata.append(("Preview samples", str(total)))
        return LabItem(
            key=item.key,
            section=item.section,
            title=detail_item.title,
            subtitle=detail_item.subtitle,
            status=detail_item.status,
            status_style=detail_item.status_style,
            metadata=tuple(metadata),
            raw=raw,
        )

    def _load_environment_detail(self, item: LabItem) -> LabItem:
        raw = dict(item.raw)
        slug = str(raw.get("slug") or item.title)
        if "/" not in slug:
            return item
        owner, name = slug.split("/", 1)
        selected_version = str(raw.get("selected_version") or "latest")
        client = self._api_client_factory(require_auth=False)
        try:
            details = client.get(f"/environmentshub/{owner}/{name}/@{selected_version}")
        except Exception:
            return item
        status: dict[str, Any] = {}
        try:
            status_response = client.get(f"/environmentshub/{owner}/{name}/status")
            status = status_response.get("data", status_response)
        except Exception as exc:
            status = {"error": str(exc)}
        versions: list[dict[str, Any]] = []
        try:
            versions_response = client.get(f"/environmentshub/{owner}/{name}/versions")
            versions_data = versions_response.get("data", versions_response)
            if isinstance(versions_data, dict):
                raw_versions = versions_data.get("versions", [])
            else:
                raw_versions = versions_data
            if isinstance(raw_versions, list):
                versions = [version for version in raw_versions if isinstance(version, dict)]
        except Exception:
            versions = []
        actions: list[dict[str, Any]] = []
        try:
            auth_client = self._api_client_factory()
            actions_response = auth_client.get(
                f"/environmentshub/{owner}/{name}/actions",
                params={"limit": 20, "offset": 0},
            )
            actions_data = actions_response.get("data", actions_response)
            if isinstance(actions_data, dict):
                raw_actions = (
                    actions_data.get("actions")
                    or actions_data.get("jobs")
                    or actions_data.get("items")
                    or []
                )
            else:
                raw_actions = actions_data
            if isinstance(raw_actions, list):
                actions = [action for action in raw_actions if isinstance(action, dict)]
        except Exception:
            actions = []

        data = details.get("data", details)
        if not isinstance(data, dict):
            data = {"value": data}
        platform = raw.get("platform")
        if not isinstance(platform, dict):
            platform = {}
        platform = {**platform, **data}
        raw = {
            **raw,
            **data,
            "platform": platform,
            "platform_detail": data,
            "selected_version": selected_version,
            "versions": versions,
            "actions": actions,
            "status": status,
        }
        metadata = list(item.metadata)
        if version := raw.get("semantic_version") or raw.get("semanticVersion"):
            metadata.append(("Semantic version", str(version)))
        if content_hash := raw.get("content_hash"):
            metadata.append(("Content hash", str(content_hash)[:12]))
        return LabItem(
            key=item.key,
            section=item.section,
            title=item.title,
            subtitle=item.subtitle,
            status=item.status,
            status_style=item.status_style,
            metadata=tuple(metadata),
            raw=raw,
        )


def load_lab_snapshot(options: LabLoadOptions) -> LabSnapshot:
    return LabDataSource().load(options)


def _row_cache_key(options: LabLoadOptions, config: Config, team: str | None) -> str:
    return lab_row_cache_key(
        workspace=options.workspace.resolve(),
        base_url=str(config.base_url),
        profile=_config_current_profile(config),
        team=team,
    )


def _account_cache_key(config: Config, team: str | None) -> str:
    return lab_account_cache_key(
        base_url=str(config.base_url),
        profile=_config_current_profile(config),
        team=team,
    )


def _hydrate_platform_sections(
    detail_cache_key: str,
    sections: list[LabSection],
    cached_sections: dict[str, LabSection],
) -> list[LabSection]:
    hydrated: list[LabSection] = []
    for section in sections:
        if section.key not in {"environments", "training", "evaluations"}:
            hydrated.append(section)
            continue
        cached = cached_sections.get(section.key)
        items = section.items
        if cached is not None and (
            _section_has_only_placeholder(section) or len(cached.items) > len(section.items)
        ):
            items = _merge_initial_items(section.key, section.items, cached.items)
        items = tuple(_hydrate_item_detail(detail_cache_key, item) for item in items)
        if items == section.items:
            hydrated.append(section)
            continue
        refreshed_at = section.refreshed_at
        row_data_origin = section.row_data_origin
        if cached is not None and cached.refreshed_at:
            refreshed_at = _newer_iso(refreshed_at, cached.refreshed_at)
            row_data_origin = _merged_origin(row_data_origin, cached.row_data_origin)
        hydrated.append(
            LabSection(
                key=section.key,
                title=section.title,
                description=section.description,
                items=items,
                status=f"{len(items)} shown",
                status_style=STATUS_INFO if items else section.status_style,
                refreshed_at=refreshed_at,
                row_data_origin=row_data_origin,
            )
        )
    return hydrated


def _mark_live_sections(sections: list[LabSection], *, refreshed_at: str) -> list[LabSection]:
    marked: list[LabSection] = []
    for section in sections:
        is_platform_section = section.key in {"environments", "training", "evaluations"}
        if is_platform_section and not _section_has_only_placeholder(section):
            marked.append(
                LabSection(
                    key=section.key,
                    title=section.title,
                    description=section.description,
                    items=section.items,
                    status=section.status,
                    status_style=section.status_style,
                    refreshed_at=_newer_iso(section.refreshed_at, refreshed_at),
                    row_data_origin=_merged_origin("live", section.row_data_origin),
                )
            )
        else:
            marked.append(section)
    return marked


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _hydrate_item_detail(detail_cache_key: str, item: LabItem) -> LabItem:
    cached = load_cached_lab_item_detail(detail_cache_key, item.key)
    if cached is None:
        return item
    if item.section == "environments":
        return _merge_environment_cached_detail(item, cached)
    return _merge_cached_item_detail(item, cached)


def _merge_cached_item_detail(item: LabItem, cached: LabItem) -> LabItem:
    raw = {**cached.raw, **item.raw}
    metadata = list(item.metadata)
    existing = {label for label, _ in metadata}
    for label, value in cached.metadata:
        if label not in existing:
            metadata.append((label, value))
            existing.add(label)
    return LabItem(
        key=item.key,
        section=item.section,
        title=item.title,
        subtitle=item.subtitle,
        status=item.status,
        status_style=item.status_style,
        metadata=tuple(metadata),
        raw=raw,
    )


def _merge_environment_cached_detail(item: LabItem, cached: LabItem) -> LabItem:
    raw = {**cached.raw, **item.raw}
    item_platform = item.raw.get("platform")
    cached_platform = cached.raw.get("platform")
    if isinstance(item_platform, dict) or isinstance(cached_platform, dict):
        raw["platform"] = {
            **(cached_platform if isinstance(cached_platform, dict) else {}),
            **(item_platform if isinstance(item_platform, dict) else {}),
        }
    for key in (
        "platform_detail",
        "versions",
        "actions",
        "status",
        "selected_version",
        "source_error",
    ):
        if key in cached.raw and key not in raw:
            raw[key] = cached.raw[key]
    metadata = list(item.metadata)
    existing = {label for label, _ in metadata}
    for label, value in cached.metadata:
        if label not in existing:
            metadata.append((label, value))
            existing.add(label)
    return LabItem(
        key=item.key,
        section=item.section,
        title=item.title,
        subtitle=item.subtitle,
        status=item.status,
        status_style=item.status_style,
        metadata=tuple(metadata),
        raw=raw,
    )


def _section_has_only_placeholder(section: LabSection) -> bool:
    if not section.items:
        return True
    return all(_is_placeholder_item(item) for item in section.items)


def _is_placeholder_item(item: LabItem) -> bool:
    return (
        item.raw.get("loading") is True
        or item.key.endswith(":error")
        or item.key.endswith(":auth-required")
        or item.title in {"Unavailable", "Sign in required"}
    )


def _cached_or_loading_section(
    key: str,
    title: str,
    description: str,
    *,
    authenticated: bool,
    local_items: tuple[LabItem, ...] = (),
    cached_section: LabSection | None = None,
) -> LabSection:
    if not authenticated and key in {"training", "evaluations"}:
        auth_section = _auth_required_section(key, title, description)
        if local_items:
            return LabSection(
                key=key,
                title=title,
                description=description,
                items=tuple([*local_items, *auth_section.items]),
                status=f"{len(local_items)} local",
                status_style=STATUS_INFO,
            )
        return auth_section

    if cached_section is not None and cached_section.items:
        items = _merge_initial_items(key, local_items, cached_section.items)
        return LabSection(
            key=key,
            title=title,
            description=description,
            items=items,
            status=f"{len(items)} cached",
            status_style=STATUS_DIM,
            refreshed_at=cached_section.refreshed_at,
            row_data_origin="mixed" if local_items else cached_section.row_data_origin,
        )

    return LabSection(
        key=key,
        title=title,
        description=description,
        items=(
            *local_items,
            LabItem(
                key=f"{key}:loading",
                section=key,
                title=f"Loading {title}",
                subtitle="Fetching platform data...",
                status="loading",
                status_style=PRIMARY,
                metadata=(("Source", "platform"),),
                raw={"loading": True},
            ),
        ),
        status="loading",
        status_style=PRIMARY,
    )


def _merge_initial_items(
    section_key: str,
    local_items: tuple[LabItem, ...],
    cached_items: tuple[LabItem, ...],
) -> tuple[LabItem, ...]:
    if not local_items:
        return cached_items
    items = [item for item in local_items if not _is_placeholder_item(item)]
    seen = {_initial_item_identity(item) for item in items}
    for item in cached_items:
        if item.section != section_key:
            item = LabItem(
                key=item.key,
                section=section_key,
                title=item.title,
                subtitle=item.subtitle,
                status=item.status,
                status_style=item.status_style,
                metadata=item.metadata,
                raw=item.raw,
            )
        identity = _initial_item_identity(item)
        if identity in seen:
            continue
        seen.add(identity)
        items.append(item)
    return tuple(items)


def _initial_item_identity(item: LabItem) -> str:
    raw_type = item.raw.get("type")
    if item.section == "environments":
        return f"environment:{item.raw.get('slug') or item.title}"
    if item.section == "evaluations" and raw_type == "local_eval":
        return f"local-eval:{item.raw.get('path') or item.title}"
    return item.key or item.title


def discover_local_eval_runs(
    workspace: Path,
    *,
    env_dir: str = "./environments",
    outputs_dir: str = "./outputs",
    limit: int = 30,
) -> list[dict[str, Any]]:
    roots: list[Path] = []
    env_root = (workspace / env_dir).resolve()
    if _safe_is_dir(env_root):
        for env_path in _safe_sorted_children(env_root):
            candidate = env_path / "outputs" / "evals"
            if _safe_is_dir(candidate):
                roots.append(candidate)

    global_root = (workspace / outputs_dir / "evals").resolve()
    if _safe_is_dir(global_root):
        roots.append(global_root)

    runs: list[dict[str, Any]] = []
    for root in roots:
        for env_model_dir in _safe_sorted_children(root):
            if not _safe_is_dir(env_model_dir) or "--" not in env_model_dir.name:
                continue
            env_id, model_part = env_model_dir.name.split("--", 1)
            model = model_part.replace("--", "/")
            for run_dir in _safe_sorted_children(env_model_dir, reverse=True):
                metadata_path = run_dir / "metadata.json"
                results_path = run_dir / "results.jsonl"
                if (
                    not _safe_is_dir(run_dir)
                    or not _safe_is_file(metadata_path)
                    or not _safe_is_file(results_path)
                ):
                    continue
                metadata = _read_json_file(metadata_path)
                runs.append(
                    {
                        "env_id": env_id,
                        "model": model,
                        "run_id": run_dir.name,
                        "path": str(run_dir),
                        "metadata": metadata,
                    }
                )
                if len(runs) >= limit:
                    return runs
    return runs


def _safe_sorted_children(path: Path, *, reverse: bool = False) -> list[Path]:
    try:
        return sorted(path.iterdir(), key=lambda child: child.name, reverse=reverse)
    except OSError:
        return []


def _safe_is_dir(path: Path) -> bool:
    try:
        return path.is_dir()
    except OSError:
        return False


def _safe_is_file(path: Path) -> bool:
    try:
        return path.is_file()
    except OSError:
        return False


def _local_eval_items(options: LabLoadOptions, *, section: str) -> list[LabItem]:
    return [
        _local_eval_item(run, idx, section=section)
        for idx, run in enumerate(
            discover_local_eval_runs(
                options.workspace,
                env_dir=options.env_dir,
                outputs_dir=options.outputs_dir,
                limit=options.limit,
            )
        )
    ]


def _combined_evaluation_items(
    *,
    platform_items: tuple[LabItem, ...],
    local_items: tuple[LabItem, ...],
    limit: int,
) -> tuple[LabItem, ...]:
    if limit <= 0:
        return ()
    if not platform_items:
        return local_items[:limit]
    if not local_items:
        return platform_items[:limit]
    local_budget = min(len(local_items), max(1, limit // 3))
    platform_budget = max(0, limit - local_budget)
    return (*platform_items[:platform_budget], *local_items[:local_budget])


def _environment_item(env: dict[str, Any], idx: int, *, section: str, scope: str) -> LabItem:
    owner = env.get("owner") or {}
    owner_name = owner.get("name") or owner.get("slug") or env.get("owner_name") or "-"
    name = env.get("name") or env.get("slug") or "-"
    env_id = f"{owner_name}/{name}" if owner_name != "-" else str(name)
    description = str(env.get("description") or "")
    updated = _format_optional_time(env.get("updated_at"))
    version = env.get("latest_version") or "-"
    stars = str(env.get("stars", 0))
    visibility = str(env.get("visibility") or "-")
    status = str(env.get("latest_ci_status") or visibility)

    return LabItem(
        key=f"environment:{env.get('id', idx)}",
        section=section,
        title=env_id,
        subtitle=description,
        status=status,
        status_style=_status_style(status),
        metadata=(
            ("Scope", scope),
            ("Version", str(version)),
            ("Stars", stars),
            ("Visibility", visibility),
            ("Updated", updated),
        ),
        raw=env,
    )


def _evaluation_item(eval_data: dict[str, Any], idx: int) -> LabItem:
    eval_id = str(eval_data.get("evaluation_id") or eval_data.get("id") or idx)
    env_names = eval_data.get("environment_names") or eval_data.get("environmentNames") or []
    env_name = _first_text(env_names) or _first_text(eval_data.get("environment_ids")) or "-"
    model = str(eval_data.get("model_name") or eval_data.get("modelName") or "-")
    status = str(eval_data.get("status") or "UNKNOWN")
    raw_metadata = eval_data.get("metadata")
    metadata: dict[str, Any] = raw_metadata if isinstance(raw_metadata, dict) else {}
    updated = _format_optional_time(eval_data.get("updated_at") or eval_data.get("updatedAt"))
    created = _format_optional_time(eval_data.get("created_at") or eval_data.get("createdAt"))
    row_date = created if created != "-" else updated
    badges = [
        {"label": "HOSTED", "style": STATUS_INFO},
        {"label": status.upper(), "style": _status_style(status)},
    ]

    return LabItem(
        key=f"evaluation:{eval_id}",
        section="evaluations",
        title=eval_id,
        subtitle=f"{env_name} · {model}",
        status=status,
        status_style=_status_style(status),
        metadata=(
            ("Environment", str(env_name)),
            ("Model", model),
            ("Type", "hosted"),
            ("Examples", str(metadata.get("num_examples", "-"))),
            ("Rollouts", str(metadata.get("rollouts_per_example", "-"))),
            ("Samples", str(eval_data.get("total_samples", eval_data.get("totalSamples", "-")))),
            ("Created", created),
            ("Updated", updated),
        ),
        raw={
            **eval_data,
            "source": "hosted",
            "row_date": "" if row_date == "-" else row_date,
            "badges": badges,
        },
    )


def _rl_run_item(run: Any, idx: int) -> LabItem:
    data = run.model_dump(mode="json") if hasattr(run, "model_dump") else dict(run)
    run_id = str(data.get("id") or idx)
    status = str(data.get("status") or "UNKNOWN")
    model = str(data.get("base_model") or data.get("baseModel") or "-")
    envs = data.get("environments") or []
    env_names = (
        ", ".join(_environment_name(env) for env in envs[:3]) if isinstance(envs, list) else "-"
    )
    if isinstance(envs, list) and len(envs) > 3:
        env_names += f", +{len(envs) - 3}"
    if not env_names:
        env_names = "-"
    created = _format_optional_time(data.get("created_at") or data.get("createdAt"))
    updated = _format_optional_time(data.get("updated_at") or data.get("updatedAt"))

    return LabItem(
        key=f"rl-run:{run_id}",
        section="training",
        title=run_id,
        subtitle=f"{model} · {env_names}",
        status=status,
        status_style=_status_style(status),
        metadata=(
            ("Name", str(data.get("name") or "-")),
            ("Model", model),
            ("Environments", env_names),
            ("Max steps", str(data.get("max_steps") or data.get("maxSteps") or "-")),
            ("Batch", str(data.get("batch_size") or data.get("batchSize") or "-")),
            ("Created", created),
            ("Updated", updated),
        ),
        raw={
            **data,
            "source": "hosted",
            "row_date": "" if created == "-" else created,
            "badges": [{"label": status.upper(), "style": _status_style(status)}],
        },
    )


def _local_eval_item(run: dict[str, Any], idx: int, *, section: str = "local-evals") -> LabItem:
    raw_metadata = run.get("metadata")
    metadata: dict[str, Any] = raw_metadata if isinstance(raw_metadata, dict) else {}
    reward = metadata.get("avg_reward")
    reward_is_numeric = isinstance(reward, int | float)
    reward_text = f"{reward:.4g}" if reward_is_numeric else "-"
    run_id = str(run.get("run_id") or idx)
    row_date = _local_path_time_ago(run.get("path"))

    return LabItem(
        key=f"local-eval:{run_id}:{idx}",
        section=section,
        title=run_id,
        subtitle=f"{run.get('env_id', '-')} · {run.get('model', '-')}",
        status="COMPLETED",
        status_style=STATUS_SUCCESS,
        metadata=(
            ("Environment", str(run.get("env_id") or "-")),
            ("Model", str(run.get("model") or "-")),
            ("Type", "local"),
            ("Avg reward", reward_text),
            ("Examples", str(metadata.get("num_examples", "-"))),
            ("Rollouts", str(metadata.get("rollouts_per_example", "-"))),
            ("Path", str(run.get("path") or "-")),
        ),
        raw={
            **run,
            "type": "local_eval",
            "source": "local",
            "row_date": row_date,
            "badges": [
                {"label": "LOCAL", "style": STATUS_LOCAL},
                {"label": "COMPLETED", "style": STATUS_SUCCESS},
            ],
        },
    )


def _workspace_context_items(
    workspace: Path,
    config: Config,
    *,
    auth_status: str,
    team: str | None,
    profiles: list[str],
    current_profile: str,
    limit: int,
) -> list[LabItem]:
    active_workspace_metadata = _read_lab_workspace_metadata(workspace)
    active_metadata = active_workspace_metadata or {}
    items = [
        _workspace_context_item(
            workspace,
            config,
            auth_status=auth_status,
            team=team,
            profiles=profiles,
            current_profile=current_profile,
            active=True,
            lab_metadata=active_metadata,
        )
    ]
    if active_workspace_metadata is None:
        items.append(_workspace_setup_item(workspace))
    items.append(_workspace_doctor_item(workspace))
    if active_workspace_metadata is not None:
        items.append(_workspace_agent_item(workspace, active_metadata))
        items.append(_workspace_agent_sync_item(workspace, active_metadata))
    items.append(_workspace_add_item(workspace))
    for inactive_workspace, lab_metadata in _discover_inactive_lab_workspaces(
        workspace, max(0, max(limit, 100) - len(items))
    ):
        items.append(
            _workspace_context_item(
                inactive_workspace,
                config,
                auth_status=auth_status,
                team=team,
                profiles=(),
                current_profile=current_profile,
                active=False,
                lab_metadata=lab_metadata,
            )
        )
    return items


def _workspace_add_item(workspace: Path) -> LabItem:
    return LabItem(
        key=f"workspace:add:{_workspace_cache_key(workspace)}",
        section="workspace",
        title="Add workspace",
        subtitle="Remember another local workspace path.",
        status="action",
        status_style=STATUS_INFO,
        metadata=(
            ("Kind", "Workspace action"),
            ("Current", str(workspace)),
        ),
        raw={
            "type": "add_workspace",
            "workspace": str(workspace),
        },
    )


def _workspace_doctor_item(workspace: Path) -> LabItem:
    command = "prime lab doctor"
    return LabItem(
        key=f"workspace:doctor:{_workspace_cache_key(workspace)}",
        section="workspace",
        title="Check workspace",
        subtitle="Run deterministic Lab checks for setup, configs, outputs, and agent assets.",
        status="doctor",
        status_style=STATUS_INFO,
        metadata=(
            ("Kind", "Workspace check"),
            ("Workspace", str(workspace)),
            ("Run", command),
        ),
        raw={
            "type": "doctor_action",
            "workspace": str(workspace),
            "command": command,
        },
    )


def _workspace_agent_item(workspace: Path, lab_metadata: dict[str, Any]) -> LabItem:
    agent = _workspace_primary_agent(lab_metadata)
    subtitle = f"{agent} · {workspace}" if agent else f"not configured · {workspace}"
    return LabItem(
        key=f"workspace:agent:{_workspace_cache_key(workspace)}",
        section="workspace",
        title="Agent",
        subtitle=subtitle,
        status="agent" if agent else "not configured",
        status_style=PRIMARY if agent else "dim",
        metadata=(
            ("Agent", str(agent or "not configured")),
            ("Workspace", str(workspace)),
        ),
        raw={
            "type": "agent_chat",
            "workspace": str(workspace),
            "agent": str(agent),
        },
    )


def _workspace_agent_sync_item(workspace: Path, lab_metadata: dict[str, Any]) -> LabItem:
    agent = _workspace_primary_agent(lab_metadata) or "codex"
    command = f"prime lab sync --agent {agent}"
    return LabItem(
        key=f"workspace:agent-sync:{_workspace_cache_key(workspace)}",
        section="workspace",
        title="Sync Lab assets",
        subtitle=f"{agent} · refresh templates, skills, docs, and local guidance",
        status="sync",
        status_style=STATUS_INFO,
        metadata=(
            ("Kind", "Lab asset sync"),
            ("Agent", agent),
            ("Workspace", str(workspace)),
            ("Run", command),
        ),
        raw={
            "type": "agent_sync",
            "workspace": str(workspace),
            "agent": agent,
            "command": command,
        },
    )


def _workspace_primary_agent(lab_metadata: dict[str, Any]) -> str:
    choices = lab_metadata.get("choices")
    return str(
        choices.get("primary_agent")
        if isinstance(choices, dict) and choices.get("primary_agent")
        else ""
    )


def _workspace_context_item(
    workspace: Path,
    config: Config,
    *,
    auth_status: str,
    team: str | None,
    profiles: list[str] | tuple[str, ...],
    current_profile: str,
    active: bool,
    lab_metadata: dict[str, Any],
) -> LabItem:
    choices = lab_metadata.get("choices")
    primary_agent = (
        choices.get("primary_agent")
        if isinstance(choices, dict) and choices.get("primary_agent")
        else None
    )
    setup_source = lab_metadata.get("setup_source")
    metadata = [
        ("Path", str(workspace)),
    ]
    if setup_source is not None:
        metadata.append(("Setup", str(setup_source)))
    if primary_agent is not None:
        metadata.append(("Primary agent", str(primary_agent)))
    if active:
        metadata.extend(
            [
                ("Auth", auth_status),
                ("Profile", current_profile),
                ("Team", team or "-"),
                ("API", config.base_url),
                ("App", config.frontend_url),
            ]
        )
    return LabItem(
        key=f"workspace:context:{_workspace_cache_key(workspace)}",
        section="workspace",
        title=workspace.name,
        subtitle=str(workspace),
        status="active" if active else "inactive",
        status_style=STATUS_SUCCESS if active else STATUS_INFO,
        metadata=tuple(metadata),
        raw={
            "type": "workspace_context",
            "workspace": str(workspace),
            "active": active,
            "auth": auth_status,
            "profile": current_profile,
            "profiles": list(profiles),
            "team": team,
            "api": config.base_url,
            "app": config.frontend_url,
            "setup_source": setup_source,
            "choices": choices if isinstance(choices, dict) else {},
        },
    )


def _workspace_setup_item(workspace: Path) -> LabItem:
    command = "prime lab setup"
    return LabItem(
        key=f"workspace:setup:{_workspace_cache_key(workspace)}",
        section="workspace",
        title="Set up Lab workspace",
        subtitle=str(workspace),
        status="setup",
        status_style=STATUS_WARNING,
        metadata=(
            ("Kind", "Setup action"),
            ("Path", str(workspace)),
            ("Run", command),
        ),
        raw={
            "type": "setup_action",
            "workspace": str(workspace),
            "command": command,
        },
    )


def _discover_inactive_lab_workspaces(
    active_workspace: Path, limit: int
) -> list[tuple[Path, dict[str, Any]]]:
    active_workspace = active_workspace.resolve()
    found: list[tuple[Path, dict[str, Any]]] = []
    seen = {active_workspace}
    for workspace in recent_workspaces():
        if workspace in seen:
            continue
        metadata = _read_lab_workspace_metadata(workspace)
        seen.add(workspace)
        found.append((workspace, metadata or {}))
    if limit <= 0:
        return found
    sibling_count = 0
    for root in _lab_workspace_search_roots(active_workspace):
        for marker_workspace in _iter_lab_workspace_markers(root):
            workspace = marker_workspace.resolve()
            if workspace in seen:
                continue
            metadata = _read_lab_workspace_metadata(workspace)
            if metadata is None:
                continue
            seen.add(workspace)
            found.append((workspace, metadata))
            record_recent_workspace(workspace)
            sibling_count += 1
            if sibling_count >= limit:
                return found
    return found


def _lab_workspace_search_roots(active_workspace: Path) -> list[Path]:
    candidates = [active_workspace.parent]
    roots: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        try:
            root = candidate.resolve()
        except OSError:
            continue
        if not _safe_is_dir(root) or root in seen:
            continue
        roots.append(root)
        seen.add(root)
    return roots


def _iter_lab_workspace_markers(root: Path) -> list[Path]:
    workspaces: list[Path] = []
    skip_dirs = {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "node_modules",
        "outputs",
    }
    max_depth = 3
    for current, dirs, _files in os.walk(root):
        current_path = Path(current)
        try:
            depth = len(current_path.relative_to(root).parts)
        except ValueError:
            depth = 0
        if _safe_is_file(current_path / ".prime" / "lab.json") or _safe_is_file(
            current_path / ".prime" / "lab_setup.json"
        ):
            workspaces.append(current_path)
        if depth >= max_depth:
            dirs[:] = []
            continue
        dirs[:] = [
            directory
            for directory in dirs
            if directory not in skip_dirs and not directory.startswith(".")
        ]
    return sorted(workspaces, key=lambda path: path.name.lower())


def _read_lab_workspace_metadata(workspace: Path) -> dict[str, Any] | None:
    metadata: dict[str, Any] = {}
    for filename in ("lab_setup.json", "lab.json"):
        path = workspace / ".prime" / filename
        if not _safe_is_file(path):
            continue
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(loaded, dict):
            metadata.update(loaded)
    return metadata or None


def _workspace_cache_key(workspace: Path) -> str:
    return hashlib.sha1(str(workspace).encode("utf-8")).hexdigest()[:12]


def _workspace_profile_items(profiles: list[str], current_profile: str) -> list[LabItem]:
    items: list[LabItem] = []
    for profile in profiles:
        current = profile == current_profile
        command = f"prime config use {profile}"
        items.append(
            LabItem(
                key=f"workspace:profile:{profile}",
                section="workspace",
                title=profile,
                subtitle="Prime CLI auth profile",
                status="current" if current else "profile",
                status_style=STATUS_SUCCESS if current else STATUS_INFO,
                metadata=(
                    ("Kind", "Auth profile"),
                    ("Current", "yes" if current else "no"),
                    ("Command", command),
                ),
                raw={
                    "type": "auth_profile",
                    "profile": profile,
                    "current": current,
                    "command": command,
                },
            )
        )
    return items


def _workspace_environment_items(
    workspace: Path, env_dir: str, limit: int, *, section: str
) -> list[LabItem]:
    return local_environment_items(workspace, env_dir, limit, section=section)


def _workspace_config_items(workspace: Path, limit: int) -> list[LabItem]:
    configs_root = workspace / "configs"
    if not configs_root.is_dir():
        return []

    items: list[LabItem] = []
    for config_kind in ("rl", "eval", "gepa"):
        kind_root = configs_root / config_kind
        if not kind_root.is_dir():
            continue
        for path in sorted(kind_root.rglob("*.toml")):
            if not path.is_file():
                continue
            items.append(_workspace_config_item(path, workspace, config_kind))
            if len(items) >= limit:
                return items
    return items


def _workspace_config_item(path: Path, workspace: Path, config_kind: str) -> LabItem:
    rel_path = _relative_path(path, workspace)
    parsed, toml_text = _read_toml_preview(path)
    command = _workspace_config_command(config_kind, rel_path)
    title = path.stem
    subtitle = _workspace_config_subtitle(config_kind, parsed) or rel_path
    return LabItem(
        key=f"workspace:config:{config_kind}:{rel_path}",
        section="workspace",
        title=title,
        subtitle=subtitle,
        status=config_kind,
        status_style=STATUS_WARNING if config_kind == "rl" else STATUS_INFO,
        metadata=(
            ("Kind", _workspace_config_kind_label(config_kind)),
            ("Path", rel_path),
            ("Run", command),
        ),
        raw={
            "type": "config_file",
            "config_kind": config_kind,
            "workspace": str(workspace.resolve()),
            "path": str(path),
            "relative_path": rel_path,
            "command": command,
            "toml": toml_text,
            "parsed": parsed,
        },
    )


def _config_profiles(config: Config) -> list[str]:
    list_environments = getattr(config, "list_environments", None)
    if not callable(list_environments):
        return [_config_current_profile(config)]
    try:
        profiles = [str(profile) for profile in list_environments()]
    except Exception:
        return [_config_current_profile(config)]
    return profiles or [_config_current_profile(config)]


def _config_current_profile(config: Config) -> str:
    return str(getattr(config, "current_environment", None) or "production")


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
    for child in sorted(path.iterdir()):
        if child.name.startswith("."):
            continue
        if child.is_file() and child.suffix in {".py", ".toml", ".md", ".json"}:
            names.append(child.name)
    return names[:12]


def _read_toml_preview(path: Path) -> tuple[dict[str, Any], str]:
    try:
        toml_text = path.read_text(encoding="utf-8")
    except OSError:
        return {}, ""
    try:
        parsed = toml.loads(toml_text)
    except toml.TomlDecodeError:
        parsed = {}
    return parsed if isinstance(parsed, dict) else {}, toml_text


def _workspace_config_command(config_kind: str, rel_path: str) -> str:
    if config_kind == "rl":
        return f"prime train run {rel_path} --yes"
    if config_kind == "eval":
        return f"prime eval run {rel_path} --hosted"
    return f"prime gepa run {rel_path}"


def _workspace_config_kind_label(config_kind: str) -> str:
    if config_kind == "rl":
        return "Training config"
    if config_kind == "eval":
        return "Evaluation config"
    return "GEPA config"


def _workspace_config_subtitle(config_kind: str, parsed: dict[str, Any]) -> str:
    if config_kind == "rl":
        model = parsed.get("model") or parsed.get("base_model") or parsed.get("baseModel")
        environments = parsed.get("environments")
        env_name = "-"
        if isinstance(environments, list) and environments:
            first_env = environments[0]
            if isinstance(first_env, dict):
                env_name = str(first_env.get("id") or first_env.get("slug") or "-")
        parts = [str(value) for value in (model, env_name) if value and value != "-"]
        return " · ".join(parts)
    if config_kind == "eval":
        evals = parsed.get("eval")
        if isinstance(evals, list) and evals:
            first_eval = evals[0]
            if isinstance(first_eval, dict):
                env = first_eval.get("env_id") or first_eval.get("environment")
                model = first_eval.get("model") or parsed.get("model")
                parts = [str(value) for value in (env, model) if value]
                if len(evals) > 1:
                    parts.append(f"{len(evals)} evals")
                return " · ".join(parts)
        env = parsed.get("env_id") or parsed.get("environment")
        model = parsed.get("model")
        return " · ".join(str(value) for value in (env, model) if value)
    env = parsed.get("env_id") or parsed.get("environment") or parsed.get("task")
    return str(env or "")


def _latest_int(values: Any) -> int | None:
    if not isinstance(values, list):
        return None
    parsed = []
    for value in values:
        try:
            parsed.append(int(value))
        except (TypeError, ValueError):
            continue
    return max(parsed) if parsed else None


def _auth_required_section(key: str, title: str, description: str) -> LabSection:
    item = LabItem(
        key=f"{key}:auth-required",
        section=key,
        title="Sign in required",
        subtitle="Run prime login to view authenticated platform data.",
        status="auth",
        status_style=STATUS_WARNING,
    )
    return LabSection(
        key=key,
        title=title,
        description=description,
        items=(item,),
        status="auth required",
        status_style=STATUS_WARNING,
    )


def _error_section(key: str, title: str, description: str, exc: Exception) -> LabSection:
    item = LabItem(
        key=f"{key}:error",
        section=key,
        title="Unavailable",
        subtitle=str(exc),
        status="error",
        status_style=STATUS_ERROR,
    )
    return LabSection(
        key=key,
        title=title,
        description=description,
        items=(item,),
        status="error",
        status_style=STATUS_ERROR,
    )


def _environment_name(env: Any) -> str:
    if not isinstance(env, dict):
        return str(env)
    return str(env.get("slug") or env.get("name") or env.get("id") or "?")


def _environment_slug(env: Any) -> str | None:
    if not isinstance(env, dict):
        return None
    slug = env.get("slug") or env.get("id")
    if isinstance(slug, str) and "/" in slug:
        return slug
    owner = env.get("owner")
    name = env.get("name")
    if isinstance(owner, str) and isinstance(name, str) and owner and name:
        return f"{owner}/{name}"
    return None


def _first_text(value: Any) -> str | None:
    if isinstance(value, list) and value:
        return str(value[0])
    if isinstance(value, str) and value:
        return value
    return None


def _format_optional_time(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return format_time_ago(value)
    except (TypeError, ValueError):
        return str(value)


def _local_path_time_ago(value: Any) -> str:
    if not value:
        return ""
    try:
        path = Path(str(value))
        mtime = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
    except OSError:
        return ""
    formatted = _format_optional_time(mtime.isoformat())
    return "" if formatted == "-" else formatted


def _read_json_file(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _status_style(status: str) -> str:
    normalized = status.upper()
    if normalized in {"COMPLETED", "SUCCESS", "READY", "PUBLIC"}:
        return STATUS_SUCCESS
    if normalized in {"RUNNING", "PENDING", "STARTING", "PRIVATE"}:
        return STATUS_WARNING
    if normalized in {"FAILED", "ERROR", "CANCELLED"}:
        return STATUS_ERROR
    return "dim"
