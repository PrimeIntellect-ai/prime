from __future__ import annotations

import json
import queue
import tarfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest
import toml
from prime_cli.commands.lab import app as lab_cli_app
from prime_cli.commands.rl import RLConfig as HostedRLConfig
from prime_cli.lab_setup import (
    LabDoctorOptions,
    LabSetupOptions,
    LabSyncOptions,
    parse_lab_doctor_args,
    parse_lab_setup_args,
    parse_lab_sync_args,
    run_lab_doctor_service,
    run_lab_setup_service,
    run_lab_sync_service,
)
from prime_lab_view.agent_adapters import agent_adapter, agent_select_options
from prime_lab_view.agent_runtime import AgentRuntime
from prime_lab_view.agent_screen import AgentChatScreen
from prime_lab_view.app import (
    EvaluationTree,
    LabOptionList,
    PrimeLabView,
    _item_details,
    _ladder_limits,
    _parse_log_records,
)
from prime_lab_view.cache import (
    cached_environment_source,
    ensure_environment_source,
    environment_source_cache_path,
    forget_recent_workspace,
    load_cached_lab_sections,
    recent_workspaces,
    record_recent_workspace,
    write_cached_lab_sections,
)
from prime_lab_view.config_screen import ConfigLaunchScreen, ConfigRunScreen
from prime_lab_view.data import LabDataSource, LabLoadOptions, discover_local_eval_runs
from prime_lab_view.environment_screen import (
    AddWorkspaceScreen,
    EnvironmentAction,
    EnvironmentScreen,
    WorkspaceScreen,
)
from prime_lab_view.eval_markdown import MathMarkdown, make_math_parser
from prime_lab_view.eval_records import LazyRunResults, LocalEvalRun
from prime_lab_view.eval_render import compute_run_overview_stats, history_groups
from prime_lab_view.eval_screen import LocalEvalRunScreen, RolloutViewer
from prime_lab_view.evaluation_browser import (
    evaluation_index,
    evaluation_model_selection_details,
    evaluation_run_selection_details,
)
from prime_lab_view.filters import FilterChoice, filter_choices
from prime_lab_view.launch_backdrop import LaunchBackdrop
from prime_lab_view.launch_screen import LaunchScreen
from prime_lab_view.models import LabItem, LabSection
from prime_lab_view.palette import TOOL_CALL
from prime_lab_view.readme import readme_links as _readme_links
from prime_lab_view.readme import readme_markdown as _readme_markdown
from prime_lab_view.rows import item_badges_text
from prime_lab_view.setup_screens import AgentSyncScreen, DoctorScreen, SetupScreen
from prime_lab_view.shell import lab_header
from prime_lab_view.source_browser import source_entries, source_preview
from prime_lab_view.training_charts import (
    chart_count as _chart_count,
)
from prime_lab_view.training_charts import (
    histogram_charts_from_raw as _histogram_charts_from_raw,
)
from prime_lab_view.training_config import (
    training_config_toml as _training_config_toml,
)
from prime_lab_view.training_config import (
    training_platform_url as _training_platform_url,
)
from prime_lab_view.training_render import training_run_widgets as _training_run_widgets
from prime_lab_view.training_screen import TrainingRunScreen, _next_log_tail_lines
from prime_lab_view.widgets import HomeLaunchPanel
from rich.console import Console
from textual.containers import Vertical, VerticalScroll
from textual.widgets import Button, Label, OptionList, Tab
from typer.testing import CliRunner


class FakeConfig:
    api_key = "token"
    team_id = "team-123"
    team_name = "research"
    base_url = "https://api.test"
    frontend_url = "https://app.test"


class FakeAPIClient:
    def __init__(self, *_args: Any, require_auth: bool = True, **_kwargs: Any) -> None:
        self.require_auth = require_auth

    def get(self, endpoint: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        params = params or {}
        if endpoint == "/environmentshub/":
            if params.get("mine_only"):
                return {
                    "data": [
                        {
                            "id": "env-private",
                            "owner": {"name": "research"},
                            "name": "private-env",
                            "description": "team env",
                            "visibility": "PRIVATE",
                            "latest_version": "0.1.0",
                            "stars": 1,
                            "updated_at": "2026-04-26T00:00:00Z",
                        }
                    ]
                }
            return {
                "data": [
                    {
                        "id": "env-private",
                        "owner": {"name": "research"},
                        "name": "private-env",
                        "description": "duplicate",
                        "visibility": "PUBLIC",
                        "latest_version": "0.1.0",
                        "stars": 2,
                    },
                    {
                        "id": "env-public",
                        "owner": {"name": "primeintellect"},
                        "name": "gsm8k",
                        "description": "math",
                        "visibility": "PUBLIC",
                        "latest_version": "1.0.0",
                        "stars": 10,
                    },
                ]
            }
        if endpoint == "/environmentshub/primeintellect/gsm8k/@latest":
            return {
                "data": {
                    "semantic_version": "1.0.0",
                    "content_hash": "abcdef123456",
                    "package_url": "https://example.test/gsm8k-1.tar.gz",
                }
            }
        if endpoint == "/environmentshub/primeintellect/gsm8k/@0.9.0":
            return {
                "data": {
                    "semantic_version": "0.9.0",
                    "content_hash": "fedcba654321",
                    "package_url": "https://example.test/gsm8k-0.tar.gz",
                }
            }
        if endpoint == "/environmentshub/primeintellect/gsm8k/status":
            return {"data": {"latest_version": {"semantic_version": "1.0.0"}}}
        if endpoint == "/environmentshub/primeintellect/gsm8k/versions":
            return {
                "data": {
                    "versions": [
                        {
                            "version": "1.0.0",
                            "semantic_version": "1.0.0",
                            "sha256": "abcdef1234567890",
                            "created_at": "2026-04-26T00:00:00Z",
                        },
                        {
                            "version": "0.9.0",
                            "semantic_version": "0.9.0",
                            "sha256": "fedcba6543210000",
                            "created_at": "2026-04-20T00:00:00Z",
                        },
                    ]
                }
            }
        if endpoint == "/environmentshub/primeintellect/gsm8k/actions":
            return {"data": {"actions": [{"name": "ci", "status": "SUCCESS"}]}}
        raise AssertionError(f"unexpected endpoint: {endpoint}")


class MineFailingAPIClient(FakeAPIClient):
    def get(self, endpoint: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        if endpoint == "/environmentshub/" and (params or {}).get("mine_only"):
            raise RuntimeError("auth query failed")
        return super().get(endpoint, params)


class FakeEvalsClient:
    def __init__(self, _api_client: Any) -> None:
        pass

    def list_evaluations(self, **_kwargs: Any) -> dict[str, Any]:
        return {
            "evaluations": [
                {
                    "evaluation_id": "eval-1",
                    "environment_names": ["gsm8k"],
                    "model_name": "openai/gpt-4.1-mini",
                    "status": "COMPLETED",
                    "is_hosted": True,
                    "metadata": {"num_examples": 5, "rollouts_per_example": 3},
                    "total_samples": 15,
                }
            ]
        }

    def get_evaluation(self, eval_id: str) -> dict[str, Any]:
        return {
            "evaluation_id": eval_id,
            "environment_names": ["gsm8k"],
            "model_name": "openai/gpt-4.1-mini",
            "status": "COMPLETED",
            "total_samples": 2,
            "avg_score": 0.5,
            "chartData": {
                "histogramData": [
                    {"range": "0.000-0.500", "count": 1},
                    {"range": "0.500-1.000", "count": 1},
                ],
                "stats": {
                    "avgReward": 0.5,
                    "medianReward": 0.5,
                    "minReward": 0.0,
                    "maxReward": 1.0,
                    "totalResults": 2,
                },
            },
            "detailedMetrics": {
                "detectedMetrics": ["accuracy"],
                "metricsStats": {
                    "accuracy": {
                        "mean": 0.5,
                        "stdDev": 0.5,
                        "count": 2,
                        "p50": 0.5,
                    }
                },
            },
        }

    def get_samples(self, _eval_id: str, page: int, limit: int) -> dict[str, Any]:
        return {"samples": [{"reward": 1.0}], "total": 1, "page": page, "limit": limit}


@dataclass
class FakeRun:
    id: str = "run-1"
    status: str = "RUNNING"
    created_at: datetime = datetime(2026, 4, 26, tzinfo=timezone.utc)

    def model_dump(self, mode: str = "python") -> dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status,
            "base_model": "openai/gpt-5-mini",
            "environments": [{"slug": "primeintellect/gsm8k"}],
            "max_steps": 100,
            "batch_size": 64,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.created_at.isoformat(),
        }


class FakeRLClient:
    def __init__(self, _api_client: Any) -> None:
        pass

    def list_runs(self, team_id: str | None = None) -> list[FakeRun]:
        assert team_id == "team-123"
        return [FakeRun()]

    def get_run(self, run_id: str) -> FakeRun:
        assert run_id == "run-1"
        return FakeRun()

    def get_progress(self, run_id: str) -> dict[str, Any]:
        assert run_id == "run-1"
        return {"latest_step": 12, "steps_with_samples": [0, 4, 8, 12]}

    def get_metrics(
        self,
        run_id: str,
        min_step: int | None = None,
        max_step: int | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        assert run_id == "run-1"
        assert min_step in {None, 13}
        assert max_step is None
        assert limit in {10, 20, 40, 80, 160, 320, 500}
        return [{"step": 12, "reward/all/mean": 0.8}]

    def get_logs(self, run_id: str, tail_lines: int = 1000) -> str:
        assert run_id == "run-1"
        assert tail_lines == 50
        return "training log line"

    def get_distributions(
        self, run_id: str, distribution_type: str | None = None, step: int | None = None
    ) -> dict[str, Any]:
        assert run_id == "run-1"
        assert distribution_type == "rewards"
        assert step is None
        return {"step": 12, "bins": [{"range": "0.0-1.0", "count": 8}]}

    def get_rollouts(
        self,
        run_id: str,
        step: int,
        page: int = 1,
        limit: int = 100,
    ) -> dict[str, Any]:
        assert run_id == "run-1"
        assert step == 12
        assert page == 1
        assert limit == 50
        return {
            "samples": [
                {
                    "reward": 0.8,
                    "prompt": [{"role": "user", "content": "2+2?"}],
                    "completion": [{"role": "assistant", "content": "4"}],
                }
            ],
            "total": 1,
            "page": page,
            "limit": limit,
        }

    def get_environment_status(self, owner: str, name: str) -> dict[str, Any]:
        return {"environment": f"{owner}/{name}", "status": "SUCCESS"}


class FailingAPIClient(FakeAPIClient):
    def get(self, endpoint: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        raise RuntimeError(f"{endpoint} unavailable")


class FailingEvalsClient(FakeEvalsClient):
    def list_evaluations(self, **_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("evaluations unavailable")


class FailingRLClient(FakeRLClient):
    def list_runs(self, team_id: str | None = None) -> list[FakeRun]:
        raise RuntimeError("training unavailable")


def make_source() -> LabDataSource:
    return LabDataSource(
        api_client_factory=FakeAPIClient,
        evals_client_factory=FakeEvalsClient,
        rl_client_factory=FakeRLClient,
        config_factory=FakeConfig,
    )


def test_lab_view_snapshot_prioritizes_training_and_auth_environments(tmp_path: Path) -> None:
    snapshot = make_source().load(LabLoadOptions(limit=10, workspace=tmp_path))

    assert [section.title for section in snapshot.sections] == [
        "Home",
        "Environments",
        "Training",
        "Evaluations",
    ]

    training = snapshot.section("training")
    assert training is not None
    assert training.items[0].title == "run-1"
    assert "primeintellect/gsm8k" in training.items[0].subtitle

    environments = snapshot.section("environments")
    assert environments is not None
    assert [item.title for item in environments.items] == [
        "research/private-env",
        "primeintellect/gsm8k",
    ]
    assert environments.items[0].metadata[0] == ("Source", "platform")
    assert environments.items[0].raw["badges"][0]["label"] == "PRIVATE"
    assert environments.items[1].metadata[0] == ("Source", "platform")
    assert environments.items[1].raw["badges"][0]["label"] == "PUBLIC"


def test_lab_view_merges_local_and_platform_environments(tmp_path: Path) -> None:
    env_path = tmp_path / "environments" / "private-env"
    (env_path / ".prime").mkdir(parents=True)
    (env_path / "README.md").write_text("# Private Env\n\nTeam environment.", encoding="utf-8")
    (env_path / "pyproject.toml").write_text(
        '[project]\nname = "private-env"\nversion = "0.1.0"\ndescription = "local desc"\n',
        encoding="utf-8",
    )
    (env_path / ".prime" / ".env-metadata.json").write_text(
        json.dumps(
            {
                "environment_id": "env-private",
                "owner": "research",
                "name": "private-env",
                "version": "0.1.0",
                "visibility": "PRIVATE",
            }
        ),
        encoding="utf-8",
    )

    snapshot = make_source().load(LabLoadOptions(limit=10, workspace=tmp_path))
    environments = snapshot.section("environments")

    assert environments is not None
    private = environments.items[0]
    assert [item.title for item in environments.items].count("research/private-env") == 1
    assert private.raw["sources"] == ["local", "platform"]
    assert [badge["label"] for badge in private.raw["badges"]] == ["LOCAL", "PRIVATE"]
    assert private.raw["local"]["relative_path"] == "environments/private-env"
    assert private.raw["local"]["readme_preview"] == "# Private Env Team environment."
    assert len(private.raw["local"]["content_hash"]) == 64


def test_lab_view_environment_badges_keep_independent_styles() -> None:
    item = LabItem(
        key="environment:alphabet",
        section="environments",
        title="primeintellect/alphabet-sort",
        subtitle="",
        status="LOCAL PUBLIC",
        status_style="bold #38bdf8",
        metadata=(),
        raw={
            "badges": [
                {"label": "LOCAL", "style": "bold #38bdf8"},
                {"label": "PUBLIC", "style": "bold #84cc16"},
            ]
        },
    )

    badges = item_badges_text(item)

    assert badges.plain == "  LOCAL PUBLIC"
    assert [str(span.style) for span in badges.spans] == ["bold #38bdf8", "bold #84cc16"]


def test_lab_view_local_environment_source_hash_ignores_generated_outputs(
    tmp_path: Path,
) -> None:
    env_path = tmp_path / "environments" / "hash-env"
    env_path.mkdir(parents=True)
    (env_path / "README.md").write_text("# Hash Env\n", encoding="utf-8")
    (env_path / "hash_env.py").write_text("VALUE = 1\n", encoding="utf-8")
    (env_path / "pyproject.toml").write_text(
        '[project]\nname = "hash-env"\nversion = "0.1.0"\n',
        encoding="utf-8",
    )

    first = make_source().load(LabLoadOptions(limit=10, workspace=tmp_path))
    first_envs = first.section("environments")
    assert first_envs is not None
    first_hash = next(
        item.raw["local"]["content_hash"] for item in first_envs.items if item.title == "hash-env"
    )

    (env_path / "outputs").mkdir()
    (env_path / "outputs" / "run.json").write_text("{}", encoding="utf-8")
    (env_path / "__pycache__").mkdir()
    (env_path / "__pycache__" / "hash_env.pyc").write_bytes(b"cache")
    (env_path / "dist").mkdir()
    (env_path / "dist" / "artifact.whl").write_bytes(b"wheel")

    second = make_source().load(LabLoadOptions(limit=10, workspace=tmp_path))
    second_envs = second.section("environments")
    assert second_envs is not None
    second_hash = next(
        item.raw["local"]["content_hash"] for item in second_envs.items if item.title == "hash-env"
    )

    assert second_hash == first_hash


def test_lab_environment_readme_hides_html_href_and_badge_urls() -> None:
    readme = """
<a href="https://github.com/PrimeIntellect-ai/verifiers/tree/main/environments/sentence_repeater">
<img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge" alt="Source Code">
</a>

### Overview
- **Environment ID**: `sentence-repeater`
https://app.primeintellect.ai/dashboard/environments/primeintellect/sentence-repeater
"""

    rendered = _readme_markdown(readme, "primeintellect/sentence-repeater")
    links = _readme_links(readme, "primeintellect/sentence-repeater")

    assert "<a" not in rendered
    assert "href=" not in rendered
    assert "img.shields.io" not in rendered
    assert "[Source Code](" in rendered
    assert links == [
        (
            "Source Code",
            "https://github.com/PrimeIntellect-ai/verifiers/tree/main/environments/sentence_repeater",
        ),
        (
            "sentence-repeater on platform",
            "https://app.primeintellect.ai/dashboard/environments/primeintellect/sentence-repeater",
        ),
    ]


def test_lab_environment_cache_downloads_source_and_writes_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    archive = _tar_bytes({"README.md": "# Cached Env\n", "env.py": "VALUE = 1\n"})

    class FakeStream:
        def __enter__(self) -> "FakeStream":
            return self

        def __exit__(self, *_args: Any) -> None:
            return None

        def raise_for_status(self) -> None:
            return None

        def iter_bytes(self, chunk_size: int) -> list[bytes]:
            return [archive[:chunk_size], archive[chunk_size:]]

    def fake_stream(method: str, url: str, timeout: float) -> FakeStream:
        assert method == "GET"
        assert url == "https://example.test/env.tar.gz"
        assert timeout == 60.0
        return FakeStream()

    monkeypatch.setattr("prime_lab_view.cache.httpx.stream", fake_stream)

    cached = ensure_environment_source(
        {
            "slug": "research/cached-env",
            "platform": {
                "package_url": "https://example.test/env.tar.gz",
                "semanticVersion": "0.2.0",
            },
        }
    )

    assert cached is not None
    assert cached.root == environment_source_cache_path("research", "cached-env", "0.2.0")
    assert (cached.root / "README.md").read_text(encoding="utf-8") == "# Cached Env\n"
    assert cached.manifest["slug"] == "research/cached-env"
    assert cached.manifest["version"] == "0.2.0"


def test_lab_environment_cache_finds_existing_source_without_package_url(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    root = environment_source_cache_path("research", "cached-env", "0.2.0")
    manifest = root / ".prime" / "lab-cache.json"
    manifest.parent.mkdir(parents=True)
    (root / "README.md").write_text("# Cached Env\n", encoding="utf-8")
    manifest.write_text(
        json.dumps(
            {
                "slug": "research/cached-env",
                "version": "0.2.0",
                "cached_at": "2026-01-01T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )

    cached = cached_environment_source({"slug": "research/cached-env", "latest_version": "0.2.0"})

    assert cached is not None
    assert cached.root == root


def test_lab_environment_cache_rejects_unsafe_archive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    archive = _tar_bytes({"../evil.txt": "nope"})

    class FakeStream:
        def __enter__(self) -> "FakeStream":
            return self

        def __exit__(self, *_args: Any) -> None:
            return None

        def raise_for_status(self) -> None:
            return None

        def iter_bytes(self, chunk_size: int) -> list[bytes]:
            return [archive[:chunk_size], archive[chunk_size:]]

    monkeypatch.setattr("prime_lab_view.cache.httpx.stream", lambda *_args, **_kwargs: FakeStream())

    with pytest.raises(ValueError, match="path with '..'"):
        ensure_environment_source(
            {
                "slug": "research/unsafe-env",
                "platform": {
                    "package_url": "https://example.test/env.tar.gz",
                    "semanticVersion": "0.2.0",
                },
            }
        )


def test_source_browser_lists_and_previews_workspace_files(tmp_path: Path) -> None:
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "run.toml").write_text("max_steps = 10\n", encoding="utf-8")

    entries = source_entries(tmp_path)
    names = [entry.name for entry in entries]

    assert "configs" in names
    preview = _render_renderable(source_preview(tmp_path / "configs" / "run.toml"))
    assert "max_steps" in preview


def test_lab_view_initial_snapshot_shows_local_context_and_platform_loading(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "outputs" / "evals" / "gsm8k--openai--gpt-4" / "run-a"
    run_dir.mkdir(parents=True)
    (run_dir / "metadata.json").write_text(
        '{"avg_reward": 0.5, "num_examples": 1, "rollouts_per_example": 2}',
        encoding="utf-8",
    )
    (run_dir / "results.jsonl").write_text('{"reward": 0.5}\n', encoding="utf-8")

    snapshot = make_source().load_initial(LabLoadOptions(limit=10, workspace=tmp_path))

    assert [section.title for section in snapshot.sections] == [
        "Home",
        "Environments",
        "Training",
        "Evaluations",
    ]
    training = snapshot.section("training")
    environments = snapshot.section("environments")
    evaluations = snapshot.section("evaluations")
    assert training is not None
    assert environments is not None
    assert evaluations is not None
    assert training.status == "loading"
    assert environments.status == "loading"
    assert evaluations.status == "loading"
    assert evaluations.items[0].title == "run-a"
    workspace = snapshot.section("workspace")
    assert workspace is not None
    assert workspace.items[0].title == tmp_path.name


def test_lab_view_initial_snapshot_hydrates_cached_platform_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    make_source().load(LabLoadOptions(limit=10, workspace=workspace))

    snapshot = make_source().load_initial(LabLoadOptions(limit=10, workspace=workspace))

    training = snapshot.section("training")
    environments = snapshot.section("environments")
    evaluations = snapshot.section("evaluations")
    assert training is not None
    assert environments is not None
    assert evaluations is not None
    assert training.items[0].title == "run-1"
    assert environments.items[0].title == "research/private-env"
    assert evaluations.items[0].title == "eval-1"
    assert training.status == "10 cached" or training.status.endswith("cached")


def test_lab_view_row_cache_never_shrinks_on_smaller_refresh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    cache_key = "row-cache-shrink-regression"
    full_items = tuple(
        LabItem(
            key=f"rl-run:{index}",
            section="training",
            title=f"run-{index}",
            subtitle="model · env",
        )
        for index in range(10)
    )
    smaller_items = full_items[:5]

    write_cached_lab_sections(
        cache_key,
        (
            LabSection(
                key="training",
                title="Training",
                description="Training runs.",
                items=full_items,
            ),
        ),
    )
    write_cached_lab_sections(
        cache_key,
        (
            LabSection(
                key="training",
                title="Training",
                description="Training runs.",
                items=smaller_items,
            ),
        ),
    )

    cached = load_cached_lab_sections(cache_key, limit=100)

    assert "training" in cached
    assert [item.title for item in cached["training"].items] == [
        f"run-{index}" for index in range(10)
    ]


def test_lab_view_platform_cache_survives_refresh_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    make_source().load(LabLoadOptions(limit=10, workspace=workspace))
    failing_source = LabDataSource(
        api_client_factory=FailingAPIClient,
        evals_client_factory=FailingEvalsClient,
        rl_client_factory=FailingRLClient,
        config_factory=FakeConfig,
    )

    snapshot = failing_source.load(LabLoadOptions(limit=5, workspace=workspace))

    training = snapshot.section("training")
    environments = snapshot.section("environments")
    evaluations = snapshot.section("evaluations")
    assert training is not None
    assert environments is not None
    assert evaluations is not None
    assert any(item.title == "run-1" for item in training.items)
    assert any(item.title == "research/private-env" for item in environments.items)
    assert any(item.title == "eval-1" for item in evaluations.items)


def test_lab_view_home_shows_inactive_sibling_workspaces(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    active = tmp_path / "active-lab"
    inactive = tmp_path / "inactive-lab"
    active.mkdir()
    (inactive / ".prime").mkdir(parents=True)
    (inactive / ".prime" / "lab.json").write_text(
        '{"setup_source": "prime lab setup", "choices": {"primary_agent": "opencode"}}',
        encoding="utf-8",
    )

    snapshot = make_source().load(LabLoadOptions(limit=10, workspace=active))
    workspace = snapshot.section("workspace")

    assert workspace is not None
    workspaces = [item for item in workspace.items if item.raw.get("type") == "workspace_context"]
    assert [item.title for item in workspaces] == ["active-lab", "inactive-lab"]
    assert workspaces[0].status == "active"
    assert workspaces[1].status == "inactive"
    assert workspaces[1].metadata == (
        ("Path", str(inactive.resolve())),
        ("Setup", "prime lab setup"),
        ("Primary agent", "opencode"),
    )
    assert inactive.resolve() in recent_workspaces()


def test_lab_view_home_shows_remembered_non_sibling_workspaces(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    active = tmp_path / "root-a" / "active-lab"
    remembered = tmp_path / "root-b" / "remembered-lab"
    active.mkdir(parents=True)
    remembered.mkdir(parents=True)
    record_recent_workspace(remembered)

    snapshot = make_source().load(LabLoadOptions(limit=5, workspace=active))
    workspace = snapshot.section("workspace")

    assert workspace is not None
    workspaces = [item for item in workspace.items if item.raw.get("type") == "workspace_context"]
    assert [item.title for item in workspaces] == ["active-lab", "remembered-lab"]
    assert workspaces[1].status == "inactive"

    forget_recent_workspace(remembered)
    assert remembered.resolve() not in recent_workspaces()


def test_lab_view_loads_training_logs_and_environment_status() -> None:
    source = make_source()
    snapshot = source.load(LabLoadOptions(limit=10))
    training = snapshot.section("training")
    assert training is not None

    item = source.load_item_detail(training.items[0], include_logs=True)

    assert item.section == "training"
    assert item.raw["progress"]["latest_step"] == 12
    assert item.raw["logs_tail"] == "training log line"
    assert item.raw["log_tail_lines"] == 50
    assert item.raw["reward_distribution"] == {
        "step": 12,
        "bins": [{"range": "0.0-1.0", "count": 8}],
    }
    assert item.raw["rollout_samples_step"] == 12
    assert item.raw["rollout_samples"]["samples"][0]["reward"] == 0.8
    assert item.raw["environment_statuses"] == [
        {"environment": "primeintellect/gsm8k", "status": "SUCCESS"}
    ]


def test_training_data_tab_uses_rollout_viewer() -> None:
    source = make_source()
    snapshot = source.load(LabLoadOptions(limit=10))
    training = snapshot.section("training")
    assert training is not None
    item = source.load_item_detail(training.items[0])

    widgets = _training_run_widgets(item, include_logs=False, active_tab="data")

    assert any(isinstance(widget, RolloutViewer) for widget in widgets)


def test_training_data_tab_normalizes_serialized_rollout_samples() -> None:
    completion = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": json.dumps(
                [
                    {
                        "id": "call-1",
                        "function": {"name": "search", "arguments": '{"query":"prime"}'},
                    }
                ]
            ),
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "found it"},
        {"role": "assistant", "content": "done"},
    ]
    item = LabItem(
        key="training:run-1",
        section="training",
        title="run-1",
        subtitle="",
        status="COMPLETED",
        status_style="green",
        raw={
            "id": "run-1",
            "progress": {"steps_with_samples": [90]},
            "rollout_samples_loaded": True,
            "rollout_samples_step": 90,
            "rollout_samples": {
                "samples": [
                    {
                        "reward": 1.0,
                        "prompt": json.dumps([{"role": "user", "content": "question"}]),
                        "completion": json.dumps(completion),
                    }
                ],
                "total": 1,
                "page": 1,
                "limit": 50,
            },
        },
    )

    widgets = _training_run_widgets(item, include_logs=False, active_tab="data")
    viewer = next(widget for widget in widgets if isinstance(widget, RolloutViewer))
    record = viewer.records[0]
    sections = viewer._history_section_data(record)

    assert isinstance(record["completion"], list)
    assert isinstance(record["completion"][0]["tool_calls"], list)
    assert sections[1].title.startswith("1. assistant")
    assert "Call" in sections[1].nested_sections[0].body
    assert "Output" in sections[1].nested_sections[0].body


def test_rollout_viewer_escapes_xml_in_markdown_body() -> None:
    viewer = RolloutViewer(
        [{"completion": [{"role": "assistant", "content": "<answer>hello endpoint</answer>"}]}]
    )

    body = viewer._make_body_widget("<answer>hello endpoint</answer>", "completion")

    assert isinstance(body, MathMarkdown)
    assert getattr(body, "_initial_markdown") == "<answer>hello endpoint</answer>"
    tokens = make_math_parser().parse("<answer>hello endpoint</answer>")
    inline = next(token for token in tokens if token.type == "inline")
    assert [(child.type, child.content) for child in inline.children or []] == [
        ("text", "<answer>hello endpoint</answer>")
    ]


def test_rollout_tool_calls_use_non_error_palette() -> None:
    assert TOOL_CALL in RolloutViewer.DEFAULT_CSS
    tool_call_block = RolloutViewer.DEFAULT_CSS.split(".tool-call-section", 1)[1].split("}", 1)[0]
    assert "$accent" not in tool_call_block


def test_lab_view_renders_platform_histogram_data() -> None:
    raw = {
        "avg_score": 0.5,
        "chartData": {
            "histogramData": [
                {
                    "binStart": 0.0,
                    "binEnd": 0.2,
                    "count": 1,
                    "range": "0.000-0.200",
                },
                {
                    "binStart": 0.2,
                    "binEnd": 0.4,
                    "count": 0,
                    "range": "0.200-0.400",
                },
            ]
        },
    }

    histograms = _histogram_charts_from_raw(raw)

    assert [label for label, _ in histograms] == ["Score Distribution"]
    assert (
        _chart_count(
            {
                "recent_metrics": [],
                "reward_distribution": {"chartData": raw["chartData"]},
            }
        )
        == 1
    )


def test_lab_view_renders_eval_details_without_raw_json() -> None:
    source = make_source()
    snapshot = source.load(LabLoadOptions(limit=10))
    evaluations = snapshot.section("evaluations")
    assert evaluations is not None

    item = source.load_item_detail(evaluations.items[0])
    rendered = _render_details(item)

    assert "Score Distribution" in rendered
    assert "Detailed Metrics" in rendered
    assert "reward/all/mean" not in rendered
    assert "Raw" not in rendered
    assert "chartData" not in rendered


def test_lab_view_renders_environment_details_without_raw_json() -> None:
    source = make_source()
    snapshot = source.load(LabLoadOptions(limit=10))
    environments = snapshot.section("environments")
    assert environments is not None

    item = source.load_item_detail(environments.items[1])
    rendered = _render_details(item)

    assert "Status" in rendered
    assert "1.0.0" in rendered
    assert "prime env install primeintellect/gsm8k" in rendered
    assert "Raw" not in rendered
    assert "content_hash" not in rendered


def test_lab_view_loads_environment_versions_and_actions() -> None:
    source = make_source()
    snapshot = source.load(LabLoadOptions(limit=10))
    environments = snapshot.section("environments")
    assert environments is not None

    item = source.load_item_detail(environments.items[1])
    versioned = source.load_item_detail(
        LabItem(
            key=item.key,
            section=item.section,
            title=item.title,
            subtitle=item.subtitle,
            status=item.status,
            status_style=item.status_style,
            metadata=item.metadata,
            raw={**item.raw, "selected_version": "0.9.0"},
        )
    )

    assert [version["version"] for version in item.raw["versions"]] == ["1.0.0", "0.9.0"]
    assert item.raw["actions"] == [{"name": "ci", "status": "SUCCESS"}]
    assert versioned.raw["semantic_version"] == "0.9.0"
    assert versioned.raw["content_hash"] == "fedcba654321"


def test_lab_view_training_sidebar_config_filters_nulls() -> None:
    config = _training_config_toml(
        {
            "name": "wiki-search",
            "base_model": "Qwen/Qwen3.5-2B",
            "max_steps": 200,
            "learning_rate": None,
            "environments": [
                {
                    "id": "primeintellect/wiki-search",
                    "version": "0.1.23",
                    "args": {},
                }
            ],
        }
    )

    assert 'model = "Qwen/Qwen3.5-2B"' in config
    assert "max_steps = 200" in config
    assert "\n[[env]]" in config
    assert "learning_rate" not in config
    assert "args" not in config
    assert "seq_len" not in config
    assert "max_tokens" not in config
    HostedRLConfig.model_validate(toml.loads(config))


def test_lab_view_training_config_uses_sampling_for_max_tokens() -> None:
    config = _training_config_toml(
        {
            "base_model": "Qwen/Qwen3.5-2B",
            "max_steps": 100,
            "seq_len": 65536,
            "max_tokens": 8192,
            "environments": ["primeintellect/alphabet-sort@0.1.8"],
        }
    )

    parsed = toml.loads(config)
    assert "seq_len" not in parsed
    assert "max_tokens" not in parsed
    assert parsed["sampling"]["max_tokens"] == 8192
    assert parsed["env"] == [{"id": "primeintellect/alphabet-sort", "version": "0.1.8"}]
    HostedRLConfig.model_validate(parsed)


def test_lab_view_parses_json_logs() -> None:
    records = _parse_log_records(
        '{"timestamp":"2026-04-26T00:00:00Z","level":"INFO","step":12,'
        '"message":"sample complete","reward":0.5}\nplain line'
    )

    assert records[0]["time"] == "2026-04-26T00:00:00Z"
    assert records[0]["level"] == "INFO"
    assert records[0]["step"] == "12"
    assert records[0]["message"] == "sample complete"
    assert records[0]["fields"] == "reward=0.5"
    assert records[1] == {"message": "plain line", "structured": False}


def test_lab_view_parses_progress_logs() -> None:
    records = _parse_log_records(
        '{"timestamp":"2026-04-26T03:15:02Z","level":"INFO","step":197,'
        '"type":"progress","desc":"Generating rollouts (train)",'
        '"current":256,"total":512,"percent":50}'
    )

    assert records[0]["message"] == "Generating rollouts (train)  256/512 (50%)"
    assert records[0]["fields"] == ""


def test_lab_view_keeps_public_environments_when_auth_catalog_fails(tmp_path: Path) -> None:
    source = LabDataSource(
        api_client_factory=MineFailingAPIClient,
        evals_client_factory=FakeEvalsClient,
        rl_client_factory=FakeRLClient,
        config_factory=FakeConfig,
    )

    snapshot = source.load(LabLoadOptions(limit=10, workspace=tmp_path))
    environments = snapshot.section("environments")

    assert environments is not None
    assert [item.title for item in environments.items] == [
        "research/private-env",
        "primeintellect/gsm8k",
    ]
    assert environments.items[0].metadata[0] == ("Source", "platform")
    assert environments.items[0].raw["badges"][0]["label"] == "PUBLIC"
    assert snapshot.warnings == ("Authenticated environments unavailable: auth query failed",)


def test_discover_local_eval_runs(tmp_path: Path) -> None:
    run_dir = tmp_path / "outputs" / "evals" / "gsm8k--openai--gpt-4" / "run-a"
    run_dir.mkdir(parents=True)
    (run_dir / "metadata.json").write_text(
        '{"avg_reward": 0.5, "num_examples": 1, "rollouts_per_example": 2}',
        encoding="utf-8",
    )
    (run_dir / "results.jsonl").write_text('{"reward": 0.5}\n', encoding="utf-8")

    runs = discover_local_eval_runs(tmp_path)

    assert runs == [
        {
            "env_id": "gsm8k",
            "model": "openai/gpt-4",
            "run_id": "run-a",
            "path": str(run_dir),
            "metadata": {"avg_reward": 0.5, "num_examples": 1, "rollouts_per_example": 2},
        }
    ]


def test_local_eval_lazy_records_and_overview_stats(tmp_path: Path) -> None:
    run_dir = tmp_path / "outputs" / "evals" / "gsm8k--openai--gpt-4" / "run-a"
    run_dir.mkdir(parents=True)
    (run_dir / "metadata.json").write_text(
        '{"avg_reward": 0.75, "num_examples": 1, "rollouts_per_example": 2}',
        encoding="utf-8",
    )
    (run_dir / "results.jsonl").write_text(
        "\n".join(
            [
                '{"reward": 0.5, "metrics": {"accuracy": 0.0}}',
                '{"reward": 1.0, "metrics": {"accuracy": 1.0}, "num_turns": 3}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    run = LocalEvalRun(env_id="gsm8k", model="openai/gpt-4", run_id="run-a", path=run_dir)

    records = LazyRunResults(run)
    try:
        assert records.count_hint() == 2
        assert records[1]["reward"] == 1.0
    finally:
        records.close()

    stats = compute_run_overview_stats(run)
    assert stats.rewards == [0.5, 1.0]
    assert {summary.name: summary.avg for summary in stats.metric_summaries} == {
        "accuracy": 0.5,
        "num_turns": 3.0,
    }


def test_local_eval_selection_details_match_verifiers_sidebar_logic(tmp_path: Path) -> None:
    run_dir = tmp_path / "outputs" / "evals" / "gsm8k--openai--gpt-4" / "run-a"
    run_dir.mkdir(parents=True)
    metadata = {
        "avg_reward": 0.75,
        "date": "2026-04-26",
        "time": "03:15:00",
        "num_examples": 1,
        "rollouts_per_example": 2,
        "sampling_args": {"temperature": 0.7, "max_tokens": 128},
        "pass_at_k": {"1": 0.5},
        "pass_all_k": {"2": 1.0},
    }
    (run_dir / "metadata.json").write_text('{"avg_reward": 0.75}', encoding="utf-8")
    (run_dir / "results.jsonl").write_text(
        '{"reward": 0.5, "metrics": {"accuracy": 0.0}}\n'
        '{"reward": 1.0, "metrics": {"accuracy": 1.0}}\n',
        encoding="utf-8",
    )
    item = _local_eval_item("eval:run-a", "run-a", run_dir, metadata)
    stats = compute_run_overview_stats(LocalEvalRun.from_item(item))

    details = _render_renderable(evaluation_run_selection_details(item, stats))

    assert "Run settings" in details
    assert "sampling.temperature" in details
    assert "Pass rates" in details
    assert "pass@1" in details
    assert "pass-all@2" in details
    assert "Rollout rewards" in details


def test_local_eval_model_details_include_setting_variations(tmp_path: Path) -> None:
    run_a = _local_eval_item(
        "eval:run-a",
        "run-a",
        tmp_path / "run-a",
        {
            "avg_reward": 0.5,
            "num_examples": 1,
            "rollouts_per_example": 1,
            "sampling_args": {"temperature": 0.2},
        },
    )
    run_b = _local_eval_item(
        "eval:run-b",
        "run-b",
        tmp_path / "run-b",
        {
            "avg_reward": 0.8,
            "num_examples": 1,
            "rollouts_per_example": 2,
            "sampling_args": {"temperature": 0.8},
        },
    )
    index = evaluation_index([run_a, run_b])

    details = _render_renderable(evaluation_model_selection_details("gsm8k", "openai/gpt-4", index))

    assert "Setting variations" in details
    assert "rollouts/example" in details
    assert "sampling.temperature" in details
    assert "0.2 (1 run)" in details
    assert "0.8 (1 run)" in details


def test_local_eval_history_groups_pair_tool_calls_with_outputs() -> None:
    completion = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "function": {"name": "search", "arguments": '{"query":"prime"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "found it"},
        {"role": "assistant", "content": "done"},
    ]

    groups = history_groups(completion)

    assert groups[0]["kind"] == "assistant-tools"
    assert groups[0]["tool_outputs"][0]["content"] == "found it"
    assert groups[1]["kind"] == "message"


def test_lab_view_ladder_limits() -> None:
    assert _ladder_limits(3) == (3,)
    assert _ladder_limits(30) == (5, 10, 20, 30)
    assert _ladder_limits(100) == (5, 10, 20, 40, 80, 100)
    assert _ladder_limits(1000) == (5, 10, 20, 40, 80, 160, 320, 640, 1000)


def test_lab_view_log_tail_doubles_after_default() -> None:
    assert _next_log_tail_lines(50) == 1000
    assert _next_log_tail_lines(1000) == 2000
    assert _next_log_tail_lines(2000) == 4000


def test_lab_view_filter_choices_preserve_prefilled_results() -> None:
    choices = [
        FilterChoice("a", "gsm8k run", "gsm8k openai completed", "gsm8k"),
        FilterChoice("b", "wiki run", "wiki qwen running", "wiki"),
    ]

    assert [choice.key for choice in filter_choices(choices, "")] == ["a", "b"]
    assert [choice.key for choice in filter_choices(choices, "qwen")] == ["b"]


def test_prime_lab_launches_viewer_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    def fake_run_lab_view(**kwargs: Any) -> None:
        calls.append(kwargs)

    monkeypatch.setattr("prime_lab_view.run_lab_view", fake_run_lab_view)

    result = CliRunner().invoke(lab_cli_app, [])

    assert result.exit_code == 0
    assert calls
    assert calls[0]["limit"] == 1000


def test_prime_lab_view_alias_launches_viewer(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    def fake_run_lab_view(**kwargs: Any) -> None:
        calls.append(kwargs)

    monkeypatch.setattr("prime_lab_view.run_lab_view", fake_run_lab_view)

    result = CliRunner().invoke(lab_cli_app, ["view", "--limit", "7"])

    assert result.exit_code == 0
    assert calls[0]["limit"] == 7


def test_prime_lab_setup_uses_setup_service(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def fake_run_lab_setup(args: list[str], **_kwargs: Any) -> int:
        calls.append(args)
        return 0

    monkeypatch.setattr("prime_cli.commands.lab.run_lab_setup", fake_run_lab_setup)

    result = CliRunner().invoke(lab_cli_app, ["setup", "--agent", "codex"])

    assert result.exit_code == 0
    assert calls == [["--agent", "codex"]]


def test_prime_lab_sync_uses_sync_service(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def fake_run_lab_sync(args: list[str], **_kwargs: Any) -> int:
        calls.append(args)
        return 0

    monkeypatch.setattr("prime_cli.commands.lab.run_lab_sync", fake_run_lab_sync)

    result = CliRunner().invoke(lab_cli_app, ["sync", "--agent", "codex"])

    assert result.exit_code == 0
    assert calls == [["--agent", "codex"]]


def test_prime_lab_doctor_uses_doctor_service(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def fake_run_lab_doctor(args: list[str], **_kwargs: Any) -> int:
        calls.append(args)
        return 0

    monkeypatch.setattr("prime_cli.commands.lab.run_lab_doctor", fake_run_lab_doctor)

    result = CliRunner().invoke(lab_cli_app, ["doctor", "--fix"])

    assert result.exit_code == 0
    assert calls == [["--fix"]]


def test_prime_lab_doctor_service_checks_and_fixes_workspace(tmp_path: Path) -> None:
    assert parse_lab_doctor_args(["--fix"]).fix is True

    result = run_lab_doctor_service(LabDoctorOptions(), workspace=tmp_path)

    assert result.exit_code == 1
    assert any(
        check.name == "Workspace metadata" and check.status == "FAIL" for check in result.checks
    )
    assert any(
        check.name == "Gitignore outputs" and check.status == "WARN" for check in result.checks
    )

    fixed = run_lab_doctor_service(LabDoctorOptions(fix=True), workspace=tmp_path)

    assert (tmp_path / "configs").is_dir()
    assert (tmp_path / "environments").is_dir()
    gitignore = (tmp_path / ".gitignore").read_text(encoding="utf-8")
    assert "./outputs" in gitignore
    assert "*.pyc" in gitignore
    assert any(
        check.name == "Configs directory" and check.status == "PASS" for check in fixed.checks
    )


def test_prime_lab_doctor_service_checks_configs_and_source_hygiene(tmp_path: Path) -> None:
    configs_dir = tmp_path / "configs" / "rl"
    configs_dir.mkdir(parents=True)
    (configs_dir / "broken.toml").write_text("model = [\n", encoding="utf-8")
    env_dir = tmp_path / "environments" / "demo"
    (env_dir / "outputs").mkdir(parents=True)
    (env_dir / "pyproject.toml").write_text('[project]\nname = "demo"\n', encoding="utf-8")
    (env_dir / "README.md").write_text("# demo\n", encoding="utf-8")

    result = run_lab_doctor_service(LabDoctorOptions(), workspace=tmp_path)

    assert any(check.name == "Config TOML" and check.status == "FAIL" for check in result.checks)
    assert any(
        check.name == "Environment source hygiene" and check.status == "WARN"
        for check in result.checks
    )


def test_prime_lab_doctor_service_warns_on_config_environment_refs(tmp_path: Path) -> None:
    configs_dir = tmp_path / "configs" / "rl"
    configs_dir.mkdir(parents=True)
    (configs_dir / "train.toml").write_text(
        'model = "openai/gpt-5-mini"\nenvironments = ["missing-env"]\n',
        encoding="utf-8",
    )
    (tmp_path / "environments").mkdir()

    missing = run_lab_doctor_service(LabDoctorOptions(), workspace=tmp_path)

    assert any(
        check.name == "Config environment refs"
        and check.status == "WARN"
        and "missing-env" in check.message
        for check in missing.checks
    )

    env_dir = tmp_path / "environments" / "missing-env"
    env_dir.mkdir()
    (env_dir / "pyproject.toml").write_text(
        '[project]\nname = "missing-env"\n',
        encoding="utf-8",
    )
    (env_dir / "README.md").write_text("# missing\n", encoding="utf-8")
    unpinned = run_lab_doctor_service(LabDoctorOptions(), workspace=tmp_path)

    assert any(
        check.name == "Config environment refs"
        and check.status == "WARN"
        and "Unpinned" in check.message
        for check in unpinned.checks
    )


def test_prime_lab_sync_service_refreshes_agent_assets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    downloads: list[Path] = []

    def fake_download(_url: str, dest: Path, _emit: Any, *, force: bool = False) -> None:
        downloads.append(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(f"{dest.name}\n", encoding="utf-8")

    monkeypatch.setattr("prime_cli.lab_setup._download_file", fake_download)

    assert parse_lab_sync_args(["--agent", "pi"]).agents == ("pi",)
    assert parse_lab_sync_args([]).agents == ()

    result = run_lab_sync_service(
        LabSyncOptions(agents=("pi",)),
        workspace=tmp_path,
        emit=lambda _text: None,
    )

    assert result.exit_code == 0
    assert (tmp_path / ".prime" / "skills" / "create-environments" / "SKILL.md").is_file()
    assert (tmp_path / ".pi" / "skills" / "create-environments").exists()
    assert (tmp_path / ".prime" / "lab" / "templates" / "configs" / "rl" / "gsm8k.toml").is_file()
    assert (tmp_path / ".prime" / "lab" / "docs" / "index.md").is_file()
    assert (tmp_path / "AGENTS.md").is_file()
    assert downloads


def test_prime_lab_sync_service_preserves_workspace_agent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / ".prime").mkdir()
    (tmp_path / ".prime" / "lab.json").write_text(
        '{"choices": {"agents": ["opencode"], "primary_agent": "opencode"}}',
        encoding="utf-8",
    )

    def fake_download(_url: str, dest: Path, _emit: Any, *, force: bool = False) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(f"{dest.name}\n", encoding="utf-8")

    monkeypatch.setattr("prime_cli.lab_setup._download_file", fake_download)

    result = run_lab_sync_service(LabSyncOptions(), workspace=tmp_path)

    metadata = json.loads((tmp_path / ".prime" / "lab.json").read_text(encoding="utf-8"))
    assert result.exit_code == 0
    assert metadata["choices"]["primary_agent"] == "opencode"
    assert (tmp_path / ".opencode" / "skills" / "create-environments").exists()


def test_prime_lab_setup_service_supports_pi_agent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_download(url: str, dest: Path, emit: Any, *, force: bool = False) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(f"downloaded from {url}\n", encoding="utf-8")

    def fake_runner(command: Any, cwd: Path, emit: Any) -> int:
        return 0

    monkeypatch.setattr("prime_cli.lab_setup._download_file", fake_download)

    assert parse_lab_setup_args(["--agent", "pi"]).agents == ("pi",)

    result = run_lab_setup_service(
        LabSetupOptions(skip_install=True, skip_agents_md=True, agents=("pi",)),
        workspace=tmp_path,
        emit=lambda _text: None,
        runner=fake_runner,
    )

    metadata = json.loads((tmp_path / ".prime" / "lab.json").read_text(encoding="utf-8"))
    assert result.exit_code == 0
    assert metadata["choices"]["primary_agent"] == "pi"
    assert (tmp_path / ".pi" / "skills" / "create-environments").exists()


def test_lab_agent_adapters_map_known_and_custom_commands() -> None:
    assert agent_adapter("codex").prompt_command("hello") == ["codex", "exec", "hello"]
    assert agent_adapter("claude").prompt_command("hello") == ["claude", "-p", "hello"]
    assert agent_adapter("claude").server_spec(Path("/workspace")).transport == "one-shot"
    assert agent_adapter("opencode").prompt_command("hello") == ["opencode", "run", "hello"]
    assert agent_adapter("pi").prompt_command("hello") == ["pi", "-p", "hello"]
    assert agent_adapter("hermes").prompt_command("hello") == ["hermes", "--oneshot", "hello"]
    pi_server = agent_adapter("pi").server_spec(Path("/workspace"))
    assert pi_server.command == (
        "pi",
        "--mode",
        "rpc",
        "--session-dir",
        "/workspace/.prime/lab/agent-sessions/pi",
    )
    assert pi_server.transport == "stdio-jsonrpc"
    assert agent_adapter("hermes").name == "hermes-agent"
    assert agent_adapter("codex").server_spec(Path("/workspace")).command == (
        "codex",
        "app-server",
        "--listen",
        "stdio://",
    )
    assert agent_adapter("codex").server_spec(Path("/workspace")).transport == "codex-app-stdio"
    assert agent_adapter("hermes-agent").server_spec(Path("/workspace")).command == (
        "hermes",
        "acp",
    )
    assert agent_adapter("hermes-agent").server_spec(Path("/workspace")).transport == "acp-stdio"
    assert agent_adapter("my-agent").prompt_command("hello") == ["my-agent", "hello"]


class _FakeJsonRpcProcess:
    stderr: list[str] = []

    def __init__(self, handler: Any, *, initial_stdout: tuple[str, ...] = ()) -> None:
        self._handler = handler
        self._stdout: queue.Queue[str | None] = queue.Queue()
        self.stdin = _FakeJsonRpcStdin(self)
        self.stdout = self
        self.command: list[str] = []
        self._terminated = False
        for line in initial_stdout:
            self._stdout.put(line if line.endswith("\n") else f"{line}\n")

    def __iter__(self) -> "_FakeJsonRpcProcess":
        return self

    def __next__(self) -> str:
        line = self._stdout.get(timeout=5)
        if line is None:
            raise StopIteration
        return line

    def emit(self, payload: dict[str, Any]) -> None:
        self._stdout.put(json.dumps(payload) + "\n")

    def handle_stdin(self, value: str) -> None:
        message = json.loads(value)
        self._handler(self, message)

    def poll(self) -> int | None:
        return 0 if self._terminated else None

    def wait(self, timeout: float | None = None) -> int:
        self._terminated = True
        self._stdout.put(None)
        return 0

    def terminate(self) -> None:
        self._terminated = True
        self._stdout.put(None)

    def kill(self) -> None:
        self.terminate()


class _FakeJsonRpcStdin:
    def __init__(self, process: _FakeJsonRpcProcess) -> None:
        self._process = process

    def write(self, value: str) -> int:
        self._process.handle_stdin(value)
        return len(value)

    def flush(self) -> None:
        return None


def _wait_for(predicate: Any, *, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def _last_message(messages: tuple[Any, ...]) -> Any:
    assert messages
    return messages[-1]


def test_agent_runtime_supports_one_shot_exec(tmp_path: Path) -> None:
    class FakeProcess:
        stdout = ["agent response\n"]

        def wait(self) -> int:
            return 0

        def poll(self) -> None:
            return None

        def terminate(self) -> None:
            return None

    done = threading.Event()
    commands: list[list[str]] = []
    latest_messages: tuple[Any, ...] = ()

    def fake_popen(command: list[str], **_kwargs: Any) -> FakeProcess:
        commands.append(command)
        return FakeProcess()

    def on_messages(messages: Any) -> None:
        nonlocal latest_messages
        latest_messages = messages
        if messages and messages[-1].role == "assistant" and messages[-1].status != "streaming":
            done.set()

    runtime = AgentRuntime(on_messages=on_messages, popen_factory=fake_popen)
    runtime.start(tmp_path, "claude")

    assert runtime.state.status == "connected"
    assert runtime.state.transport == "one-shot"

    runtime.send_prompt("hello")

    assert done.wait(timeout=2)
    assert commands == [["claude", "-p", "hello"]]
    assert latest_messages
    assert latest_messages[-1].content == "agent response\n"


def test_agent_runtime_supports_codex_app_stdio_chat(tmp_path: Path) -> None:
    messages: tuple[Any, ...] = ()

    def handler(process: _FakeJsonRpcProcess, message: dict[str, Any]) -> None:
        request_id = message.get("id")
        method = message.get("method")
        if method == "initialize":
            process.emit({"jsonrpc": "2.0", "id": request_id, "result": {}})
        elif method == "thread/start":
            process.emit(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {"thread": {"id": "thread-1"}},
                }
            )
        elif method == "turn/start":
            process.emit(
                {
                    "jsonrpc": "2.0",
                    "method": "item/agentMessage/delta",
                    "params": {"turnId": "turn-1", "delta": "codex response"},
                }
            )
            process.emit(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {"turn": {"id": "turn-1", "status": "completed"}},
                }
            )
        else:
            process.emit(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"message": f"unexpected {method}"},
                }
            )

    def fake_popen(command: list[str], **_kwargs: Any) -> _FakeJsonRpcProcess:
        process = _FakeJsonRpcProcess(handler)
        process.command = command
        return process

    def on_messages(value: Any) -> None:
        nonlocal messages
        messages = value

    runtime = AgentRuntime(on_messages=on_messages, popen_factory=fake_popen)
    runtime.start(tmp_path, "codex")

    assert _wait_for(lambda: runtime.state.status == "connected")
    assert runtime.state.session_id == "thread-1"

    runtime.send_prompt("hello")

    assert _wait_for(lambda: messages and messages[-1].status != "streaming")
    message = _last_message(messages)
    assert message.role == "assistant"
    assert message.content == "codex response"
    runtime.stop()


def test_agent_runtime_supports_hermes_acp_stdio_chat(tmp_path: Path) -> None:
    messages: tuple[Any, ...] = ()

    def handler(process: _FakeJsonRpcProcess, message: dict[str, Any]) -> None:
        request_id = message.get("id")
        method = message.get("method")
        if method == "initialize":
            process.emit({"jsonrpc": "2.0", "id": request_id, "result": {"authMethods": []}})
        elif method == "session/new":
            process.emit(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {"sessionId": "session-1"},
                }
            )
        elif method == "session/prompt":
            process.emit(
                {
                    "jsonrpc": "2.0",
                    "method": "session/update",
                    "params": {
                        "update": {
                            "sessionUpdate": "agent_message_chunk",
                            "content": {"type": "text", "text": "hermes response"},
                        }
                    },
                }
            )
            process.emit({"jsonrpc": "2.0", "id": request_id, "result": {}})
        else:
            process.emit(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"message": f"unexpected {method}"},
                }
            )

    def fake_popen(command: list[str], **_kwargs: Any) -> _FakeJsonRpcProcess:
        process = _FakeJsonRpcProcess(handler)
        process.command = command
        return process

    def on_messages(value: Any) -> None:
        nonlocal messages
        messages = value

    runtime = AgentRuntime(on_messages=on_messages, popen_factory=fake_popen)
    runtime.start(tmp_path, "hermes")

    assert _wait_for(lambda: runtime.state.status == "connected")
    assert runtime.state.agent == "hermes-agent"
    assert runtime.state.session_id == "session-1"

    runtime.send_prompt("hello")

    assert _wait_for(lambda: messages and messages[-1].status != "streaming")
    message = _last_message(messages)
    assert message.role == "assistant"
    assert message.content == "hermes response"
    runtime.stop()


def test_agent_runtime_records_opencode_http_endpoint(tmp_path: Path) -> None:
    processes: list[_FakeJsonRpcProcess] = []
    messages: tuple[Any, ...] = ()

    class FakeOneShotProcess:
        stdout = ["opencode response\n"]

        def wait(self) -> int:
            return 0

        def poll(self) -> None:
            return None

        def terminate(self) -> None:
            return None

    def fake_popen(command: list[str], **_kwargs: Any) -> Any:
        if command[:2] == ["opencode", "run"]:
            return FakeOneShotProcess()
        process = _FakeJsonRpcProcess(lambda *_args: None, initial_stdout=("http://127.0.0.1:0",))
        process.command = command
        processes.append(process)
        return process

    def on_messages(value: Any) -> None:
        nonlocal messages
        messages = value

    runtime = AgentRuntime(on_messages=on_messages, popen_factory=fake_popen)
    runtime.start(tmp_path, "opencode")

    assert _wait_for(lambda: runtime.state.status == "connected")
    assert runtime.state.transport == "acp-http"
    assert runtime.state.endpoint == "http://127.0.0.1:0"
    assert processes[0].command == ["opencode", "acp", "--hostname", "127.0.0.1", "--port", "0"]
    runtime.send_prompt("hello")
    assert _wait_for(lambda: messages and messages[-1].status != "streaming")
    assert _last_message(messages).content == "opencode response\n"
    runtime.stop()


def test_agent_runtime_starts_pi_rpc_with_session_dir(tmp_path: Path) -> None:
    processes: list[_FakeJsonRpcProcess] = []
    messages: tuple[Any, ...] = ()

    class FakeOneShotProcess:
        stdout = ["pi response\n"]

        def wait(self) -> int:
            return 0

        def poll(self) -> None:
            return None

        def terminate(self) -> None:
            return None

    def fake_popen(command: list[str], **_kwargs: Any) -> Any:
        if command[:2] == ["pi", "-p"]:
            return FakeOneShotProcess()
        process = _FakeJsonRpcProcess(lambda *_args: None)
        process.command = command
        processes.append(process)
        return process

    def on_messages(value: Any) -> None:
        nonlocal messages
        messages = value

    runtime = AgentRuntime(on_messages=on_messages, popen_factory=fake_popen)
    runtime.start(tmp_path, "pi")

    assert runtime.state.status == "connected"
    assert runtime.state.transport == "stdio-jsonrpc"
    assert processes[0].command == [
        "pi",
        "--mode",
        "rpc",
        "--session-dir",
        str(tmp_path / ".prime" / "lab" / "agent-sessions" / "pi"),
    ]
    assert (tmp_path / ".prime" / "lab" / "agent-sessions" / "pi").is_dir()
    runtime.send_prompt("hello")
    assert _wait_for(lambda: messages and messages[-1].status != "streaming")
    assert _last_message(messages).content == "pi response\n"
    runtime.stop()


def test_lab_agent_select_options_keep_registered_agent_first() -> None:
    options = agent_select_options("opencode")

    assert options[0] == ("OpenCode", "opencode")
    assert ("Codex", "codex") in options
    assert len({value for _, value in options}) == len(options)


def test_lab_brand_header_uses_lab_and_prime_marks() -> None:
    rendered = _render_renderable(lab_header("Training"))

    assert "L A B" in rendered
    assert "Training" in rendered
    assert "PRIME Intellect" in rendered


@pytest.mark.asyncio
async def test_prime_lab_view_mounts(tmp_path: Path) -> None:
    snapshot = make_source().load(LabLoadOptions(limit=10, workspace=tmp_path))
    app = PrimeLabView(lambda: snapshot)

    async with app.run_test() as pilot:
        await pilot.pause()
        assert app._snapshot == snapshot


@pytest.mark.asyncio
async def test_prime_lab_view_home_launch_panel_uses_home_state(tmp_path: Path) -> None:
    (tmp_path / ".prime").mkdir()
    (tmp_path / ".prime" / "lab.json").write_text(
        '{"choices": {"primary_agent": "codex"}}',
        encoding="utf-8",
    )
    snapshot = make_source().load_initial(LabLoadOptions(limit=10, workspace=tmp_path))
    app = PrimeLabView(lambda: snapshot, initial_loader=lambda: snapshot)

    async with app.run_test(size=(140, 44)) as pilot:
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, LaunchScreen)
        panel = screen.query_one("#home-launch", HomeLaunchPanel)

        assert panel.display is True
        assert any(label == "workspaces" and count >= 1 for label, count in panel._state.counts)


@pytest.mark.asyncio
async def test_prime_lab_view_home_launch_panel_shows_for_loaded_workspace(
    tmp_path: Path,
) -> None:
    snapshot = make_source().load(LabLoadOptions(limit=10, workspace=tmp_path))
    app = PrimeLabView(lambda: snapshot, initial_loader=lambda: snapshot)

    async with app.run_test(size=(140, 44)) as pilot:
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, LaunchScreen)

        assert screen.query_one("#home-launch", HomeLaunchPanel).display is True
        assert screen.query_one("#launch-actions").display is True
        assert screen.query_one("#launch-hotkeys").display is True
        assert app.check_action("search", ()) is False
        assert app.check_action("load_more_rows", ()) is False


@pytest.mark.asyncio
async def test_prime_lab_view_w_reopens_launch_screen(tmp_path: Path) -> None:
    snapshot = make_source().load(LabLoadOptions(limit=10, workspace=tmp_path))
    app = PrimeLabView(lambda: snapshot, initial_loader=lambda: snapshot)

    async with app.run_test(size=(140, 44)) as pilot:
        await pilot.pause()
        assert isinstance(app.screen, LaunchScreen)

        await pilot.press("enter")
        await pilot.pause()
        assert not isinstance(app.screen, LaunchScreen)

        await pilot.press("w")
        await pilot.pause()
        assert isinstance(app.screen, LaunchScreen)


@pytest.mark.asyncio
async def test_prime_lab_view_auto_starts_configured_agent(tmp_path: Path) -> None:
    (tmp_path / ".prime").mkdir()
    (tmp_path / ".prime" / "lab.json").write_text(
        '{"choices": {"primary_agent": "claude"}}',
        encoding="utf-8",
    )
    snapshot = make_source().load_initial(LabLoadOptions(limit=10, workspace=tmp_path))
    app = PrimeLabView(lambda: snapshot, initial_loader=lambda: snapshot)

    async with app.run_test(size=(140, 44)) as pilot:
        await pilot.pause()

        assert app._agent_state.agent == "claude"
        assert app._agent_state.status == "connected"
        assert app._agent_state.transport == "one-shot"
        assert "Claude Code connected" in _render_renderable(app._statusbar_text())


@pytest.mark.asyncio
async def test_agent_chat_uses_centered_stage_without_sidebar(tmp_path: Path) -> None:
    (tmp_path / ".prime").mkdir()
    (tmp_path / ".prime" / "lab.json").write_text(
        '{"choices": {"primary_agent": "claude"}}',
        encoding="utf-8",
    )
    snapshot = make_source().load(LabLoadOptions(limit=10, workspace=tmp_path))
    app = PrimeLabView(lambda: snapshot, initial_loader=lambda: snapshot)

    async with app.run_test(size=(140, 44)) as pilot:
        await pilot.pause()
        app._dismiss_launch_screen()
        await pilot.pause()
        section = snapshot.section("workspace")
        assert section is not None
        agent_item = next(item for item in section.items if item.raw.get("type") == "agent_chat")
        app._show_item(agent_item)
        app.action_load_detail()
        await pilot.pause()

        assert isinstance(app.screen, AgentChatScreen)
        stage = app.screen.query_one("#agent-stage")
        assert isinstance(stage, Vertical)
        assert not app.screen.query("#agent-controls")
        assert isinstance(app.screen.query_one("#agent-chat"), VerticalScroll)
        assert app.screen.query_one("#agent-atmosphere").display is True


def test_launch_backdrop_renders_stable_terminal_field() -> None:
    rows = LaunchBackdrop(frame=3).render_parts(72, 14)

    assert len(rows) == 14
    assert all(sum(len(value) for value, _ in row) == 72 for row in rows)
    plain = "\n".join("".join(value for value, _ in row) for row in rows)
    assert any(char in plain for char in {"┃", "╎"})
    assert any(char in plain for char in {"•", "◆"})


@pytest.mark.asyncio
async def test_prime_lab_view_home_launch_panel_dismisses_into_workspace(
    tmp_path: Path,
) -> None:
    snapshot = make_source().load(LabLoadOptions(limit=10, workspace=tmp_path))
    app = PrimeLabView(lambda: snapshot, initial_loader=lambda: snapshot)

    async with app.run_test(size=(140, 44)) as pilot:
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

        assert not isinstance(app.screen, LaunchScreen)
        assert app._active_section_key == "workspace"
        assert str(app.query_one("#section-title", Label).render()) == "Home"
        assert app.query_one("#item-list", OptionList).display is True
        assert app.query_one("#topbar").display is True
        assert app.query_one("#nav-pane").display is True
        assert app.query_one("#section-title").display is True
        assert app.query_one("#section-subtitle").display is True
        assert app.query_one("#inspector-pane").display is True
        assert app.query_one("#statusbar").display is True
        assert app.query("Footer").first().display is True


@pytest.mark.asyncio
async def test_prime_lab_view_launch_grid_opens_quickstart_flows(tmp_path: Path) -> None:
    snapshot = make_source().load(LabLoadOptions(limit=10, workspace=tmp_path))
    app = PrimeLabView(lambda: snapshot, initial_loader=lambda: snapshot)

    async with app.run_test(size=(140, 44)) as pilot:
        await pilot.pause()
        await pilot.click("#launch-evaluate")
        await pilot.pause()

        assert isinstance(app.screen, ConfigRunScreen)
        assert app.screen._config_kind == "eval"


@pytest.mark.asyncio
async def test_prime_lab_view_launch_grid_build_opens_agent_templates(tmp_path: Path) -> None:
    snapshot = make_source().load(LabLoadOptions(limit=10, workspace=tmp_path))
    app = PrimeLabView(lambda: snapshot, initial_loader=lambda: snapshot)

    async with app.run_test(size=(140, 44)) as pilot:
        await pilot.pause()
        await pilot.click("#launch-build")
        await pilot.pause()

        assert isinstance(app.screen, AgentChatScreen)
        assert app.screen.query("#agent-template-0")


@pytest.mark.asyncio
async def test_prime_lab_view_launch_grid_training_and_explore(tmp_path: Path) -> None:
    snapshot = make_source().load(LabLoadOptions(limit=10, workspace=tmp_path))
    app = PrimeLabView(lambda: snapshot, initial_loader=lambda: snapshot)

    async with app.run_test(size=(140, 44)) as pilot:
        await pilot.pause()
        await pilot.click("#launch-train")
        await pilot.pause()

        assert isinstance(app.screen, ConfigRunScreen)
        assert app.screen._config_kind == "rl"

    app = PrimeLabView(lambda: snapshot, initial_loader=lambda: snapshot)
    async with app.run_test(size=(140, 44)) as pilot:
        await pilot.pause()
        await pilot.click("#launch-explore")
        await pilot.pause()

        assert not isinstance(app.screen, LaunchScreen)
        assert app._active_section_key == "environments"


@pytest.mark.asyncio
async def test_prime_lab_view_ladder_loads_platform_sections(tmp_path: Path) -> None:
    source = make_source()
    loaded_limits: list[int] = []

    def load_limit(limit: int) -> Any:
        loaded_limits.append(limit)
        return source.load(LabLoadOptions(limit=limit, workspace=tmp_path))

    app = PrimeLabView(
        lambda: source.load(LabLoadOptions(limit=20, workspace=tmp_path)),
        initial_loader=lambda: source.load_initial(LabLoadOptions(limit=20, workspace=tmp_path)),
        ladder_loader=load_limit,
        ladder_limits=(5, 10, 20),
    )

    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.pause()

    assert loaded_limits == [5, 10, 20]


@pytest.mark.asyncio
async def test_prime_lab_view_can_load_more_platform_rows(tmp_path: Path) -> None:
    source = make_source()
    loaded_limits: list[int] = []

    def load_limit(limit: int) -> Any:
        loaded_limits.append(limit)
        return source.load(LabLoadOptions(limit=limit, workspace=tmp_path))

    app = PrimeLabView(
        lambda: source.load(LabLoadOptions(limit=100, workspace=tmp_path)),
        initial_loader=lambda: source.load_initial(LabLoadOptions(limit=100, workspace=tmp_path)),
        ladder_loader=load_limit,
        ladder_limits=(5, 10, 20, 40, 80, 100),
    )

    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.pause()
        app.action_load_more_rows()
        await pilot.pause()

    assert loaded_limits[-1] == 200


@pytest.mark.asyncio
async def test_prime_lab_view_opens_training_run_screen(tmp_path: Path) -> None:
    source = make_source()
    snapshot = source.load(LabLoadOptions(limit=10, workspace=tmp_path))
    app = PrimeLabView(
        lambda: snapshot,
        lambda item, include_logs, log_tail_lines, metrics_limit, metrics_min_step: (
            source.load_item_detail(
                item,
                include_logs=include_logs,
                log_tail_lines=log_tail_lines,
                metrics_limit=metrics_limit,
                metrics_min_step=metrics_min_step,
            )
        ),
    )

    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        app._active_section_key = "training"
        app._render_active_section()
        await pilot.pause()
        app.action_load_detail()
        await pilot.pause()

        assert isinstance(app.screen, TrainingRunScreen)


@pytest.mark.asyncio
async def test_training_run_screen_edits_config_natively(tmp_path: Path) -> None:
    source = make_source()
    snapshot = source.load(LabLoadOptions(limit=10, workspace=tmp_path))
    app = PrimeLabView(
        lambda: snapshot,
        lambda item, include_logs, log_tail_lines, metrics_limit, metrics_min_step: (
            source.load_item_detail(
                item,
                include_logs=include_logs,
                log_tail_lines=log_tail_lines,
                metrics_limit=metrics_limit,
                metrics_min_step=metrics_min_step,
            )
        ),
    )

    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        app._active_section_key = "training"
        app._render_active_section()
        await pilot.pause()
        app.action_load_detail()
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, TrainingRunScreen)
        training = next(section for section in snapshot.sections if section.key == "training")
        detail = source.load_item_detail(training.items[0])
        screen._set_detail(detail, 0.1)
        await pilot.pause()
        assert str(screen.query_one("#run-edit-config", Button).label) == "Modify and run"
        assert str(screen.query_one("#run-train", Button).label) == "Train"
        await pilot.click("#run-train")
        await pilot.pause()

        assert isinstance(app.screen, ConfigRunScreen)
        assert app.screen._item.title == "training-config.toml"
        assert app.screen._item.raw["config_kind"] == "rl"
        assert app.screen._item.raw["workspace"] == str(tmp_path.resolve())


def test_training_platform_url_uses_dashboard_route() -> None:
    assert (
        _training_platform_url("https://app.test/", {"id": "run-123"})
        == "https://app.test/dashboard/training/run-123"
    )


@pytest.mark.asyncio
async def test_prime_lab_view_opens_environment_screen(tmp_path: Path) -> None:
    source = make_source()
    snapshot = source.load(LabLoadOptions(limit=10, workspace=tmp_path))
    app = PrimeLabView(
        lambda: snapshot,
        lambda item, include_logs, log_tail_lines, metrics_limit, metrics_min_step: (
            source.load_item_detail(
                item,
                include_logs=include_logs,
                log_tail_lines=log_tail_lines,
                metrics_limit=metrics_limit,
                metrics_min_step=metrics_min_step,
            )
        ),
    )

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        app._active_section_key = "environments"
        app._render_active_section()
        await pilot.pause()
        app.action_load_detail()
        await pilot.pause()

        assert isinstance(app.screen, EnvironmentScreen)
        screen = app.screen
        assert [tab.id for tab in screen.query(Tab)] == ["code", "leaderboard"]

        screen._render_tab()
        await pilot.pause()
        assert {"train", "evaluate", "install", "view", "discussions", "actions", "refresh"} <= {
            action.key for action in screen._action_by_id.values()
        }

        await pilot.click("#env-action-0")
        await pilot.pause()

        assert isinstance(app.screen, ConfigRunScreen)
        assert app.screen._item.raw["config_kind"] == "rl"


@pytest.mark.asyncio
async def test_environment_install_action_uses_native_follow_screen(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeProcess:
        stdout = ["installing\n"]

        def wait(self) -> int:
            return 0

        def poll(self) -> None:
            return None

        def terminate(self) -> None:
            return None

    commands: list[list[str]] = []

    def fake_popen(command: list[str], **_kwargs: Any) -> FakeProcess:
        commands.append(command)
        return FakeProcess()

    monkeypatch.setattr("prime_lab_view.config_screen.subprocess.Popen", fake_popen)

    source = make_source()
    snapshot = source.load(LabLoadOptions(limit=10, workspace=tmp_path))
    app = PrimeLabView(
        lambda: snapshot,
        lambda item, include_logs, log_tail_lines, metrics_limit, metrics_min_step: (
            source.load_item_detail(
                item,
                include_logs=include_logs,
                log_tail_lines=log_tail_lines,
                metrics_limit=metrics_limit,
                metrics_min_step=metrics_min_step,
            )
        ),
    )

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        app._active_section_key = "environments"
        app._render_active_section()
        await pilot.pause()
        app.action_load_detail()
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, EnvironmentScreen)

        screen._run_environment_action(EnvironmentAction("install", "Install", ""))
        await pilot.pause()
        await pilot.pause()

        assert isinstance(app.screen, ConfigLaunchScreen)
        assert commands and commands[0][:3] == ["prime", "env", "install"]


@pytest.mark.asyncio
async def test_prime_lab_view_opens_setup_screen_for_uninitialized_workspace(
    tmp_path: Path,
) -> None:
    snapshot = make_source().load(LabLoadOptions(limit=10, workspace=tmp_path))
    app = PrimeLabView(lambda: snapshot)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app._dismiss_launch_screen()
        await pilot.pause()
        section = snapshot.section("workspace")
        assert section is not None
        setup_item = next(item for item in section.items if item.raw.get("type") == "setup_action")
        app._show_item(setup_item)
        app.action_load_detail()
        await pilot.pause()

        assert isinstance(app.screen, SetupScreen)


@pytest.mark.asyncio
async def test_prime_lab_view_opens_workspace_browser(tmp_path: Path) -> None:
    (tmp_path / ".prime").mkdir()
    (tmp_path / ".prime" / "lab.json").write_text("{}", encoding="utf-8")
    snapshot = make_source().load(LabLoadOptions(limit=10, workspace=tmp_path))
    app = PrimeLabView(lambda: snapshot)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app._dismiss_launch_screen()
        await pilot.pause()
        workspace_item = next(
            item for item in app._visible_items if item.raw.get("type") == "workspace_context"
        )
        app._show_item(workspace_item)
        app.action_load_detail()
        await pilot.pause()

        assert isinstance(app.screen, WorkspaceScreen)


@pytest.mark.asyncio
async def test_prime_lab_view_opens_add_workspace_screen(tmp_path: Path) -> None:
    snapshot = make_source().load(LabLoadOptions(limit=10, workspace=tmp_path))
    app = PrimeLabView(lambda: snapshot)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app._dismiss_launch_screen()
        await pilot.pause()
        section = snapshot.section("workspace")
        assert section is not None
        add_item = next(item for item in section.items if item.raw.get("type") == "add_workspace")
        app._show_item(add_item)
        app.action_load_detail()
        await pilot.pause()

        assert isinstance(app.screen, AddWorkspaceScreen)


@pytest.mark.asyncio
async def test_prime_lab_view_opens_agent_sync_screen(tmp_path: Path) -> None:
    (tmp_path / ".prime").mkdir()
    (tmp_path / ".prime" / "lab.json").write_text(
        '{"choices": {"primary_agent": "codex"}}',
        encoding="utf-8",
    )
    snapshot = make_source().load(LabLoadOptions(limit=10, workspace=tmp_path))
    app = PrimeLabView(lambda: snapshot)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app._dismiss_launch_screen()
        await pilot.pause()
        section = snapshot.section("workspace")
        assert section is not None
        sync_item = next(item for item in section.items if item.raw.get("type") == "agent_sync")
        app._show_item(sync_item)
        app.action_load_detail()
        await pilot.pause()

        assert isinstance(app.screen, AgentSyncScreen)


@pytest.mark.asyncio
async def test_prime_lab_view_opens_doctor_screen(tmp_path: Path) -> None:
    snapshot = make_source().load(LabLoadOptions(limit=10, workspace=tmp_path))
    app = PrimeLabView(lambda: snapshot)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app._dismiss_launch_screen()
        await pilot.pause()
        section = snapshot.section("workspace")
        assert section is not None
        doctor_item = next(
            item for item in section.items if item.raw.get("type") == "doctor_action"
        )
        app._show_item(doctor_item)
        app.action_load_detail()
        await pilot.pause()

        assert isinstance(app.screen, DoctorScreen)


@pytest.mark.asyncio
async def test_prime_lab_view_opens_config_run_screen(tmp_path: Path) -> None:
    config_path = tmp_path / "configs" / "rl" / "train.toml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        'model = "openai/gpt-5-mini"\nmax_steps = 10\nenvironments = ["gsm8k"]\n',
        encoding="utf-8",
    )
    snapshot = make_source().load(LabLoadOptions(limit=10, workspace=tmp_path))
    app = PrimeLabView(lambda: snapshot)

    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause()
        app.set_home_group("configs")
        await pilot.pause()
        config_item = next(item for item in app._visible_items if item.raw["type"] == "config_file")
        app._show_item(config_item)
        app.action_load_detail()
        await pilot.pause()

        assert isinstance(app.screen, ConfigRunScreen)
        assert config_item.raw["workspace"] == str(tmp_path.resolve())


@pytest.mark.asyncio
async def test_config_run_launch_uses_native_follow_screen(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeProcess:
        stdout = ["launching\n", "done\n"]

        def wait(self) -> int:
            return 0

        def poll(self) -> None:
            return None

        def terminate(self) -> None:
            return None

    commands: list[list[str]] = []

    def fake_popen(command: list[str], **_kwargs: Any) -> FakeProcess:
        commands.append(command)
        return FakeProcess()

    monkeypatch.setattr("prime_lab_view.config_screen.subprocess.Popen", fake_popen)

    config_path = tmp_path / "configs" / "rl" / "train.toml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        'model = "openai/gpt-5-mini"\nmax_steps = 10\nenvironments = ["gsm8k"]\n',
        encoding="utf-8",
    )
    snapshot = make_source().load(LabLoadOptions(limit=10, workspace=tmp_path))
    app = PrimeLabView(lambda: snapshot)

    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause()
        app.set_home_group("configs")
        await pilot.pause()
        config_item = next(item for item in app._visible_items if item.raw["type"] == "config_file")
        app._show_item(config_item)
        app.action_load_detail()
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, ConfigRunScreen)

        screen.launch()
        await pilot.pause()
        await pilot.pause()

        assert isinstance(app.screen, ConfigLaunchScreen)
        assert commands and commands[0][:3] == ["prime", "rl", "run"]
        assert (tmp_path / ".prime" / "lab" / "configs" / "rl" / "train.toml").is_file()


@pytest.mark.asyncio
async def test_config_launch_follows_training_logs_from_hint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeProcess:
        def __init__(self, stdout: list[str], returncode: int = 0) -> None:
            self.stdout = stdout
            self._returncode = returncode

        def wait(self) -> int:
            return self._returncode

        def poll(self) -> None:
            return None

        def terminate(self) -> None:
            return None

    commands: list[list[str]] = []

    def fake_popen(command: list[str], **_kwargs: Any) -> FakeProcess:
        commands.append(command)
        if command[:3] == ["prime", "rl", "logs"]:
            return FakeProcess(["Watching logs for run abc123run...\n", "step 1 reward 0.5\n"])
        return FakeProcess(["Creating RL training run...\n", "  prime rl logs abc123run -f\n"])

    monkeypatch.setattr("prime_lab_view.config_screen.subprocess.Popen", fake_popen)

    config_path = tmp_path / "configs" / "rl" / "train.toml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        'model = "openai/gpt-5-mini"\nmax_steps = 10\nenvironments = ["gsm8k"]\n',
        encoding="utf-8",
    )
    snapshot = make_source().load(LabLoadOptions(limit=10, workspace=tmp_path))
    app = PrimeLabView(lambda: snapshot)

    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause()
        app.set_home_group("configs")
        await pilot.pause()
        config_item = next(item for item in app._visible_items if item.raw["type"] == "config_file")
        app._show_item(config_item)
        app.action_load_detail()
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, ConfigRunScreen)

        screen.launch()
        for _ in range(10):
            await pilot.pause()
            if len(commands) >= 2:
                break

        assert commands[:2] == [
            ["prime", "rl", "run", ".prime/lab/configs/rl/train.toml"],
            ["prime", "rl", "logs", "abc123run", "-f"],
        ]
        assert isinstance(app.screen, ConfigLaunchScreen)
        assert "Following run logs with: prime rl logs abc123run -f" in app.screen._output
        assert "step 1 reward 0.5" in app.screen._output


@pytest.mark.asyncio
async def test_config_launch_retries_training_logs_until_ready(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeProcess:
        def __init__(self, stdout: list[str], returncode: int = 0) -> None:
            self.stdout = stdout
            self._returncode = returncode

        def wait(self) -> int:
            return self._returncode

        def poll(self) -> None:
            return None

        def terminate(self) -> None:
            return None

    commands: list[list[str]] = []
    log_attempts = 0

    def fake_popen(command: list[str], **_kwargs: Any) -> FakeProcess:
        nonlocal log_attempts
        commands.append(command)
        if command[:3] == ["prime", "rl", "logs"]:
            log_attempts += 1
            if log_attempts == 1:
                return FakeProcess(["No logs available yet.\n"], 1)
            return FakeProcess(["logs ready\n"], 0)
        return FakeProcess(["  prime rl logs retryrun -f\n"])

    monkeypatch.setattr("prime_lab_view.config_screen.subprocess.Popen", fake_popen)
    monkeypatch.setattr("prime_lab_view.config_screen._LOG_RETRY_DELAYS", (0.01,))

    config_path = tmp_path / "configs" / "rl" / "train.toml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        'model = "openai/gpt-5-mini"\nmax_steps = 10\nenvironments = ["gsm8k"]\n',
        encoding="utf-8",
    )
    snapshot = make_source().load(LabLoadOptions(limit=10, workspace=tmp_path))
    app = PrimeLabView(lambda: snapshot)

    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause()
        app.set_home_group("configs")
        await pilot.pause()
        config_item = next(item for item in app._visible_items if item.raw["type"] == "config_file")
        app._show_item(config_item)
        app.action_load_detail()
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, ConfigRunScreen)

        screen.launch()
        for _ in range(20):
            await pilot.pause()
            if log_attempts >= 2:
                break

        assert commands.count(["prime", "rl", "logs", "retryrun", "-f"]) == 2
        assert isinstance(app.screen, ConfigLaunchScreen)
        assert "Logs are not ready yet; retrying in 0.01s." in app.screen._output
        assert "logs ready" in app.screen._output


@pytest.mark.asyncio
async def test_prime_lab_view_opens_local_eval_run_screen(tmp_path: Path) -> None:
    run_dir = tmp_path / "outputs" / "evals" / "gsm8k--openai--gpt-4" / "run-a"
    run_dir.mkdir(parents=True)
    (run_dir / "metadata.json").write_text(
        '{"avg_reward": 0.5, "num_examples": 1, "rollouts_per_example": 1}',
        encoding="utf-8",
    )
    (run_dir / "results.jsonl").write_text(
        '{"reward": 0.5, "prompt": [{"role": "user", "content": "2+2?"}], '
        '"completion": [{"role": "assistant", "content": "4"}]}\n',
        encoding="utf-8",
    )
    source = make_source()
    snapshot = source.load(LabLoadOptions(limit=10, workspace=tmp_path))
    app = PrimeLabView(
        lambda: snapshot,
        lambda item, include_logs, log_tail_lines, metrics_limit, metrics_min_step: (
            source.load_item_detail(
                item,
                include_logs=include_logs,
                log_tail_lines=log_tail_lines,
                metrics_limit=metrics_limit,
                metrics_min_step=metrics_min_step,
            )
        ),
    )

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        app._active_section_key = "evaluations"
        app._render_active_section()
        await pilot.pause()
        local_item = next(
            item for item in app._visible_items if item.raw.get("type") == "local_eval"
        )
        app._show_item(local_item)
        app.action_load_detail()
        await pilot.pause()

        assert isinstance(app.screen, LocalEvalRunScreen)


@pytest.mark.asyncio
async def test_prime_lab_view_evaluations_by_env_groups_runs(tmp_path: Path) -> None:
    run_dir = tmp_path / "outputs" / "evals" / "gsm8k--openai--gpt-4" / "run-a"
    run_dir.mkdir(parents=True)
    (run_dir / "metadata.json").write_text(
        '{"avg_reward": 0.5, "num_examples": 1, "rollouts_per_example": 1}',
        encoding="utf-8",
    )
    (run_dir / "results.jsonl").write_text('{"reward": 0.5}\n', encoding="utf-8")
    source = make_source()
    snapshot = source.load(LabLoadOptions(limit=10, workspace=tmp_path))
    app = PrimeLabView(lambda: snapshot)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        app._active_section_key = "evaluations"
        app.set_evaluation_view("env")
        await pilot.pause()

        tree = app.query_one("#evaluation-tree", EvaluationTree)
        assert tree.display
        assert not app.query_one("#item-list", LabOptionList).display
        assert "gsm8k" in app._evaluation_tree_index
        assert "openai/gpt-4.1-mini" in app._evaluation_tree_index["gsm8k"]
        assert str(app.query_one("#inspector-title", Label).render()) == "Selection Details"


@pytest.mark.asyncio
async def test_prime_lab_view_by_env_enter_opens_highlighted_local_eval(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "outputs" / "evals" / "gsm8k--openai--gpt-4" / "run-a"
    run_dir.mkdir(parents=True)
    (run_dir / "metadata.json").write_text(
        '{"avg_reward": 0.5, "num_examples": 1, "rollouts_per_example": 1}',
        encoding="utf-8",
    )
    (run_dir / "results.jsonl").write_text(
        '{"reward": 0.5, "prompt": [{"role": "user", "content": "2+2?"}], '
        '"completion": [{"role": "assistant", "content": "4"}]}\n',
        encoding="utf-8",
    )
    source = make_source()
    snapshot = source.load(LabLoadOptions(limit=10, workspace=tmp_path))
    app = PrimeLabView(lambda: snapshot)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        app._active_section_key = "evaluations"
        app.set_evaluation_view("env")
        await pilot.pause()

        tree = app.query_one("#evaluation-tree", EvaluationTree)
        tree.focus()
        await pilot.press("enter")
        await pilot.pause()

        assert isinstance(app.screen, LocalEvalRunScreen)


@pytest.mark.asyncio
async def test_prime_lab_view_click_opens_local_eval_run_screen(tmp_path: Path) -> None:
    run_dir = tmp_path / "outputs" / "evals" / "gsm8k--openai--gpt-4" / "run-a"
    run_dir.mkdir(parents=True)
    (run_dir / "metadata.json").write_text(
        '{"avg_reward": 0.5, "num_examples": 1, "rollouts_per_example": 1}',
        encoding="utf-8",
    )
    (run_dir / "results.jsonl").write_text(
        '{"reward": 0.5, "prompt": [{"role": "user", "content": "2+2?"}], '
        '"completion": [{"role": "assistant", "content": "4"}]}\n',
        encoding="utf-8",
    )
    source = make_source()
    snapshot = source.load(LabLoadOptions(limit=10, workspace=tmp_path))
    app = PrimeLabView(
        lambda: snapshot,
        lambda item, include_logs, log_tail_lines, metrics_limit, metrics_min_step: (
            source.load_item_detail(
                item,
                include_logs=include_logs,
                log_tail_lines=log_tail_lines,
                metrics_limit=metrics_limit,
                metrics_min_step=metrics_min_step,
            )
        ),
    )

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        app._active_section_key = "evaluations"
        app._render_active_section()
        await pilot.pause()
        local_index = next(
            idx
            for idx, item in enumerate(app._visible_items)
            if item.raw.get("type") == "local_eval"
        )
        option_list = app.query_one("#item-list", LabOptionList)
        assert await pilot.click(option_list, offset=(2, local_index * 2 + 1))
        await pilot.pause()

        assert isinstance(app.screen, LocalEvalRunScreen)


@pytest.mark.asyncio
async def test_prime_lab_view_mouse_click_requires_visible_training_sidebar(
    tmp_path: Path,
) -> None:
    source = make_source()
    snapshot = source.load(LabLoadOptions(limit=10, workspace=tmp_path))
    training = snapshot.section("training")
    assert training is not None
    run = training.items[0]

    app = PrimeLabView(
        lambda: snapshot,
        lambda item, include_logs, log_tail_lines, metrics_limit, metrics_min_step: (
            source.load_item_detail(
                item,
                include_logs=include_logs,
                log_tail_lines=log_tail_lines,
                metrics_limit=metrics_limit,
                metrics_min_step=metrics_min_step,
            )
        ),
    )

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        app._prefetching_detail_key = run.key
        app._active_section_key = "training"
        app._render_active_section()
        await pilot.pause()

        app._detail_cache[run.key] = source.load_item_detail(run)
        app._selected_item = None
        app._expand_ready_run_key = None

        option_list = app.query_one("#item-list", LabOptionList)
        assert await pilot.click(option_list, offset=(2, 1))
        await pilot.pause()

        assert not isinstance(app.screen, TrainingRunScreen)
        assert app._selected_item is not None
        assert app._selected_item.key == run.key
        assert app._expand_ready_run_key == run.key

        assert await pilot.click(option_list, offset=(2, 1))
        await pilot.pause()

        assert isinstance(app.screen, TrainingRunScreen)


def _render_details(item: Any) -> str:
    console = Console(record=True, width=120)
    console.print(_item_details(item))
    return console.export_text()


def _render_renderable(renderable: Any) -> str:
    console = Console(record=True, width=120)
    console.print(renderable)
    return console.export_text()


def _tar_bytes(files: dict[str, str]) -> bytes:
    buffer = BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name, text in files.items():
            data = text.encode("utf-8")
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, BytesIO(data))
    return buffer.getvalue()


def _local_eval_item(
    key: str,
    run_id: str,
    run_dir: Path,
    metadata: dict[str, Any],
    *,
    env_id: str = "gsm8k",
    model: str = "openai/gpt-4",
) -> LabItem:
    return LabItem(
        key=key,
        section="evaluations",
        title=run_id,
        subtitle=f"{model} · {env_id}",
        status="",
        status_style="dim",
        metadata=(
            ("Environment", env_id),
            ("Model", model),
            ("Avg reward", str(metadata.get("avg_reward", "-"))),
        ),
        raw={
            "type": "local_eval",
            "env_id": env_id,
            "model": model,
            "run_id": run_id,
            "path": str(run_dir),
            "metadata": metadata,
        },
    )
