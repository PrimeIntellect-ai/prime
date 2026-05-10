from __future__ import annotations

import builtins
import json
import os
import queue
import shlex
import sys
import tarfile
import threading
import time
import types
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any, Callable

import prime_lab_app.agent_widget_model as agent_widget_model
import pytest
import toml
from prime_cli.api.rl import RLClient
from prime_cli.commands.env import _environment_fork_chain, _environment_ref
from prime_cli.commands.lab import app as lab_cli_app
from prime_cli.commands.rl import RLConfig as HostedRLConfig
from prime_cli.lab_mcp import _serve_lab_mcp_stdio, lab_mcp_tool_definitions
from prime_cli.lab_setup import (
    LabDoctorOptions,
    LabSetupOptions,
    LabSetupResult,
    LabSyncOptions,
    parse_lab_doctor_args,
    parse_lab_setup_args,
    parse_lab_sync_args,
    run_lab_doctor_service,
    run_lab_setup_service,
    run_lab_sync_service,
)
from prime_lab_app.agent_acp import (
    acp_lab_mcp_servers,
    acp_session_params,
    acp_session_support,
    acp_update_event,
)
from prime_lab_app.agent_adapters import (
    agent_adapter,
    agent_mcp_config_path,
    agent_select_options,
    pi_lab_extension_path,
)
from prime_lab_app.agent_capabilities import agent_capability, known_agent_names
from prime_lab_app.agent_cards import (
    AgentWidgetCard,
    _choice_followup_status,
    _widget_card_body,
    _widget_card_heading,
)
from prime_lab_app.agent_mcp_bridge import (
    LabMcpIpcServer,
    _dump_simple_yaml,
    call_lab_mcp_tool,
    lab_mcp_runtime_dir,
    write_amp_mcp_config,
    write_droid_mcp_config,
    write_hermes_mcp_config,
    write_opencode_mcp_config,
)
from prime_lab_app.agent_runtime import (
    AgentChatMessage,
    AgentConnectionState,
    AgentRuntime,
    _dedupe_streamed_text,
    _extract_stream_delta,
    _is_lab_widget_tool_result_text,
    _merge_stream_text,
)
from prime_lab_app.agent_screen import (
    AgentChatScreen,
    AgentPrompt,
    _agent_thinking_turn,
    _chat_transcript,
    _choice_followup_prompt,
)
from prime_lab_app.agent_sessions import (
    append_agent_prompt_history,
    create_agent_session,
    latest_agent_session,
    load_agent_prompt_history,
    workspace_session_key,
    write_agent_session,
)
from prime_lab_app.agent_validation import validate_agent_surfaces
from prime_lab_app.agent_widget_actions import build_agent_widget_config
from prime_lab_app.agent_widget_model import (
    _training_model_options_for_name,
    build_agent_widget_model,
    widget_training_model_option_parts,
)
from prime_lab_app.agent_widgets import (
    handle_lab_widget_tool_call,
    lab_dynamic_tools,
    lab_widget_action_from_tool_call,
    lab_widget_developer_instructions,
    lab_widget_diagnostic_prompt,
)
from prime_lab_app.app import (
    EvaluationTree,
    LabOptionList,
    PrimeLabView,
    _item_details,
    _ladder_limits,
    _parse_log_records,
    _search_platform_environments,
)
from prime_lab_app.cache import (
    cached_environment_source,
    ensure_environment_source,
    environment_source_blob_cache_path,
    environment_source_cache_path,
    forget_recent_workspace,
    lab_row_cache_path,
    load_cached_lab_sections,
    recent_workspaces,
    record_recent_workspace,
    write_cached_lab_sections,
)
from prime_lab_app.chat_parts import ReferencePart, message_parts
from prime_lab_app.config_factory import evaluation_config, format_lab_config, rl_config
from prime_lab_app.config_screen import (
    ConfigLaunchScreen,
    ConfigRunScreen,
    build_config_from_fields,
    launch_command_for_config,
)
from prime_lab_app.data import LabDataSource, LabLoadOptions, discover_local_eval_runs
from prime_lab_app.environment_screen import (
    AddWorkspaceScreen,
    EnvironmentAction,
    EnvironmentScreen,
    WorkspaceScreen,
)
from prime_lab_app.eval_markdown import MathMarkdown, make_math_parser
from prime_lab_app.eval_records import LazyRunResults, LocalEvalRun
from prime_lab_app.eval_render import compute_run_overview_stats, history_groups
from prime_lab_app.eval_screen import LocalEvalRunScreen, RolloutCopyScreen, RolloutViewer
from prime_lab_app.evaluation_browser import (
    evaluation_index,
    evaluation_model_selection_details,
    evaluation_run_selection_details,
)
from prime_lab_app.filters import FilterChoice, filter_choices
from prime_lab_app.launch_backdrop import LaunchBackdrop
from prime_lab_app.launch_runner import ConfigLaunchRunner, extract_training_log_follow_command
from prime_lab_app.launch_screen import LaunchScreen
from prime_lab_app.models import LabItem, LabSection, LabSnapshot
from prime_lab_app.palette import TOOL_CALL
from prime_lab_app.platform_preview import preview_lab_config
from prime_lab_app.readme import readme_links as _readme_links
from prime_lab_app.readme import readme_markdown as _readme_markdown
from prime_lab_app.rows import item_badges_text
from prime_lab_app.setup_screens import AgentSyncScreen, DoctorScreen, SetupScreen, _setup_body
from prime_lab_app.shell import (
    compact_path,
    configured_workspace_agent,
    lab_header,
    warning_popover_text,
)
from prime_lab_app.snapshots import merge_snapshot_rows
from prime_lab_app.source_browser import source_entries, source_preview
from prime_lab_app.training_charts import (
    ChartSpec,
    LabPlotWidget,
)
from prime_lab_app.training_charts import (
    chart_count as _chart_count,
)
from prime_lab_app.training_charts import (
    histogram_charts_from_raw as _histogram_charts_from_raw,
)
from prime_lab_app.training_config import (
    training_config_toml as _training_config_toml,
)
from prime_lab_app.training_config import (
    training_platform_url as _training_platform_url,
)
from prime_lab_app.training_render import (
    training_progress_summary,
)
from prime_lab_app.training_render import (
    training_run_widgets as _training_run_widgets,
)
from prime_lab_app.training_screen import (
    TrainingRunScreen,
    _merge_training_detail,
    _next_log_tail_lines,
)
from prime_lab_app.widgets import (
    ClearableInput,
    EvaluationViewToggle,
    HomeGroupToggle,
    HomeLaunchPanel,
    ScopeToggle,
)
from rich.console import Console
from textual.containers import Vertical, VerticalScroll
from textual.widgets import Button, Label, OptionList, Select, Static, Tree
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
        "Settings",
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


def test_lab_view_caps_combined_evaluation_rows(tmp_path: Path) -> None:
    for idx in range(4):
        run_dir = tmp_path / "outputs" / "evals" / f"env-{idx}--model" / f"run-{idx}"
        run_dir.mkdir(parents=True)
        (run_dir / "metadata.json").write_text(
            '{"avg_reward": 0.5, "num_examples": 1}',
            encoding="utf-8",
        )
        (run_dir / "results.jsonl").write_text('{"reward": 0.5}\n', encoding="utf-8")

    class ManyEvalsClient(FakeEvalsClient):
        def list_evaluations(self, **_kwargs: Any) -> dict[str, Any]:
            return {
                "evaluations": [
                    {
                        "evaluation_id": f"eval-{idx}",
                        "environment_names": [f"env-{idx}"],
                        "model_name": "model",
                        "status": "COMPLETED",
                        "is_hosted": True,
                    }
                    for idx in range(6)
                ]
            }

    source = LabDataSource(
        api_client_factory=FakeAPIClient,
        evals_client_factory=ManyEvalsClient,
        rl_client_factory=FakeRLClient,
        config_factory=FakeConfig,
    )

    snapshot = source.load(LabLoadOptions(limit=3, workspace=tmp_path))
    evaluations = snapshot.section("evaluations")

    assert evaluations is not None
    assert len(evaluations.items) == 3
    assert [item.raw.get("source") for item in evaluations.items] == ["hosted", "hosted", "local"]
    assert evaluations.status == "3 shown"


@pytest.mark.asyncio
async def test_prime_lab_app_environment_scope_defaults_to_account_and_can_show_public(
    tmp_path: Path,
) -> None:
    snapshot = make_source().load(LabLoadOptions(limit=10, workspace=tmp_path))
    app = PrimeLabView(lambda: snapshot, initial_loader=lambda: snapshot)

    async with app.run_test(size=(140, 44)) as pilot:
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

        app._active_section_key = "environments"
        app._render_active_section()
        await pilot.pause()

        assert app.query_one("#scope-toggle", ScopeToggle).display is True
        assert [item.title for item in app._visible_items] == ["research/private-env"]

        app.set_scope_view("public")
        await pilot.pause()

        assert "primeintellect/gsm8k" in {item.title for item in app._visible_items}


def test_lab_view_evaluation_rows_mark_source_and_keep_status_consistent(tmp_path: Path) -> None:
    run_dir = tmp_path / "outputs" / "evals" / "gsm8k--openai--gpt-4" / "run-a"
    run_dir.mkdir(parents=True)
    (run_dir / "metadata.json").write_text(
        '{"avg_reward": 0.5, "num_examples": 1}',
        encoding="utf-8",
    )
    (run_dir / "results.jsonl").write_text('{"reward": 0.5}\n', encoding="utf-8")

    snapshot = make_source().load(LabLoadOptions(limit=10, workspace=tmp_path))
    evaluations = snapshot.section("evaluations")

    assert evaluations is not None
    hosted = next(item for item in evaluations.items if item.raw.get("source") == "hosted")
    local = next(item for item in evaluations.items if item.raw.get("source") == "local")

    assert hosted.metadata[2] == ("Type", "hosted")
    assert [badge["label"] for badge in hosted.raw["badges"]] == ["HOSTED", "COMPLETED"]
    assert local.status == "COMPLETED"
    assert local.metadata[2] == ("Type", "local")
    assert [badge["label"] for badge in local.raw["badges"]] == ["LOCAL", "COMPLETED"]


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


def test_environment_metadata_tracks_fork_chain() -> None:
    origin = _environment_ref(
        "primeintellect",
        "alphabet-sort",
        environment_id="env-original",
        version="0.1.8",
    )
    upstream = _environment_ref(
        "research",
        "alphabet-sort",
        environment_id="env-fork",
        version="0.1.9",
    )

    chain = _environment_fork_chain(
        {
            "origin": origin,
            "fork_chain": [origin],
        },
        upstream,
    )

    assert chain == [origin, upstream]


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

    monkeypatch.setattr("prime_lab_app.cache.httpx.stream", fake_stream)

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
    assert cached.root == environment_source_blob_cache_path(cached.manifest["content_hash"])
    assert (cached.root / "README.md").read_text(encoding="utf-8") == "# Cached Env\n"
    assert cached.manifest["slug"] == "research/cached-env"
    assert cached.manifest["version"] == "0.2.0"
    pointer_manifest = (
        environment_source_cache_path("research", "cached-env", "0.2.0")
        / ".prime"
        / "lab-cache.json"
    )
    assert (
        json.loads(pointer_manifest.read_text(encoding="utf-8"))["content_hash"]
        == (cached.manifest["content_hash"])
    )


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


def test_lab_environment_cache_resolves_version_pointer_to_hash_blob(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    pointer = environment_source_cache_path("research", "cached-env", "0.2.0")
    pointer_manifest = pointer / ".prime" / "lab-cache.json"
    blob_root = tmp_path / ".prime" / "lab" / "cache" / "sources"
    content_hash = "a" * 64
    source_root = environment_source_blob_cache_path(content_hash)
    source_manifest = blob_root / content_hash / "manifest.json"
    source_root.mkdir(parents=True)
    (source_root / "README.md").write_text("# Blob Env\n", encoding="utf-8")
    pointer_manifest.parent.mkdir(parents=True)
    pointer_manifest.write_text(
        json.dumps(
            {
                "slug": "research/cached-env",
                "version": "0.2.0",
                "content_hash": content_hash,
                "cached_at": "2026-01-01T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    source_manifest.write_text(
        json.dumps(
            {
                "slug": "research/cached-env",
                "version": "0.2.0",
                "content_hash": content_hash,
                "cached_at": "2026-01-01T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )

    cached = cached_environment_source({"slug": "research/cached-env", "latest_version": "0.2.0"})

    assert cached is not None
    assert cached.root == source_root
    assert (cached.root / "README.md").read_text(encoding="utf-8") == "# Blob Env\n"


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

    monkeypatch.setattr("prime_lab_app.cache.httpx.stream", lambda *_args, **_kwargs: FakeStream())

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
        "Settings",
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
    assert training.row_data_origin == "disk"
    assert training.refreshed_at


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
    assert cached["training"].row_data_origin == "disk"
    assert cached["training"].refreshed_at


def test_lab_view_row_cache_stores_per_section_freshness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    cache_key = "row-cache-freshness"

    write_cached_lab_sections(
        cache_key,
        (
            LabSection(
                key="training",
                title="Training",
                description="Training runs.",
                items=(LabItem(key="rl-run:1", section="training", title="run-1"),),
                refreshed_at="2026-05-01T00:00:00+00:00",
                row_data_origin="live",
            ),
        ),
    )

    payload = json.loads(lab_row_cache_path(cache_key).read_text(encoding="utf-8"))
    cached = load_cached_lab_sections(cache_key, limit=10)

    assert payload["sections"]["training"]["refreshed_at"] == "2026-05-01T00:00:00+00:00"
    assert cached["training"].refreshed_at == "2026-05-01T00:00:00+00:00"
    assert cached["training"].row_data_origin == "disk"


def test_lab_view_live_sections_include_freshness(tmp_path: Path) -> None:
    snapshot = make_source().load(LabLoadOptions(limit=10, workspace=tmp_path))
    training = snapshot.section("training")

    assert training is not None
    assert training.refreshed_at
    assert training.row_data_origin == "live"


def test_merge_snapshot_rows_preserves_newer_freshness(tmp_path: Path) -> None:
    previous = LabSnapshot(
        workspace=tmp_path,
        base_url="https://api.test",
        frontend_url="https://app.test",
        authenticated=True,
        team="team",
        sections=(
            LabSection(
                key="training",
                title="Training",
                description="Training runs.",
                items=(
                    LabItem(key="rl-run:1", section="training", title="run-1"),
                    LabItem(key="rl-run:2", section="training", title="run-2"),
                ),
                refreshed_at="2026-05-01T00:00:00+00:00",
                row_data_origin="disk",
            ),
        ),
    )
    incoming = LabSnapshot(
        workspace=tmp_path,
        base_url="https://api.test",
        frontend_url="https://app.test",
        authenticated=True,
        team="team",
        sections=(
            LabSection(
                key="training",
                title="Training",
                description="Training runs.",
                items=(LabItem(key="rl-run:1", section="training", title="run-1"),),
                refreshed_at="2026-05-02T00:00:00+00:00",
                row_data_origin="live",
            ),
        ),
    )

    merged = merge_snapshot_rows(previous, incoming)
    training = merged.section("training")

    assert training is not None
    assert [item.title for item in training.items] == ["run-1", "run-2"]
    assert training.refreshed_at == "2026-05-02T00:00:00+00:00"
    assert training.row_data_origin == "mixed"


def test_merge_snapshot_rows_drops_error_placeholders_when_cache_exists(
    tmp_path: Path,
) -> None:
    previous = LabSnapshot(
        workspace=tmp_path,
        base_url="https://api.test",
        frontend_url="https://app.test",
        authenticated=True,
        team="team",
        sections=(
            LabSection(
                key="training",
                title="Training",
                description="Training runs.",
                items=(LabItem(key="rl-run:1", section="training", title="run-1"),),
                refreshed_at="2026-05-01T00:00:00+00:00",
                row_data_origin="disk",
            ),
        ),
    )
    incoming = LabSnapshot(
        workspace=tmp_path,
        base_url="https://api.test",
        frontend_url="https://app.test",
        authenticated=True,
        team="team",
        sections=(
            LabSection(
                key="training",
                title="Training",
                description="Training runs.",
                items=(
                    LabItem(
                        key="training:error",
                        section="training",
                        title="Unavailable",
                        raw={"error": "boom"},
                    ),
                ),
                refreshed_at="2026-05-02T00:00:00+00:00",
                row_data_origin="live",
            ),
        ),
    )

    merged = merge_snapshot_rows(previous, incoming)
    training = merged.section("training")

    assert training is not None
    assert [item.title for item in training.items] == ["run-1"]


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
    assert "Unavailable" not in [item.title for item in training.items]
    assert "Unavailable" not in [item.title for item in environments.items]
    assert "Unavailable" not in [item.title for item in evaluations.items]


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


def test_agent_surface_validation_reports_missing_native_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "prime_lab_app.agent_capabilities.shutil.which",
        lambda _binary: "/bin/tool",
    )

    class Completed:
        returncode = 0

    results = validate_agent_surfaces(
        tmp_path,
        agents=("cursor",),
        runner=lambda *_args, **_kwargs: Completed(),
    )

    assert len(results) == 1
    assert results[0].agent == "cursor"
    assert results[0].ok is False
    assert "Missing native surface files" in results[0].message


def test_agent_surface_validation_passes_when_binary_and_surface_exist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "prime_lab_app.agent_capabilities.shutil.which",
        lambda _binary: "/bin/tool",
    )
    (tmp_path / ".cursor").mkdir()
    (tmp_path / ".cursor" / "mcp.json").write_text("{}", encoding="utf-8")

    class Completed:
        returncode = 0

    results = validate_agent_surfaces(
        tmp_path,
        agents=("cursor",),
        runner=lambda *_args, **_kwargs: Completed(),
    )

    assert len(results) == 1
    assert results[0].ok is True


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


def test_training_progress_counts_step_zero_as_completed() -> None:
    summary = training_progress_summary(
        {"max_steps": 10, "status": "RUNNING"},
        {"latest_step": 0},
        [],
    )

    assert summary["current_step"] == 0
    assert summary["steps_completed"] == 1
    assert summary["progress_percent"] == 10.0


def test_training_log_reload_preserves_accumulated_metric_pages() -> None:
    previous = LabItem(
        key="training:run-1",
        section="training",
        title="run-1",
        subtitle="",
        status="RUNNING",
        status_style="green",
        raw={
            "recent_metrics": [{"step": 0, "loss": 2.0}, {"step": 1, "loss": 1.5}],
            "metrics_loaded": True,
            "metrics_min_step": 1,
        },
    )
    incoming = replace(
        previous,
        raw={
            "recent_metrics": [{"step": 0, "loss": 1.9}],
            "metrics_loaded": True,
            "metrics_min_step": None,
            "logs_tail": "latest log",
            "logs_loaded": True,
        },
    )

    merged = _merge_training_detail(previous, incoming)

    assert [metric["step"] for metric in merged.raw["recent_metrics"]] == [0, 1]
    assert merged.raw["recent_metrics"][0]["loss"] == 1.9
    assert merged.raw["logs_tail"] == "latest log"


def test_training_chart_failure_marks_plot_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_draw(_plot: Any, _chart: ChartSpec) -> None:
        raise RuntimeError("plot failed")

    monkeypatch.setattr("prime_lab_app.training_charts.draw_plot", fail_draw)
    plot = LabPlotWidget(
        ChartSpec(
            title="Reward",
            kind="line",
            x=(1, 2),
            y=(0.1, 0.2),
            xlabel="step",
            ylabel="reward",
        )
    )

    plot._draw_chart()

    assert plot.border_title == "Chart unavailable"
    assert plot.border_subtitle == "Could not render chart"


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


def test_y_copies_on_rollout_copy_surfaces() -> None:
    for surface in (RolloutCopyScreen, RolloutViewer, LocalEvalRunScreen):
        copy_keys = {binding.key for binding in surface.BINDINGS if binding.action == "copy"}
        assert "y" in copy_keys


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
    assert "prime env install primeintellect/gsm8k" not in rendered
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


def test_lab_config_factory_renders_shared_eval_and_rl_templates() -> None:
    eval_toml = format_lab_config(
        evaluation_config(env_id="primeintellect/wordle", num_examples=-1, max_tokens=None)
    )
    rl_toml = format_lab_config(rl_config(env_id="primeintellect/wordle", version="1.0.0"))

    parsed_eval = toml.loads(eval_toml)
    parsed_rl = toml.loads(rl_toml)

    assert parsed_eval["eval"][0]["env_id"] == "primeintellect/wordle"
    assert parsed_eval["eval"][0]["num_examples"] == -1
    assert "sampling_args" not in parsed_eval["eval"][0]
    assert parsed_rl["env"] == [{"id": "primeintellect/wordle", "version": "1.0.0"}]
    assert parsed_rl["sampling"]["max_tokens"] == 512


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


def test_local_eval_run_screen_keeps_rollouts_lazy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "outputs" / "evals" / "gsm8k--openai--gpt-4" / "run-a"
    run_dir.mkdir(parents=True)
    (run_dir / "metadata.json").write_text(
        '{"avg_reward": 0.75, "num_examples": 2, "rollouts_per_example": 1}',
        encoding="utf-8",
    )
    (run_dir / "results.jsonl").write_text(
        '{"reward": 0.5}\n{"reward": 1.0}\n',
        encoding="utf-8",
    )
    run = LocalEvalRun(env_id="gsm8k", model="openai/gpt-4", run_id="run-a", path=run_dir)

    def fail_len(_records: LazyRunResults) -> int:
        raise AssertionError("Local eval screen should not count every rollout at startup")

    monkeypatch.setattr(LazyRunResults, "__len__", fail_len)
    screen = LocalEvalRunScreen(run)
    try:
        assert screen._record_count == 2
        assert screen._rollout_records[0]["reward"] == 0.5
    finally:
        screen.records.close()


def test_local_eval_run_screen_counts_rollouts_without_metadata_hint(tmp_path: Path) -> None:
    run_dir = tmp_path / "outputs" / "evals" / "gsm8k--openai--gpt-4" / "run-a"
    run_dir.mkdir(parents=True)
    (run_dir / "metadata.json").write_text('{"avg_reward": 0.75}', encoding="utf-8")
    (run_dir / "results.jsonl").write_text(
        '{"reward": 0.5}\n{"reward": 1.0}\n',
        encoding="utf-8",
    )
    run = LocalEvalRun(env_id="gsm8k", model="openai/gpt-4", run_id="run-a", path=run_dir)

    screen = LocalEvalRunScreen(run)
    try:
        assert screen._record_count is None
        assert len(screen._rollout_records) == 2
        assert screen._rollout_records.loaded(1) is None
        assert screen._rollout_records[1]["reward"] == 1.0
    finally:
        screen.records.close()


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

    monkeypatch.setattr("prime_lab_app.run_lab_view", fake_run_lab_view)

    result = CliRunner().invoke(lab_cli_app, [])

    assert result.exit_code == 0
    assert calls
    assert calls[0]["limit"] == 1000


def test_prime_lab_app_alias_launches_viewer(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    def fake_run_lab_view(**kwargs: Any) -> None:
        calls.append(kwargs)

    monkeypatch.setattr("prime_lab_app.run_lab_view", fake_run_lab_view)

    result = CliRunner().invoke(lab_cli_app, ["view", "--limit", "7"])

    assert result.exit_code == 0
    assert calls[0]["limit"] == 7


def test_prime_lab_setup_uses_setup_service(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def fake_run_lab_setup(args: list[str], **_kwargs: Any) -> int:
        calls.append(args)
        return 0

    monkeypatch.setattr("prime_cli.lab_setup.run_lab_setup", fake_run_lab_setup)

    result = CliRunner().invoke(lab_cli_app, ["setup", "--agent", "codex"])

    assert result.exit_code == 0
    assert calls == [["--agent", "codex"]]


def test_prime_lab_sync_uses_sync_service(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def fake_run_lab_sync(args: list[str], **_kwargs: Any) -> int:
        calls.append(args)
        return 0

    monkeypatch.setattr("prime_cli.lab_setup.run_lab_sync", fake_run_lab_sync)

    result = CliRunner().invoke(lab_cli_app, ["sync", "--agent", "codex"])

    assert result.exit_code == 0
    assert calls == [["--agent", "codex"]]


def test_prime_lab_doctor_uses_doctor_service(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def fake_run_lab_doctor(args: list[str], **_kwargs: Any) -> int:
        calls.append(args)
        return 0

    monkeypatch.setattr("prime_cli.lab_setup.run_lab_doctor", fake_run_lab_doctor)

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
    gitignore_lines = set(gitignore.splitlines())
    assert ".env" in gitignore_lines
    assert "/outputs/" in gitignore_lines
    assert "/prime-rl/" in gitignore_lines
    assert "*.py[cod]" in gitignore_lines
    assert any(
        check.name == "Configs directory" and check.status == "PASS" for check in fixed.checks
    )


def test_prime_lab_doctor_checks_all_configured_agent_surfaces(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("prime_lab_app.agent_capabilities.shutil.which", lambda _binary: "/bin/x")
    (tmp_path / ".prime").mkdir()
    (tmp_path / ".prime" / "lab.json").write_text(
        json.dumps(
            {
                "choices": {
                    "agents": ["codex", "claude"],
                    "primary_agent": "codex",
                    "use_multiple_agents": True,
                }
            }
        ),
        encoding="utf-8",
    )

    result = run_lab_doctor_service(LabDoctorOptions(), workspace=tmp_path)
    checks = {check.name: check for check in result.checks}

    assert checks["Codex native tools"].status == "PASS"
    assert checks["Claude native tools"].status == "WARN"
    assert ".prime/lab/agent-mcp/claude.json" in checks["Claude native tools"].message
    assert "Coding agent" not in checks


def test_prime_lab_doctor_service_checks_configs_and_source_hygiene(tmp_path: Path) -> None:
    configs_dir = tmp_path / "configs" / "rl"
    configs_dir.mkdir(parents=True)
    (configs_dir / "broken.toml").write_text("model = 'openai/gpt-5-mini'\n[", encoding="utf-8")
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
    (tmp_path / "environments" / "other-env").mkdir(parents=True)

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
    resolved = run_lab_doctor_service(LabDoctorOptions(), workspace=tmp_path)

    assert any(
        check.name == "Config environment refs" and check.status == "PASS"
        for check in resolved.checks
    )


def _fake_lab_skill_download_json(url: str) -> Any:
    config_tree: dict[str, list[tuple[str, str]]] = {
        "configs": [
            ("endpoints.toml", "file"),
            ("eval", "dir"),
            ("gepa", "dir"),
            ("local", "dir"),
            ("rl", "dir"),
            ("zero3.yaml", "file"),
        ],
        "configs/eval": [
            ("debug.toml", "file"),
            ("minimal.toml", "file"),
            ("multi-env.toml", "file"),
            ("wordle.toml", "file"),
        ],
        "configs/gepa": [
            ("base.toml", "file"),
            ("wordle.toml", "file"),
        ],
        "configs/local": [("prime-rl", "dir")],
        "configs/local/prime-rl": [("wiki-search.toml", "file")],
        "configs/rl": [
            ("alphabet-sort.toml", "file"),
            ("gsm8k.toml", "file"),
            ("math-python.toml", "file"),
            ("reverse-text.toml", "file"),
            ("wiki-search.toml", "file"),
            ("wordle.toml", "file"),
        ],
    }
    if "/git/trees/" in url:
        tree = [
            {"path": "skills", "type": "tree"},
            {"path": "skills/create-environments", "type": "tree"},
            {"path": "skills/create-environments/SKILL.md", "type": "blob"},
        ]
        for source_path, entries in config_tree.items():
            tree.append({"path": source_path, "type": "tree"})
            for name, entry_type in entries:
                tree.append(
                    {
                        "path": f"{source_path}/{name}",
                        "type": "tree" if entry_type == "dir" else "blob",
                    }
                )
        return {"tree": tree}
    return []


def test_prime_lab_sync_service_refreshes_agent_assets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    downloads: list[Path] = []

    def fake_download(_url: str, dest: Path, _emit: Any, *, force: bool = False) -> None:
        downloads.append(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(f"{dest.name}\n", encoding="utf-8")

    monkeypatch.setattr("prime_cli.lab_setup._download_file", fake_download)
    monkeypatch.setattr("prime_cli.lab_setup._download_json", _fake_lab_skill_download_json)
    monkeypatch.setattr(
        "prime_lab_app.agent_capabilities.shutil.which",
        lambda command: "/bin/pi" if command == "pi" else None,
    )
    emitted: list[str] = []

    assert parse_lab_sync_args(["--agent", "pi"]).agents == ("pi",)
    assert parse_lab_sync_args([]).agents == ()

    result = run_lab_sync_service(
        LabSyncOptions(agents=("pi",)),
        workspace=tmp_path,
        emit=emitted.append,
    )

    assert result.exit_code == 0
    assert (tmp_path / ".prime" / "skills" / "create-environments" / "SKILL.md").is_file()
    assert (tmp_path / ".pi" / "skills" / "create-environments").exists()
    assert not any("pi-acp" in line for line in emitted)
    assert (tmp_path / ".prime" / "lab" / "templates" / "configs" / "rl" / "gsm8k.toml").is_file()
    assert (tmp_path / ".prime" / "lab" / "docs" / "index.md").is_file()
    assert (tmp_path / "AGENTS.md").is_file()
    assert downloads


def test_prime_lab_sync_service_preserves_workspace_agent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / ".prime").mkdir()
    (tmp_path / ".prime" / "lab.json").write_text(
        '{"choices": {"agents": ["opencode"], "primary_agent": "opencode"}}',
        encoding="utf-8",
    )

    def fake_download(_url: str, dest: Path, _emit: Any, *, force: bool = False) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(f"{dest.name}\n", encoding="utf-8")

    monkeypatch.setattr("prime_cli.lab_setup._download_file", fake_download)
    monkeypatch.setattr("prime_cli.lab_setup._download_json", _fake_lab_skill_download_json)

    result = run_lab_sync_service(LabSyncOptions(), workspace=tmp_path)

    metadata = json.loads((tmp_path / ".prime" / "lab.json").read_text(encoding="utf-8"))
    assert result.exit_code == 0
    assert metadata["choices"]["primary_agent"] == "opencode"
    assert (tmp_path / ".opencode" / "skills" / "create-environments").exists()
    assert metadata["setup_source"] == "prime lab sync"


def test_prime_lab_setup_service_supports_pi_agent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    def fake_download(url: str, dest: Path, emit: Any, *, force: bool = False) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(f"downloaded from {url}\n", encoding="utf-8")

    monkeypatch.setattr("prime_cli.lab_setup._download_file", fake_download)
    monkeypatch.setattr("prime_cli.lab_setup._download_json", _fake_lab_skill_download_json)
    monkeypatch.setattr(
        "prime_lab_app.agent_capabilities.shutil.which",
        lambda command: "/bin/pi" if command == "pi" else None,
    )
    emitted: list[str] = []

    assert parse_lab_setup_args(["--agent", "pi"]).agents == ("pi",)

    result = run_lab_setup_service(
        LabSetupOptions(skip_install=True, skip_agents_md=True, agents=("pi",)),
        workspace=tmp_path,
        emit=emitted.append,
    )

    metadata = json.loads((tmp_path / ".prime" / "lab.json").read_text(encoding="utf-8"))
    assert result.exit_code == 0
    assert metadata["choices"]["primary_agent"] == "pi"
    assert (tmp_path / ".pi" / "skills" / "create-environments").exists()
    assert pi_lab_extension_path(tmp_path).is_file()
    extension_source = pi_lab_extension_path(tmp_path).read_text(encoding="utf-8")
    assert "process.env.PRIME_LAB_RUNTIME_DIR || os.tmpdir()" in extension_source
    assert 'process.env.PRIME_LAB_RUNTIME_DIR || "/tmp"' not in extension_source
    assert not any("pi-acp" in line for line in emitted)


def test_prime_lab_setup_service_supports_claude_code_agent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    def fake_download(url: str, dest: Path, emit: Any, *, force: bool = False) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(f"downloaded from {url}\n", encoding="utf-8")

    def fake_runner(command: Any, cwd: Path, emit: Any) -> int:
        return 0

    monkeypatch.setattr("prime_cli.lab_setup._download_file", fake_download)
    monkeypatch.setattr("prime_cli.lab_setup._download_json", _fake_lab_skill_download_json)

    assert parse_lab_setup_args(["--agent", "claude-code"]).agents == ("claude",)

    result = run_lab_setup_service(
        LabSetupOptions(skip_install=True, skip_agents_md=True, agents=("claude",)),
        workspace=tmp_path,
        emit=lambda _text: None,
        runner=fake_runner,
    )

    metadata = json.loads((tmp_path / ".prime" / "lab.json").read_text(encoding="utf-8"))
    assert result.exit_code == 0
    assert metadata["choices"]["primary_agent"] == "claude"
    assert (tmp_path / ".claude" / "skills" / "create-environments").exists()


def test_prime_lab_setup_service_supports_hermes_agent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    def fake_download(url: str, dest: Path, emit: Any, *, force: bool = False) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(f"downloaded from {url}\n", encoding="utf-8")

    def fake_runner(command: Any, cwd: Path, emit: Any) -> int:
        return 0

    monkeypatch.setattr("prime_cli.lab_setup._download_file", fake_download)
    monkeypatch.setattr("prime_cli.lab_setup._download_json", _fake_lab_skill_download_json)

    assert parse_lab_setup_args(["--agent", "hermes-agent"]).agents == ("hermes",)

    result = run_lab_setup_service(
        LabSetupOptions(skip_install=True, skip_agents_md=True, agents=("hermes",)),
        workspace=tmp_path,
        emit=lambda _text: None,
        runner=fake_runner,
    )

    metadata = json.loads((tmp_path / ".prime" / "lab.json").read_text(encoding="utf-8"))
    assert result.exit_code == 0
    assert metadata["choices"]["primary_agent"] == "hermes"
    assert (tmp_path / ".hermes" / "skills" / "create-environments").exists()


def test_lab_agent_adapters_map_known_and_custom_commands() -> None:
    assert agent_adapter("codex").prompt_command("hello") == ["codex", "exec", "hello"]
    assert agent_adapter("claude").prompt_command("hello") == ["claude", "-p", "hello"]
    assert agent_adapter("claude").server_spec(Path("/workspace")).transport == "resumable-cli"
    assert agent_adapter("codex").lab_widget_contract == "codex-dynamic-tools"
    assert agent_adapter("claude").lab_widget_contract == "mcp-stdio-tools"
    assert agent_adapter("claude-code").lab_widget_contract == "mcp-stdio-tools"
    assert agent_adapter("cursor").lab_widget_contract == "mcp-stdio-tools"
    assert agent_adapter("opencode").lab_widget_contract == "mcp-stdio-tools"
    assert agent_adapter("pi").lab_widget_contract == "pi-extension-tools"
    assert agent_adapter("hermes-agent").lab_widget_contract == "mcp-stdio-tools"
    workspace = Path("/workspace")
    allowed_lab_tools = ",".join(
        (
            "mcp__prime_lab__choose",
            "mcp__prime_lab__search_environments",
            "mcp__prime_lab__train_model",
            "mcp__prime_lab__edit_config",
            "mcp__prime_lab__preview_action",
            "mcp__prime_lab__launch_run",
            "mcp__prime_lab__show_patch",
            "mcp__prime_lab__inspect_rollouts",
        )
    )
    assert agent_adapter("claude-code").stream_command("hello", workspace=workspace) == [
        "claude",
        "-p",
        "--output-format",
        "stream-json",
        "--verbose",
        "--include-partial-messages",
        "--allowedTools",
        allowed_lab_tools,
        "--mcp-config",
        "/workspace/.prime/lab/agent-mcp/claude.json",
        "--",
        "hello",
    ]
    assert agent_adapter("amp-code").stream_command("hello", workspace=workspace) == [
        "amp",
        "--stream-json",
        "--mcp-config",
        "/workspace/.prime/lab/agent-mcp/amp.json",
        "--execute",
        "hello",
    ]
    assert agent_adapter("claude-code").name == "claude"
    assert agent_adapter("claude-cli").name == "claude"
    assert agent_adapter("cursor").stream_command("hello", "session-1") == [
        "cursor-agent",
        "-p",
        "--output-format",
        "stream-json",
        "--stream-partial-output",
        "--trust",
        "--approve-mcps",
        "--force",
        "--resume",
        "session-1",
        "hello",
    ]
    assert agent_adapter("opencode").prompt_command("hello") == ["opencode", "run", "hello"]
    assert agent_adapter("opencode").server_spec(workspace).command == (
        "opencode",
        "acp",
        "--cwd",
        "/workspace",
    )
    assert agent_adapter("opencode").server_spec(workspace).transport == "acp-stdio"
    assert agent_adapter("pi").prompt_command("hello") == ["pi", "--print", "hello"]
    assert agent_adapter("pi").stream_command("hello", workspace=workspace) == [
        "pi",
        "--print",
        "--mode",
        "json",
        "--no-session",
        "hello",
    ]
    assert agent_adapter("hermes").prompt_command("hello") == ["hermes", "--oneshot", "hello"]
    assert agent_adapter("hermes-agent").server_spec(workspace).command == (
        "hermes",
        "acp",
        "--accept-hooks",
    )
    assert agent_adapter("hermes-agent").server_spec(workspace).transport == "acp-stdio"
    assert agent_adapter("factory-droid").name == "droid"
    assert agent_adapter("factory-droid").lab_widget_contract == "mcp-stdio-tools"
    assert agent_adapter("factory-droid").stream_command("hello", workspace=workspace) == [
        "droid",
        "exec",
        "--output-format",
        "stream-json",
        "--cwd",
        "/workspace",
        "hello",
    ]
    assert agent_adapter("amp-code").name == "amp"
    pi_server = agent_adapter("pi").server_spec(Path("/workspace"))
    assert pi_server.command == ()
    assert pi_server.transport == "resumable-cli"
    assert agent_adapter("hermes").name == "hermes"
    assert agent_adapter("codex").server_spec(Path("/workspace")).command == (
        "codex",
        "app-server",
        "--listen",
        "stdio://",
    )
    assert agent_adapter("codex").server_spec(Path("/workspace")).transport == "codex-app-stdio"
    assert agent_adapter("my-agent").prompt_command("hello") == ["my-agent", "hello"]


def test_lab_agent_capabilities_centralize_supported_agents(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "prime_lab_app.agent_capabilities.shutil.which",
        lambda command: "/bin/tool" if command in {"pi", "codex"} else None,
    )

    assert known_agent_names() == (
        "amp",
        "claude",
        "codex",
        "cursor",
        "droid",
        "hermes",
        "opencode",
        "pi",
    )
    pi = agent_capability("pi")
    assert pi.label == "Pi Coding Agent"
    assert pi.native_surface == "pi_extension"
    assert pi.status == "supported"
    assert pi.missing_requirements() == ()
    assert pi.resolved_surface_paths(tmp_path) == (pi_lab_extension_path(tmp_path.resolve()),)
    assert agent_capability("amp").native_surface == "mcp_config"
    assert agent_capability("amp").resolved_surface_paths(tmp_path) == (
        tmp_path.resolve() / ".prime" / "lab" / "agent-mcp" / "amp.json",
    )
    droid = agent_capability("factory-droid")
    assert droid.name == "droid"
    assert droid.status == "supported"
    assert droid.native_surface == "droid_mcp_config"
    assert droid.resolved_surface_paths(tmp_path) == (tmp_path.resolve() / ".factory" / "mcp.json",)
    assert agent_capability("codex").native_surface == "codex_app_server"
    assert agent_capability("claude-code").name == "claude"
    assert agent_capability("claude").native_surface == "mcp_config"
    assert agent_capability("custom-agent").status == "not_supported"
    assert agent_capability("cursor").resolved_surface_paths(tmp_path) == (
        tmp_path.resolve() / ".cursor" / "mcp.json",
    )


def test_agent_session_cache_scopes_by_workspace_and_agent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    codex_session = create_agent_session(workspace, "codex")
    codex_session = write_agent_session(
        codex_session,
        AgentConnectionState(
            agent="codex",
            label="Codex",
            status="connected",
            transport="codex-app-stdio",
            workspace=workspace,
            session_id="thread-1",
        ),
        (
            AgentChatMessage("user", "hello"),
            AgentChatMessage(
                "system",
                "Widget requested",
                "widget",
                {"type": "widget_requested", "payload": {"title": "Pick env"}},
            ),
            AgentChatMessage("assistant", "hi"),
        ),
        base_url="https://api.test",
        team="research",
        authenticated=True,
    )
    claude_session = create_agent_session(workspace, "claude")

    loaded_codex = latest_agent_session(workspace, "codex")
    loaded_claude = latest_agent_session(workspace, "claude")
    assert workspace_session_key(workspace)
    assert loaded_codex == codex_session
    assert loaded_claude == claude_session
    assert loaded_codex is not None
    assert loaded_codex.native_session_id == "thread-1"
    assert loaded_codex.messages[-1].content == "hi"
    assert loaded_codex.messages[1].metadata["payload"]["title"] == "Pick env"


def test_agent_prompt_history_is_global_and_recent_unique(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    first_workspace = tmp_path / "first"
    second_workspace = tmp_path / "second"
    first_workspace.mkdir()
    second_workspace.mkdir()

    append_agent_prompt_history(first_workspace, "codex", "first prompt")
    append_agent_prompt_history(second_workspace, "cursor", "second prompt")
    append_agent_prompt_history(first_workspace, "codex", "first prompt")

    assert load_agent_prompt_history(limit=10) == ("second prompt", "first prompt")


def test_agent_acp_helpers_normalize_lab_mcp_and_updates(tmp_path: Path) -> None:
    servers = acp_lab_mcp_servers(tmp_path)
    assert servers[0]["name"] == "prime_lab"
    assert servers[0]["args"][-2:] == ["--workspace", str(tmp_path.resolve())]
    assert servers[0]["env"] == {}

    params = acp_session_params(tmp_path, session_id="session-1")
    assert params["sessionId"] == "session-1"
    assert params["mcpServers"][0]["name"] == "prime_lab"
    assert acp_session_params(tmp_path, mcp_servers=[])["mcpServers"] == []

    support = acp_session_support(
        {
            "agentCapabilities": {
                "sessionCapabilities": {"resume": {}, "close": {}},
                "loadSession": True,
            }
        }
    )
    assert support.resume is True
    assert support.load is True
    assert support.close is True

    delta = acp_update_event(
        {
            "sessionUpdate": "agent_message_chunk",
            "content": {"type": "text", "text": "hello"},
        }
    )
    assert delta.kind == "assistant_delta"
    assert delta.text == "hello"

    tool = acp_update_event(
        {
            "sessionUpdate": "tool_call_update",
            "toolCallId": "tool-1",
            "title": "Read config",
            "status": "completed",
            "content": [{"type": "content", "content": {"type": "text", "text": "ok"}}],
        }
    )
    assert tool.kind == "tool_update"
    assert tool.tool_call_id == "tool-1"
    assert tool.title == "Read config"
    assert tool.status == "completed"
    assert tool.text == "ok"


def test_lab_mcp_runtime_dir_without_getuid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PRIME_LAB_RUNTIME_DIR", str(tmp_path))
    monkeypatch.setenv("USERNAME", "windows-user")
    monkeypatch.delattr(os, "getuid", raising=False)

    runtime_dir = lab_mcp_runtime_dir(tmp_path / "workspace")

    assert runtime_dir.parent.name.startswith("prime-lab-")
    assert runtime_dir.parent.parent == tmp_path
    assert len(runtime_dir.name) == 24


def test_agent_runtime_filters_wrapped_lab_widget_tool_results() -> None:
    ack = {
        "success": True,
        "contentItems": [
            {
                "type": "inputText",
                "text": json.dumps(
                    {
                        "ok": True,
                        "tool": "choose",
                        "kind": "choice_picker",
                        "title": "Lab tool diagnostic",
                    }
                ),
            }
        ],
    }
    wrapped = json.dumps({"result": json.dumps(ack)})

    assert _is_lab_widget_tool_result_text(wrapped) is True


def test_agent_runtime_suppresses_empty_lab_tool_update_after_widget() -> None:
    messages: tuple[Any, ...] = ()

    def on_messages(value: Any) -> None:
        nonlocal messages
        messages = value

    runtime = AgentRuntime(on_messages=on_messages)
    runtime._messages = [
        AgentChatMessage(
            "system",
            "Lab tool diagnostic",
            "widget",
            {"tool": "choose", "kind": "choice_picker"},
        )
    ]

    runtime._record_acp_tool_event("prime_lab_choose", "completed", "")

    assert len(messages) == 0
    assert runtime.messages() == (
        AgentChatMessage(
            "system",
            "Lab tool diagnostic",
            "widget",
            {"tool": "choose", "kind": "choice_picker"},
        ),
    )


def test_agent_widget_choice_body_does_not_repeat_heading(tmp_path: Path) -> None:
    title = "What would you like to train on?"
    message = AgentChatMessage(
        "system",
        "Choice ready",
        "widget",
        {
            "kind": "choice_picker",
            "title": title,
            "payload": {
                "kind": "choice_picker",
                "title": title,
                "prompt": title,
                "candidates": [
                    {"id": "search", "label": "Search for an environment"},
                    {"id": "known", "label": "I know the env"},
                    {"id": "local", "label": "Use a local environment"},
                ],
            },
        },
    )
    model = build_agent_widget_model(message, tmp_path)

    assert title in _render_renderable(_widget_card_heading(model))
    body = _render_renderable(_widget_card_body(model))

    assert title not in body
    assert "Choices" in body
    assert "3" in body


def test_agent_widget_choice_followup_status_names_next_input() -> None:
    assert _choice_followup_status("Search for an environment") == (
        "Selected Search for an environment. "
        "Click Enter to continue, or add details below first if you want."
    )
    assert _choice_followup_status("I know the env") == (
        "Selected I know the env. Click Enter to continue, or add details below first if you want."
    )
    assert _choice_followup_status("Use a local environment") == (
        "Selected Use a local environment. "
        "Click Enter to continue, or add details below first if you want."
    )


def test_agent_widget_choice_followup_prompt_carries_selection_context() -> None:
    action = {"choice_label": "reverse-text"}

    assert _choice_followup_prompt(action, "") == "I chose: reverse-text."
    assert _choice_followup_prompt(action, "use Qwen") == "I chose: reverse-text.\n\nuse Qwen"
    assert _choice_followup_prompt(None, "plain prompt") == "plain prompt"


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


def test_agent_runtime_rejects_malformed_jsonrpc_response_ids(tmp_path: Path) -> None:
    states: list[AgentConnectionState] = []
    runtime = AgentRuntime(on_state=states.append)
    runtime._agent = "codex"
    runtime._label = "Codex"
    runtime._workspace = tmp_path

    runtime._handle_jsonrpc_message({"jsonrpc": "2.0", "id": {"bad": True}, "result": {}})

    assert states[-1].status == "error"
    assert "unsupported JSON-RPC id" in states[-1].message


@pytest.mark.parametrize(
    "agent_name",
    ("my-agent",),
)
def test_agent_runtime_triages_agents_without_lab_tool_surfaces(
    tmp_path: Path,
    agent_name: str,
) -> None:
    popen_called = False
    latest_messages: tuple[Any, ...] = ()

    def fake_popen(_command: list[str], **_kwargs: Any) -> Any:
        nonlocal popen_called
        popen_called = True
        raise AssertionError("unsupported agents should not start a process")

    def on_messages(messages: Any) -> None:
        nonlocal latest_messages
        latest_messages = messages

    runtime = AgentRuntime(on_messages=on_messages, popen_factory=fake_popen)
    runtime.start(tmp_path, agent_name)

    assert runtime.state.status == "unsupported"
    assert "not yet supported" in runtime.state.message
    assert latest_messages
    assert latest_messages[-1].status == "warning"
    runtime.send_prompt("hello")
    assert latest_messages[-1].status == "error"
    assert "not yet supported" in latest_messages[-1].content
    assert not popen_called


def test_lab_mcp_ipc_roundtrip(tmp_path: Path) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    def handler(tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        calls.append((tool, arguments))
        return {"ok": True, "tool": tool, "arguments": arguments}

    server = LabMcpIpcServer(tmp_path, handler)
    server.start()
    try:
        assert (server.path.parent.parent.stat().st_mode & 0o777) == 0o700
        assert (server.path.parent.stat().st_mode & 0o777) == 0o700
        assert (server.path.stat().st_mode & 0o777) == 0o600
        result = call_lab_mcp_tool(tmp_path, "choose", {"title": "Pick one"})
    finally:
        server.stop()

    assert result == {"ok": True, "tool": "choose", "arguments": {"title": "Pick one"}}
    assert calls == [("choose", {"title": "Pick one"})]


def test_lab_mcp_tool_definitions_include_widget_tools() -> None:
    tools = lab_mcp_tool_definitions()
    names = {tool["name"] for tool in tools}

    assert {
        "choose",
        "search_environments",
        "train_model",
        "edit_config",
        "preview_action",
        "launch_run",
        "show_patch",
        "inspect_rollouts",
    }.issubset(names)
    choose_tool = next(tool for tool in tools if tool["name"] == "choose")
    assert choose_tool["inputSchema"]["type"] == "object"
    train_tool = next(tool for tool in tools if tool["name"] == "train_model")
    train_schema = train_tool["inputSchema"]
    assert train_schema["additionalProperties"] is False
    assert train_schema["required"] == [
        "env",
        "model",
        "max_steps",
        "batch_size",
        "rollouts_per_example",
        "max_tokens",
    ]
    assert set(train_schema["properties"]) == {
        "env",
        "model",
        "max_steps",
        "batch_size",
        "rollouts_per_example",
        "max_tokens",
    }
    assert "environments" not in train_schema["properties"]
    assert "config" not in train_schema["properties"]
    assert "title" not in train_schema["properties"]
    edit_tool = next(tool for tool in tools if tool["name"] == "edit_config")
    assert edit_tool["inputSchema"]["additionalProperties"] is False
    assert edit_tool["inputSchema"]["properties"]["config_kind"]["enum"] == ["eval", "rl", "gepa"]
    launch_tool = next(tool for tool in tools if tool["name"] == "launch_run")
    assert launch_tool["inputSchema"]["properties"]["config_kind"]["enum"] == [
        "eval",
        "rl",
        "gepa",
    ]


def test_lab_config_tools_accept_training_configs() -> None:
    action = lab_widget_action_from_tool_call(
        {
            "namespace": "lab",
            "tool": "edit_config",
            "callId": "edit-rl",
            "arguments": {
                "title": "Edit training config",
                "config_kind": "rl",
                "config": {"model": "openai/gpt-oss-20b", "env": [{"id": "wordle"}]},
            },
        }
    )

    assert action["kind"] == "config_editor"
    assert action["payload"]["config_kind"] == "rl"


def test_lab_train_model_tool_uses_explicit_training_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "prime_lab_app.agent_widget_model._training_model_options",
        lambda: _training_model_options_for_name("openai/gpt-oss-20b"),
    )
    action = lab_widget_action_from_tool_call(
        {
            "namespace": "lab",
            "tool": "train_model",
            "callId": "train-1",
            "arguments": {
                "env": "primeintellect/wordle",
                "model": "openai/gpt-oss-20b",
                "max_steps": 10,
                "rollouts_per_example": 8,
                "batch_size": 256,
                "max_tokens": 2048,
            },
        }
    )

    assert action["kind"] == "run_launcher"
    assert action["title"] == "Train wordle"
    assert action["payload"]["config_kind"] == "rl"
    assert action["payload"]["config"] == {
        "model": "openai/gpt-oss-20b",
        "env": [{"id": "primeintellect/wordle"}],
        "max_steps": 10,
        "rollouts_per_example": 8,
        "batch_size": 256,
        "sampling": {"max_tokens": 2048},
    }
    model = build_agent_widget_model(
        AgentChatMessage("system", "Action ready", "widget", action),
        tmp_path,
    )
    fields = {field.name: field for field in model.fields}
    build = build_agent_widget_config(model, {})

    selected_model, selected_controls = widget_training_model_option_parts(fields["model"].value)
    assert selected_model == "openai/gpt-oss-20b"
    assert selected_controls == {"reasoning_effort": "medium"}
    assert build.parsed["sampling"] == {
        "max_tokens": 2048,
        "reasoning_effort": "medium",
    }


def test_lab_environment_search_uses_prime_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    class Completed:
        returncode = 0
        stdout = json.dumps(
            {
                "environments": [
                    {
                        "environment": "primeintellect/wordle",
                        "description": "Wordle environment",
                        "action_status": "SUCCESS",
                    }
                ]
            }
        )

    def fake_run(command: list[str], **kwargs: Any) -> Completed:
        calls.append(command)
        assert kwargs["cwd"] == str(tmp_path)
        assert kwargs["timeout"] == 12
        return Completed()

    monkeypatch.setattr("prime_lab_app.app.subprocess.run", fake_run)

    rows = _search_platform_environments("wordle", 5, tmp_path)

    assert rows == [
        {
            "id": "primeintellect/wordle",
            "label": "primeintellect/wordle",
            "description": "Wordle environment",
            "status": "SUCCESS",
            "scope": "mine",
        }
    ]
    assert calls[0] == [
        "prime",
        "env",
        "list",
        "--output",
        "json",
        "--num",
        "5",
        "--sort",
        "updated_at",
        "--order",
        "desc",
        "--search",
        "wordle",
        "--mine",
    ]


def test_lab_mcp_stdio_server_lists_and_forwards_tools(tmp_path: Path) -> None:
    server = LabMcpIpcServer(
        tmp_path,
        lambda tool, arguments: {"ok": True, "tool": tool, "arguments": arguments},
    )
    server.start()
    try:
        request_stream = StringIO(
            "\n".join(
                json.dumps(payload)
                for payload in (
                    {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
                    {"jsonrpc": "2.0", "method": "notifications/initialized"},
                    {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
                    {
                        "jsonrpc": "2.0",
                        "id": 3,
                        "method": "tools/call",
                        "params": {"name": "choose", "arguments": {"title": "Pick"}},
                    },
                )
            )
            + "\n"
        )
        response_stream = StringIO()
        _serve_lab_mcp_stdio(tmp_path, request_stream, response_stream)
    finally:
        server.stop()

    responses = [
        json.loads(line) for line in response_stream.getvalue().splitlines() if line.strip()
    ]
    assert [response["id"] for response in responses] == [1, 2, 3]
    assert responses[0]["result"]["serverInfo"]["name"] == "prime-lab"
    assert any(tool["name"] == "choose" for tool in responses[1]["result"]["tools"])
    tool_result = responses[2]["result"]["structuredContent"]
    assert tool_result == {"ok": True, "tool": "choose", "arguments": {"title": "Pick"}}


def test_lab_mcp_stdio_marks_failed_tool_result_as_error(tmp_path: Path) -> None:
    server = LabMcpIpcServer(
        tmp_path,
        lambda tool, _arguments: {"success": False, "tool": tool, "error": "invalid choice"},
    )
    server.start()
    try:
        request_stream = StringIO(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/call",
                    "params": {"name": "choose", "arguments": {"title": ""}},
                }
            )
            + "\n"
        )
        response_stream = StringIO()
        _serve_lab_mcp_stdio(tmp_path, request_stream, response_stream)
    finally:
        server.stop()

    response = json.loads(response_stream.getvalue())
    assert response["result"]["isError"] is True
    assert response["result"]["structuredContent"] == {
        "success": False,
        "tool": "choose",
        "error": "invalid choice",
    }


def test_agent_native_surface_writers_create_agent_specific_configs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    opencode_path = write_opencode_mcp_config(tmp_path)
    hermes_path = write_hermes_mcp_config(tmp_path)
    amp_path = write_amp_mcp_config(tmp_path)
    droid_path = write_droid_mcp_config(tmp_path)

    opencode_config = json.loads(opencode_path.read_text(encoding="utf-8"))
    assert opencode_config["mcp"]["prime_lab"]["type"] == "local"
    assert opencode_config["mcp"]["prime_lab"]["command"] == [
        sys.executable,
        "-c",
        "from prime_cli.main import run; run()",
        "lab",
        "mcp",
        "--workspace",
        str(tmp_path.resolve()),
    ]
    assert opencode_config["mcp"]["prime_lab"]["enabled"] is True
    assert opencode_config["mcp"]["prime_lab"]["timeout"] == 60000
    hermes_config = hermes_path.read_text(encoding="utf-8")
    assert "mcp_servers" in hermes_config
    assert "prime_lab" in hermes_config
    assert str(tmp_path.resolve()) in hermes_config
    amp_config = json.loads(amp_path.read_text(encoding="utf-8"))
    assert amp_config["prime_lab"]["args"] == [
        "-c",
        "from prime_cli.main import run; run()",
        "lab",
        "mcp",
        "--workspace",
        str(tmp_path.resolve()),
    ]
    assert "mcpServers" not in amp_config
    assert "amp.mcpServers" not in amp_config
    droid_config = json.loads(droid_path.read_text(encoding="utf-8"))
    assert droid_config["mcpServers"]["prime_lab"]["type"] == "stdio"
    assert droid_config["mcpServers"]["prime_lab"]["args"] == [
        "-c",
        "from prime_cli.main import run; run()",
        "lab",
        "mcp",
        "--workspace",
        str(tmp_path.resolve()),
    ]


def test_hermes_config_preserves_existing_yaml_without_pyyaml(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "config.yaml"
    original = "theme: dark\nhooks:\n  enabled: true\n"
    config_path.write_text(original, encoding="utf-8")
    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "yaml":
            raise ModuleNotFoundError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    assert write_hermes_mcp_config(tmp_path, config_path) == config_path
    assert config_path.read_text(encoding="utf-8") == original


def test_simple_yaml_fallback_serializes_nested_values_as_valid_yaml() -> None:
    import yaml

    payload = {
        "mcpServers": {
            "prime_lab": {
                "type": "stdio",
                "command": "/path with spaces/python",
                "args": ["-c", "from prime_cli.main import run; run()", {"nested": "a: b # c"}],
                "disabled": False,
            }
        }
    }

    dumped = _dump_simple_yaml(payload)

    assert yaml.safe_load(dumped) == payload
    assert not dumped.lstrip().startswith("{")


def test_agent_runtime_handles_external_mcp_widget_calls(tmp_path: Path) -> None:
    messages: tuple[Any, ...] = ()
    actions: list[dict[str, Any]] = []

    def on_messages(value: Any) -> None:
        nonlocal messages
        messages = value

    runtime = AgentRuntime(on_messages=on_messages, on_action=actions.append)
    response = runtime.handle_external_lab_tool(
        "choose",
        {"prompt": "Pick env", "options": [{"id": "env", "label": "Env"}]},
    )

    assert '"ok": true' in response["contentItems"][0]["text"]
    assert messages
    assert messages[-1].status == "widget"
    assert messages[-1].metadata["kind"] == "choice_picker"
    assert messages[-1].metadata["payload"]["title"] == "Pick env"
    assert actions[-1]["tool"] == "choose"
    assert actions[-1]["source"] == "native_tool"


def test_agent_runtime_widget_call_closes_active_assistant_turn(tmp_path: Path) -> None:
    messages: tuple[Any, ...] = ()

    def on_messages(value: Any) -> None:
        nonlocal messages
        messages = value

    runtime = AgentRuntime(on_messages=on_messages)
    runtime._workspace = tmp_path
    runtime._messages = [AgentChatMessage("assistant", "Preparing launch.", "streaming")]

    response = runtime.handle_external_lab_tool(
        "choose",
        {
            "title": "Pick a run",
            "candidates": [{"id": "a", "label": "Run A"}],
            "default_id": "a",
        },
    )

    assert response["success"] is True
    assert len(messages) == 2
    assert messages[0] == AgentChatMessage("assistant", "Preparing launch.")
    assert messages[1].status == "widget"
    assert messages[1].metadata["kind"] == "choice_picker"


def test_agent_runtime_supports_cursor_with_workspace_mcp_config(tmp_path: Path) -> None:
    commands: list[list[str]] = []
    messages: tuple[Any, ...] = ()
    existing_config = tmp_path / ".cursor" / "mcp.json"
    existing_config.parent.mkdir(parents=True)
    existing_config.write_text(
        json.dumps({"mcpServers": {"existing": {"command": "echo", "args": ["ok"]}}}),
        encoding="utf-8",
    )

    class FakeCursorProcess:
        def __init__(self) -> None:
            self.stdin = None
            self.stderr = iter(())
            self.stdout = iter(
                [
                    json.dumps(
                        {
                            "type": "assistant",
                            "message": {
                                "role": "assistant",
                                "content": [{"type": "text", "text": "cursor ready"}],
                            },
                            "session_id": "cursor-session",
                            "timestamp_ms": 1,
                        }
                    )
                    + "\n",
                    json.dumps(
                        {
                            "type": "assistant",
                            "message": {
                                "role": "assistant",
                                "content": [{"type": "text", "text": "cursor ready"}],
                            },
                            "session_id": "cursor-session",
                        }
                    )
                    + "\n",
                    json.dumps(
                        {
                            "type": "result",
                            "subtype": "success",
                            "result": "cursor ready",
                            "session_id": "cursor-session",
                        }
                    )
                    + "\n",
                ]
            )

        def poll(self) -> int | None:
            return None

        def wait(self, timeout: float | None = None) -> int:
            return 0

    def fake_popen(command: list[str], **_kwargs: Any) -> FakeCursorProcess:
        commands.append(command)
        return FakeCursorProcess()

    def on_messages(value: Any) -> None:
        nonlocal messages
        messages = value

    runtime = AgentRuntime(on_messages=on_messages, popen_factory=fake_popen)
    runtime.start(tmp_path, "cursor")

    assert runtime.state.status == "connected"
    cursor_config = json.loads(existing_config.read_text(encoding="utf-8"))
    assert cursor_config["mcpServers"]["existing"]["command"] == "echo"
    assert cursor_config["mcpServers"]["prime_lab"]["args"] == [
        "-c",
        "from prime_cli.main import run; run()",
        "lab",
        "mcp",
        "--workspace",
        str(tmp_path.resolve()),
    ]

    runtime.send_prompt("hello")

    assert _wait_for(lambda: messages and messages[-1].status != "streaming")
    assert "--approve-mcps" in commands[-1]
    assert "--force" in commands[-1]
    assert "--workspace" in commands[-1]
    assert str(tmp_path.resolve()) in commands[-1]
    assert "Native Lab tools are available" in commands[-1][-1]
    assert messages
    assert messages[-1].content == "cursor ready"


def test_agent_runtime_supports_opencode_with_workspace_mcp_config(tmp_path: Path) -> None:
    commands: list[list[str]] = []
    messages: tuple[Any, ...] = ()
    (tmp_path / "opencode.json").write_text(
        json.dumps({"theme": "system", "mcp": {"existing": {"type": "local"}}}),
        encoding="utf-8",
    )
    session_new_params: list[dict[str, Any]] = []
    prompt_params: list[dict[str, Any]] = []
    auth_calls: list[dict[str, Any]] = []

    def handler(process: _FakeJsonRpcProcess, message: dict[str, Any]) -> None:
        request_id = message["id"]
        method = message["method"]
        if method == "initialize":
            process.emit(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "authMethods": [
                            {
                                "id": "opencode-login",
                                "name": "Login with opencode",
                            }
                        ]
                    },
                }
            )
            return
        if method == "authenticate":
            auth_calls.append(message["params"])
            process.emit(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": -32603,
                        "message": "Internal error",
                        "data": {"details": "Authentication not implemented"},
                    },
                }
            )
            return
        if method == "session/new":
            params = message["params"]
            session_new_params.append(params)
            process.emit(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {"sessionId": "opencode-session"},
                }
            )
            return
        if method == "session/prompt":
            prompt_params.append(message["params"])
            process.emit(
                {
                    "jsonrpc": "2.0",
                    "method": "session/update",
                    "params": {
                        "sessionId": "opencode-session",
                        "update": {
                            "sessionUpdate": "agent_message_chunk",
                            "content": {"type": "text", "text": "opencode ready"},
                        },
                    },
                }
            )
            process.emit({"jsonrpc": "2.0", "id": request_id, "result": {}})
            return
        process.emit({"jsonrpc": "2.0", "id": request_id, "result": {}})

    def fake_popen(command: list[str], **_kwargs: Any) -> _FakeJsonRpcProcess:
        commands.append(command)
        return _FakeJsonRpcProcess(handler)

    def on_messages(value: Any) -> None:
        nonlocal messages
        messages = value

    runtime = AgentRuntime(on_messages=on_messages, popen_factory=fake_popen)
    runtime.start(tmp_path, "opencode")

    assert _wait_for(lambda: runtime.state.status == "connected")
    opencode_config = json.loads((tmp_path / "opencode.json").read_text(encoding="utf-8"))
    assert opencode_config["theme"] == "system"
    assert opencode_config["mcp"]["prime_lab"]["type"] == "local"
    assert commands[0] == ["opencode", "acp", "--cwd", str(tmp_path.resolve())]
    assert session_new_params[0]["cwd"] == str(tmp_path.resolve())
    assert session_new_params[0]["mcpServers"] == []
    assert auth_calls == []

    runtime.send_prompt("hello")

    assert _wait_for(lambda: messages and messages[-1].status != "streaming")
    assert prompt_params
    assert prompt_params[0]["sessionId"] == "opencode-session"
    assert "Native Lab tools are available" in prompt_params[0]["prompt"][0]["text"]
    assert "do not stop at environment search prose" in prompt_params[0]["prompt"][0]["text"]
    assert messages
    assert messages[-1].content == "opencode ready"
    runtime.stop()


def test_agent_runtime_supports_pi_with_project_extension_tools(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    messages: tuple[Any, ...] = ()
    commands: list[list[str]] = []

    class FakeCliProcess:
        def __init__(self) -> None:
            self.stdin = None
            self.stderr = iter(())
            self.stdout = iter(
                [
                    json.dumps({"type": "message_update", "delta": "pi ready"}) + "\n",
                ]
            )

        def poll(self) -> int | None:
            return None

        def wait(self, timeout: float | None = None) -> int:
            return 0

    def fake_popen(command: list[str], **_kwargs: Any) -> FakeCliProcess:
        commands.append(command)
        return FakeCliProcess()

    def on_messages(value: Any) -> None:
        nonlocal messages
        messages = value

    runtime = AgentRuntime(on_messages=on_messages, popen_factory=fake_popen)
    runtime.start(tmp_path, "pi")

    assert runtime.state.status == "connected"
    assert runtime.state.transport == "resumable-cli"
    extension_path = pi_lab_extension_path(tmp_path)
    assert extension_path.is_file()
    assert "registerTool" in extension_path.read_text(encoding="utf-8")

    runtime.send_prompt("hello")

    assert _wait_for(lambda: messages and messages[-1].status != "streaming")
    assert commands[-1][:4] == ["pi", "--print", "--mode", "json"]
    assert "--no-session" in commands[-1]
    assert "Native Lab tools are available" in commands[-1][-1]
    assert messages
    assert messages[-1].content == "pi ready"


def test_agent_runtime_supports_droid_with_project_mcp_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    messages: tuple[Any, ...] = ()
    commands: list[list[str]] = []

    class FakeCliProcess:
        def __init__(self) -> None:
            self.stdin = None
            self.stderr = iter(())
            self.stdout = iter(
                [
                    json.dumps({"type": "message_update", "delta": "droid ready"}) + "\n",
                ]
            )

        def poll(self) -> int | None:
            return None

        def wait(self, timeout: float | None = None) -> int:
            return 0

    def fake_popen(command: list[str], **_kwargs: Any) -> FakeCliProcess:
        commands.append(command)
        return FakeCliProcess()

    def on_messages(value: Any) -> None:
        nonlocal messages
        messages = value

    runtime = AgentRuntime(on_messages=on_messages, popen_factory=fake_popen)
    runtime.start(tmp_path, "factory-droid")

    assert runtime.state.status == "connected"
    assert runtime.state.transport == "resumable-cli"
    droid_config = json.loads((tmp_path / ".factory" / "mcp.json").read_text(encoding="utf-8"))
    assert droid_config["mcpServers"]["prime_lab"]["type"] == "stdio"

    runtime.send_prompt("hello")

    assert _wait_for(lambda: messages and messages[-1].status != "streaming")
    assert commands[-1][:4] == ["droid", "exec", "--output-format", "stream-json"]
    assert commands[-1][4:6] == ["--cwd", str(tmp_path.resolve())]
    assert "Native Lab tools are available" in commands[-1][-1]
    assert messages[-1].content == "droid ready"


def test_agent_runtime_surfaces_resumable_cli_json_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    messages: tuple[Any, ...] = ()

    class FakeCliProcess:
        def __init__(self) -> None:
            self.stdin = None
            self.stderr = iter(())
            self.stdout = iter(
                [
                    json.dumps(
                        {
                            "type": "error",
                            "source": "agent_loop",
                            "message": "402 Payment Required",
                        }
                    )
                    + "\n",
                    json.dumps({"type": "error", "source": "cli", "message": "Exec failed"}) + "\n",
                ]
            )

        def poll(self) -> int | None:
            return None

        def wait(self, timeout: float | None = None) -> int:
            return 1

    def on_messages(value: Any) -> None:
        nonlocal messages
        messages = value

    def fake_popen(_command: list[str], **_kwargs: Any) -> FakeCliProcess:
        return FakeCliProcess()

    runtime = AgentRuntime(on_messages=on_messages, popen_factory=fake_popen)
    runtime.start(tmp_path, "factory-droid")
    runtime.send_prompt("hello")

    assert _wait_for(lambda: messages and messages[-1].status == "error")
    assert "Factory Droid Agent request failed with 1" in messages[-1].content
    assert "402 Payment Required" in messages[-1].content
    assert "Exec failed" in messages[-1].content


def test_agent_runtime_supports_hermes_with_mcp_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    commands: list[list[str]] = []
    messages: tuple[Any, ...] = ()
    session_new_params: list[dict[str, Any]] = []
    prompt_params: list[dict[str, Any]] = []

    def handler(process: _FakeJsonRpcProcess, message: dict[str, Any]) -> None:
        request_id = message["id"]
        method = message["method"]
        if method == "initialize":
            process.emit({"jsonrpc": "2.0", "id": request_id, "result": {"authMethods": []}})
            return
        if method == "session/new":
            session_new_params.append(message["params"])
            process.emit(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {"sessionId": "hermes-session"},
                }
            )
            return
        if method == "session/prompt":
            prompt_params.append(message["params"])
            process.emit(
                {
                    "jsonrpc": "2.0",
                    "method": "session/update",
                    "params": {
                        "sessionId": "hermes-session",
                        "update": {
                            "sessionUpdate": "agent_message_chunk",
                            "content": {"type": "text", "text": "hermes ready"},
                        },
                    },
                }
            )
            process.emit({"jsonrpc": "2.0", "id": request_id, "result": {}})
            return
        process.emit({"jsonrpc": "2.0", "id": request_id, "result": {}})

    def fake_popen(command: list[str], **_kwargs: Any) -> _FakeJsonRpcProcess:
        commands.append(command)
        return _FakeJsonRpcProcess(handler)

    def on_messages(value: Any) -> None:
        nonlocal messages
        messages = value

    runtime = AgentRuntime(on_messages=on_messages, popen_factory=fake_popen)
    runtime.start(tmp_path, "hermes-agent")

    assert _wait_for(lambda: runtime.state.status == "connected")
    hermes_config = (tmp_path / "home" / ".hermes" / "config.yaml").read_text(encoding="utf-8")
    assert "prime_lab" in hermes_config
    assert str(tmp_path.resolve()) in hermes_config
    assert commands[0] == ["hermes", "acp", "--accept-hooks"]
    assert session_new_params[0]["mcpServers"][0]["name"] == "prime_lab"
    assert session_new_params[0]["mcpServers"][0]["env"] == {}

    runtime.send_prompt("hello")

    assert _wait_for(lambda: messages and messages[-1].status != "streaming")
    assert prompt_params
    assert "Native Lab tools are available" in prompt_params[0]["prompt"][0]["text"]
    assert messages
    assert messages[-1].content == "hermes ready"
    runtime.stop()


def test_agent_runtime_supports_claude_code_with_lab_mcp_config(tmp_path: Path) -> None:
    commands: list[list[str]] = []
    messages: tuple[Any, ...] = ()

    class FakeCliProcess:
        def __init__(self) -> None:
            self.stdin = None
            self.stderr = iter(())
            self.stdout = iter(
                [
                    json.dumps(
                        {
                            "type": "assistant",
                            "message": {
                                "role": "assistant",
                                "content": [{"type": "text", "text": "ready"}],
                            },
                            "session_id": "claude-session",
                        }
                    )
                    + "\n"
                ]
            )

        def poll(self) -> int | None:
            return None

        def wait(self, timeout: float | None = None) -> int:
            return 0

    def fake_popen(command: list[str], **_kwargs: Any) -> FakeCliProcess:
        commands.append(command)
        return FakeCliProcess()

    def on_messages(value: Any) -> None:
        nonlocal messages
        messages = value

    runtime = AgentRuntime(on_messages=on_messages, popen_factory=fake_popen)
    runtime.start(tmp_path, "claude-code")

    assert runtime.state.status == "connected"
    assert runtime.state.agent == "claude"
    assert runtime.state.transport == "resumable-cli"
    config_path = agent_mcp_config_path(tmp_path, "claude")
    assert config_path.exists()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert config["mcpServers"]["prime_lab"]["args"] == [
        "-c",
        "from prime_cli.main import run; run()",
        "lab",
        "mcp",
        "--workspace",
        str(tmp_path.resolve()),
    ]

    runtime.send_prompt("hello")

    assert _wait_for(lambda: messages and messages[-1].status != "streaming")
    assert "--mcp-config" in commands[-1]
    assert str(config_path) in commands[-1]
    assert "Native Lab tools are available" in commands[-1][-1]
    assert messages
    assert messages[-1].content == "ready"


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


def test_agent_runtime_exposes_codex_lab_widget_tools(tmp_path: Path) -> None:
    messages: tuple[Any, ...] = ()
    actions: list[dict[str, Any]] = []
    thread_start_params: dict[str, Any] = {}
    tool_responses: list[dict[str, Any]] = []
    process_holder: list[_FakeJsonRpcProcess] = []
    tool_response_seen = threading.Event()

    def handler(process: _FakeJsonRpcProcess, message: dict[str, Any]) -> None:
        request_id = message.get("id")
        method = message.get("method")
        if method == "initialize":
            process.emit({"jsonrpc": "2.0", "id": request_id, "result": {}})
        elif method == "thread/start":
            params = message.get("params")
            if isinstance(params, dict):
                thread_start_params.update(params)
            process.emit(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {"thread": {"id": "thread-1"}},
                }
            )
        elif method is None and request_id == 77:
            result = message.get("result")
            if isinstance(result, dict):
                tool_responses.append(result)
            tool_response_seen.set()
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
        process_holder.append(process)
        return process

    def on_messages(value: Any) -> None:
        nonlocal messages
        messages = value

    runtime = AgentRuntime(
        on_messages=on_messages,
        on_action=actions.append,
        popen_factory=fake_popen,
    )
    runtime.start(tmp_path, "codex")

    assert _wait_for(lambda: runtime.state.status == "connected")
    assert "edit_config" in str(thread_start_params.get("developerInstructions"))
    assert "do not stop after searching" in str(thread_start_params.get("developerInstructions"))
    dynamic_tools = thread_start_params.get("dynamicTools")
    assert isinstance(dynamic_tools, list)
    tool_names = {tool["name"] for tool in dynamic_tools if isinstance(tool, dict)}
    assert {
        "choose",
        "edit_config",
        "preview_action",
        "launch_run",
        "show_patch",
        "inspect_rollouts",
    }.issubset(tool_names)
    assert all(tool["namespace"] == "lab" for tool in dynamic_tools if isinstance(tool, dict))

    process_holder[0].emit(
        {
            "jsonrpc": "2.0",
            "id": 77,
            "method": "item/tool/call",
            "params": {
                "namespace": "lab",
                "tool": "choose",
                "callId": "widget-1",
                "arguments": {
                    "title": "Pick a config",
                    "candidates": [{"id": "a"}, {"id": "b"}],
                },
            },
        }
    )

    assert tool_response_seen.wait(timeout=2)
    assert tool_responses[-1]["success"] is True
    assert _wait_for(lambda: messages and messages[-1].status == "widget")
    latest_message = _last_message(messages)
    assert "Pick a config" in latest_message.content
    assert actions[-1]["type"] == "widget_requested"
    assert actions[-1]["tool"] == "choose"
    assert actions[-1]["kind"] == "choice_picker"
    assert actions[-1]["title"] == "Pick a config"
    runtime.stop()


def test_agent_chat_transcript_renders_assistant_markdown() -> None:
    transcript = _chat_transcript(
        (
            AgentChatMessage("user", "show me code"),
            AgentChatMessage(
                "assistant",
                "Here is a helper:\n\n```python\nprint('hello')\n```\n\n- done",
            ),
        ),
        AgentConnectionState(status="connected", label="Codex"),
    )

    rendered = _render_renderable(transcript)

    assert "Assistant" not in rendered
    assert "User" not in rendered
    assert "show me code" in rendered
    assert "print" in rendered
    assert "hello" in rendered
    assert "```" not in rendered


def test_agent_thinking_turn_animates_empty_streaming_response() -> None:
    first = _render_renderable(_agent_thinking_turn(0))
    second = _render_renderable(_agent_thinking_turn(1))

    assert "Thinking" in first
    assert "Thinking" in second
    assert first != second


def test_agent_chat_transcript_renders_lab_widget_card() -> None:
    transcript = _chat_transcript(
        (
            AgentChatMessage(
                "system",
                "Widget requested",
                "widget",
                {
                    "type": "widget_requested",
                    "kind": "run_launcher",
                    "title": "Eval: alphabet-sort",
                    "payload": {
                        "kind": "run_launcher",
                        "title": "Eval: alphabet-sort",
                        "config_kind": "eval",
                        "config_path": "configs/eval/alphabet-sort.toml",
                    },
                },
            ),
        ),
        AgentConnectionState(status="connected", label="Cursor"),
    )

    rendered = _render_renderable(transcript)

    assert "Lab widget" not in rendered
    assert "alphabet-sort" in rendered
    assert "Eval: alphabet-sort" not in rendered
    assert "Run launcher" not in rendered
    assert "configs/eval/alphabet-sort.toml" in rendered


def test_agent_chat_parts_parse_lab_references() -> None:
    parts = message_parts(
        AgentChatMessage(
            "user",
            "Compare @env:primeintellect/gsm8k with @config:configs/eval/gsm8k.toml",
        )
    )

    references = [part for part in parts if isinstance(part, ReferencePart)]

    assert [(part.ref_type, part.ref_id) for part in references] == [
        ("environment", "primeintellect/gsm8k"),
        ("config", "configs/eval/gsm8k.toml"),
    ]


def test_agent_stream_delta_deduplicates_idless_full_messages() -> None:
    seen_messages: dict[str, str] = {}

    first = _extract_stream_delta(
        {"message": {"role": "assistant", "content": "Checking config parsing."}},
        seen_messages,
    )
    duplicate = _extract_stream_delta(
        {"message": {"role": "assistant", "content": "Checking config parsing."}},
        seen_messages,
    )
    next_message = _extract_stream_delta(
        {"message": {"role": "assistant", "content": "Writing config."}},
        seen_messages,
    )

    assert first == "Checking config parsing."
    assert duplicate == ""
    assert next_message == "Writing config."


def test_agent_stream_delta_handles_claude_code_partial_then_final() -> None:
    seen_messages: dict[str, str] = {}

    first = _extract_stream_delta(
        {
            "type": "stream_event",
            "event": {
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": "LAB_AGENT_"},
            },
            "session_id": "claude-session",
        },
        seen_messages,
    )
    second = _extract_stream_delta(
        {
            "type": "stream_event",
            "event": {
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": "SMOKE"},
            },
            "session_id": "claude-session",
        },
        seen_messages,
    )
    final = _extract_stream_delta(
        {
            "type": "assistant",
            "message": {
                "id": "msg-1",
                "role": "assistant",
                "content": [{"type": "text", "text": "LAB_AGENT_SMOKE"}],
            },
            "session_id": "claude-session",
        },
        seen_messages,
    )

    assert first == "LAB_AGENT_"
    assert second == "SMOKE"
    assert final == ""


def test_agent_stream_delta_handles_cursor_chunks_then_final_snapshot() -> None:
    seen_messages: dict[str, str] = {}
    chunks = [
        {"text": "LAB_AGENT_S", "timestamp_ms": 1},
        {"text": "MO", "timestamp_ms": 2},
        {"text": "KE", "timestamp_ms": 3},
        {"text": "_CURSOR", "timestamp_ms": 4},
    ]

    deltas = [
        _extract_stream_delta(
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": chunk["text"]}],
                },
                "timestamp_ms": chunk["timestamp_ms"],
                "session_id": "cursor-session",
            },
            seen_messages,
        )
        for chunk in chunks
    ]
    final = _extract_stream_delta(
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "LAB_AGENT_SMOKE_CURSOR"}],
            },
            "session_id": "cursor-session",
        },
        seen_messages,
    )

    assert "".join(deltas) == "LAB_AGENT_SMOKE_CURSOR"
    assert final == ""


def test_agent_stream_delta_ignores_cursor_thinking_events() -> None:
    seen_messages: dict[str, str] = {}

    thought = _extract_stream_delta(
        {
            "type": "thinking",
            "subtype": "delta",
            "text": "Planning the eval answer.",
            "timestamp_ms": 1,
        },
        seen_messages,
    )
    assistant = _extract_stream_delta(
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "Here are eval options."}],
            },
            "timestamp_ms": 2,
        },
        seen_messages,
    )

    assert thought == ""
    assert assistant == "Here are eval options."


def test_agent_stream_delta_ignores_lab_widget_ack_payload() -> None:
    text = _extract_stream_delta(
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            {
                                "ok": True,
                                "tool": "choose",
                                "kind": "choice_picker",
                                "title": "Smoke test",
                                "message": "displayed",
                            }
                        ),
                    }
                ],
            },
        },
        {},
    )

    assert text == ""


def test_agent_stream_delta_ignores_final_snapshot_after_streamed_chunks() -> None:
    seen_messages: dict[str, str] = {}

    status = _extract_stream_delta(
        {
            "type": "assistant",
            "message": {"role": "assistant", "content": [{"type": "text", "text": "Checking."}]},
            "timestamp_ms": 1,
        },
        seen_messages,
    )
    answer = _extract_stream_delta(
        {
            "type": "assistant",
            "message": {"role": "assistant", "content": [{"type": "text", "text": "\n\nAnswer."}]},
            "timestamp_ms": 2,
        },
        seen_messages,
    )
    final = _extract_stream_delta(
        {
            "type": "assistant",
            "message": {"role": "assistant", "content": [{"type": "text", "text": "Answer."}]},
        },
        seen_messages,
    )

    assert status == "Checking."
    assert answer == "\n\nAnswer."
    assert final == ""


def test_agent_stream_merge_ignores_duplicate_cursor_snapshots() -> None:
    content = _merge_stream_text("Checking config parsing.", "Checking config parsing.")
    content = _merge_stream_text(content, "Checking config parsing.\nWriting config.")
    content = _merge_stream_text(content, "\nWriting config.")

    assert content == "Checking config parsing.\nWriting config."


def test_agent_stream_merge_ignores_replayed_prior_blocks() -> None:
    content = _merge_stream_text(
        "Exploring eval setup.\n\nChecking environment defaults.",
        "Exploring eval setup.",
    )
    content = _merge_stream_text(content, "\n\nChecking environment defaults.")

    assert content == "Exploring eval setup.\n\nChecking environment defaults."


def test_agent_stream_finalize_collapses_adjacent_duplicate_blocks() -> None:
    content = "Exploring configs.\n\nExploring configs.\n\nWriting config."

    assert _dedupe_streamed_text(content) == "Exploring configs.\n\nWriting config."


def test_agent_runtime_collapses_duplicate_blocks_while_streaming() -> None:
    messages: list[tuple[AgentChatMessage, ...]] = []
    runtime = AgentRuntime(on_messages=messages.append)

    runtime._append_streaming_assistant_text("Exploring configs.")
    runtime._append_streaming_assistant_text("\n\nExploring configs.")
    runtime._append_streaming_assistant_text("\n\nWriting config.")

    assert messages[-1][-1] == AgentChatMessage(
        "assistant",
        "Exploring configs.\n\nWriting config.",
        "streaming",
    )


def test_agent_runtime_strips_ansi_escape_codes_from_stream_events() -> None:
    text = _extract_stream_delta(
        {
            "type": "assistant",
            "message": {
                "id": "message-1",
                "content": [{"type": "text", "text": "\x1b[31mLAB\x1b[0m"}],
            },
        },
        {},
    )

    assert text == "LAB"


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
async def test_prime_lab_app_mounts(tmp_path: Path) -> None:
    snapshot = make_source().load(LabLoadOptions(limit=10, workspace=tmp_path))
    app = PrimeLabView(lambda: snapshot)

    async with app.run_test() as pilot:
        await pilot.pause()
        assert app._snapshot == snapshot


@pytest.mark.asyncio
async def test_prime_lab_app_home_launch_panel_uses_home_state(tmp_path: Path) -> None:
    (tmp_path / ".prime").mkdir()
    (tmp_path / ".prime" / "lab.json").write_text(
        '{"choices": {"primary_agent": "claude"}}',
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
        launch_text = _render_renderable(panel.render())
        assert "✓  ·" not in launch_text


@pytest.mark.asyncio
async def test_prime_lab_app_home_launch_panel_shows_for_loaded_workspace(
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
        hotkeys = str(screen.query_one("#launch-hotkeys").render())
        assert "agent" in hotkeys
        assert "refresh" not in hotkeys
        assert app.check_action("refresh", ()) is False
        assert app.check_action("search", ()) is False
        assert app.check_action("load_more_rows", ()) is False


@pytest.mark.asyncio
async def test_prime_lab_app_w_reopens_launch_screen(tmp_path: Path) -> None:
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
async def test_prime_lab_app_s_opens_workspace_settings(tmp_path: Path) -> None:
    snapshot = make_source().load(LabLoadOptions(limit=10, workspace=tmp_path))
    app = PrimeLabView(lambda: snapshot, initial_loader=lambda: snapshot)

    async with app.run_test(size=(140, 44)) as pilot:
        await pilot.pause()
        assert isinstance(app.screen, LaunchScreen)

        await pilot.press("s")
        await pilot.pause()

        assert not isinstance(app.screen, LaunchScreen)
        assert app._active_section_key == "workspace"


@pytest.mark.asyncio
async def test_prime_lab_app_auto_starts_configured_agent(tmp_path: Path) -> None:
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
        assert app._agent_state.transport == "resumable-cli"
        status_text = _render_renderable(app._statusbar_text())
        assert "✓ Claude" in status_text
        assert "Claude connected" not in status_text


@pytest.mark.asyncio
async def test_agent_chat_uses_centered_stage_without_sidebar(tmp_path: Path) -> None:
    (tmp_path / ".prime").mkdir()
    (tmp_path / ".prime" / "lab.json").write_text(
        '{"choices": {"primary_agent": "claude"}}',
        encoding="utf-8",
    )
    base_snapshot = make_source().load(LabLoadOptions(limit=10, workspace=tmp_path))
    snapshot = replace(
        base_snapshot,
        warnings=("Training runs unavailable: retry later",),
    )
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
        note = _render_renderable(app.screen.query_one("#agent-experimental-note", Static).content)
        assert "Agent mode is experimental" in note
        prompt = app.screen.query_one("#agent-prompt", AgentPrompt)
        assert prompt.placeholder == "Message Claude, Enter to send  •  /  ?  @"
        warning_popover = app.screen.query_one("#agent-warning-popover")
        assert warning_popover.display is False
        status_text = _render_renderable(app.screen.query_one("#agent-statusbar", Static).content)
        assert "1 warning" in status_text
        assert "Welcome" not in status_text

        await pilot.hover("#agent-statusbar")
        await pilot.pause()

        assert warning_popover.display is True
        rendered = _render_renderable(
            app.screen.query_one("#agent-warning-popover-body", Static).render()
        )
        assert "Warnings" in rendered
        assert "Training runs unavailable" in rendered

        await pilot.hover("#agent-prompt")
        await pilot.pause()

        assert warning_popover.display is False


def test_launch_backdrop_renders_stable_terminal_field() -> None:
    rows = LaunchBackdrop(frame=3).render_parts(72, 14)

    assert len(rows) == 14
    assert all(sum(len(value) for value, _ in row) == 72 for row in rows)
    plain = "\n".join("".join(value for value, _ in row) for row in rows)
    assert any(char in plain for char in {"┃", "╎"})
    assert any(char in plain for char in {"•", "◆"})


@pytest.mark.asyncio
async def test_prime_lab_app_home_launch_panel_dismisses_into_workspace(
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
        assert str(app.query_one("#section-title", Label).render()) == "Settings"
        workspace_path = _render_renderable(app.query_one("#workspace-path").render())
        assert "✓ research" in workspace_path
        assert "production" in workspace_path
        assert compact_path(tmp_path) in workspace_path
        status_text = _render_renderable(app._statusbar_text())
        assert "research" in status_text
        assert compact_path(tmp_path) in status_text
        section_tree = app.query_one("#section-tree", Tree)
        section_labels = [
            _render_renderable(node.label).strip() for node in section_tree.root.children
        ]
        assert section_labels == ["Environments", "Training", "Evaluations", "Settings"]
        assert app.query_one("#item-list", OptionList).display is True
        assert app.query_one("#topbar").display is True
        assert app.query_one("#nav-pane").display is True
        assert app.query_one("#section-title").display is True
        assert app.query_one("#section-subtitle").display is True


@pytest.mark.asyncio
async def test_prime_lab_app_settings_is_reachable_from_section_nav(tmp_path: Path) -> None:
    snapshot = make_source().load(LabLoadOptions(limit=10, workspace=tmp_path))
    app = PrimeLabView(lambda: snapshot, initial_loader=lambda: snapshot)

    async with app.run_test(size=(140, 44)) as pilot:
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        app._active_section_key = "environments"
        app._render_tree()
        app._render_active_section()
        await pilot.pause()

        section_tree = app.query_one("#section-tree", Tree)
        settings_node = next(
            node
            for node in section_tree.root.children
            if _render_renderable(node.label).strip() == "Settings"
        )
        section_tree.move_cursor(settings_node)
        await pilot.press("enter")
        await pilot.pause()

        assert app._active_section_key == "workspace"
        assert str(app.query_one("#section-title", Label).render()) == "Settings"
        assert app.query_one("#inspector-pane").display is True
        assert app.query_one("#statusbar").display is True
        assert app.query("Footer").first().display is True


@pytest.mark.asyncio
async def test_prime_lab_app_settings_subcolumn_vertical_keys_enter_rows(
    tmp_path: Path,
) -> None:
    snapshot = make_source().load(LabLoadOptions(limit=10, workspace=tmp_path))
    app = PrimeLabView(lambda: snapshot, initial_loader=lambda: snapshot)

    async with app.run_test(size=(140, 44)) as pilot:
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

        toggle = app.query_one("#home-toggle", HomeGroupToggle)
        toggle.focus()
        active_group = app._home_group

        await pilot.press("up")
        await pilot.pause()

        assert app.focused is toggle
        assert app._home_group == active_group

        await pilot.press("down")
        await pilot.pause()

        assert isinstance(app.focused, LabOptionList)
        assert app._home_group == active_group


@pytest.mark.asyncio
async def test_prime_lab_app_warning_status_opens_warning_viewer(tmp_path: Path) -> None:
    base_snapshot = make_source().load(LabLoadOptions(limit=10, workspace=tmp_path))
    snapshot = replace(
        base_snapshot,
        warnings=("Training runs unavailable: retry later",),
    )
    app = PrimeLabView(lambda: snapshot, initial_loader=lambda: snapshot)

    async with app.run_test(size=(140, 44)) as pilot:
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

        warning_viewer = app.query_one("#warning-viewer")
        assert warning_viewer.display is False
        status_text = _render_renderable(app._statusbar_text())
        assert "1 warning" in status_text
        assert "1 warnings" not in status_text
        await pilot.hover("#statusbar")
        await pilot.pause()

        assert warning_viewer.display is True
        await pilot.hover("#item-list")
        await pilot.pause()

        assert warning_viewer.display is False
        await pilot.click("#statusbar")
        await pilot.pause()

        assert warning_viewer.display is True
        rendered = _render_renderable(app.query_one("#warning-viewer-body", Static).render())
        assert "Warnings" in rendered
        assert "Training runs unavailable" in rendered


def test_warning_popover_text_formats_multiline_validation_errors() -> None:
    text = warning_popover_text(
        (
            "Evaluations unavailable: Validation failed:\n"
            "- query.limit: Input should be less than or equal to 100",
        )
    )
    rendered = _render_renderable(text)

    assert "Warnings" in rendered
    assert "1. Evaluations unavailable: Validation failed:" in rendered
    assert "- query.limit: Input should be less than or equal to 100" in rendered


@pytest.mark.asyncio
async def test_prime_lab_app_launch_grid_opens_quickstart_flows(tmp_path: Path) -> None:
    snapshot = make_source().load(LabLoadOptions(limit=10, workspace=tmp_path))
    app = PrimeLabView(lambda: snapshot, initial_loader=lambda: snapshot)

    async with app.run_test(size=(140, 44)) as pilot:
        await pilot.pause()
        await pilot.click("#launch-evaluate")
        await pilot.pause()

        assert isinstance(app.screen, ConfigRunScreen)
        assert app.screen._config_kind == "eval"


@pytest.mark.asyncio
async def test_prime_lab_app_launch_grid_build_opens_agent_templates(tmp_path: Path) -> None:
    (tmp_path / ".prime").mkdir()
    (tmp_path / ".prime" / "lab.json").write_text(
        '{"choices": {"primary_agent": "claude"}}',
        encoding="utf-8",
    )
    snapshot = make_source().load(LabLoadOptions(limit=10, workspace=tmp_path))
    app = PrimeLabView(lambda: snapshot, initial_loader=lambda: snapshot)

    async with app.run_test(size=(140, 44)) as pilot:
        await pilot.pause()
        button = app.screen.query_one("#launch-agent", Button)
        assert "Build with Claude" in str(button.label)

        await pilot.click("#launch-agent")
        await pilot.pause()

        assert isinstance(app.screen, AgentChatScreen)
        assert not app.screen.query("#agent-send")
        assert not app.screen.query("#agent-select")
        prompt = app.screen.query_one("#agent-prompt", AgentPrompt)
        prompt.load_text("?")
        await pilot.pause()
        menu = _render_renderable(app.screen.query_one("#agent-command-menu", Static).content)
        assert "Build env" in menu

        await pilot.press("enter")
        await pilot.pause()

        assert "Help me build a new verifiers environment" in prompt.text


@pytest.mark.asyncio
async def test_prime_lab_app_agent_reference_menu_inserts_lab_reference(tmp_path: Path) -> None:
    (tmp_path / ".prime").mkdir()
    (tmp_path / ".prime" / "lab.json").write_text(
        '{"choices": {"primary_agent": "claude"}}',
        encoding="utf-8",
    )
    snapshot = make_source().load(LabLoadOptions(limit=10, workspace=tmp_path))
    app = PrimeLabView(lambda: snapshot, initial_loader=lambda: snapshot)

    async with app.run_test(size=(140, 44)) as pilot:
        await pilot.pause()
        await pilot.click("#launch-agent")
        await pilot.pause()

        prompt = app.screen.query_one("#agent-prompt", AgentPrompt)
        prompt.load_text("@")
        await pilot.pause()

        menu = _render_renderable(app.screen.query_one("#agent-command-menu", Static).content)
        assert "@env:research/private-env" in menu

        await pilot.press("enter")
        await pilot.pause()

        assert prompt.text == "@env:research/private-env "


@pytest.mark.asyncio
async def test_prime_lab_app_agent_prompt_history_uses_empty_up_down(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / ".prime").mkdir()
    (tmp_path / ".prime" / "lab.json").write_text(
        '{"choices": {"primary_agent": "claude"}}',
        encoding="utf-8",
    )
    snapshot = make_source().load(LabLoadOptions(limit=10, workspace=tmp_path))
    app = PrimeLabView(lambda: snapshot, initial_loader=lambda: snapshot)

    async with app.run_test(size=(140, 44)) as pilot:
        await pilot.pause()
        await pilot.click("#launch-agent")
        await pilot.pause()

        assert isinstance(app.screen, AgentChatScreen)
        app._agent_messages = (
            AgentChatMessage("user", "first prompt"),
            AgentChatMessage("assistant", "first response"),
            AgentChatMessage("user", "second prompt"),
        )
        app.screen._refresh_runtime_view()
        prompt = app.screen.query_one("#agent-prompt", AgentPrompt)
        prompt.clear_prompt()
        prompt.focus()
        await pilot.pause()

        await pilot.press("up")
        await pilot.pause()
        assert prompt.text == "second prompt"

        prompt.clear_prompt()
        await pilot.press("up")
        await pilot.press("up")
        await pilot.pause()
        assert prompt.text == "first prompt"

        await pilot.press("down")
        await pilot.pause()
        assert prompt.text == "second prompt"

        await pilot.press("down")
        await pilot.pause()
        assert prompt.text == ""


@pytest.mark.asyncio
async def test_prime_lab_app_agent_prompt_history_uses_global_log_for_new_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / ".prime").mkdir()
    (tmp_path / ".prime" / "lab.json").write_text(
        '{"choices": {"primary_agent": "claude"}}',
        encoding="utf-8",
    )
    append_agent_prompt_history(tmp_path, "cursor", "remembered prompt")
    snapshot = make_source().load(LabLoadOptions(limit=10, workspace=tmp_path))
    app = PrimeLabView(lambda: snapshot, initial_loader=lambda: snapshot)

    async with app.run_test(size=(140, 44)) as pilot:
        await pilot.pause()
        await pilot.click("#launch-agent")
        await pilot.pause()

        assert isinstance(app.screen, AgentChatScreen)
        app._agent_messages = ()
        app.screen._refresh_runtime_view()
        prompt = app.screen.query_one("#agent-prompt", AgentPrompt)
        prompt.clear_prompt()
        prompt.focus()
        await pilot.pause()

        await pilot.press("up")
        await pilot.pause()

        assert prompt.text == "remembered prompt"


@pytest.mark.asyncio
async def test_prime_lab_app_agent_slash_menu_uses_arrow_selection(tmp_path: Path) -> None:
    (tmp_path / ".prime").mkdir()
    (tmp_path / ".prime" / "lab.json").write_text(
        '{"choices": {"primary_agent": "claude"}}',
        encoding="utf-8",
    )
    snapshot = make_source().load(LabLoadOptions(limit=10, workspace=tmp_path))
    app = PrimeLabView(lambda: snapshot, initial_loader=lambda: snapshot)

    async with app.run_test(size=(140, 44)) as pilot:
        await pilot.pause()
        await pilot.click("#launch-agent")
        await pilot.pause()

        prompt = app.screen.query_one("#agent-prompt", AgentPrompt)
        prompt.load_text("/agent")
        await pilot.pause()

        await pilot.press("down")
        await pilot.press("down")
        await pilot.press("enter")
        await pilot.pause()

        assert app._agent_state.agent == "codex"


@pytest.mark.asyncio
async def test_prime_lab_app_agent_prompt_expands_and_preserves_large_paste(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / ".prime").mkdir()
    (tmp_path / ".prime" / "lab.json").write_text(
        '{"choices": {"primary_agent": "claude"}}',
        encoding="utf-8",
    )
    snapshot = make_source().load(LabLoadOptions(limit=10, workspace=tmp_path))
    app = PrimeLabView(lambda: snapshot, initial_loader=lambda: snapshot)
    sent: list[str] = []
    quit_calls = 0

    def fake_quit() -> None:
        nonlocal quit_calls
        quit_calls += 1

    app._send_agent_prompt = sent.append  # type: ignore[method-assign]
    app.action_quit = fake_quit  # type: ignore[method-assign]

    async with app.run_test(size=(140, 44)) as pilot:
        await pilot.pause()
        await pilot.click("#launch-agent")
        await pilot.pause()

        prompt = app.screen.query_one("#agent-prompt", AgentPrompt)
        assert prompt.styles.height.value == 1

        prompt.load_text("line one")
        prompt.action_insert_newline()
        await pilot.pause()
        assert prompt.styles.height.value == 2

        pasted = "\n".join(f"line {index}" for index in range(10))
        prompt.apply_large_paste(pasted)
        await pilot.pause()

        assert prompt.text == "[10 lines pasted]"
        assert prompt.submitted_text == pasted
        assert prompt.has_class("large-paste")

        await pilot.press("enter")
        await pilot.pause()

        assert sent == [pasted]
        assert prompt.text == ""
        assert prompt.styles.height.value == 1

        prompt.load_text("clear me")
        prompt.focus()
        await pilot.press("ctrl+c")
        await pilot.pause()

        assert isinstance(app.screen, AgentChatScreen)
        assert prompt.text == "clear me"
        assert quit_calls == 1

        prompt.clear_prompt()
        prompt.focus()
        await pilot.press("b")
        await pilot.pause()

        assert isinstance(app.screen, AgentChatScreen)
        assert prompt.text == "b"

        prompt.clear_prompt()
        await pilot.press("ctrl+c")
        await pilot.pause()

        assert quit_calls == 2


@pytest.mark.asyncio
async def test_prime_lab_app_switching_agent_uses_separate_cached_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / ".prime").mkdir()
    (tmp_path / ".prime" / "lab.json").write_text(
        '{"choices": {"primary_agent": "claude"}}',
        encoding="utf-8",
    )
    snapshot = make_source().load(LabLoadOptions(limit=10, workspace=tmp_path))
    app = PrimeLabView(lambda: snapshot, initial_loader=lambda: snapshot)

    async with app.run_test(size=(140, 44)) as pilot:
        await pilot.pause()
        await pilot.click("#launch-agent")
        await pilot.pause()

        claude_session = app._agent_session
        assert claude_session is not None
        assert claude_session.agent == "claude"

        prompt = app.screen.query_one("#agent-prompt", AgentPrompt)
        prompt.load_text("/agent cursor")
        await pilot.press("enter")
        await pilot.pause()

        cursor_session = app._agent_session
        assert cursor_session is not None
        assert cursor_session.agent == "cursor"
        assert cursor_session.path != claude_session.path
        assert latest_agent_session(tmp_path, "claude") == claude_session
        assert latest_agent_session(tmp_path, "cursor") == cursor_session
        assert app._snapshot is not None
        assert configured_workspace_agent(app._snapshot) == "cursor"
        metadata = json.loads((tmp_path / ".prime" / "lab.json").read_text(encoding="utf-8"))
        assert metadata["choices"]["primary_agent"] == "cursor"

        app._set_snapshot(snapshot)
        await pilot.pause()

        assert app._agent_state.agent == "cursor"
        assert app._agent_state.status == "connected"


@pytest.mark.asyncio
async def test_prime_lab_app_agent_clear_command_starts_fresh_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / ".prime").mkdir()
    (tmp_path / ".prime" / "lab.json").write_text(
        '{"choices": {"primary_agent": "claude"}}',
        encoding="utf-8",
    )
    snapshot = make_source().load(LabLoadOptions(limit=10, workspace=tmp_path))
    app = PrimeLabView(lambda: snapshot, initial_loader=lambda: snapshot)

    async with app.run_test(size=(140, 44)) as pilot:
        await pilot.pause()
        await pilot.click("#launch-agent")
        await pilot.pause()

        assert isinstance(app.screen, AgentChatScreen)
        old_session = app._agent_session
        assert old_session is not None
        app._agent_messages = (AgentChatMessage(role="user", content="old prompt"),)
        app.screen._refresh_runtime_view()
        await pilot.pause()

        prompt = app.screen.query_one("#agent-prompt", AgentPrompt)
        prompt.load_text("/")
        await pilot.pause()
        menu = _render_renderable(app.screen.query_one("#agent-command-menu", Static).content)
        assert "/clear" in menu
        assert "/new" not in menu

        prompt.load_text("/clear")
        await pilot.press("enter")
        await pilot.pause()

        new_session = app._agent_session
        assert new_session is not None
        assert new_session.path != old_session.path
        assert app._agent_messages == ()
        assert latest_agent_session(tmp_path, "claude") == new_session
        assert app.screen.query_one("#agent-atmosphere").display is True


@pytest.mark.asyncio
async def test_prime_lab_app_agent_diagnose_command_sends_native_tool_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / ".prime").mkdir()
    (tmp_path / ".prime" / "lab.json").write_text(
        '{"choices": {"primary_agent": "claude"}}',
        encoding="utf-8",
    )
    snapshot = make_source().load(LabLoadOptions(limit=10, workspace=tmp_path))
    app = PrimeLabView(lambda: snapshot, initial_loader=lambda: snapshot)
    sent: list[str] = []
    app._send_agent_prompt = sent.append  # type: ignore[method-assign]

    async with app.run_test(size=(140, 44)) as pilot:
        await pilot.pause()
        await pilot.click("#launch-agent")
        await pilot.pause()

        assert isinstance(app.screen, AgentChatScreen)
        prompt = app.screen.query_one("#agent-prompt", AgentPrompt)
        prompt.load_text("/diagnose")
        await pilot.press("enter")
        await pilot.pause()

        assert sent == [lab_widget_diagnostic_prompt()]
        assert prompt.text == ""
        assert app._agent_session is not None
        actions_path = app._agent_session.path / "actions.jsonl"
        actions = [
            json.loads(line)
            for line in actions_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert actions[-1]["type"] == "agent_tool_diagnostic_started"
        assert actions[-1]["agent"] == "claude"


@pytest.mark.asyncio
async def test_prime_lab_app_launch_grid_agent_configures_when_missing(tmp_path: Path) -> None:
    snapshot = make_source().load(LabLoadOptions(limit=10, workspace=tmp_path))
    app = PrimeLabView(lambda: snapshot, initial_loader=lambda: snapshot)

    async with app.run_test(size=(140, 44)) as pilot:
        await pilot.pause()
        button = app.screen.query_one("#launch-agent", Button)
        assert "Configure Agent" in str(button.label)

        await pilot.click("#launch-agent")
        await pilot.pause()

        assert isinstance(app.screen, SetupScreen)


@pytest.mark.asyncio
async def test_prime_lab_app_chat_hotkey_opens_agent_chat(tmp_path: Path) -> None:
    (tmp_path / ".prime").mkdir()
    (tmp_path / ".prime" / "lab.json").write_text(
        '{"choices": {"primary_agent": "claude"}}',
        encoding="utf-8",
    )
    snapshot = make_source().load(LabLoadOptions(limit=10, workspace=tmp_path))
    app = PrimeLabView(lambda: snapshot, initial_loader=lambda: snapshot)

    async with app.run_test(size=(140, 44)) as pilot:
        await pilot.pause()
        await pilot.press("c")
        await pilot.pause()

        assert isinstance(app.screen, AgentChatScreen)
        header = _render_renderable(app.screen.query_one("#agent-header", Static).content)
        assert "Agent" in header
        assert str(tmp_path) not in header


@pytest.mark.asyncio
async def test_prime_lab_app_chat_mounts_lab_widget_cards(tmp_path: Path) -> None:
    (tmp_path / ".prime").mkdir()
    (tmp_path / ".prime" / "lab.json").write_text(
        '{"choices": {"primary_agent": "cursor"}}',
        encoding="utf-8",
    )
    snapshot = make_source().load(LabLoadOptions(limit=10, workspace=tmp_path))
    app = PrimeLabView(lambda: snapshot, initial_loader=lambda: snapshot)
    widget_messages = (
        AgentChatMessage(
            "system",
            "Widget requested",
            "widget",
            {
                "type": "widget_requested",
                "kind": "run_launcher",
                "title": "Eval: alphabet-sort",
                "payload": {
                    "kind": "run_launcher",
                    "title": "Eval: alphabet-sort",
                    "config_kind": "eval",
                    "config_path": "configs/eval/alphabet-sort.toml",
                },
            },
        ),
    )

    async with app.run_test(size=(140, 44)) as pilot:
        await pilot.pause()
        await pilot.press("c")
        await pilot.pause()

        assert isinstance(app.screen, AgentChatScreen)
        app._set_agent_messages(widget_messages)
        app.screen._refresh_runtime_view()
        await pilot.pause()

        cards = list(app.screen.query(AgentWidgetCard))
        assert cards
        card_text = _render_renderable(cards[0].query(Static).first().content)
        assert "Lab widget" not in card_text
        assert "alphabet-sort" in card_text
        assert "Eval:" not in card_text
        assert "Evaluation" not in card_text
        assert "Command" not in card_text
        input_values = {
            input_widget.name: input_widget.value for input_widget in cards[0].query(ClearableInput)
        }
        assert input_values["envs"] == "primeintellect/alphabet-sort"
        assert input_values["num_examples"] == "50"
        assert input_values["rollouts_per_example"] == "3"
        assert "max_concurrent" in input_values
        assert input_values["max_concurrent"] == "auto"
        assert input_values["model"] == ""
        assert not list(cards[0].query(Select))
        button_labels = {str(button.label) for button in app.screen.query(Button)}
        assert "Launch" in button_labels
        assert "Stop" in button_labels


@pytest.mark.asyncio
async def test_prime_lab_app_ctrl_c_quits_from_agent_chat_screen(
    tmp_path: Path,
) -> None:
    (tmp_path / ".prime").mkdir()
    (tmp_path / ".prime" / "lab.json").write_text(
        '{"choices": {"primary_agent": "cursor"}}',
        encoding="utf-8",
    )
    snapshot = make_source().load(LabLoadOptions(limit=10, workspace=tmp_path))
    app = PrimeLabView(lambda: snapshot, initial_loader=lambda: snapshot)
    quit_calls = 0

    def fake_quit() -> None:
        nonlocal quit_calls
        quit_calls += 1

    app.action_quit = fake_quit  # type: ignore[method-assign]

    async with app.run_test(size=(140, 44)) as pilot:
        await pilot.pause()
        await pilot.press("c")
        await pilot.pause()

        assert isinstance(app.screen, AgentChatScreen)
        app.screen.query_one("#agent-header", Static).focus()
        await pilot.press("ctrl+c")
        await pilot.pause()

        assert quit_calls == 1


@pytest.mark.asyncio
async def test_prime_lab_app_chat_choice_picker_records_selection(tmp_path: Path) -> None:
    (tmp_path / ".prime").mkdir()
    (tmp_path / ".prime" / "lab.json").write_text(
        '{"choices": {"primary_agent": "cursor"}}',
        encoding="utf-8",
    )
    snapshot = make_source().load(LabLoadOptions(limit=10, workspace=tmp_path))
    app = PrimeLabView(lambda: snapshot, initial_loader=lambda: snapshot)
    sent: list[str] = []
    app._send_agent_prompt = sent.append  # type: ignore[method-assign]
    widget_messages = (
        AgentChatMessage(
            "system",
            "Choice ready",
            "widget",
            {
                "type": "widget_requested",
                "widget_id": "choice-1",
                "kind": "choice_picker",
                "title": "Pick an environment",
                "payload": {
                    "kind": "choice_picker",
                    "title": "Pick an environment",
                    "candidates": [
                        {"id": "reverse-text", "label": "reverse-text"},
                        {"id": "wordle", "label": "wordle"},
                    ],
                    "default_id": "reverse-text",
                },
            },
        ),
    )

    async with app.run_test(size=(140, 44)) as pilot:
        await pilot.pause()
        await pilot.press("c")
        await pilot.pause()

        assert isinstance(app.screen, AgentChatScreen)
        app._set_agent_messages(widget_messages)
        app.screen._refresh_runtime_view()
        await pilot.pause()

        card = app.screen.query_one(AgentWidgetCard)
        buttons = list(card.query(Button))
        labels = [str(button.label) for button in buttons]
        assert labels == ["reverse-text", "wordle", "Enter"]
        enter_button = next(button for button in buttons if button.name == "choice-enter")
        assert enter_button.disabled

        card.query(Button).first().press()
        await pilot.pause()

        assert app._agent_session is not None
        actions = [
            json.loads(line)
            for line in (app._agent_session.path / "actions.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        assert actions[-1]["type"] == "agent_widget_choice_selected"
        assert actions[-1]["choice_id"] == "reverse-text"
        status = _render_renderable(card.query_one(".agent-widget-status", Static).content)
        assert "Selected reverse-text" in status
        assert "Click Enter to continue, or add details below first if you want." in status
        assert app.screen.focused is app.screen.query_one("#agent-prompt", AgentPrompt)
        assert not enter_button.disabled

        prompt = app.screen.query_one("#agent-prompt", AgentPrompt)
        prompt.load_text("use Qwen")
        enter_button.press()
        await pilot.pause()

        assert sent == ["I chose: reverse-text.\n\nuse Qwen"]
        assert prompt.text == ""
        assert enter_button.disabled


@pytest.mark.asyncio
async def test_prime_lab_app_chat_config_editor_uses_inline_controls(tmp_path: Path) -> None:
    (tmp_path / ".prime").mkdir()
    (tmp_path / ".prime" / "lab.json").write_text(
        '{"choices": {"primary_agent": "cursor"}}',
        encoding="utf-8",
    )
    snapshot = make_source().load(LabLoadOptions(limit=10, workspace=tmp_path))
    app = PrimeLabView(lambda: snapshot, initial_loader=lambda: snapshot)
    widget_messages = (
        AgentChatMessage(
            "system",
            "Config ready",
            "widget",
            {
                "type": "widget_requested",
                "kind": "config_editor",
                "title": "Eval: alphabet-sort",
                "payload": {
                    "kind": "config_editor",
                    "title": "Eval: alphabet-sort",
                    "config_kind": "eval",
                    "env_id": "alphabet-sort",
                    "defaults": {"num_examples": 10, "rollouts_per_example": 2},
                },
            },
        ),
    )

    async with app.run_test(size=(140, 44)) as pilot:
        await pilot.pause()
        await pilot.press("c")
        await pilot.pause()

        assert isinstance(app.screen, AgentChatScreen)
        app._set_agent_messages(widget_messages)
        app.screen._refresh_runtime_view()
        await pilot.pause()

        card = app.screen.query_one(AgentWidgetCard)
        field_values = {
            input_widget.name: input_widget.value for input_widget in card.query(ClearableInput)
        }
        button_labels = [str(button.label) for button in card.query(Button)]

        assert field_values["num_examples"] == "10"
        assert field_values["rollouts_per_example"] == "2"
        assert button_labels == ["Launch", "Stop"]


@pytest.mark.asyncio
async def test_prime_lab_app_chat_widget_launches_from_inline_config(
    tmp_path: Path,
) -> None:
    (tmp_path / ".prime").mkdir()
    (tmp_path / ".prime" / "lab.json").write_text(
        '{"choices": {"primary_agent": "cursor"}}',
        encoding="utf-8",
    )
    snapshot = make_source().load(LabLoadOptions(limit=10, workspace=tmp_path))
    app = PrimeLabView(lambda: snapshot, initial_loader=lambda: snapshot)
    widget_messages = (
        AgentChatMessage(
            "system",
            "Action ready",
            "widget",
            {
                "type": "widget_requested",
                "kind": "run_launcher",
                "title": "Eval: reverse-text",
                "payload": {
                    "kind": "run_launcher",
                    "title": "Eval: reverse-text",
                    "config_kind": "eval",
                    "env_id": "reverse-text",
                    "defaults": {
                        "model": "openai/gpt-4.1-mini",
                        "num_examples": 25,
                        "rollouts_per_example": 2,
                        "max_tokens": 256,
                    },
                },
            },
        ),
    )

    async with app.run_test(size=(140, 44)) as pilot:
        await pilot.pause()
        await pilot.press("c")
        await pilot.pause()
        assert isinstance(app.screen, AgentChatScreen)
        app._set_agent_messages(widget_messages)
        app.screen._refresh_runtime_view()
        await pilot.pause()

        card = app.screen.query_one(AgentWidgetCard)
        fields = {input_widget.name: input_widget for input_widget in card.query(ClearableInput)}
        fields["num_examples"].value = "7"
        fields["rollouts_per_example"].value = "4"
        launch_plan = card.current_launch_plan()
        command = launch_plan.command
        generated = tmp_path / ".prime" / "lab" / "configs" / "eval" / "reverse-text.toml"
        parsed = toml.loads(generated.read_text(encoding="utf-8"))
        assert command == (
            "prime eval run .prime/lab/configs/eval/reverse-text.toml --hosted --follow"
        )
        assert parsed["model"] == "openai/gpt-4.1-mini"
        assert parsed["num_examples"] == 7
        assert parsed["rollouts_per_example"] == 4
        assert parsed["eval"][0]["env_id"] == "primeintellect/reverse-text"
        assert parsed["eval"][0]["sampling_args"]["max_tokens"] == 256


def test_prime_lab_app_chat_widget_completes_partial_eval_config(tmp_path: Path) -> None:
    message = AgentChatMessage(
        "system",
        "Action ready",
        "widget",
        {
            "kind": "run_launcher",
            "title": "Eval: reverse-text",
            "payload": {
                "kind": "run_launcher",
                "config_kind": "eval",
                "config": {"eval": [{"env_id": "reverse-text"}]},
            },
        },
    )

    model = build_agent_widget_model(message, tmp_path)
    values = model.config_context["values"] if model.config_context is not None else {}

    assert model.title == "Evaluate reverse-text"
    assert values["envs"] == "reverse-text"
    assert values["model"] == ""
    assert values["num_examples"] == "50"
    assert values["rollouts_per_example"] == "3"
    assert values["max_tokens"] == "1024"
    assert values["max_concurrent"] == "auto"


def test_prime_lab_app_chat_widget_uses_default_environment_for_rl_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "prime_lab_app.agent_widget_model._training_model_options",
        lambda: _training_model_options_for_name("future/gpt-oss-next"),
    )
    message = AgentChatMessage(
        "system",
        "Action ready",
        "widget",
        {
            "kind": "run_launcher",
            "title": "Train: wordle",
            "payload": {
                "kind": "run_launcher",
                "config_kind": "rl",
                "defaults": {
                    "model": "future/gpt-oss-next",
                    "env": [{"id": "primeintellect/wordle"}],
                },
            },
        },
    )

    model = build_agent_widget_model(message, tmp_path)
    values = model.config_context["values"] if model.config_context is not None else {}
    build = build_agent_widget_config(model, {})

    assert model.title == "Train wordle"
    assert values["envs"] == "primeintellect/wordle"
    selected_model, selected_controls = widget_training_model_option_parts(values["model"])
    assert selected_model == "future/gpt-oss-next"
    assert selected_controls == {"reasoning_effort": "medium"}
    assert build.parsed["env"] == [{"id": "primeintellect/wordle"}]
    assert build.parsed["model"] == "future/gpt-oss-next"
    assert build.parsed["sampling"]["reasoning_effort"] == "medium"
    assert "name" not in build.parsed
    assert "name =" not in build.toml_text


def test_prime_lab_app_chat_widget_strips_agent_training_run_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "prime_lab_app.agent_widget_model._training_model_options",
        lambda: _training_model_options_for_name("future/gpt-oss-next"),
    )
    message = AgentChatMessage(
        "system",
        "Action ready",
        "widget",
        {
            "kind": "run_launcher",
            "title": "Train: wordle",
            "payload": {
                "kind": "run_launcher",
                "config_kind": "rl",
                "config": {
                    "name": "wordle-short-name",
                    "model": "future/gpt-oss-next",
                    "env": [{"id": "primeintellect/wordle"}],
                    "max_steps": 10,
                    "batch_size": 32,
                    "rollouts_per_example": 4,
                    "sampling": {
                        "max_tokens": 1024,
                        "reasoning_effort": "medium",
                    },
                },
            },
        },
    )

    model = build_agent_widget_model(message, tmp_path)
    build = build_agent_widget_config(model, {})

    assert "name" not in build.parsed
    assert "wordle-short-name" not in build.toml_text


def test_training_model_options_expand_reasoning_variants() -> None:
    qwen35 = _training_model_options_for_name("future/Qwen3.5-New")
    assert [label for label, _value in qwen35] == [
        "future/Qwen3.5-New (thinking)",
        "future/Qwen3.5-New (instruct)",
    ]
    assert [widget_training_model_option_parts(value) for _label, value in qwen35] == [
        ("future/Qwen3.5-New", {"enable_thinking": "true"}),
        ("future/Qwen3.5-New", {"enable_thinking": "false"}),
    ]

    qwen36 = _training_model_options_for_name("future/Qwen3.6-New")
    assert [label for label, _value in qwen36] == [
        "future/Qwen3.6-New (thinking)",
        "future/Qwen3.6-New (instruct)",
    ]
    assert [widget_training_model_option_parts(value) for _label, value in qwen36] == [
        ("future/Qwen3.6-New", {"enable_thinking": "true"}),
        ("future/Qwen3.6-New", {"enable_thinking": "false"}),
    ]

    nemotron = _training_model_options_for_name("future/Nemotron-Next")
    assert [label for label, _value in nemotron] == [
        "future/Nemotron-Next (thinking)",
        "future/Nemotron-Next (instruct)",
    ]
    assert [widget_training_model_option_parts(value) for _label, value in nemotron] == [
        ("future/Nemotron-Next", {"enable_thinking": "true"}),
        ("future/Nemotron-Next", {"enable_thinking": "false"}),
    ]

    gpt_oss = _training_model_options_for_name("future/gpt-oss-next")
    assert [label for label, _value in gpt_oss] == [
        "future/gpt-oss-next (low)",
        "future/gpt-oss-next (medium)",
        "future/gpt-oss-next (high)",
    ]
    assert [widget_training_model_option_parts(value) for _label, value in gpt_oss] == [
        ("future/gpt-oss-next", {"reasoning_effort": "low"}),
        ("future/gpt-oss-next", {"reasoning_effort": "medium"}),
        ("future/gpt-oss-next", {"reasoning_effort": "high"}),
    ]


def test_training_model_options_does_not_cache_auth_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class NoAuthConfig:
        api_key = ""
        team_id = None

    class AuthConfig:
        api_key = "token"
        team_id = "team-123"

    class FakeModelsClient:
        def __init__(self, _api_client: Any) -> None:
            pass

        def list_models(self, team_id: str | None = None) -> list[Any]:
            assert team_id == "team-123"
            return [types.SimpleNamespace(name="future/gpt-oss-next")]

    monkeypatch.setattr(agent_widget_model, "_TRAINING_MODEL_OPTIONS_CACHE", None)
    monkeypatch.setattr(agent_widget_model, "_TRAINING_MODEL_NAMES_CACHE", None)
    monkeypatch.setattr(agent_widget_model, "_TRAINING_MODEL_OPTIONS_CACHE_KEY", None)
    monkeypatch.setattr(agent_widget_model, "_TRAINING_MODEL_NAMES_CACHE_KEY", None)
    monkeypatch.setattr(agent_widget_model, "_TRAINING_MODEL_OPTION_METADATA", {})
    monkeypatch.setattr(agent_widget_model, "Config", NoAuthConfig)

    assert agent_widget_model._training_model_options() == ()
    assert agent_widget_model._TRAINING_MODEL_OPTIONS_CACHE is None
    assert agent_widget_model._TRAINING_MODEL_NAMES_CACHE is None

    monkeypatch.setattr(agent_widget_model, "Config", AuthConfig)
    monkeypatch.setattr(agent_widget_model, "APIClient", lambda: object())
    monkeypatch.setattr(agent_widget_model, "RLClient", FakeModelsClient)

    options = agent_widget_model._training_model_options()

    assert [label for label, _value in options] == [
        "future/gpt-oss-next (low)",
        "future/gpt-oss-next (medium)",
        "future/gpt-oss-next (high)",
    ]
    assert agent_widget_model.training_model_names() == ("future/gpt-oss-next",)
    assert agent_widget_model._TRAINING_MODEL_OPTIONS_CACHE == options


def test_training_model_options_cache_scopes_by_active_account(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = {"profile": "production", "team_id": "team-a"}
    calls: list[str | None] = []

    class DynamicConfig:
        def __init__(self) -> None:
            self.api_key = "token"
            self.base_url = "https://api.test"
            self.current_environment = state["profile"]
            self.team_id = state["team_id"]
            self.team_name = ""

    class FakeModelsClient:
        def __init__(self, _api_client: Any) -> None:
            pass

        def list_models(self, team_id: str | None = None) -> list[Any]:
            calls.append(team_id)
            return [types.SimpleNamespace(name=f"future/{team_id}-model")]

    monkeypatch.setattr(agent_widget_model, "_TRAINING_MODEL_OPTIONS_CACHE", None)
    monkeypatch.setattr(agent_widget_model, "_TRAINING_MODEL_NAMES_CACHE", None)
    monkeypatch.setattr(agent_widget_model, "_TRAINING_MODEL_OPTIONS_CACHE_KEY", None)
    monkeypatch.setattr(agent_widget_model, "_TRAINING_MODEL_NAMES_CACHE_KEY", None)
    monkeypatch.setattr(agent_widget_model, "_TRAINING_MODEL_OPTION_METADATA", {})
    monkeypatch.setattr(agent_widget_model, "Config", DynamicConfig)
    monkeypatch.setattr(agent_widget_model, "APIClient", lambda: object())
    monkeypatch.setattr(agent_widget_model, "RLClient", FakeModelsClient)

    assert agent_widget_model.training_model_names() == ("future/team-a-model",)

    state["team_id"] = "team-b"

    options = agent_widget_model._training_model_options()

    assert calls == ["team-a", "team-b"]
    assert options == (("future/team-b-model", "future/team-b-model"),)


def test_lab_agent_training_tools_include_current_model_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    models = ("openai/gpt-oss-120b", "qwen/qwen3-8b")
    monkeypatch.setattr(agent_widget_model, "training_model_names", lambda: models)

    tools = {tool["name"]: tool for tool in lab_dynamic_tools()}
    model_schema = tools["train_model"]["inputSchema"]["properties"]["model"]
    instructions = lab_widget_developer_instructions()

    assert model_schema["enum"] == list(models)
    assert "openai/gpt-oss-120b" in instructions
    assert "qwen/qwen3-8b" in instructions


def test_lab_agent_training_tool_rejects_unavailable_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        agent_widget_model,
        "training_model_names",
        lambda: ("openai/gpt-oss-120b",),
    )

    status, summary, payload = handle_lab_widget_tool_call(
        {
            "namespace": "lab",
            "tool": "train_model",
            "callId": "train-1",
            "arguments": {
                "env": "primeintellect/wiki-search",
                "model": "Qwen3-4B",
                "max_steps": 10,
                "batch_size": 32,
                "rollouts_per_example": 4,
                "max_tokens": 1024,
            },
        }
    )

    assert status == "error"
    assert "Qwen3-4B" in summary
    assert "openai/gpt-oss-120b" in payload["contentItems"][0]["text"]


def test_prime_lab_app_chat_widget_shows_all_train_models_for_rl_dropdown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "endpoints.toml").write_text(
        '[[endpoint]]\nmodel = "endpoint/should-not-drive-training"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "prime_lab_app.agent_widget_model._training_model_options",
        lambda: (
            *_training_model_options_for_name("future/gpt-oss-next"),
            *_training_model_options_for_name("future/Qwen3.5-New"),
            *_training_model_options_for_name("other/plain-model"),
        ),
    )
    message = AgentChatMessage(
        "system",
        "Action ready",
        "widget",
        {
            "kind": "run_launcher",
            "title": "Train: wordle",
            "payload": {
                "kind": "run_launcher",
                "config_kind": "rl",
                "defaults": {
                    "model": "future/Qwen3.5-New",
                    "env": [{"id": "primeintellect/wordle"}],
                },
            },
        },
    )

    model = build_agent_widget_model(message, tmp_path)
    fields = {field.name: field for field in model.fields}

    assert fields["envs"].widget == "input"
    assert fields["envs"].value == "primeintellect/wordle"
    assert fields["model"].widget == "select"
    selected_model, selected_controls = widget_training_model_option_parts(fields["model"].value)
    assert selected_model == "future/Qwen3.5-New"
    assert selected_controls == {"enable_thinking": "true"}
    assert [label for label, _value in fields["model"].options] == [
        "future/gpt-oss-next (low)",
        "future/gpt-oss-next (medium)",
        "future/gpt-oss-next (high)",
        "future/Qwen3.5-New (thinking)",
        "future/Qwen3.5-New (instruct)",
        "other/plain-model",
    ]
    assert [
        widget_training_model_option_parts(value) for _label, value in fields["model"].options
    ] == [
        ("future/gpt-oss-next", {"reasoning_effort": "low"}),
        ("future/gpt-oss-next", {"reasoning_effort": "medium"}),
        ("future/gpt-oss-next", {"reasoning_effort": "high"}),
        ("future/Qwen3.5-New", {"enable_thinking": "true"}),
        ("future/Qwen3.5-New", {"enable_thinking": "false"}),
        ("other/plain-model", {}),
    ]
    assert "enable_thinking" not in fields
    assert "reasoning_effort" not in fields
    build = build_agent_widget_config(model, {})

    assert build.parsed["model"] == "future/Qwen3.5-New"
    assert build.parsed["sampling"]["enable_thinking"] is True


def test_prime_lab_app_chat_widget_drops_unavailable_rl_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "prime_lab_app.agent_widget_model._training_model_options",
        lambda: (
            *_training_model_options_for_name("future/gpt-oss-next"),
            *_training_model_options_for_name("future/Qwen3.5-New"),
        ),
    )
    message = AgentChatMessage(
        "system",
        "Action ready",
        "widget",
        {
            "kind": "run_launcher",
            "payload": {
                "kind": "run_launcher",
                "config_kind": "rl",
                "defaults": {
                    "model": "unavailable-requested-model",
                    "env": [{"id": "primeintellect/wordle"}],
                },
            },
        },
    )

    model = build_agent_widget_model(message, tmp_path)
    fields = {field.name: field for field in model.fields}

    selected_model, selected_controls = widget_training_model_option_parts(fields["model"].value)
    assert selected_model == "future/gpt-oss-next"
    assert selected_controls == {"reasoning_effort": "medium"}
    assert [label for label, _value in fields["model"].options] == [
        "future/gpt-oss-next (low)",
        "future/gpt-oss-next (medium)",
        "future/gpt-oss-next (high)",
        "future/Qwen3.5-New (thinking)",
        "future/Qwen3.5-New (instruct)",
    ]
    build = build_agent_widget_config(model, {})

    assert build.parsed["model"] == "future/gpt-oss-next"
    assert build.parsed["sampling"]["reasoning_effort"] == "medium"
    assert "unavailable-requested-model" not in {value for _label, value in fields["model"].options}


def test_config_builder_requires_rl_model_and_environment() -> None:
    values = {
        "config-name": "missing-required",
        "config-model": "",
        "config-envs": "",
        "config-max-steps": "5",
        "config-rollouts": "1",
        "config-batch-size": "1",
        "config-max-tokens": "64",
    }

    build = build_config_from_fields({}, "rl", lambda field_id: values.get(field_id, ""))

    assert "model is required" in build.errors
    assert "at least one environment is required" in build.errors


@pytest.mark.asyncio
async def test_prime_lab_app_chat_preserves_widget_edits_when_transcript_appends(
    tmp_path: Path,
) -> None:
    (tmp_path / ".prime").mkdir()
    (tmp_path / ".prime" / "lab.json").write_text(
        '{"choices": {"primary_agent": "cursor"}}',
        encoding="utf-8",
    )
    snapshot = make_source().load(LabLoadOptions(limit=10, workspace=tmp_path))
    app = PrimeLabView(lambda: snapshot, initial_loader=lambda: snapshot)
    widget_message = AgentChatMessage(
        "system",
        "Action ready",
        "widget",
        {
            "type": "widget_requested",
            "kind": "run_launcher",
            "title": "Eval: reverse-text",
            "payload": {
                "kind": "run_launcher",
                "title": "Eval: reverse-text",
                "config_kind": "eval",
                "env_id": "reverse-text",
                "defaults": {"num_examples": 25},
            },
        },
    )

    async with app.run_test(size=(140, 44)) as pilot:
        await pilot.pause()
        await pilot.press("c")
        await pilot.pause()
        assert isinstance(app.screen, AgentChatScreen)
        app._set_agent_messages((widget_message,))
        app.screen._refresh_runtime_view()
        await pilot.pause()

        card = app.screen.query_one(AgentWidgetCard)
        fields = {input_widget.name: input_widget for input_widget in card.query(ClearableInput)}
        fields["num_examples"].value = "9"
        app._set_agent_messages((widget_message, AgentChatMessage("assistant", "Ready.")))
        app.screen._refresh_runtime_view()
        await pilot.pause()

        card_after = app.screen.query_one(AgentWidgetCard)
        fields_after = {
            input_widget.name: input_widget for input_widget in card_after.query(ClearableInput)
        }
        assert card_after is card
        assert fields_after["num_examples"].value == "9"


@pytest.mark.asyncio
async def test_prime_lab_app_chat_widget_prefills_local_env_and_endpoint_model(
    tmp_path: Path,
) -> None:
    (tmp_path / ".prime").mkdir()
    (tmp_path / ".prime" / "lab.json").write_text(
        '{"choices": {"primary_agent": "cursor"}}',
        encoding="utf-8",
    )
    env_dir = tmp_path / "environments" / "reverse_text"
    env_dir.mkdir(parents=True)
    (env_dir / "pyproject.toml").write_text(
        '[project]\nname = "reverse-text"\n',
        encoding="utf-8",
    )
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "endpoints.toml").write_text(
        "\n".join(
            [
                "[[endpoint]]",
                'endpoint_id = "local-model"',
                'model = "local/reverse-model"',
                'url = "https://example.invalid/v1"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    snapshot = make_source().load(LabLoadOptions(limit=10, workspace=tmp_path))
    app = PrimeLabView(lambda: snapshot, initial_loader=lambda: snapshot)
    widget_messages = (
        AgentChatMessage(
            "system",
            "Action ready",
            "widget",
            {
                "type": "widget_requested",
                "kind": "run_launcher",
                "title": "Eval: reverse-text",
                "payload": {
                    "kind": "run_launcher",
                    "title": "Eval: reverse-text",
                    "config_kind": "eval",
                    "env_id": "reverse-text",
                },
            },
        ),
    )

    async with app.run_test(size=(140, 44)) as pilot:
        await pilot.pause()
        await pilot.press("c")
        await pilot.pause()
        assert isinstance(app.screen, AgentChatScreen)
        app._set_agent_messages(widget_messages)
        app.screen._refresh_runtime_view()
        await pilot.pause()

        card = app.screen.query_one(AgentWidgetCard)
        fields = {
            input_widget.name: input_widget.value for input_widget in card.query(ClearableInput)
        }
        selects = {select.name: select.value for select in card.query(Select)}
        assert fields["envs"] == "reverse-text"
        assert "envs" not in selects
        assert selects["model"] == "local/reverse-model"


@pytest.mark.asyncio
async def test_prime_lab_app_chat_widget_prefills_existing_eval_config(
    tmp_path: Path,
) -> None:
    (tmp_path / ".prime").mkdir()
    (tmp_path / ".prime" / "lab.json").write_text(
        '{"choices": {"primary_agent": "cursor"}}',
        encoding="utf-8",
    )
    config_path = tmp_path / "configs" / "eval" / "reverse-text.toml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        "\n".join(
            [
                'model = "existing/model"',
                "num_examples = 12",
                "rollouts_per_example = 5",
                "max_concurrent = 3",
                "save_results = true",
                "",
                "[[eval]]",
                'env_id = "reverse-text"',
                "",
                "[eval.sampling_args]",
                "max_tokens = 2048",
                "",
            ]
        ),
        encoding="utf-8",
    )
    snapshot = make_source().load(LabLoadOptions(limit=10, workspace=tmp_path))
    app = PrimeLabView(lambda: snapshot, initial_loader=lambda: snapshot)
    widget_messages = (
        AgentChatMessage(
            "system",
            "Action ready",
            "widget",
            {
                "type": "widget_requested",
                "kind": "run_launcher",
                "title": "Eval: reverse-text",
                "payload": {
                    "kind": "run_launcher",
                    "title": "Eval: reverse-text",
                    "config_kind": "eval",
                    "env_id": "reverse-text",
                },
            },
        ),
    )

    async with app.run_test(size=(140, 44)) as pilot:
        await pilot.pause()
        await pilot.press("c")
        await pilot.pause()
        assert isinstance(app.screen, AgentChatScreen)
        app._set_agent_messages(widget_messages)
        app.screen._refresh_runtime_view()
        await pilot.pause()

        card = app.screen.query_one(AgentWidgetCard)
        fields = {
            input_widget.name: input_widget.value for input_widget in card.query(ClearableInput)
        }
        model_select = card.query_one(Select)
        assert fields["envs"] == "reverse-text"
        assert fields["num_examples"] == "12"
        assert fields["rollouts_per_example"] == "5"
        assert fields["max_tokens"] == "2048"
        assert fields["max_concurrent"] == "3"
        assert model_select.value == "existing/model"


@pytest.mark.asyncio
async def test_prime_lab_app_chat_widget_launch_button_streams_inline_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / ".prime").mkdir()
    (tmp_path / ".prime" / "lab.json").write_text(
        '{"choices": {"primary_agent": "cursor"}}',
        encoding="utf-8",
    )
    snapshot = make_source().load(LabLoadOptions(limit=10, workspace=tmp_path))
    app = PrimeLabView(lambda: snapshot, initial_loader=lambda: snapshot)
    widget_messages = (
        AgentChatMessage(
            "system",
            "Action ready",
            "widget",
            {
                "type": "widget_requested",
                "kind": "run_launcher",
                "title": "Eval: reverse-text",
                "payload": {
                    "kind": "run_launcher",
                    "title": "Eval: reverse-text",
                    "config_kind": "eval",
                    "env_id": "reverse-text",
                    "defaults": {"model": "openai/gpt-4.1-mini"},
                },
            },
        ),
    )
    calls: list[str] = []

    class FakeLaunchRunner:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            calls.append(str(kwargs["command"]))

        def run(self) -> None:
            self.kwargs["append_output"]("launching\n")
            self.kwargs["append_output"]("done\n")
            self.kwargs["finish"]("launch", 0)

        def stop(self) -> None:
            self.kwargs["finish"]("stopped", None)

    monkeypatch.setattr("prime_lab_app.agent_cards.ConfigLaunchRunner", FakeLaunchRunner)

    async with app.run_test(size=(140, 44)) as pilot:
        await pilot.pause()
        await pilot.press("c")
        await pilot.pause()
        assert isinstance(app.screen, AgentChatScreen)
        app._set_agent_messages(widget_messages)
        app.screen._refresh_runtime_view()
        await pilot.pause()

        card = app.screen.query_one(AgentWidgetCard)
        launch_plan = card.current_launch_plan()
        assert launch_plan.command == (
            "prime eval run .prime/lab/configs/eval/reverse-text.toml --hosted --follow"
        )
        assert card._launch_running is False
        card.query_one(".agent-widget-action-launch", Button).press()
        await pilot.pause()
        await pilot.pause()

        log_text = _render_renderable(card.query_one(".agent-widget-log", Static).content)
        status_text = _render_renderable(card.query_one(".agent-widget-status", Static).content)
        assert calls == [
            "prime eval run .prime/lab/configs/eval/reverse-text.toml --hosted --follow"
        ]
        assert "launching" in log_text
        assert "done" in log_text
        assert "Completed" in status_text
        assert app._agent_session is not None
        actions = [
            json.loads(line)
            for line in (app._agent_session.path / "actions.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        assert [action["status"] for action in actions[-2:]] == ["started", "completed"]
        assert actions[-1]["type"] == "agent_inline_launch"
        assert actions[-1]["config_kind"] == "eval"
        assert actions[-1]["config_path"] == ".prime/lab/configs/eval/reverse-text.toml"


@pytest.mark.asyncio
async def test_prime_lab_app_chat_training_launch_opens_run_screen(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "prime_lab_app.agent_widget_model._training_model_options",
        lambda: _training_model_options_for_name("openai/gpt-oss-120b"),
    )
    (tmp_path / ".prime").mkdir()
    (tmp_path / ".prime" / "lab.json").write_text(
        '{"choices": {"primary_agent": "cursor"}}',
        encoding="utf-8",
    )
    snapshot = make_source().load(LabLoadOptions(limit=10, workspace=tmp_path))
    app = PrimeLabView(
        lambda: snapshot,
        lambda item, include_logs, log_tail_lines, metrics_limit, metrics_min_step: item,
        initial_loader=lambda: snapshot,
    )
    widget_messages = (
        AgentChatMessage(
            "system",
            "Action ready",
            "widget",
            lab_widget_action_from_tool_call(
                {
                    "namespace": "lab",
                    "tool": "train_model",
                    "callId": "train-1",
                    "arguments": {
                        "env": "primeintellect/wiki-search",
                        "model": "openai/gpt-oss-120b",
                        "max_steps": 10,
                        "batch_size": 32,
                        "rollouts_per_example": 4,
                        "max_tokens": 1024,
                    },
                }
            ),
        ),
    )

    class FakeLaunchRunner:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

        def run(self) -> None:
            assert self.kwargs["training_run_created"] is not None
            self.kwargs["training_run_created"]("abc123run")
            self.kwargs["finish"]("launch", 0)

        def stop(self) -> None:
            self.kwargs["finish"]("stopped", None)

    monkeypatch.setattr("prime_lab_app.agent_cards.ConfigLaunchRunner", FakeLaunchRunner)

    async with app.run_test(size=(140, 44)) as pilot:
        await pilot.pause()
        await pilot.press("c")
        await pilot.pause()
        assert isinstance(app.screen, AgentChatScreen)
        app._set_agent_messages(widget_messages)
        app.screen._refresh_runtime_view()
        await pilot.pause()

        card = app.screen.query_one(AgentWidgetCard)
        view_button = next(button for button in card.query(Button) if button.name == "view-run")
        assert view_button.disabled is True

        app.screen.query_one(".agent-widget-action-launch", Button).press()
        for _ in range(10):
            await pilot.pause()
            if isinstance(app.screen, TrainingRunScreen):
                break

        assert isinstance(app.screen, TrainingRunScreen)
        assert app.screen._base_item.title == "abc123run"
        app.pop_screen()
        await pilot.pause()
        assert isinstance(app.screen, AgentChatScreen)
        card = app.screen.query_one(AgentWidgetCard)
        view_button = next(button for button in card.query(Button) if button.name == "view-run")
        assert view_button.disabled is False

        view_button.press()
        await pilot.pause()

        assert isinstance(app.screen, TrainingRunScreen)
        assert app.screen._base_item.title == "abc123run"
        assert app._agent_session is not None
        actions = [
            json.loads(line)
            for line in (app._agent_session.path / "actions.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        launch_actions = [
            action for action in actions if action.get("type") == "agent_inline_launch"
        ]
        assert [action["status"] for action in launch_actions[-3:]] == [
            "started",
            "run_created",
            "completed",
        ]
        assert launch_actions[-2]["run_id"] == "abc123run"
        assert launch_actions[-1]["run_id"] == "abc123run"


@pytest.mark.asyncio
async def test_prime_lab_app_chat_widget_stops_launch_when_unmounted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / ".prime").mkdir()
    (tmp_path / ".prime" / "lab.json").write_text(
        '{"choices": {"primary_agent": "cursor"}}',
        encoding="utf-8",
    )
    snapshot = make_source().load(LabLoadOptions(limit=10, workspace=tmp_path))
    app = PrimeLabView(lambda: snapshot, initial_loader=lambda: snapshot)
    widget_message = AgentChatMessage(
        "system",
        "Action ready",
        "widget",
        {
            "type": "widget_requested",
            "kind": "run_launcher",
            "title": "Eval: reverse-text",
            "payload": {
                "kind": "run_launcher",
                "title": "Eval: reverse-text",
                "config_kind": "eval",
                "env_id": "reverse-text",
                "defaults": {"model": "openai/gpt-4.1-mini"},
            },
        },
    )
    runners: list[Any] = []

    class FakeLaunchRunner:
        def __init__(self, **_kwargs: Any) -> None:
            self.stopped = False
            runners.append(self)

        def run(self) -> None:
            deadline = time.monotonic() + 1
            while not self.stopped and time.monotonic() < deadline:
                time.sleep(0.01)

        def stop(self) -> None:
            self.stopped = True

    monkeypatch.setattr("prime_lab_app.agent_cards.ConfigLaunchRunner", FakeLaunchRunner)

    async with app.run_test(size=(140, 44)) as pilot:
        await pilot.pause()
        await pilot.press("c")
        await pilot.pause()
        assert isinstance(app.screen, AgentChatScreen)
        app._set_agent_messages((widget_message,))
        app.screen._refresh_runtime_view()
        await pilot.pause()

        card = app.screen.query_one(AgentWidgetCard)
        card.query_one(".agent-widget-action-launch", Button).press()
        await pilot.pause()
        assert runners

        app._set_agent_messages(())
        app.screen._refresh_runtime_view()
        await pilot.pause()

        assert runners[0].stopped is True


@pytest.mark.asyncio
async def test_prime_lab_app_launch_grid_training_and_explore(tmp_path: Path) -> None:
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
async def test_prime_lab_app_ladder_loads_platform_sections(tmp_path: Path) -> None:
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
async def test_prime_lab_app_can_load_more_platform_rows(tmp_path: Path) -> None:
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
async def test_prime_lab_app_opens_training_run_screen(tmp_path: Path) -> None:
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


def test_training_run_screen_reports_platform_open_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    notifications: list[tuple[str, str | None]] = []
    screen = TrainingRunScreen(
        LabItem(
            key="training:run-123",
            section="training",
            title="run-123",
            subtitle="",
            status="RUNNING",
            raw={"id": "run-123"},
        ),
        lambda item, *_args: item,
        frontend_url="https://app.test",
    )

    monkeypatch.setattr("prime_lab_app.training_screen.webbrowser.open", lambda _url: False)
    monkeypatch.setattr(
        screen,
        "notify",
        lambda message, **kwargs: notifications.append((message, kwargs.get("severity"))),
    )

    screen.action_open_platform()

    assert notifications == [("Could not open the training run in a browser.", "warning")]


@pytest.mark.asyncio
async def test_prime_lab_app_opens_environment_screen(tmp_path: Path) -> None:
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
        assert not list(screen.query("#env-tabs"))

        screen._render_tab()
        await pilot.pause()
        assert {"train", "evaluate", "sync", "platform"} == {
            action.key for action in screen._action_by_id.values()
        }

        await pilot.click("#env-action-0")
        await pilot.pause()

        assert isinstance(app.screen, ConfigRunScreen)
        assert app.screen._item.raw["config_kind"] == "rl"


@pytest.mark.asyncio
async def test_environment_sync_action_uses_native_follow_screen(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeProcess:
        stdout = ["syncing\n"]

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

    monkeypatch.setattr("prime_lab_app.launch_runner.subprocess.Popen", fake_popen)

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

        screen._run_environment_action(
            EnvironmentAction("sync", "Sync", "prime env pull primeintellect/gsm8k")
        )
        await pilot.pause()
        await pilot.pause()

        assert isinstance(app.screen, ConfigLaunchScreen)
        assert commands and commands[0][:3] == ["prime", "env", "pull"]


@pytest.mark.asyncio
async def test_prime_lab_app_opens_setup_screen_for_uninitialized_workspace(
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
async def test_setup_screen_uses_agent_select_without_prime_rl_option(
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

        screen = app.screen
        assert isinstance(screen, SetupScreen)
        setup_body = _render_renderable(_setup_body(setup_item))
        assert "Choose an agent, then run setup in this workspace." in setup_body
        assert "Press Enter to run setup" not in setup_body
        picker_label = _render_renderable(screen.query_one("#setup-agent-label", Static).content)
        assert "Choose your coding agent" in picker_label
        assert not screen.query("#setup-prime-rl")

        agent_select = screen.query_one("#setup-agent", Select)
        assert agent_select.value == "codex"
        assert agent_select._options[0] == ("Codex", "codex")
        assert all("Agent:" not in str(label) for label, _value in agent_select._options)

        agent_select.value = "claude"
        await pilot.pause()
        assert screen._selected_agent() == "claude"

        agent_select.focus()
        await pilot.press("enter")
        await pilot.pause()
        assert agent_select.expanded is True
        assert screen._command_running is False


@pytest.mark.asyncio
async def test_setup_screen_run_uses_selected_agent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = make_source().load(LabLoadOptions(limit=10, workspace=tmp_path))
    app = PrimeLabView(lambda: snapshot)
    captured: list[tuple[LabSetupOptions, Path]] = []

    def fake_run_lab_setup_service(
        options: LabSetupOptions,
        *,
        workspace: Path,
        emit: Callable[[str], None],
    ) -> LabSetupResult:
        _ = emit
        captured.append((options, workspace))
        return LabSetupResult(exit_code=0, workspace=workspace)

    monkeypatch.setattr(
        "prime_lab_app.setup_screens.run_lab_setup_service",
        fake_run_lab_setup_service,
    )

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

        screen = app.screen
        assert isinstance(screen, SetupScreen)
        agent_select = screen.query_one("#setup-agent", Select)
        agent_select.value = "claude"
        await pilot.pause()

        screen.action_run_setup()
        for _ in range(20):
            if captured:
                break
            await pilot.pause()

        assert captured
        options, workspace = captured[0]
        assert options.agents == ("claude",)
        assert workspace == tmp_path.resolve()


@pytest.mark.asyncio
async def test_prime_lab_app_opens_workspace_browser(tmp_path: Path) -> None:
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
async def test_prime_lab_app_opens_add_workspace_screen(tmp_path: Path) -> None:
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
async def test_prime_lab_app_opens_agent_sync_screen(tmp_path: Path) -> None:
    (tmp_path / ".prime").mkdir()
    (tmp_path / ".prime" / "lab.json").write_text(
        '{"choices": {"primary_agent": "claude"}}',
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
async def test_prime_lab_app_opens_doctor_screen(tmp_path: Path) -> None:
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
async def test_prime_lab_app_opens_config_run_screen(tmp_path: Path) -> None:
    config_path = tmp_path / "configs" / "rl" / "train.toml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        'model = "openai/gpt-5-mini"\nmax_steps = 10\n[[env]]\nid = "primeintellect/gsm8k"\n',
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
async def test_config_run_keeps_b_as_text_when_input_focused(tmp_path: Path) -> None:
    config_path = tmp_path / "configs" / "rl" / "train.toml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        'model = ""\nmax_steps = 10\n[[env]]\nid = "primeintellect/gsm8k"\n',
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
        model_input = app.screen.query_one("#config-model", ClearableInput)
        model_input.focus()
        await pilot.press("b")
        await pilot.pause()

        assert isinstance(app.screen, ConfigRunScreen)
        assert model_input.value == "b"


@pytest.mark.asyncio
async def test_b_goes_back_from_setup_screen(tmp_path: Path) -> None:
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
        await pilot.press("b")
        await pilot.pause()

        assert not isinstance(app.screen, SetupScreen)


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

    monkeypatch.setattr("prime_lab_app.launch_runner.subprocess.Popen", fake_popen)

    config_path = tmp_path / "configs" / "rl" / "train.toml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        'model = "openai/gpt-5-mini"\nmax_steps = 10\n[[env]]\nid = "primeintellect/gsm8k"\n',
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
        assert commands and commands[0][:3] == ["prime", "train", "run"]
        assert (tmp_path / ".prime" / "lab" / "configs" / "rl" / "train.toml").is_file()


def test_config_launch_unmount_stops_active_runner(tmp_path: Path) -> None:
    runners: list[Any] = []

    class FakeLaunchRunner:
        def __init__(self) -> None:
            self.stopped = False
            runners.append(self)

        def stop(self) -> None:
            self.stopped = True

    screen = ConfigLaunchScreen(
        command="prime train run train.toml",
        workspace=tmp_path,
        follow_training_logs=True,
    )
    runner = FakeLaunchRunner()
    screen._runner = runner
    screen._running = True

    screen.on_unmount()
    screen._append_output("late output\n")
    screen._finish_runner("launch", 0)

    assert runners[0].stopped is True
    assert screen._closed is True
    assert screen._running is False


def test_launch_command_quotes_config_paths_with_spaces() -> None:
    assert shlex.split(launch_command_for_config("rl", ".prime/lab/configs/rl/my run.toml")) == [
        "prime",
        "train",
        "run",
        ".prime/lab/configs/rl/my run.toml",
        "--yes",
    ]
    assert shlex.split(launch_command_for_config("eval", "configs/eval/my eval.toml")) == [
        "prime",
        "eval",
        "run",
        "configs/eval/my eval.toml",
        "--hosted",
    ]
    assert shlex.split(launch_command_for_config("gepa", "configs/gepa/my prompt.toml")) == [
        "prime",
        "gepa",
        "run",
        "configs/gepa/my prompt.toml",
    ]


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
        if command[:3] == ["prime", "train", "logs"]:
            return FakeProcess(["Watching logs for run abc123run...\n", "step 1 reward 0.5\n"])
        return FakeProcess(["Creating RL training run...\n", "  prime train logs abc123run -f\n"])

    monkeypatch.setattr("prime_lab_app.launch_runner.subprocess.Popen", fake_popen)

    config_path = tmp_path / "configs" / "rl" / "train.toml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        'model = "openai/gpt-5-mini"\nmax_steps = 10\n[[env]]\nid = "primeintellect/gsm8k"\n',
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
            ["prime", "train", "run", ".prime/lab/configs/rl/train.toml", "--yes"],
            ["prime", "train", "logs", "abc123run", "-f"],
        ]
        assert isinstance(app.screen, ConfigLaunchScreen)
        assert "Following run logs with: prime train logs abc123run -f" in app.screen._output
        assert "step 1 reward 0.5" in app.screen._output


def test_training_log_follow_command_uses_dashboard_and_run_id_hints() -> None:
    from_url = extract_training_log_follow_command(
        "Created run: https://app.test/dashboard/training/urlrun123"
    )
    from_hint = extract_training_log_follow_command("Training run id: hint_run_123")
    from_train_logs = extract_training_log_follow_command(
        "View logs with: prime train logs run123x -f"
    )

    assert from_url is not None
    assert from_url.run_id == "urlrun123"
    assert from_url.argv == ("prime", "train", "logs", "urlrun123", "-f")
    assert from_hint is not None
    assert from_hint.run_id == "hint_run_123"
    assert from_hint.argv == ("prime", "train", "logs", "hint_run_123", "-f")
    assert from_train_logs is not None
    assert from_train_logs.run_id == "run123x"


def test_config_launch_runner_opens_training_run_instead_of_following_inline_logs(
    tmp_path: Path,
) -> None:
    class FakeProcess:
        stdout = ["Training run id: abc123run\n"]

        def wait(self) -> int:
            return 0

        def poll(self) -> None:
            return None

        def terminate(self) -> None:
            return None

    commands: list[list[str]] = []
    opened: list[str] = []
    finishes: list[tuple[str, int | None]] = []

    def fake_popen(command: list[str], **_kwargs: Any) -> FakeProcess:
        commands.append(command)
        return FakeProcess()

    runner = ConfigLaunchRunner(
        command="prime train run train.toml --yes",
        workspace=tmp_path,
        follow_training_logs=True,
        append_output=lambda _text: None,
        update_status=lambda _text, _style: None,
        finish=lambda kind, returncode: finishes.append((kind, returncode)),
        training_run_created=opened.append,
        popen_factory=fake_popen,
    )

    runner.run()

    assert commands == [["prime", "train", "run", "train.toml", "--yes"]]
    assert opened == ["abc123run"]
    assert finishes == [("launch", 0)]


def test_rl_client_preview_run_uses_preview_endpoint() -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    class FakeClient:
        def post(self, path: str, *, json: dict[str, Any]) -> dict[str, Any]:
            calls.append((path, json))
            return {"ok": True, "warnings": []}

    result = RLClient(FakeClient()).preview_run({"model": {"name": "openai/gpt-5-mini"}})

    assert result == {"ok": True, "warnings": []}
    assert calls == [("/rft/runs/preview", {"model": {"name": "openai/gpt-5-mini"}})]


def test_lab_platform_preview_reports_success_and_warnings() -> None:
    class FakeRLPreviewClient:
        def __init__(self, _api_client: object) -> None:
            pass

        def preview_run(self, payload: dict[str, Any]) -> dict[str, Any]:
            assert payload["model"] == "openai/gpt-5-mini"
            return {"message": "Preview ready", "warnings": ["Queue is busy"]}

    result = preview_lab_config(
        "rl",
        {"model": "openai/gpt-5-mini"},
        api_client_factory=lambda: object(),
        rl_client_factory=FakeRLPreviewClient,
    )

    assert result.status == "warning"
    assert result.message == "Preview ready"
    assert result.warnings == ("Queue is busy",)


def test_lab_platform_preview_falls_back_when_unavailable() -> None:
    class FakeAPIClient:
        def post(self, _path: str, *, json: dict[str, Any]) -> dict[str, Any]:
            raise RuntimeError("404 Not Found")

    result = preview_lab_config(
        "eval",
        {"model": "openai/gpt-5-mini"},
        api_client_factory=FakeAPIClient,
    )

    assert result.status == "unavailable"
    assert "Platform preview unavailable" in result.message


@pytest.mark.asyncio
async def test_config_launch_reports_missing_training_log_hint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeProcess:
        stdout = ["Created training run.\n"]

        def wait(self) -> int:
            return 0

        def poll(self) -> None:
            return None

        def terminate(self) -> None:
            return None

    def fake_popen(_command: list[str], **_kwargs: Any) -> FakeProcess:
        return FakeProcess()

    monkeypatch.setattr("prime_lab_app.launch_runner.subprocess.Popen", fake_popen)

    config_path = tmp_path / "configs" / "rl" / "train.toml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        'model = "openai/gpt-5-mini"\nmax_steps = 10\n[[env]]\nid = "primeintellect/gsm8k"\n',
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
            if (
                isinstance(app.screen, ConfigLaunchScreen)
                and "no live log command" in app.screen._output
            ):
                break

        assert isinstance(app.screen, ConfigLaunchScreen)
        assert "no live log command was detected" in app.screen._output


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
        if command[:3] == ["prime", "train", "logs"]:
            log_attempts += 1
            if log_attempts == 1:
                return FakeProcess(["No logs available yet.\n"], 1)
            return FakeProcess(["logs ready\n"], 0)
        return FakeProcess(["  prime train logs retryrun -f\n"])

    monkeypatch.setattr("prime_lab_app.launch_runner.subprocess.Popen", fake_popen)
    monkeypatch.setattr("prime_lab_app.launch_runner._LOG_RETRY_DELAYS", (0.01,))

    config_path = tmp_path / "configs" / "rl" / "train.toml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        'model = "openai/gpt-5-mini"\nmax_steps = 10\n[[env]]\nid = "primeintellect/gsm8k"\n',
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

        assert commands.count(["prime", "train", "logs", "retryrun", "-f"]) == 2
        assert isinstance(app.screen, ConfigLaunchScreen)
        assert "Logs are not ready yet; retrying in 0.01s." in app.screen._output
        assert "logs ready" in app.screen._output


@pytest.mark.asyncio
async def test_prime_lab_app_opens_local_eval_run_screen(tmp_path: Path) -> None:
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
async def test_prime_lab_app_evaluations_by_env_groups_runs(tmp_path: Path) -> None:
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
async def test_prime_lab_app_evaluation_view_arrows_change_selector(tmp_path: Path) -> None:
    snapshot = make_source().load(LabLoadOptions(limit=10, workspace=tmp_path))
    app = PrimeLabView(lambda: snapshot)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        app._active_section_key = "evaluations"
        app._render_active_section()
        toggle = app.query_one("#evaluation-toggle", EvaluationViewToggle)
        toggle.focus()
        await pilot.pause()

        assert app._evaluation_view == "runs"

        await pilot.press("left")
        await pilot.pause()

        assert app._evaluation_view == "env"
        assert app.focused is toggle


@pytest.mark.asyncio
async def test_prime_lab_app_by_env_enter_opens_highlighted_local_eval(
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
async def test_prime_lab_app_click_opens_local_eval_run_screen(tmp_path: Path) -> None:
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
async def test_prime_lab_app_mouse_click_requires_visible_training_sidebar(
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
    console = Console(record=True, width=120, file=StringIO())
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
