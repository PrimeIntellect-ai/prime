from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
from prime_cli.lab_view.app import (
    LabOptionList,
    PrimeLabView,
    TrainingRunScreen,
    _chart_count,
    _histogram_charts_from_raw,
    _item_details,
    _ladder_limits,
    _next_log_tail_lines,
    _parse_log_records,
    _training_config_toml,
)
from prime_cli.lab_view.data import LabDataSource, LabLoadOptions, discover_local_eval_runs
from rich.console import Console


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
            return {"data": {"semantic_version": "1.0.0", "content_hash": "abcdef123456"}}
        if endpoint == "/environmentshub/primeintellect/gsm8k/status":
            return {"data": {"latest_version": {"semantic_version": "1.0.0"}}}
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

    def get_environment_status(self, owner: str, name: str) -> dict[str, Any]:
        return {"environment": f"{owner}/{name}", "status": "SUCCESS"}


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
    assert environments.items[0].metadata[0] == ("Scope", "mine")
    assert environments.items[1].metadata[0] == ("Scope", "public")


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


def test_lab_view_home_shows_cached_sibling_workspaces(tmp_path: Path) -> None:
    active = tmp_path / "active-lab"
    cached = tmp_path / "cached-lab"
    active.mkdir()
    (cached / ".prime").mkdir(parents=True)
    (cached / ".prime" / "lab.json").write_text(
        '{"setup_source": "prime lab setup", "choices": {"primary_agent": "opencode"}}',
        encoding="utf-8",
    )

    snapshot = make_source().load(LabLoadOptions(limit=10, workspace=active))
    workspace = snapshot.section("workspace")

    assert workspace is not None
    workspaces = [item for item in workspace.items if item.raw.get("type") == "workspace_context"]
    assert [item.title for item in workspaces] == ["active-lab", "cached-lab"]
    assert workspaces[0].status == "active"
    assert workspaces[1].status == "cached"
    assert workspaces[1].metadata == (
        ("Path", str(cached.resolve())),
        ("Setup", "prime lab setup"),
        ("Primary agent", "opencode"),
    )


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
    assert item.raw["environment_statuses"] == [
        {"environment": "primeintellect/gsm8k", "status": "SUCCESS"}
    ]


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

    assert "Environment" in rendered
    assert "Status" in rendered
    assert "1.0.0" in rendered
    assert "Raw" not in rendered
    assert "content_hash" not in rendered


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
    assert "\n[[environments]]" in config
    assert "learning_rate" not in config
    assert "args" not in config


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
    assert environments.items[0].metadata[0] == ("Scope", "public")
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


def test_lab_view_ladder_limits() -> None:
    assert _ladder_limits(3) == (3,)
    assert _ladder_limits(30) == (5, 10, 20, 30)
    assert _ladder_limits(100) == (5, 10, 20, 40, 80, 100)


def test_lab_view_log_tail_doubles_after_default() -> None:
    assert _next_log_tail_lines(50) == 1000
    assert _next_log_tail_lines(1000) == 2000
    assert _next_log_tail_lines(2000) == 4000


@pytest.mark.asyncio
async def test_prime_lab_view_mounts(tmp_path: Path) -> None:
    snapshot = make_source().load(LabLoadOptions(limit=10, workspace=tmp_path))
    app = PrimeLabView(lambda: snapshot)

    async with app.run_test() as pilot:
        await pilot.pause()
        assert app._snapshot == snapshot


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
        app._active_section_key = "training"
        app._render_active_section()
        await pilot.pause()
        app.action_load_detail()
        await pilot.pause()

        assert isinstance(app.screen, TrainingRunScreen)


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
