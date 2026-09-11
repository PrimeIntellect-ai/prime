import json
from pathlib import Path

import pytest
from conftest import RecordingHandler

from prime_runs.backend import EvalsBackend, RftBackend
from prime_runs.config_privacy import config_summary, metadata_summary
from prime_runs.models import EnvironmentRef, RunSpec, RunStatus

FIXTURE = json.loads((Path(__file__).parent / "data/config_privacy.json").read_text())


def test_shared_wire_contract_does_not_mutate_config():
    before = json.dumps(FIXTURE["input"])
    assert config_summary(FIXTURE["input"]) == FIXTURE["config"]
    assert metadata_summary(FIXTURE["input"]) == FIXTURE["metadata"]
    assert json.dumps(FIXTURE["input"]) == before


def test_eval_create_update_and_failure_only_send_summaries(make_platform_client, eval_routes):
    handler = RecordingHandler(eval_routes)
    backend = EvalsBackend(make_platform_client(handler), frontend_url="https://example.invalid")
    source = FIXTURE["input"]
    spec = RunSpec(
        name="public identity", environments=[EnvironmentRef(id="env-123")], config=source
    )
    handle = backend.create(spec)
    backend.update(handle.id, config=source)
    backend.finalize(handle.id, status=RunStatus.FAILED, error="SECRET_ERROR", config=source)
    for request in handler.requests:
        assert b"SECRET" not in request.content
        assert b"PRIVATE" not in request.content
    created = handler.bodies_for("/api/v1/evaluations/")[0]
    assert created["metadata"] == FIXTURE["metadata"]
    assert handler.bodies_for("/api/v1/evaluations/eval-abc")[0]["metadata"] == FIXTURE["metadata"]
    assert source["env"]["agent"]["harness"]["env"]["OPENAI_API_KEY"] == "SECRET_HARNESS"


def test_an_empty_summary_replaces_old_metadata(make_platform_client, eval_routes):
    handler = RecordingHandler(eval_routes)
    backend = EvalsBackend(make_platform_client(handler), frontend_url="https://example.invalid")
    backend.update("eval-abc", config={"config_source": {"text": "SECRET"}})
    assert handler.bodies_for("/api/v1/evaluations/eval-abc") == [{"metadata": {}}]


def test_training_config_does_not_bypass_source_protection(make_platform_client, rft_routes):
    handler = RecordingHandler(rft_routes)
    backend = RftBackend(
        make_platform_client(handler), frontend_url="https://example.invalid", team_id="team-1"
    )
    backend.create(
        RunSpec(
            kind="train",
            model="public/model",
            config={
                **FIXTURE["input"],
                "trainer": {"lr": 1e-6, "env": {"key": "SECRET_TRAINER"}},
            },
        )
    )
    assert b"SECRET" not in handler.requests[0].content
    assert b"PRIVATE" not in handler.requests[0].content
    assert handler.bodies_for("/api/v1/rft/external-runs")[0]["run_config"]["trainer"] == {
        "lr": 1e-6
    }


@pytest.mark.parametrize(
    "value", [None, {}, [], "SECRET", True, float("inf"), float("nan"), 2**100]
)
def test_numeric_controls_reject_opaque_values(value):
    assert config_summary({"max_tokens": value, "sampling": {"temperature": value}}) == {}


def test_state_columns_preserve_exact_keys_with_bounded_names_and_count():
    columns = ["tool-output", "grader.score", "ground truth", "答案", "😀" * 128]
    assert metadata_summary(
        {"state_columns": [*columns, "", "x" * 129, "😀" * 129, None, {"key": "SECRET"}]}
    ) == {"state_columns": columns}
    many_columns = [f"column {index}" for index in range(129)]
    assert metadata_summary({"state_columns": many_columns}) == {
        "state_columns": many_columns[:128]
    }


@pytest.mark.parametrize(
    ("finished_at", "expected"),
    [
        ("2026-09-11", None),
        ("2026-09-11T12:34:56", None),
        ("2026-09-11T07:04:56Z", "2026-09-11T07:04:56+00:00"),
        ("2026-09-11T12:34:56+05:30", "2026-09-11T07:04:56+00:00"),
        ("2026-09-11T12:34:56+05:30:15", "2026-09-11T07:04:41+00:00"),
        ("0001-01-01T00:00:00+01:00", None),
    ],
)
def test_terminal_timestamps_require_a_timezone_and_normalize_to_utc(finished_at, expected):
    terminal = metadata_summary(
        {"prime_runs": {"status": "completed", "finished_at": finished_at}}
    )["prime_runs"]
    assert terminal == (
        {"status": "completed", "finished_at": expected}
        if expected is not None
        else {"status": "completed"}
    )
