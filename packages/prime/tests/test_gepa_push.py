import json

import pytest
from prime_cli.commands.gepa import (
    _load_gepa_run,
    _push_gepa_run,
    _validate_gepa_run_dir,
)
from prime_cli.main import app
from typer.testing import CliRunner

runner = CliRunner()


def _write_gepa_run(tmp_path, metadata=None, samples=None):
    metadata = {
        "schema_version": "verifiers.gepa.v1",
        "eval_kind": "gepa",
        "framework": "verifiers",
        "optimizer": "gepa",
        "optimization_target": "system_prompt",
        "env_id": "primeintellect/wordle",
        "model": "openai/gpt-4.1-mini",
        "reflection_model": "openai/gpt-4.1",
        "timestamp": "2026-04-28T12:00:00",
        "num_candidates": 2,
        "best_idx": 1,
        "best_score": 0.82,
        **(metadata or {}),
    }
    samples = samples or [
        {
            "example_id": 0,
            "reward": 0.82,
            "score": 0.82,
            "info": {
                "eval_kind": "gepa",
                "sample_type": "candidate",
                "optimization_target": "system_prompt",
                "candidate_idx": 1,
                "is_best": True,
                "system_prompt": "Solve carefully.",
                "system_prompt_sha256": "abc123",
                "diff_from_initial": "+ carefully",
                "parent_candidate_idxs": [0],
                "val_subscores": [0.8, 0.84],
                "num_val_examples": 2,
            },
        }
    ]

    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    (tmp_path / "results.jsonl").write_text("\n".join(json.dumps(sample) for sample in samples))
    (tmp_path / "system_prompt.txt").write_text("Solve carefully.")
    (tmp_path / "pareto_frontier.jsonl").write_text(json.dumps({"candidate_idx": 1}))
    return tmp_path


def test_load_gepa_run_preserves_metadata_and_samples(tmp_path):
    run_dir = _write_gepa_run(tmp_path)

    data = _load_gepa_run(run_dir)

    assert (
        data["eval_name"] == "primeintellect/wordle--openai/gpt-4.1-mini--gepa--2026-04-28T12:00:00"
    )
    assert data["model_name"] == "openai/gpt-4.1-mini"
    assert data["env_id"] == "primeintellect/wordle"
    assert data["framework"] == "verifiers"
    assert data["metadata"]["eval_kind"] == "gepa"
    assert data["metadata"]["schema_version"] == "verifiers.gepa.v1"
    assert data["metadata"]["artifacts"]["system_prompt"] == "Solve carefully."
    assert data["metadata"]["artifacts"]["pareto_frontier_count"] == 1
    assert data["results"] == [
        {
            "example_id": 0,
            "reward": 0.82,
            "score": 0.82,
            "info": {
                "eval_kind": "gepa",
                "sample_type": "candidate",
                "optimization_target": "system_prompt",
                "candidate_idx": 1,
                "is_best": True,
                "system_prompt": "Solve carefully.",
                "system_prompt_sha256": "abc123",
                "diff_from_initial": "+ carefully",
                "parent_candidate_idxs": [0],
                "val_subscores": [0.8, 0.84],
                "num_val_examples": 2,
            },
        }
    ]


def test_push_gepa_run_constructs_request_without_task_fields(tmp_path, monkeypatch):
    run_dir = _write_gepa_run(tmp_path)
    captured = {}

    class DummyEvalsClient:
        def __init__(self, _api_client):
            pass

        def create_evaluation(self, **kwargs):
            captured["create"] = kwargs
            return {"evaluation_id": "eval-123"}

        def push_samples(self, evaluation_id, samples):
            captured["pushed_evaluation_id"] = evaluation_id
            captured["samples"] = samples

        def finalize_evaluation(self, evaluation_id):
            captured["finalized_evaluation_id"] = evaluation_id

    monkeypatch.setattr("prime_cli.commands.gepa.APIClient", lambda: object())
    monkeypatch.setattr("prime_cli.commands.gepa.EvalsClient", DummyEvalsClient)

    eval_id = _push_gepa_run(str(run_dir))

    assert eval_id == "eval-123"
    assert (
        captured["create"]["name"]
        == "primeintellect/wordle--openai/gpt-4.1-mini--gepa--2026-04-28T12:00:00"
    )
    assert captured["create"]["environments"] == [{"slug": "primeintellect/wordle"}]
    assert captured["create"]["model_name"] == "openai/gpt-4.1-mini"
    assert captured["create"]["dataset"] == "primeintellect/wordle"
    assert captured["create"]["framework"] == "verifiers"
    assert "task_type" not in captured["create"]
    assert captured["create"]["metadata"]["eval_kind"] == "gepa"
    assert captured["create"]["metadata"]["optimizer"] == "gepa"
    assert captured["pushed_evaluation_id"] == "eval-123"
    assert captured["finalized_evaluation_id"] == "eval-123"

    assert captured["samples"][0]["example_id"] == 0
    assert "task" not in captured["samples"][0]
    assert "task_type" not in captured["samples"][0]


def test_gepa_push_cli_json_outputs_only_json(tmp_path, monkeypatch):
    run_dir = _write_gepa_run(tmp_path)
    captured = {}

    monkeypatch.setattr("prime_cli.main.check_for_update", lambda: (False, None))

    class DummyEvalsClient:
        def __init__(self, _api_client):
            pass

        def create_evaluation(self, **kwargs):
            captured["create"] = kwargs
            return {"evaluation_id": "eval-123"}

        def push_samples(self, evaluation_id, samples):
            captured["pushed_evaluation_id"] = evaluation_id
            captured["samples"] = samples

        def finalize_evaluation(self, evaluation_id):
            captured["finalized_evaluation_id"] = evaluation_id

    monkeypatch.setattr("prime_cli.commands.gepa.APIClient", lambda: object())
    monkeypatch.setattr("prime_cli.commands.gepa.EvalsClient", DummyEvalsClient)

    result = runner.invoke(
        app,
        ["gepa", "push", str(run_dir), "--public", "--output", "json"],
        env={"PRIME_DISABLE_VERSION_CHECK": "1"},
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {"evaluation_id": "eval-123"}
    assert captured["create"]["is_public"] is True
    assert captured["pushed_evaluation_id"] == "eval-123"
    assert captured["finalized_evaluation_id"] == "eval-123"


def test_push_gepa_run_sends_eval_kind_when_client_supports_it(tmp_path, monkeypatch):
    run_dir = _write_gepa_run(tmp_path)
    captured = {}

    class DummyEvalsClient:
        def __init__(self, _api_client):
            pass

        def create_evaluation(self, eval_kind=None, **kwargs):
            captured["eval_kind"] = eval_kind
            captured["create"] = kwargs
            return {"evaluation_id": "eval-123"}

        def push_samples(self, _evaluation_id, _samples):
            pass

        def finalize_evaluation(self, _evaluation_id):
            pass

    monkeypatch.setattr("prime_cli.commands.gepa.APIClient", lambda: object())
    monkeypatch.setattr("prime_cli.commands.gepa.EvalsClient", DummyEvalsClient)

    _push_gepa_run(str(run_dir))

    assert captured["eval_kind"] == "gepa"
    assert captured["create"]["metadata"]["eval_kind"] == "gepa"


def test_validate_gepa_run_dir_requires_metadata(tmp_path):
    (tmp_path / "results.jsonl").write_text("")

    with pytest.raises(ValueError) as exc_info:
        _validate_gepa_run_dir(str(tmp_path))

    assert "missing metadata.json" in str(exc_info.value)


def test_validate_gepa_run_dir_requires_results(tmp_path):
    (tmp_path / "metadata.json").write_text("{}")

    with pytest.raises(ValueError) as exc_info:
        _validate_gepa_run_dir(str(tmp_path))

    assert "missing results.jsonl" in str(exc_info.value)
