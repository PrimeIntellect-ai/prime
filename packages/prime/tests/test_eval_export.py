import json
import zipfile

from prime_cli.main import app
from prime_cli.utils.eval_export import build_inspect_log, build_verifiers_rows, write_inspect_eval
from typer.testing import CliRunner

runner = CliRunner()


def _evaluation() -> dict:
    return {
        "evaluation_id": "eval-1",
        "run_id": "run-1",
        "status": "COMPLETED",
        "environment_names": ["primeintellect/gsm8k"],
        "model_name": "openai/gpt-4.1-mini",
        "version_id": "v1",
        "metadata": {"rollouts_per_example": 2},
        "created_at": "2026-05-11T10:00:00Z",
        "completed_at": "2026-05-11T10:05:00Z",
    }


def _sample(reward: float = 0.8) -> dict:
    return {
        "example_id": "gsm8k:test:1",
        "prompt": [{"role": "user", "content": "What is 1+1?"}],
        "completion": [{"role": "assistant", "content": "2"}],
        "answer": "2",
        "reward": reward,
        "metrics": {"correctness": reward},
        "rollout_number": 0,
    }


def test_build_verifiers_rows_preserves_rollout_fields():
    rows = build_verifiers_rows(_evaluation(), [_sample()], "run-1")

    assert rows == [
        {
            "prompt": [{"role": "user", "content": "What is 1+1?"}],
            "answer": "2",
            "reward": 0.8,
            "rollout_index": 0,
            "example_id": "gsm8k:test:1",
            "completion": [{"role": "assistant", "content": "2"}],
            "run_id": "run-1",
            "model": "openai/gpt-4.1-mini",
            "env": "primeintellect/gsm8k",
            "env_version": "v1",
            "timestamp": "2026-05-11T10:05:00Z",
            "rewards": {"correctness": 0.8},
        }
    ]


def test_write_inspect_eval_creates_log_json(tmp_path):
    path = tmp_path / "run-1.eval"
    log = build_inspect_log(_evaluation(), [_sample()], "run-1")

    write_inspect_eval(path, log)

    with zipfile.ZipFile(path) as archive:
        assert archive.namelist() == ["log.json"]
        payload = json.loads(archive.read("log.json"))

    assert payload["version"] == 2
    assert payload["eval"]["run_id"] == "run-1"
    assert payload["samples"][0]["scores"]["reward"]["value"] == 0.8


def test_eval_export_command_filters_rewards(monkeypatch, tmp_path):
    class DummyConfig:
        team_id = None

    class DummyClient:
        def __init__(self, *_args, **_kwargs):
            pass

        def get_evaluation(self, eval_id):
            assert eval_id == "run-1"
            return _evaluation()

        def get_samples(self, eval_id, page=1, limit=1000):
            assert eval_id == "eval-1"
            assert page == 1
            return {
                "samples": [_sample(0.4), _sample(0.9)],
                "total": 2,
                "page": 1,
                "limit": limit,
                "total_pages": 1,
            }

    monkeypatch.setattr("prime_cli.commands.evals.Config", lambda: DummyConfig())
    monkeypatch.setattr("prime_cli.commands.evals.APIClient", lambda: object())
    monkeypatch.setattr("prime_cli.commands.evals.EvalsClient", DummyClient)

    output_path = tmp_path / "filtered.jsonl"
    result = runner.invoke(
        app,
        [
            "eval",
            "export",
            "run-1",
            "--min-reward",
            "0.8",
            "--output",
            str(output_path),
        ],
        env={"PRIME_DISABLE_VERSION_CHECK": "1"},
    )

    assert result.exit_code == 0, result.output
    lines = output_path.read_text().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["reward"] == 0.9
