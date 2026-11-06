import inspect
import sys
from pathlib import Path
from typing import Any

import pytest
from prime_cli.commands.evals import _create_hosted_evaluations
from prime_cli.main import app
from prime_cli.utils.config_snapshot import (
    MAX_COMMAND_CHARS,
    MAX_CONFIG_FILES_BYTES,
    NOTICE,
    REDACTED,
    build_config_snapshot,
    launch_command,
    scrub_command_args,
)
from prime_cli.utils.hosted_eval import HostedEvalConfig
from prime_evals import EvalsClient
from typer.testing import CliRunner

runner = CliRunner()
TEST_ENV = {"PRIME_DISABLE_VERSION_CHECK": "1", "COLUMNS": "200"}

TOML = '# lr sweep, see notes\n[model]\nname = "Qwen/Qwen3-0.6B"  # small\n'


@pytest.mark.parametrize(
    "args,expected",
    [
        (["-e", "WANDB_API_KEY=abc"], ["-e", f"WANDB_API_KEY={REDACTED}"]),
        (["-eHF_TOKEN=abc"], [f"-eHF_TOKEN={REDACTED}"]),
        (["--env-var=K=v=w"], [f"--env-var=K={REDACTED}"]),
        # A name read from the local environment, and an env file, are not secrets.
        (
            ["-e", "WANDB_API_KEY", "-e", "secrets.env"],
            ["-e", "WANDB_API_KEY", "-e", "secrets.env"],
        ),
        (["--custom-secrets", '{"A":"b"}'], ["--custom-secrets", REDACTED]),
        (["--header", "Authorization: Bearer x"], ["--header", f"Authorization: {REDACTED}"]),
        (
            ["--wandb-api-key", "abc", "--hf-token=xyz"],
            ["--wandb-api-key", REDACTED, f"--hf-token={REDACTED}"],
        ),
        # Where a secret lives is kept; so is an unrelated "tokens" count.
        (
            ["-k", "OPENAI_API_KEY", "--api-key-var", "K"],
            ["-k", "OPENAI_API_KEY", "--api-key-var", "K"],
        ),
        (["--max-tokens", "512", "-t", "64"], ["--max-tokens", "512", "-t", "64"]),
        (
            ["-a", '{"split":"test","api_key":"abc","nested":{"token":"t"}}'],
            ["-a", f'{{"split":"test","api_key":"{REDACTED}","nested":{{"token":"{REDACTED}"}}}}'],
        ),
        (["-a", '{"split": "test"}'], ["-a", '{"split": "test"}']),
        (["note", "sk-abcdefghijklmnopqrstuvwx"], ["note", REDACTED]),
        (["--", "-e", "K=v"], ["--", "-e", "K=v"]),
    ],
)
def test_scrub_command_args(args, expected):
    assert scrub_command_args(args) == expected


def test_launch_command_is_prime_and_what_the_user_typed():
    argv = ["/usr/local/bin/prime", "train", "my config.toml", "-e", "K=v"]
    assert launch_command(argv) == f"prime train 'my config.toml' -e 'K={REDACTED}'"


def test_overlong_command_is_left_out():
    assert launch_command(["prime", "x" * MAX_COMMAND_CHARS]) is None


def test_snapshot_keeps_files_byte_for_byte(tmp_path: Path):
    path = tmp_path / "rl.toml"
    path.write_text(TOML)

    snapshot, warning = build_config_snapshot([str(path)], ["prime", "train", str(path)])

    assert warning is None
    assert snapshot == {
        "files": [{"path": str(path), "content": TOML}],
        "command": f"prime train {path}",
    }


def test_oversized_files_are_left_out_but_the_command_is_kept(tmp_path: Path):
    path = tmp_path / "big.toml"
    path.write_text("x" * (MAX_CONFIG_FILES_BYTES + 1))

    snapshot, warning = build_config_snapshot([str(path)], ["prime", "train", "big.toml"])

    assert snapshot == {"files": [], "command": "prime train big.toml"}
    assert warning and "KiB" in warning


def test_unreadable_file_is_left_out(tmp_path: Path):
    path = tmp_path / "bad.toml"
    path.write_bytes(b"\xff\xfe")

    snapshot, warning = build_config_snapshot([str(path)], ["prime"])

    assert snapshot["files"] == []
    assert warning and "UTF-8" in warning


def _capture_train(monkeypatch) -> list[dict[str, Any]]:
    from prime_cli.api.training import HostedTrainingRunResponse

    captured: list[dict[str, Any]] = []

    def create_run(self, payload):
        captured.append(payload)
        return HostedTrainingRunResponse(run_id="r1", token_value="t")

    monkeypatch.setattr("prime_cli.api.training.HostedTrainingClient.create_run", create_run)
    monkeypatch.setattr(
        "prime_cli.api.training.HostedTrainingClient.prepare_local_validation",
        lambda self, payload: {"mode": payload["mode"], "submission_supported": True},
    )
    monkeypatch.setattr(
        "prime_cli.utils.training_compiler.compile_locally",
        lambda raw, preparation: {"mode": preparation["mode"]},
    )
    return captured


def _train_config(tmp_path: Path) -> Path:
    cfg = tmp_path / "rl.toml"
    cfg.write_text(TOML + "\n[deployment]\nnum_train_gpus = 1\nnum_infer_gpus = 1\n")
    return cfg


def test_train_sends_the_file_and_scrubbed_command(monkeypatch, tmp_path: Path):
    captured = _capture_train(monkeypatch)
    cfg = _train_config(tmp_path)
    args = ["train", str(cfg), "-e", "HF_TOKEN=hf-secret", "-y"]
    monkeypatch.setattr(sys, "argv", ["prime", *args])

    result = runner.invoke(app, args, env=TEST_ENV)

    assert result.exit_code == 0, result.output
    assert NOTICE in result.output.replace("\n", " ")
    assert captured[0]["configSnapshot"] == {
        "files": [{"path": str(cfg), "content": cfg.read_text()}],
        "command": f"prime train {cfg} -e 'HF_TOKEN={REDACTED}' -y",
    }
    assert "hf-secret" not in str(captured[0]["configSnapshot"])


def test_train_opt_out_sends_nothing(monkeypatch, tmp_path: Path):
    captured = _capture_train(monkeypatch)
    cfg = _train_config(tmp_path)

    result = runner.invoke(app, ["train", str(cfg), "-y", "--no-config-upload"], env=TEST_ENV)

    assert result.exit_code == 0, result.output
    assert "configSnapshot" not in captured[0]
    assert "Config tab" not in result.output


def _capture_hosted_eval(monkeypatch) -> list[dict[str, Any]]:
    captured: list[dict[str, Any]] = []

    def create(config, environment_ids=None, config_snapshot=None):
        captured.append({"environment_ids": environment_ids, "config_snapshot": config_snapshot})
        return {"evaluation_id": f"eval-{len(captured)}"}

    monkeypatch.setattr("prime_cli.commands.evals._create_hosted_evaluations", create)
    monkeypatch.setattr(
        "prime_cli.commands.evals._resolve_hosted_environment",
        lambda environment, env_dir_path=None, env_path=None: (f"owner/{environment}", environment),
    )
    return captured


def test_hosted_eval_sends_the_config_file_with_every_create(monkeypatch, tmp_path: Path):
    captured = _capture_hosted_eval(monkeypatch)
    cfg = tmp_path / "eval.toml"
    content = (
        "# two envs\n"
        'model = "openai/gpt-4.1-mini"\n'
        "num_examples = 3\n"
        "rollouts_per_example = 1\n\n"
        '[[eval]]\nenv_id = "gsm8k"\n\n'
        '[[eval]]\nenv_id = "math"\nnum_examples = 5  # differs, so a second request\n'
    )
    cfg.write_text(content)
    args = ["eval", "run", str(cfg), "--hosted", "--custom-secrets", '{"K":"v"}']
    monkeypatch.setattr(sys, "argv", ["prime", *args])

    result = runner.invoke(app, args, env=TEST_ENV)

    assert result.exit_code == 0, result.output
    assert len(captured) == 2
    expected = {
        "files": [{"path": str(cfg), "content": content}],
        "command": f"prime eval run {cfg} --hosted --custom-secrets '{REDACTED}'",
    }
    assert [call["config_snapshot"] for call in captured] == [expected, expected]


def test_hosted_eval_without_a_config_file_sends_only_the_command(monkeypatch):
    captured = _capture_hosted_eval(monkeypatch)
    args = ["eval", "run", "gsm8k", "--hosted", "-n", "2"]
    monkeypatch.setattr(sys, "argv", ["prime", *args])

    result = runner.invoke(app, args, env=TEST_ENV)

    assert result.exit_code == 0, result.output
    assert captured[0]["config_snapshot"] == {
        "files": [],
        "command": "prime eval run gsm8k --hosted -n 2",
    }


def test_hosted_eval_opt_out_sends_nothing(monkeypatch):
    captured = _capture_hosted_eval(monkeypatch)

    result = runner.invoke(
        app, ["eval", "run", "gsm8k", "--hosted", "--no-config-upload"], env=TEST_ENV
    )

    assert result.exit_code == 0, result.output
    assert captured[0]["config_snapshot"] is None


def _hosted_config() -> HostedEvalConfig:
    return HostedEvalConfig(
        environment_id="env-1",
        inference_model="openai/gpt-4.1-mini",
        num_examples=1,
        rollouts_per_example=1,
    )


def test_create_hosted_evaluations_passes_the_snapshot_to_a_current_sdk(monkeypatch):
    calls: list[dict[str, Any]] = []

    def create_hosted_evaluation(self, environment_ids, model, eval_config, **kwargs):
        calls.append(kwargs)
        return {"evaluation_id": "eval-1"}

    monkeypatch.setattr("prime_cli.commands.evals.APIClient", lambda: None)
    monkeypatch.setattr(EvalsClient, "__init__", lambda self, api_client: None)
    monkeypatch.setattr(EvalsClient, "create_hosted_evaluation", create_hosted_evaluation)

    _create_hosted_evaluations(_hosted_config(), config_snapshot={"files": []})

    assert calls == [{"name": None, "config_snapshot": {"files": []}}]


def test_create_hosted_evaluations_skips_the_snapshot_for_an_older_sdk(monkeypatch):
    calls: list[dict[str, Any]] = []

    def create_hosted_evaluation(
        self, environment_ids, model, eval_config, *, name=None, team_id=None
    ):
        calls.append({"name": name})
        return {"evaluation_id": "eval-1"}

    monkeypatch.setattr("prime_cli.commands.evals.APIClient", lambda: None)
    monkeypatch.setattr(EvalsClient, "__init__", lambda self, api_client: None)
    monkeypatch.setattr(EvalsClient, "create_hosted_evaluation", create_hosted_evaluation)

    _create_hosted_evaluations(_hosted_config(), config_snapshot={"files": []})

    assert calls == [{"name": None}]


def test_prime_evals_sends_the_snapshot():
    sent: list[dict[str, Any]] = []

    class Client:
        config = type("Config", (), {"team_id": None})()

        def request(self, method, path, json):
            sent.append(json)
            return {"evaluation_id": "eval-1"}

    evals = EvalsClient.__new__(EvalsClient)
    evals.client = Client()
    assert "config_snapshot" in inspect.signature(evals.create_hosted_evaluation).parameters

    evals.create_hosted_evaluation(["env-1"], "m", {}, config_snapshot={"files": []})
    evals.create_hosted_evaluation(["env-1"], "m", {})

    assert sent[0]["config_snapshot"] == {"files": []}
    assert "config_snapshot" not in sent[1]
