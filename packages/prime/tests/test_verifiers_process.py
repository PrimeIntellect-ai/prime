import json
import subprocess

import click
import pytest
from prime_cli.verifiers_process import (
    exec_eval_process,
    load_eval_artifacts,
    load_run_info,
)


class ExecCalled(Exception):
    pass


def test_exec_eval_process_resolves_and_reuses_run_id(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        if command[-1] == "--protocol-version":
            payload = {
                "protocol_version": 1,
                "trace_schema_version": 1,
                "operations": ["run", "resolve"],
            }
        else:
            payload = {
                "operation": "resolve",
                "protocol_version": 1,
                "trace_schema_version": 1,
                "run_id": "resolved-run-id",
                "output_dir": str(tmp_path / "outputs" / "resolved-run-id"),
                "resume": False,
                "config": {"model": "test-model"},
            }
        return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")

    def fake_exec(executable, command, env):
        calls.append(("exec", executable, command, env))
        raise ExecCalled

    monkeypatch.setattr(
        "prime_cli.verifiers_process.resolve_workspace_python", lambda _cwd: "python"
    )
    monkeypatch.setattr("prime_cli.verifiers_process.subprocess.run", fake_run)
    monkeypatch.setattr("prime_cli.verifiers_process.os.execvpe", fake_exec)

    with pytest.raises(ExecCalled):
        exec_eval_process(["gsm8k-v1", "--dry-run"], plain=True)

    assert calls[0][0] == ["python", "-m", "verifiers.v1.cli.eval.main", "--protocol-version"]
    assert calls[1][0] == [
        "python",
        "-m",
        "verifiers.v1.cli.eval.main",
        "resolve",
        "--format",
        "json",
        "gsm8k-v1",
        "--dry-run",
    ]
    assert calls[2][2] == [
        "python",
        "-m",
        "verifiers.v1.cli.eval.main",
        "run",
        "gsm8k-v1",
        "--dry-run",
        "--uuid",
        "resolved-run-id",
    ]
    assert calls[2][3]["NO_COLOR"] == "1"
    assert calls[2][3]["PYDANTIC_CONFIG_PLAIN"] == "1"


def test_exec_eval_process_does_not_resolve_help(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    calls = []
    protocol = {
        "protocol_version": 1,
        "trace_schema_version": 1,
        "operations": ["run", "resolve"],
    }

    def fake_run(command, **_kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, json.dumps(protocol), "")

    def fake_exec(_executable, command, _env):
        calls.append(command)
        raise ExecCalled

    monkeypatch.setattr(
        "prime_cli.verifiers_process.resolve_workspace_python", lambda _cwd: "python"
    )
    monkeypatch.setattr("prime_cli.verifiers_process.subprocess.run", fake_run)
    monkeypatch.setattr("prime_cli.verifiers_process.os.execvpe", fake_exec)

    with pytest.raises(ExecCalled):
        exec_eval_process(["--help"])

    assert len(calls) == 2
    assert calls[-1][-2:] == ["run", "--help"]


def test_exec_eval_process_rejects_invalid_resolve_response(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    responses = iter(
        [
            {
                "protocol_version": 1,
                "trace_schema_version": 1,
                "operations": ["run", "resolve"],
            },
            {
                "operation": "resolve",
                "protocol_version": 1,
                "trace_schema_version": 2,
                "run_id": "run-id",
                "output_dir": "outputs/run-id",
                "resume": False,
                "config": {},
            },
        ]
    )

    monkeypatch.setattr(
        "prime_cli.verifiers_process.resolve_workspace_python", lambda _cwd: "python"
    )
    monkeypatch.setattr(
        "prime_cli.verifiers_process.subprocess.run",
        lambda command, **_kwargs: subprocess.CompletedProcess(
            command, 0, json.dumps(next(responses)), ""
        ),
    )

    with pytest.raises(click.ClickException, match="Unsupported Verifiers resolve response"):
        exec_eval_process(["gsm8k-v1"])


def test_load_eval_artifacts_validates_run_info(tmp_path) -> None:
    (tmp_path / "run.json").write_text(
        json.dumps(
            {
                "schema": "verifiers.eval-run/v1",
                "protocol_version": 1,
                "trace_schema_version": 1,
                "run_id": "persisted-run-id",
            }
        )
    )
    (tmp_path / "config.toml").write_text('model = "test-model"\n')

    run_info, config = load_eval_artifacts(tmp_path)

    assert run_info == load_run_info(tmp_path)
    assert run_info["run_id"] == "persisted-run-id"
    assert config == {"model": "test-model"}


def test_load_run_info_rejects_schema_mismatch(tmp_path) -> None:
    (tmp_path / "run.json").write_text(
        json.dumps(
            {
                "schema": "verifiers.eval-run/v1",
                "protocol_version": 1,
                "trace_schema_version": 2,
                "run_id": "run-id",
            }
        )
    )

    with pytest.raises(ValueError, match="Invalid Verifiers eval run info"):
        load_run_info(tmp_path)
