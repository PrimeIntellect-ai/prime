import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from prime_cli.main import app
from prime_cli.utils import training_compiler
from typer.testing import CliRunner


def test_worker_does_not_inherit_api_credentials_or_schema_path(monkeypatch):
    monkeypatch.setenv("PRIME_API_KEY", "secret")
    monkeypatch.setenv("HF_TOKEN", "secret")
    monkeypatch.setenv("PYTHONPATH", "wrong-schema")
    env = training_compiler._worker_environment()
    assert "PRIME_API_KEY" not in env
    assert "HF_TOKEN" not in env
    assert "PYTHONPATH" not in env


def test_worker_provenance_mismatch_is_a_hard_error(tmp_path):
    compiler = tmp_path / "compiler"
    (compiler / "rl_k8s").mkdir(parents=True)
    (compiler / "rl_k8s" / "__init__.py").touch()
    (compiler / "rl_k8s" / "compiled.py").write_text(
        'def compile_config(*args, **kwargs):\n'
        '    raise AssertionError("must check provenance first")\n'
    )
    metadata = compiler / "prime_rl_configs-0.1.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text("Name: prime-rl-configs\nVersion: 0.1\n")
    (metadata / "direct_url.json").write_text(json.dumps({"vcs_info": {"commit_id": "b" * 40}}))
    with pytest.raises(training_compiler.CompilationError, match="schema provenance mismatch"):
        training_compiler.compile_locally(
            {},
            {
                "contract_version": 1,
                "mode": "rl",
                "runtime": {
                    "schema_commit": "a" * 40,
                    "image_commit": "a" * 40,
                },
            },
            compiler_path=compiler,
            schema_python=Path(sys.executable),
        )


def test_cache_reuses_complete_install_and_retries_failed_install(monkeypatch, tmp_path):
    compiler = tmp_path / "compiler"
    compiler.mkdir()
    (compiler / "pyproject.toml").write_text('[project]\nname = "test-compiler"\n')
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setattr(training_compiler.shutil, "which", lambda _: "/test/uv")
    preparation = {
        "schema_install": {
            "python": "3.12",
            "requirements": ["schema@commit"],
            "constraints": ["pydantic==2.9.2"],
        }
    }
    commands = []
    fail = True

    def run(command, **kwargs):
        commands.append(command)
        if command[1] == "venv":
            target = Path(command[-1])
            (target / "bin").mkdir(parents=True)
            (target / "bin/python").touch()
        else:
            assert (
                Path(command[command.index("--constraints") + 1]).read_text() == "pydantic==2.9.2\n"
            )
            if fail:
                raise training_compiler.CompilationError("install failed")

    monkeypatch.setattr(training_compiler, "_run", run)
    with pytest.raises(training_compiler.CompilationError, match="install failed"):
        training_compiler._schema_python(preparation, compiler)
    assert not list((tmp_path / "cache").rglob(".ready"))
    fail = False
    python = training_compiler._schema_python(preparation, compiler)
    assert python.exists()
    assert len(commands) == 4
    assert training_compiler._schema_python(preparation, compiler) == python
    assert len(commands) == 4


def test_preparation_client_rejects_unsupported_contract():
    from prime_cli.api.training import HostedTrainingClient
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        HostedTrainingClient(
            SimpleNamespace(post=lambda *a, **kw: {"contract_version": 2})
        ).prepare_local_validation({})


def test_validate_prepares_runtime_without_sending_config_or_creating_run(monkeypatch, tmp_path):
    from prime_cli.api.training import HostedTrainingClient
    from prime_cli.commands import rl

    config = tmp_path / "rl.toml"
    config.write_text('image_tag = "branch"\n[trainer.optim]\nlr = 0.1\n')
    monkeypatch.setattr(rl, "Config", lambda: SimpleNamespace(team_id="team"))
    monkeypatch.setattr(rl, "APIClient", lambda: object())
    seen = []

    def prepare(self, payload):
        seen.append(payload)
        return {"contract_version": 1, "mode": "rl", "runtime": {"schema_commit": "a" * 40}}

    monkeypatch.setattr(HostedTrainingClient, "prepare_local_validation", prepare)

    def no_run(*a, **kw):
        raise AssertionError("validation must not submit a run")

    monkeypatch.setattr(HostedTrainingClient, "create_run", no_run)

    def compile(config, preparation, **kw):
        assert config == {"trainer": {"optim": {"lr": 0.1}}}
        return {"schema_commit": "a" * 40, "submission_supported": False}

    monkeypatch.setattr(training_compiler, "compile_locally", compile)
    artifact = tmp_path / "artifact.json"
    result = CliRunner().invoke(
        app,
        [
            "train",
            "validate",
            str(config),
            "--compiler-path",
            str(tmp_path),
            "--artifact-out",
            str(artifact),
            "-o",
            "json",
        ],
        env={"PRIME_DISABLE_VERSION_CHECK": "1"},
    )
    assert result.exit_code == 0, result.output
    assert seen == [{"mode": "rl", "teamId": "team", "imageTag": "branch"}]
    assert json.loads(result.stdout) == json.loads(artifact.read_text())


def test_invalid_toml_fails_before_preparation(monkeypatch, tmp_path):
    from prime_cli.api.training import HostedTrainingClient

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid TOML must fail locally")

    monkeypatch.setattr(HostedTrainingClient, "prepare_local_validation", forbidden)
    config = tmp_path / "bad.toml"
    config.write_text("[invalid")
    result = CliRunner().invoke(
        app,
        [
            "train",
            "validate",
            str(config),
            "--compiler-path",
            str(tmp_path),
            "-o",
            "json",
        ],
        env={"PRIME_DISABLE_VERSION_CHECK": "1"},
    )
    assert result.exit_code == 1
    assert "error" in json.loads(result.stdout)
