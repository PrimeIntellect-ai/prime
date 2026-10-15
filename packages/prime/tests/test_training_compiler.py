import json
import os
import signal
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from prime_cli.main import app
from prime_cli.utils import training_compiler
from typer.testing import CliRunner


@pytest.mark.parametrize("shared_serializer", [False, True])
def test_normalization_preserves_selected_schema_fields_without_fft(monkeypatch, shared_serializer):
    from types import ModuleType

    from prime_cli.utils import training_schema_worker as worker
    from pydantic import BaseModel, Field

    class Deployment(BaseModel):
        type: str = "multi_node"

    class Component(BaseModel):
        resume: dict | None = None
        new_option: dict = Field(default_factory=lambda: {"disabled": None})

    class Run(BaseModel):
        name: str | None = None

    class FakeRL(BaseModel):
        deployment: Deployment = Field(default_factory=Deployment)
        trainer: Component = Field(default_factory=Component)
        orchestrator: Component = Field(default_factory=Component)
        inference: Component = Field(default_factory=Component)
        output_dir: str = "/data/outputs"
        clean: bool = False
        run: Run = Field(default_factory=Run)

        @property
        def run_dir(self):
            return str(Path(self.output_dir) / self.run.name)

    modules = {
        name: ModuleType(name)
        for name in (
            "prime_rl",
            "prime_rl.configs",
            "prime_rl.configs.rl",
            "prime_rl.configs.sft",
            "prime_rl.utils",
            "prime_rl.utils.config",
        )
    }
    modules["prime_rl.configs.rl"].RLConfig = FakeRL
    modules["prime_rl.configs.sft"].SFTConfig = FakeRL
    modules["prime_rl.utils"].config = modules["prime_rl.utils.config"]
    calls = []
    if shared_serializer:

        def dump(config):
            calls.append(config)
            return {
                role: getattr(config, role).model_dump(mode="json")
                for role in ("trainer", "orchestrator", "inference")
            }

        modules["prime_rl.utils.config"].dump_rl_components = dump
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    artifact = worker.normalize({"config": {}, "mode": "rl", "schema_commit": "a" * 40})
    assert artifact["contract_version"] == 4
    assert "hosted" not in artifact and "submission_supported" not in artifact
    assert artifact["components"]["trainer"]["new_option"] == {"disabled": None}
    assert artifact["components"]["trainer"]["resume"] is None
    assert "trainer" not in artifact["resolved"]
    assert "deployment" not in artifact["components"]["inference"]
    if shared_serializer:
        assert len(calls) == 1
    else:
        assert artifact["components"]["inference"]["router"] is None
    another = worker.normalize({"config": {}, "mode": "rl", "schema_commit": "a" * 40})
    assert artifact["run_dir"] != another["run_dir"]
    named = worker.normalize(
        {"config": {"run": {"name": "experiment"}}, "mode": "rl", "schema_commit": "a" * 40}
    )
    assert named["run_dir"] == "/data/outputs/experiment"


def test_worker_does_not_inherit_api_credentials_or_schema_path(monkeypatch):
    monkeypatch.setenv("PRIME_API_KEY", "secret")
    monkeypatch.setenv("HF_TOKEN", "secret")
    monkeypatch.setenv("PYTHONPATH", "wrong-schema")
    env = training_compiler._worker_environment()
    assert "PRIME_API_KEY" not in env
    assert "HF_TOKEN" not in env
    assert "PYTHONPATH" not in env


def test_artifact_runtime_comes_from_the_preparation(monkeypatch):
    monkeypatch.setattr(
        training_compiler,
        "_schema_request",
        lambda *args: {"artifact": {"contract_version": 4, "runtime": "forged"}},
    )
    result = training_compiler.compile_locally(
        {}, {"runtime": {"schema_commit": "a" * 40}, "submission_supported": True}
    )
    assert result == {"contract_version": 4, "runtime": {"schema_commit": "a" * 40}}


@pytest.mark.parametrize("k8s", [{"release": "mine"}, {"storage_mount": "/scratch"}])
def test_hosted_settings_in_the_config_are_rejected(k8s):
    from prime_cli.utils import training_schema_worker as worker

    with pytest.raises(ValueError, match=r"\[k8s\] is not supported for hosted runs"):
        worker.normalize({"config": {"k8s": k8s}, "mode": "rl", "schema_commit": "a" * 40})


def test_worker_provenance_mismatch_is_a_hard_error(monkeypatch, tmp_path):
    # Neither the working directory nor PYTHONPATH may execute before the
    # provenance check, even when they shadow a worker's stdlib imports.
    untrusted = tmp_path / "untrusted"
    untrusted.mkdir()
    (untrusted / "json.py").write_text('raise AssertionError("untrusted import")\n')
    monkeypatch.chdir(untrusted)
    monkeypatch.setenv("PYTHONPATH", str(untrusted))
    monkeypatch.setattr(
        training_compiler,
        "_WORKER",
        "from importlib import metadata; from types import SimpleNamespace\n"
        "metadata.distribution = lambda _: SimpleNamespace(read_text=lambda _: "
        + repr(json.dumps({"vcs_info": {"commit_id": "b" * 40}}))
        + ")\n"
        + training_compiler._WORKER,
    )
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
            schema_python=Path(sys.executable),
        )


@pytest.mark.skipif(os.name == "nt", reason="POSIX process-group cleanup")
@pytest.mark.parametrize("interrupted", [False, True])
def test_installer_stops_descendants_before_returning(monkeypatch, tmp_path, interrupted):
    started = tmp_path / "started"
    release = tmp_path / "release"
    marker = tmp_path / "written"
    child = (
        "import os, time; from pathlib import Path\n"
        f"Path({str(started)!r}).write_text(str(os.getpid()))\n"
        f"while not Path({str(release)!r}).exists(): time.sleep(0.01)\n"
        f"Path({str(marker)!r}).touch()\n"
    )
    parent = (
        "import subprocess, sys, time\n"
        f"subprocess.Popen([sys.executable, '-I', '-c', {child!r}])\n"
        "time.sleep(60)\n"
    )
    if interrupted:
        communicate = training_compiler.subprocess.Popen.communicate

        def interrupt_once(process, *args, **kwargs):
            monkeypatch.setattr(training_compiler.subprocess.Popen, "communicate", communicate)
            deadline = time.monotonic() + 5
            while not started.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            raise KeyboardInterrupt

        monkeypatch.setattr(training_compiler.subprocess.Popen, "communicate", interrupt_once)
    try:
        expected = KeyboardInterrupt if interrupted else training_compiler.CompilationError
        with pytest.raises(expected):
            training_compiler._run([sys.executable, "-I", "-c", parent], timeout=2)
        assert started.exists(), "descendant must start before testing cleanup"
        release.touch()
        time.sleep(0.2)
        assert not marker.exists(), "descendant survived installer cleanup"
    finally:
        if started.exists():
            try:
                os.kill(int(started.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_cache_reuses_complete_install_and_retries_failed_install(monkeypatch, tmp_path):
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
        assert "--no-config" in command, "schema setup must ignore ambient uv settings"
        if command[1] == "venv":
            target = Path(command[-1])
            (target / "bin").mkdir(parents=True)
            (target / "bin/python").touch()
        else:
            assert "--compile-bytecode" in command
            assert (
                Path(command[command.index("--constraints") + 1]).read_text() == "pydantic==2.9.2\n"
            )
            assert command[-1] == "schema@commit"
            if fail:
                raise training_compiler.CompilationError("install failed")

    monkeypatch.setattr(training_compiler, "_run", run)
    with pytest.raises(training_compiler.CompilationError, match="install failed"):
        training_compiler._schema_python(preparation)
    assert not list((tmp_path / "cache").rglob(".ready"))
    fail = False
    python = training_compiler._schema_python(preparation)
    assert python.exists()
    assert len(commands) == 4
    assert training_compiler._schema_python(preparation) == python
    assert len(commands) == 4


@pytest.mark.parametrize("failure", [None, "error", "interrupt"])
def test_existing_cache_bytecode_upgrade_retries_without_reinstalling(
    monkeypatch, tmp_path, failure
):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setattr(training_compiler.shutil, "which", lambda _: "/test/uv")
    preparation = {
        "schema_install": {"python": "3.12", "requirements": ["schema"], "constraints": []}
    }
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        if command[:2] == ["uv", "venv"]:
            target = Path(command[-1])
            executable = target / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            executable.parent.mkdir(parents=True)
            executable.touch()
        elif command[0] != "uv":
            assert command[1:4] == ["-I", "-B", "-c"]
            if failure == "error":
                raise training_compiler.CompilationError("bytecode failed")
            if failure == "interrupt":
                raise KeyboardInterrupt

    monkeypatch.setattr(training_compiler, "_run", run)
    python = training_compiler._schema_python(preparation)
    target = python.parent.parent
    ready = target / ".ready"
    marker = target / ".bytecode-v1.ready"
    assert marker.is_file()
    marker.unlink()  # Simulate an environment created before bytecode support.
    commands.clear()
    if failure:
        error = KeyboardInterrupt if failure == "interrupt" else training_compiler.CompilationError
        with pytest.raises(error):
            training_compiler._schema_python(preparation)
        assert ready.is_file() and python.is_file()
        assert not marker.exists()
        assert not list(target.parent.glob("*.lock"))
        failure = None
    assert training_compiler._schema_python(preparation) == python
    assert marker.is_file()
    assert all(command[0] == str(python) for command in commands)
    count = len(commands)
    assert training_compiler._schema_python(preparation) == python
    assert len(commands) == count


def test_bytecode_upgrade_compiles_without_importing_or_rejecting_unused_sources(tmp_path):
    import subprocess

    source = tmp_path / "module.py"
    source.write_text('raise AssertionError("must not import during bytecode upgrade")\n')
    (tmp_path / "unused.py").write_text("invalid Python!\n")
    script = (
        "import sysconfig\n"
        f"sysconfig.get_path = lambda _: {str(tmp_path)!r}\n" + training_compiler._BYTECODE_COMPILER
    )
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", script], capture_output=True, text=True, check=True
    )
    assert list((tmp_path / "__pycache__").glob("module.*.pyc"))
    assert not result.stdout
    assert not result.stderr


def test_preparation_client_rejects_unsupported_contract():
    from prime_cli.api.training import HostedTrainingClient
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        HostedTrainingClient(
            SimpleNamespace(post=lambda *a, **kw: {"contract_version": 2})
        ).prepare_local_validation({})


@pytest.mark.parametrize("mode", ["rl", "sft"])
@pytest.mark.parametrize("failure", [None, "prepare", "schema", "unsupported"])
@pytest.mark.parametrize("output", ["table", "json"])
def test_run_validates_before_volume_creation_and_submits_normalized_config(
    monkeypatch, tmp_path, mode, failure, output
):
    import toml
    from prime_cli.api.training import HostedTrainingClient, HostedTrainingRunResponse
    from prime_cli.commands import rl
    from prime_cli.core.client import APIError

    runtime = {"data": {"type": "fake"}} if mode == "sft" else {"trainer": {}, "orchestrator": {}}
    runtime["deployment"] = {"type": "single_node", "num_train_gpus": 1, "num_infer_gpus": 1}
    path = tmp_path / "run.toml"
    path.write_text(
        toml.dumps(
            {
                "name": "review",
                "image_tag": "from-config",
                "source_ref": "from-config",
                "volume": "research",
                "gpu_type": "H100",
                **runtime,
            }
        )
    )
    monkeypatch.setattr(
        rl, "Config", lambda: SimpleNamespace(team_id="team", frontend_url="http://dashboard")
    )
    monkeypatch.setattr(rl, "APIClient", lambda: object())
    calls = []
    preparation = {"mode": mode, "submission_supported": failure != "unsupported"}
    artifact = {
        "contract_version": 3,
        "mode": mode,
        "schema_commit": "a" * 40,
        "submission_supported": True,
    }

    def prepare(self, payload):
        calls.append("prepare")
        assert payload == {
            "mode": mode,
            "teamId": "team",
            "imageTag": "selected",
            "sourceRef": "branch",
        }
        if failure == "prepare":
            raise APIError("runtime unavailable")
        return preparation

    def compile(config, prepared):
        calls.append("schema")
        assert config == runtime
        assert prepared == preparation
        if failure == "schema":
            raise training_compiler.CompilationError("invalid schema config")
        return artifact

    def create(self, payload):
        calls.append("create")
        assert payload["normalizedConfig"] == artifact
        assert payload["imageTag"] == "selected"
        assert payload["sourceRef"] == "branch"
        assert payload["volume"] == "research"
        assert payload["gpuType"] == "H100"
        assert payload["name"] == "review"
        return HostedTrainingRunResponse(run_id="run", token_value="secret")

    monkeypatch.setattr(HostedTrainingClient, "prepare_local_validation", prepare)
    monkeypatch.setattr(training_compiler, "compile_locally", compile)
    monkeypatch.setattr(HostedTrainingClient, "create_run", create)
    monkeypatch.setattr(rl, "_ensure_volume", lambda *a, **kw: calls.append("volume"))
    result = CliRunner().invoke(
        app,
        [
            "train",
            str(path),
            "--sft" if mode == "sft" else "--fft",
            "--image-tag",
            "selected",
            "--ref",
            "branch",
            "--yes",
            "-o",
            output,
        ],
        env={"PRIME_DISABLE_VERSION_CHECK": "1"},
    )
    assert result.exit_code == (1 if failure else 0), result.output
    if failure:
        assert calls == (["prepare", "schema"] if failure == "schema" else ["prepare"])
    else:
        assert calls == ["prepare", "schema", "volume", "create"]
        if output == "json":
            assert json.loads(result.stdout) == {"run": {"runId": "run", "sourceRef": "branch"}}
            assert "secret" not in result.stdout


@pytest.mark.parametrize(
    ("runtime_config", "flags", "mode"),
    [
        ({"trainer": {"optim": {"lr": 0.1}}}, [], "rl"),
        ({"data": {"type": "fake"}}, [], "sft"),
        ({}, ["--sft"], "sft"),
    ],
)
def test_validate_prepares_runtime_without_sending_config_or_creating_run(
    monkeypatch, tmp_path, runtime_config, flags, mode
):
    import toml
    from prime_cli.api.training import HostedTrainingClient
    from prime_cli.commands import rl

    config = tmp_path / "rl.toml"
    config.write_text(toml.dumps({"image_tag": "branch", **runtime_config}))
    monkeypatch.setattr(rl, "Config", lambda: SimpleNamespace(team_id="team"))
    monkeypatch.setattr(rl, "APIClient", lambda: object())
    seen = []

    def prepare(self, payload):
        seen.append(payload)
        return {"contract_version": 1, "mode": mode, "runtime": {"schema_commit": "a" * 40}}

    monkeypatch.setattr(HostedTrainingClient, "prepare_local_validation", prepare)

    def no_run(*a, **kw):
        raise AssertionError("validation must not submit a run")

    monkeypatch.setattr(HostedTrainingClient, "create_run", no_run)

    def compile(config, preparation, **kw):
        assert config == runtime_config
        return {"schema_commit": "a" * 40, "submission_supported": False}

    monkeypatch.setattr(training_compiler, "compile_locally", compile)
    artifact = tmp_path / "artifact.json"
    result = CliRunner().invoke(
        app,
        [
            "train",
            "validate",
            str(config),
            "--artifact-out",
            str(artifact),
            "-o",
            "json",
            *flags,
        ],
        env={"PRIME_DISABLE_VERSION_CHECK": "1"},
    )
    assert result.exit_code == 0, result.output
    assert seen == [{"mode": mode, "teamId": "team", "imageTag": "branch"}]
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
            "-o",
            "json",
        ],
        env={"PRIME_DISABLE_VERSION_CHECK": "1"},
    )
    assert result.exit_code == 1
    assert "error" in json.loads(result.stdout)


@pytest.mark.parametrize("failure", [None, "prepare", "schema"])
@pytest.mark.parametrize("output", [None, "table", "json"])
@pytest.mark.parametrize("plain", [False, True])
def test_validate_writes_normalized_config_without_rendering_or_submitting(
    monkeypatch, tmp_path, failure, output, plain
):
    from prime_cli.api.training import HostedTrainingClient
    from prime_cli.commands import rl
    from prime_cli.core.client import APIError

    config = tmp_path / "rl.toml"
    config.write_text("[trainer]\nmax_steps = 2\n")
    artifact_path = tmp_path / "artifact.json"
    monkeypatch.setattr(rl, "Config", lambda: SimpleNamespace(team_id="team"))

    def forbidden(*args, **kwargs):
        raise AssertionError("validation must not render deployment values or create a run")

    monkeypatch.setattr(rl, "APIClient", lambda: SimpleNamespace(post=forbidden))

    def prepare(self, payload):
        assert payload == {"mode": "rl", "teamId": "team"}
        if failure == "prepare":
            raise APIError("runtime unavailable")
        return {
            "contract_version": 1,
            "mode": "rl",
            "runtime": {"schema_commit": "a" * 40},
        }

    monkeypatch.setattr(HostedTrainingClient, "prepare_local_validation", prepare)
    artifact = {"contract_version": 3, "schema_commit": "a" * 40, "submission_supported": False}
    compilations = []

    def compile(config, preparation, **kwargs):
        assert "compiler_path" not in kwargs
        assert config == {"trainer": {"max_steps": 2}}
        compilations.append(config)
        if failure == "schema":
            raise training_compiler.CompilationError("trainer: invalid config")
        return artifact

    monkeypatch.setattr(training_compiler, "compile_locally", compile)
    monkeypatch.setattr(HostedTrainingClient, "create_run", forbidden)
    result = CliRunner().invoke(
        app,
        [
            *(["--plain"] if plain else []),
            "train",
            "validate",
            str(config),
            "--artifact-out",
            str(artifact_path),
            *(["-o", output] if output else []),
        ],
        env={"PRIME_DISABLE_VERSION_CHECK": "1"},
    )
    assert result.exit_code == (1 if failure else 0), result.output
    assert len(compilations) == (0 if failure == "prepare" else 1)
    if output == "json":
        assert "Validating training config" not in result.stdout
        json.loads(result.stdout)
    elif plain:
        assert "Validating training config" in result.stdout
    if failure:
        assert not artifact_path.exists()
        expected = "runtime unavailable" if failure == "prepare" else "trainer: invalid config"
        if output == "json":
            assert expected in json.loads(result.stdout)["error"]
        else:
            assert expected in result.stdout
    else:
        assert json.loads(artifact_path.read_text()) == artifact
        if output == "json":
            assert json.loads(result.stdout) == artifact
