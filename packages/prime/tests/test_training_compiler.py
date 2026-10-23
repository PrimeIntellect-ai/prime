import json
import os
import re
import signal
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from prime_cli.main import app
from prime_cli.utils import training_compiler
from typer.testing import CliRunner


@pytest.fixture
def fake_schema(monkeypatch):
    from types import ModuleType

    from pydantic import BaseModel, Field

    class Deployment(BaseModel):
        type: str = "multi_node"

    class Component(BaseModel):
        resume: dict | None = None
        new_option: dict = Field(default_factory=lambda: {"disabled": None})
        deployment: Deployment | None = None

    class FakeRL(BaseModel):
        deployment: Deployment = Field(default_factory=Deployment)
        trainer: Component = Field(default_factory=Component)
        orchestrator: Component = Field(default_factory=Component)
        inference: Component = Field(default_factory=Component)
        output_dir: str = "/data/outputs"
        clean: bool = False

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
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return modules["prime_rl.utils.config"]


@pytest.mark.parametrize("shared_serializer", [False, True])
def test_normalization_preserves_selected_schema_fields_without_fft(fake_schema, shared_serializer):
    from prime_cli.utils import training_schema_worker as worker

    calls = []
    if shared_serializer:

        def dump(config):
            calls.append(config)
            return {
                role: getattr(config, role).model_dump(mode="json")
                for role in ("trainer", "orchestrator", "inference")
            }

        fake_schema.dump_rl_components = dump
    artifact = worker.normalize({"config": {}, "mode": "rl", "schema_commit": "a" * 40})
    component = {"resume": None, "new_option": {"disabled": None}, "deployment": None}
    assert artifact == {
        "contract_version": 4,
        "mode": "rl",
        "schema_commit": "a" * 40,
        "deployment": {"type": "multi_node"},
        "components": {
            "trainer": component,
            "orchestrator": component,
            "inference": component
            if shared_serializer
            else {
                "resume": None,
                "new_option": {"disabled": None},
                "router": None,
            },
        },
    }
    assert len(calls) == (1 if shared_serializer else 0)


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"clean": True}, "clean = true is unsupported for hosted runs"),
        (
            {"inference": {"deployment": {"type": "disaggregated"}}},
            'inference.deployment type = "disaggregated" is unsupported',
        ),
    ],
)
def test_unsupported_hosted_rl_settings_are_rejected_locally(fake_schema, config, message):
    from prime_cli.utils import training_schema_worker as worker

    with pytest.raises(ValueError, match=message):
        worker.normalize({"config": config, "mode": "rl", "schema_commit": "a" * 40})


@pytest.mark.parametrize(
    ("config", "accepted"),
    [
        ({}, True),
        ({"deployment": {"num_infer_gpus": 0}}, False),
        ({"inference": None}, False),
        ({"deployment": {"num_infer_gpus": 0}, "trainer": {"data": {"fake": {}}}}, True),
        ({"inference": None, "trainer": {"data": {"fake": {}}}}, True),
        ({"deployment": {"type": "multi_node", "num_infer_gpus": 0}}, True),
    ],
)
def test_hosted_single_node_rl_without_inference_needs_fake_data(fake_schema, config, accepted):
    from prime_cli.utils import training_schema_worker as worker
    from pydantic import BaseModel

    class Deployment(BaseModel):
        type: str = "single_node"
        num_infer_gpus: int = 1

    class Data(BaseModel):
        fake: dict | None = None

    class Trainer(BaseModel):
        data: Data = Data()

    class Component(BaseModel):
        pass

    class RL(BaseModel):
        deployment: Deployment = Deployment()
        trainer: Trainer = Trainer()
        orchestrator: Component = Component()
        inference: Component | None = Component()
        slurm: dict | None = None
        clean: bool = False

    sys.modules["prime_rl.configs.rl"].RLConfig = RL
    request = {"config": config, "mode": "rl", "schema_commit": "a" * 40}
    if accepted:
        assert (
            worker.normalize(request)["deployment"]
            == Deployment.model_validate(config.get("deployment", {})).model_dump()
        )
    else:
        with pytest.raises(ValueError, match="Hosted RL runs need inference GPUs"):
            worker.normalize(request)


def test_sft_online_eval_is_deferred_and_the_rest_is_validated(fake_schema):
    from prime_cli.utils import training_schema_worker as worker
    from pydantic import BaseModel, ConfigDict, model_validator

    class Strict(BaseModel):
        model_config = ConfigDict(extra="forbid")

    class Deployment(Strict):
        type: str = "single_node"

    class EvalNeedingPlugins(Strict):
        interval: int = 1
        source: list[dict] = []

        @model_validator(mode="after")
        def needs_source_plugins(self):
            raise ValueError("eval sources need environment plugins")

    class SFTSchemaRejectingOnlineEval(Strict):
        deployment: Deployment = Deployment()
        optim: dict = {}
        eval: EvalNeedingPlugins | None = None
        inference: dict | None = None

    sys.modules["prime_rl.configs.sft"].SFTConfig = SFTSchemaRejectingOnlineEval
    request = {
        "mode": "sft",
        "schema_commit": "a" * 40,
        "sft_online_eval_keys": sorted(training_compiler.SFT_ONLINE_EVAL_KEYS),
    }
    online_eval = {"eval": {"interval": 25, "source": [{"name": "reverse-text"}]}, "inference": {}}
    artifact = worker.normalize({**request, "config": {"optim": {"lr": 2e-5}, **online_eval}})
    assert artifact["components"] == {
        "trainer": {
            "deployment": {"type": "single_node"},
            "optim": {"lr": 2e-5},
            "eval": {"interval": 25, "source": [{"name": "reverse-text"}]},
            "inference": {},
        }
    }
    with pytest.raises(ValueError, match="lrr"):
        worker.normalize({**request, "config": {"lrr": 1, **online_eval}})
    online_eval["eval"]["intervall"] = online_eval["eval"].pop("interval")
    with pytest.raises(ValueError, match=r"eval\.intervall: Extra inputs"):
        worker.normalize({**request, "config": online_eval})


def test_worker_does_not_inherit_api_credentials_or_schema_path(monkeypatch):
    monkeypatch.setenv("PRIME_API_KEY", "secret")
    monkeypatch.setenv("HF_TOKEN", "secret")
    monkeypatch.setenv("PYTHONPATH", "wrong-schema")
    env = training_compiler._credential_free_environment()
    assert "PRIME_API_KEY" not in env
    assert "HF_TOKEN" not in env
    assert "PYTHONPATH" not in env


@pytest.mark.parametrize("stage", ["worker", "installer"])
def test_schema_subprocesses_keep_windows_user_directories(monkeypatch, stage):
    directories = {
        "USERPROFILE": r"C:\Users\trainer",
        "APPDATA": r"C:\Users\trainer\AppData\Roaming",
        "LOCALAPPDATA": r"C:\Users\trainer\AppData\Local",
    }
    secrets = ("PRIME_API_KEY", "HF_TOKEN", "PYTHONPATH", "SSH_AUTH_SOCK")
    monkeypatch.delenv("HOME", raising=False)
    for name, value in directories.items():
        monkeypatch.setenv(name, value)
    for name in secrets:
        monkeypatch.setenv(name, "must-not-reach-schema-process")
    names = [*directories, *secrets]
    command = [
        sys.executable,
        "-c",
        f"import json, os; print(json.dumps({{k: os.environ.get(k) for k in {names!r}}}))",
    ]
    result = training_compiler._run_process(
        command,
        timeout=10,
        env=training_compiler._installer_environment() if stage == "installer" else None,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == directories | {name: None for name in secrets}


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


def test_worker_rejects_provenance_mismatch_without_running_cwd_or_pythonpath_code(
    monkeypatch, tmp_path
):
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


@pytest.mark.parametrize(
    ("install", "commit"),
    [
        ({"vcs_info": {"commit_id": "a" * 40}}, "a" * 40),
        (
            {
                "url": f"https://github.com/PrimeIntellect-ai/prime-rl/archive/{'a' * 40}.tar.gz",
                "archive_info": {},
                "subdirectory": "packages/prime-rl-configs",
            },
            "a" * 40,
        ),
        (
            {
                "url": f"https://github.com/PrimeIntellect-ai/prime-rl/archive/{'a' * 40}.tar.gz",
                "archive_info": {},
                "subdirectory": "packages/other",
            },
            None,
        ),
        (
            {
                "url": f"https://github.com/someone/prime-rl/archive/{'a' * 40}.tar.gz",
                "archive_info": {},
                "subdirectory": "packages/prime-rl-configs",
            },
            None,
        ),
    ],
)
def test_worker_reads_schema_commit_from_git_or_archive_installs(install, commit):
    from prime_cli.utils import training_schema_worker as worker

    assert worker.installed_commit(install) == commit


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
            training_compiler._run(
                [sys.executable, "-I", "-c", parent], inherited_lock=None, timeout=2
            )
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


@pytest.mark.parametrize("mode", ["rl", "sft"])
@pytest.mark.parametrize("failure", [None, "prepare", "schema"])
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
    preparation = {"mode": mode, "submission_supported": True}
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

    def compile(config, prepared, **kwargs):
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


@pytest.mark.parametrize(
    ("case", "mode", "extra", "exit_code", "normalized"),
    [
        ("supported", "rl", {}, 0, True),
        ("unsupported", "rl", {}, 0, False),
        ("http_404", "rl", {}, 0, False),
        ("http_405", "rl", {}, 0, False),
        ("http_500", "rl", {}, 1, False),
        ("supported", "sft", {}, 0, True),
        ("supported", "sft", {"eval": {"interval": 10}}, 0, False),
        ("supported", "sft", {"inference": {}}, 0, False),
        ("supported", "sft", {"weight_broadcast": {"type": "nccl"}}, 0, False),
        ("schema_error", "rl", {}, 1, False),
    ],
)
def test_run_falls_back_to_raw_config_when_local_validation_is_unavailable(
    monkeypatch, tmp_path, case, mode, extra, exit_code, normalized
):
    import httpx
    import toml
    from prime_cli.api.training import HostedTrainingClient, HostedTrainingRunResponse
    from prime_cli.commands import rl
    from prime_cli.core.client import APIClient

    body = {"data": {"type": "fake"}} if mode == "sft" else {"trainer": {}, "orchestrator": {}}
    path = tmp_path / "run.toml"
    path.write_text(toml.dumps({"name": "review", **body, **extra}))
    monkeypatch.setattr(
        rl, "Config", lambda: SimpleNamespace(team_id="team", frontend_url="http://dashboard")
    )
    artifact = {"contract_version": 3, "mode": mode, "schema_commit": "a" * 40}
    submitted = []

    def api_client():
        client = APIClient(api_key="test")
        status = int(case.removeprefix("http_"))
        client.client = httpx.Client(
            transport=httpx.MockTransport(lambda request: httpx.Response(status, json={}))
        )
        return client

    def prepare(self, payload):
        return {"mode": mode, "submission_supported": case != "unsupported"}

    def compile(config, prepared, **kwargs):
        if case == "schema_error":
            raise training_compiler.CompilationError("invalid schema config")
        return artifact

    def create(self, payload):
        submitted.append(payload)
        return HostedTrainingRunResponse(run_id="run", token_value="secret")

    if case.startswith("http_"):
        monkeypatch.setattr(rl, "APIClient", api_client)
    else:
        monkeypatch.setattr(rl, "APIClient", lambda: object())
        monkeypatch.setattr(HostedTrainingClient, "prepare_local_validation", prepare)
    monkeypatch.setattr(training_compiler, "compile_locally", compile)
    monkeypatch.setattr(HostedTrainingClient, "create_run", create)
    result = CliRunner().invoke(
        app,
        ["train", str(path), "--sft" if mode == "sft" else "--fft", "--yes"],
        env={"PRIME_DISABLE_VERSION_CHECK": "1"},
    )
    assert result.exit_code == exit_code, result.output
    if exit_code:
        assert submitted == []
        return
    assert len(submitted) == 1
    assert submitted[0].get("normalizedConfig") == (artifact if normalized else None)
    assert ("Local validation is unavailable" in result.output) is (not normalized)


def test_installer_keeps_network_settings_and_rewrites_ssh_git_urls(monkeypatch):
    for name in ("HTTPS_PROXY", "https_proxy", "SSL_CERT_FILE", "PRIME_API_KEY", "UV_INDEX_URL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy:3128")
    monkeypatch.setenv("SSL_CERT_FILE", "/etc/ca.pem")
    monkeypatch.setenv("PRIME_API_KEY", "secret")
    monkeypatch.setenv("UV_INDEX_URL", "https://ambient.example/simple")
    monkeypatch.setenv("SSH_AUTH_SOCK", "/tmp/agent")
    monkeypatch.setenv("GIT_CONFIG_COUNT", "5")
    env = training_compiler._installer_environment()
    assert {k: v for k, v in env.items() if k.startswith(("GIT_", "HTTPS", "SSL"))} == {
        "HTTPS_PROXY": "http://proxy:3128",
        "SSL_CERT_FILE": "/etc/ca.pem",
        "GIT_CONFIG_COUNT": "2",
        "GIT_CONFIG_KEY_0": "url.https://github.com/.insteadOf",
        "GIT_CONFIG_VALUE_0": "git@github.com:",
        "GIT_CONFIG_KEY_1": "url.https://github.com/.insteadOf",
        "GIT_CONFIG_VALUE_1": "ssh://git@github.com/",
    }
    assert not {"PRIME_API_KEY", "UV_INDEX_URL", "SSH_AUTH_SOCK"} & env.keys()


def test_install_commands_use_installer_environment_and_worker_keeps_stripped_one(monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy:3128")
    seen = []

    def run_process(command, *, timeout, input=None, env=None, pass_fds=()):
        seen.append(env)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(training_compiler, "_run_process", run_process)
    training_compiler._run(["uv", "venv"], inherited_lock=None)
    assert seen == [training_compiler._installer_environment()]
    assert seen[0]["HTTPS_PROXY"] == "http://proxy:3128"
    assert "HTTPS_PROXY" not in training_compiler._credential_free_environment()


def test_cache_lock_is_released_when_holder_is_sigkilled(tmp_path):
    import subprocess

    lock = tmp_path / "schema.lockfile"
    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import sys, time\n"
            "from pathlib import Path\n"
            "from prime_cli.utils.training_compiler import _cache_lock\n"
            "with _cache_lock(Path(sys.argv[1]), 5):\n"
            "    print('held', flush=True)\n"
            "    time.sleep(60)\n",
            str(lock),
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout.readline() == "held\n"
        with pytest.raises(training_compiler.CompilationError, match="Timed out"):
            with training_compiler._cache_lock(lock, 0.3):
                pass
        holder.kill()
        holder.wait()
        started = time.monotonic()
        with training_compiler._cache_lock(lock, 5):
            pass
        assert time.monotonic() - started < 2
    finally:
        holder.kill()
        holder.stdout.close()


@pytest.mark.skipif(os.name == "nt", reason="Windows region locks cannot be inherited")
def test_installer_that_outlives_a_killed_cli_keeps_the_cache_locked(tmp_path):
    import subprocess

    lock = tmp_path / "schema.lockfile"
    installed = tmp_path / "installed"
    cli = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import sys\n"
            "from pathlib import Path\n"
            "from prime_cli.utils.training_compiler import _cache_lock, _run\n"
            "with _cache_lock(Path(sys.argv[1]), 5) as lock:\n"
            "    print('installing', flush=True)\n"
            "    installer = 'import pathlib, sys, time; time.sleep(2); '\n"
            "    installer += 'pathlib.Path(sys.argv[1]).touch()'\n"
            "    _run([sys.executable, '-c', installer, sys.argv[2]], inherited_lock=lock)\n",
            str(lock),
            str(installed),
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert cli.stdout.readline() == "installing\n"
        time.sleep(0.3)
        cli.kill()
        cli.wait()
        with pytest.raises(training_compiler.CompilationError, match="Timed out"):
            with training_compiler._cache_lock(lock, 0.5):
                pass
        with training_compiler._cache_lock(lock, 10):
            assert installed.exists()
    finally:
        cli.kill()
        cli.stdout.close()


DAY = 86400.0


@pytest.mark.parametrize(
    ("ages", "deleted"),
    [
        ({"current": 30, "a": 1, "b": 2, "c": 3, "d": 4, "e": 20}, ["d", "e"]),
        ({"current": 0, "a": 15, "b": 1}, ["a"]),
        ({"current": 0, "a": 13, "b": 1}, []),
    ],
)
def test_cache_policy_keeps_current_and_recent_environments(ages, deleted):
    now = 100 * DAY
    last_used = {key: now - age * DAY for key, age in ages.items()}
    assert training_compiler.envs_to_delete(last_used, "current", now) == deleted


def _fake_environment(cache, key, age, now):
    target = cache / key
    (target / "bin").mkdir(parents=True)
    (target / "bin/python").touch()
    for marker in (".ready", ".last-used"):
        (target / marker).touch()
        os.utime(target / marker, (now - age * DAY, now - age * DAY))
    return target


def test_cache_sweep_removes_stale_environments_but_not_busy_ones(tmp_path):
    cache = tmp_path / "training-schemas"
    now = time.time()
    current, stale, busy = "c" * 64, "a" * 64, "b" * 64
    for key, age in ((current, 0), (stale, 20), (busy, 20)):
        _fake_environment(cache, key, age, now)
    (cache / f"{'d' * 64}.trash-123").mkdir()
    with training_compiler._cache_lock(cache / f"{busy}.lockfile", 1, shared=True):
        training_compiler._sweep_cache(cache, current, now)
    assert sorted(path.name for path in cache.iterdir() if path.is_dir()) == [busy, current]
    assert (cache / f"{stale}.lockfile").exists()
    training_compiler._sweep_cache(cache, current, now)
    assert (cache / busy).is_dir(), "a second sweep within a day does nothing"
    os.utime(cache / ".last-sweep", (now - 2 * DAY, now - 2 * DAY))
    training_compiler._sweep_cache(cache, current, now)
    assert sorted(path.name for path in cache.iterdir() if path.is_dir()) == [current]


def test_worker_environment_is_shared_while_in_use_and_sweeps_afterwards(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    preparation = {
        "schema_install": {"python": "3.12", "requirements": ["schema"], "constraints": []}
    }
    cache, key, target, python = training_compiler._schema_paths(preparation["schema_install"])
    now = time.time()
    _fake_environment(cache, key, 30, now)
    stale = _fake_environment(cache, "a" * 64, 30, now)
    lock = cache / f"{key}.lockfile"
    with training_compiler._schema_environment(preparation) as used:
        assert used == python
        with training_compiler._cache_lock(lock, 0, shared=True):
            pass
        with pytest.raises(training_compiler.CompilationError, match="Timed out"):
            with training_compiler._cache_lock(lock, 0):
                pass
        assert stale.is_dir()
    assert now - (target / ".last-used").stat().st_mtime < 60
    assert not stale.exists()
    assert target.is_dir()


def test_hosted_config_strips_cli_keys_and_deprecated_settings_without_touching_the_toml():
    from prime_cli.commands import rl

    raw = {
        "name": "run",
        "image_tag": "main",
        "env_file": ".env",
        "volume": "ckpts",
        "orchestrator": {"batch_size": 8, "buffer": {"seed": 1}, "trajectory_strategy": "x"},
        "trainer": {"max_steps": 2},
    }
    original = json.loads(json.dumps(raw))
    config = rl._hosted_config(raw)
    assert config == {
        "name": "run",
        "orchestrator": {"batch_size": 8},
        "trainer": {"max_steps": 2},
    }
    assert rl._schema_input(config) == {
        "orchestrator": {"batch_size": 8},
        "trainer": {"max_steps": 2},
    }
    assert raw == original


def test_validate_drops_deprecated_settings_like_submission(monkeypatch, tmp_path):
    from prime_cli.api.training import HostedTrainingClient
    from prime_cli.commands import rl

    config = tmp_path / "rl.toml"
    config.write_text(
        'name = "run"\n[orchestrator]\nbatch_size = 8\n[orchestrator.buffer]\nseed = 1\n'
    )
    monkeypatch.setattr(rl, "Config", lambda: SimpleNamespace(team_id="team"))
    monkeypatch.setattr(
        HostedTrainingClient,
        "prepare_local_validation",
        lambda self, payload: {"mode": "rl", "runtime": {"schema_commit": "a" * 40}},
    )
    compiled = []

    def compile(config, preparation, **kwargs):
        compiled.append(config)
        return {"schema_commit": "a" * 40}

    monkeypatch.setattr(training_compiler, "compile_locally", compile)
    result = CliRunner().invoke(
        app, ["train", "validate", str(config)], env={"PRIME_DISABLE_VERSION_CHECK": "1"}
    )
    assert result.exit_code == 0, result.output
    assert compiled == [{"orchestrator": {"batch_size": 8}}]
    assert "`[buffer]` is deprecated and ignored" in result.output


def test_worker_imports_only_what_the_schema_interpreter_provides():
    import ast

    tree = ast.parse(
        Path(training_compiler.__file__).with_name("training_schema_worker.py").read_text()
    )
    imported = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0
    }
    relative = [
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.level
    ]
    assert relative == []
    assert imported - sys.stdlib_module_names == {"prime_rl", "pydantic", "pydantic_config"}


def test_deferred_eval_settings_are_checked_without_their_sources(fake_schema):
    from prime_cli.utils import training_schema_worker as worker
    from pydantic import BaseModel, ConfigDict, model_validator

    class Strict(BaseModel):
        model_config = ConfigDict(extra="forbid")

    class Eval(Strict):
        interval: int = 1
        source: list[dict] = []

        @model_validator(mode="after")
        def needs_source_plugins(self):
            if not self.source:
                raise ValueError("At least one eval source is required.")
            raise ValueError("taskset 'reverse-text' not found")

    class Orchestrator(Strict):
        eval: Eval | None = None

    class Deployment(Strict):
        type: str = "single_node"
        num_infer_gpus: int = 1

    class RL(Strict):
        deployment: Deployment = Deployment()
        trainer: Strict = Strict()
        orchestrator: Orchestrator = Orchestrator()
        inference: Strict | None = Strict()
        clean: bool = False

    sys.modules["prime_rl.configs.rl"].RLConfig = RL
    request = {"mode": "rl", "schema_commit": "a" * 40}
    source = [{"name": "reverse-text"}]
    artifact = worker.normalize(
        {**request, "config": {"orchestrator": {"eval": {"interval": 10, "source": source}}}}
    )
    assert artifact["components"]["orchestrator"] == {"eval": {"interval": 10, "source": source}}
    with pytest.raises(ValueError, match=r"orchestrator\.eval\.intervall: Extra inputs"):
        worker.normalize(
            {**request, "config": {"orchestrator": {"eval": {"intervall": 10, "source": source}}}}
        )
    with pytest.raises(ValueError, match=r"orchestrator\.eval: Input should be a valid dictionary"):
        worker.normalize({**request, "config": {"orchestrator": {"eval": [{"interval": 10}]}}})


def test_deferred_sources_get_their_group_defaults(fake_schema):
    from prime_cli.utils import training_schema_worker as worker
    from pydantic import BaseModel

    class Group(BaseModel):
        env: dict = {}

        @classmethod
        def inherit_group_defaults(cls, data):
            sources = [{"env": {**data["env"], **s["env"]}} for s in data["source"]]
            return {**data, "source": sources}

    class Orchestrator(BaseModel):
        train: Group | None = None
        eval: Group | None = None

    class Component(BaseModel):
        pass

    class Deployment(BaseModel):
        type: str = "multi_node"

    class RL(BaseModel):
        deployment: Deployment = Deployment()
        trainer: Component = Component()
        orchestrator: Orchestrator = Orchestrator()
        inference: Component = Component()
        clean: bool = False

    sys.modules["prime_rl.configs.rl"].RLConfig = RL
    config = {
        "orchestrator": {
            "train": {"env": {"retries": 2}, "source": [{"env": {"id": "a"}}]},
            "eval": {"env": {"timeout": 1}, "source": [{"env": {"id": "b", "timeout": 5}}]},
        }
    }
    artifact = worker.normalize({"config": config, "mode": "rl", "schema_commit": "a" * 40})
    assert artifact["components"]["orchestrator"] == {
        "train": {"env": {"retries": 2}, "source": [{"env": {"retries": 2, "id": "a"}}]},
        "eval": {"env": {"timeout": 1}, "source": [{"env": {"id": "b", "timeout": 5}}]},
    }


def test_validate_pr_overrides_config_source_ref_like_submission(monkeypatch, tmp_path):
    from prime_cli.api.training import HostedTrainingClient
    from prime_cli.commands import rl

    config = tmp_path / "rl.toml"
    config.write_text('source_ref = "dev"\n[trainer]\nmax_steps = 2\n')
    monkeypatch.setattr(rl, "Config", lambda: SimpleNamespace(team_id="team"))
    monkeypatch.setattr(rl, "resolve_pull_request_head", lambda number: "b" * 40)
    seen = []

    def prepare(self, payload):
        seen.append(payload)
        return {"mode": "rl", "runtime": {"schema_commit": "b" * 40}}

    monkeypatch.setattr(HostedTrainingClient, "prepare_local_validation", prepare)
    monkeypatch.setattr(
        training_compiler, "compile_locally", lambda *a, **kw: {"schema_commit": "b" * 40}
    )
    env = {"PRIME_DISABLE_VERSION_CHECK": "1"}
    result = CliRunner().invoke(app, ["train", "validate", str(config), "--pr", "42"], env=env)
    assert result.exit_code == 0, result.output
    assert seen == [{"mode": "rl", "teamId": "team", "sourceRef": "b" * 40}]
    result = CliRunner().invoke(
        app, ["train", "validate", str(config), "--pr", "42", "--ref", "main"], env=env
    )
    assert result.exit_code == 1
    assert "--ref and --pr are mutually exclusive." in result.output
    config.write_text('source_ref = "bad..ref"\n[trainer]\nmax_steps = 2\n')
    result = CliRunner().invoke(app, ["train", "validate", str(config), "--pr", "42"], env=env)
    assert result.exit_code == 1
    assert f"source_ref in {config}: 'bad..ref' is not a valid git ref" in "".join(
        result.output.split("\n")
    )


def test_malformed_preparation_is_an_api_error():
    from prime_cli.api.training import HostedTrainingClient
    from prime_cli.core import APIError

    client = HostedTrainingClient(SimpleNamespace(post=lambda *a, **kw: {"contract_version": 2}))
    with pytest.raises(APIError, match="Failed to parse local validation preparation"):
        client.prepare_local_validation({"mode": "rl"})


def test_validate_json_output_stays_parseable(monkeypatch, tmp_path):
    from prime_cli.api.training import HostedTrainingClient
    from prime_cli.commands import rl
    from prime_cli.core import APIError

    config = tmp_path / "rl.toml"
    config.write_text("[orchestrator]\nbatch_size = 8\n[orchestrator.buffer]\nseed = 1\n")
    monkeypatch.setattr(rl, "Config", lambda: SimpleNamespace(team_id="team"))
    monkeypatch.setattr(
        HostedTrainingClient,
        "prepare_local_validation",
        lambda self, payload: {"mode": "rl", "runtime": {"schema_commit": "a" * 40}},
    )
    monkeypatch.setattr(
        training_compiler, "compile_locally", lambda *a, **kw: {"schema_commit": "a" * 40}
    )
    env = {"PRIME_DISABLE_VERSION_CHECK": "1"}
    result = CliRunner().invoke(app, ["train", "validate", str(config), "-o", "json"], env=env)
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == {"schema_commit": "a" * 40}
    assert "`[buffer]` is deprecated" in result.stderr

    def merged(number):
        raise APIError(f"PR #{number} is already merged")

    monkeypatch.setattr(rl, "resolve_pull_request_head", merged)
    result = CliRunner().invoke(
        app, ["train", "validate", str(config), "--pr", "42", "-o", "json"], env=env
    )
    assert result.exit_code == 1
    assert json.loads(result.stdout) == {"error": "PR #42 is already merged"}


@pytest.mark.parametrize(
    ("toml_line", "message"),
    [
        ("image_tag = 1", "image_tag in {path} must be a string, got int."),
        ("gpu_type = 2", "gpu_type in {path} must be a string, got int."),
    ],
)
def test_validate_checks_toml_metadata_like_submission(monkeypatch, tmp_path, toml_line, message):
    from prime_cli.api.training import HostedTrainingClient

    def forbidden(*args, **kwargs):
        raise AssertionError("an invalid TOML value must fail before preparation")

    monkeypatch.setattr(HostedTrainingClient, "prepare_local_validation", forbidden)
    config = tmp_path / "rl.toml"
    config.write_text(f"{toml_line}\n[trainer]\nmax_steps = 2\n")
    result = CliRunner().invoke(
        app,
        ["train", "validate", str(config), "--image-tag", "main", "-o", "json"],
        env={"PRIME_DISABLE_VERSION_CHECK": "1"},
    )
    assert result.exit_code == 1
    assert json.loads(result.stdout) == {"error": message.format(path=config)}


@pytest.mark.parametrize("toml_volume,warned", [("", True), ('volume = "ckpts"\n', False)])
def test_validate_warns_when_volume_size_has_no_volume(monkeypatch, tmp_path, toml_volume, warned):
    from prime_cli.api.training import HostedTrainingClient
    from prime_cli.commands import rl

    config = tmp_path / "rl.toml"
    config.write_text(f'volume_size = "500Gi"\n{toml_volume}[trainer]\nmax_steps = 2\n')
    monkeypatch.setattr(rl, "Config", lambda: SimpleNamespace(team_id="team"))
    monkeypatch.setattr(
        HostedTrainingClient,
        "prepare_local_validation",
        lambda self, payload: {"mode": "rl", "runtime": {"schema_commit": "a" * 40}},
    )
    monkeypatch.setattr(
        training_compiler, "compile_locally", lambda *a, **kw: {"schema_commit": "a" * 40}
    )
    result = CliRunner().invoke(
        app,
        ["train", "validate", str(config), "-o", "json"],
        env={"PRIME_DISABLE_VERSION_CHECK": "1"},
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == {"schema_commit": "a" * 40}
    assert ("`volume_size` needs a volume" in result.stderr) is warned


@pytest.mark.skipif(
    not os.getenv("PRIME_TEST_SCHEMA_PYTHON"),
    reason="set PRIME_TEST_SCHEMA_PYTHON (a prime-rl-configs venv) and PRIME_TEST_SCHEMA_COMMIT",
)
def test_flags_resolve_against_the_real_schema_like_uv_run():
    """prime-rl flags go through the schema's own parser: values, the bare
    `--no-<optional>` form and kebab-case paths match `uv run sft`, and
    unknown or hosted-only paths are refused."""
    commit = os.environ["PRIME_TEST_SCHEMA_COMMIT"]
    preparation = {
        "contract_version": 1,
        "mode": "sft",
        "runtime": {"schema_commit": commit, "image_commit": commit},
    }
    config = {"model": {"name": "PrimeIntellect/Qwen3-0.6B"}, "ckpt": {}}

    def compile(*overrides):
        return training_compiler.compile_locally(
            config,
            preparation,
            schema_python=Path(os.environ["PRIME_TEST_SCHEMA_PYTHON"]),
            overrides=list(overrides),
        )["components"]["trainer"]

    trainer = compile("--max-steps=7", "--no-ckpt", "--data.batch-size", "16")
    assert (trainer["max_steps"], trainer["ckpt"], trainer["data"]["batch_size"]) == (7, None, 16)
    for flags, message in (
        (["--data.nope", "1"], "nope"),
        (["--k8s.namespace", "x"], "[k8s] is not supported"),
        (["--help"], "--help is not a config flag"),
    ):
        with pytest.raises(training_compiler.CompilationError, match=re.escape(message)):
            compile(*flags)
