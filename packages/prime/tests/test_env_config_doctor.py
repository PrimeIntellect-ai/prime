import sys
from pathlib import Path
from typing import Any

import prime_cli.env_config as env_config
import pytest
from prime_cli.main import app
from prime_cli.verifiers_bridge import ResolvedEnvironment
from typer.testing import CliRunner

runner = CliRunner()


class FakeSurface:
    def __init__(self, env_name: str, resolved: bool = False) -> None:
        self.env_name = env_name
        self.resolved = resolved

    def to_text(self) -> str:
        mode = "resolved" if self.resolved else "template"
        return "\n".join(
            [
                f"Environment config: {self.env_name} ({mode})",
                "",
                "env.args",
                "  split: str = train",
                "",
                "env.taskset",
                "  dataset: str = examples.jsonl",
                "",
                "env.harness",
                "  max_turns: int = 10",
            ]
        )

    def to_toml(self, resolved: bool = False) -> str:
        split = '"eval"' if resolved else '"train"'
        return "\n".join(
            [
                "[env.args]",
                f"split = {split}",
                "",
                "[env.taskset]",
                'dataset = "examples.jsonl"',
                "",
                "[env.harness]",
                "max_turns = 10",
            ]
        )

    def model_dump(self, mode: str = "json") -> dict[str, Any]:
        return {
            "env": {
                "args": {"split": {"type": "str", "default": "train"}},
                "taskset": {"dataset": {"type": "str", "required": True}},
                "harness": {"max_turns": {"type": "int", "default": 10}},
            },
            "resolved": self.resolved,
        }


class FakeDiagnostic:
    def __init__(
        self,
        code: str,
        message: str,
        *,
        severity: str = "error",
        path: str | None = None,
    ) -> None:
        self.code = code
        self.message = message
        self.severity = severity
        self.path = path


@pytest.fixture(autouse=True)
def disable_version_check(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PRIME_DISABLE_VERSION_CHECK", "1")


def _resolved(env_name: str, env_path: Path | None = None) -> ResolvedEnvironment:
    return ResolvedEnvironment(
        original=env_name,
        env_name=env_name,
        install_mode="local" if env_path is not None else "none",
        env_display_id=env_name,
        local_env_path=env_path,
    )


def _install_fake_introspection(
    monkeypatch: pytest.MonkeyPatch,
    *,
    diagnostics: list[Any] | None = None,
    pyproject_diagnostics: list[Any] | None = None,
) -> None:
    diagnostics = diagnostics or []
    pyproject_diagnostics = pyproject_diagnostics or []

    class FakeAPI:
        @staticmethod
        def inspect_environment_config(
            env_id: str,
            config: object,
            resolved: bool = False,
            **_kwargs: object,
        ) -> FakeSurface:
            assert config is not None
            return FakeSurface(env_id, resolved=resolved)

        @staticmethod
        def diagnose_environment_config(**_kwargs: object) -> dict[str, Any]:
            return {"diagnostics": diagnostics}

        @staticmethod
        def diagnose_pyproject_config(**_kwargs: object) -> dict[str, Any]:
            return {"diagnostics": pyproject_diagnostics}

    monkeypatch.setattr(env_config, "_load_verifiers_config_introspection", lambda: FakeAPI)


def _install_config_stubs(monkeypatch: pytest.MonkeyPatch, env_name: str) -> None:
    _install_fake_introspection(monkeypatch)
    monkeypatch.setattr(
        env_config,
        "_resolve_for_eval",
        lambda env_ref, env_dir_path, *, messages_to_stderr: _resolved(env_name),
    )
    monkeypatch.setattr(env_config, "_env_config_instance", lambda raw=None: {"typed": True})


def test_env_config_text_output(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_config_stubs(monkeypatch, "tiny-env")

    result = runner.invoke(app, ["env", "config", "tiny-env"])

    assert result.exit_code == 0, result.output
    assert "Environment config: tiny-env (template)" in result.output
    assert "env.args" in result.output
    assert "env.taskset" in result.output
    assert "env.harness" in result.output


def test_env_config_json_stdout_is_machine_readable(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_config_stubs(monkeypatch, "tiny-env")

    result = runner.invoke(app, ["env", "config", "tiny-env", "--format", "json"])

    assert result.exit_code == 0, result.output
    payload = __import__("json").loads(result.output)
    assert payload["env"]["taskset"]["dataset"]["required"] is True
    assert payload["resolved"] is False


def test_env_config_toml_template_uses_supported_namespaces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_config_stubs(monkeypatch, "tiny-env")

    result = runner.invoke(app, ["env", "config", "tiny-env", "--format", "toml"])

    assert result.exit_code == 0, result.output
    assert "[env.args]" in result.output
    assert "[env.taskset]" in result.output
    assert "[env.harness]" in result.output
    assert "[tool.verifiers.harness]" not in result.output


def test_env_config_resolved_reaches_verifiers_renderer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_config_stubs(monkeypatch, "tiny-env")

    result = runner.invoke(app, ["env", "config", "tiny-env", "--format", "toml", "--resolved"])

    assert result.exit_code == 0, result.output
    assert 'split = "eval"' in result.output


def _write_env_package(tmp_path: Path, name: str, body: str) -> Path:
    env_path = tmp_path / name
    package_path = tmp_path / name.replace("-", "_")
    package_path.mkdir()
    (package_path / "__init__.py").write_text(body)
    env_path.mkdir()
    (env_path / "pyproject.toml").write_text('[project]\nname = "tiny-env"\n')
    return env_path


def _install_doctor_stubs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    env_name: str,
    env_path: Path,
    *,
    diagnostics: list[Any] | None = None,
    pyproject_diagnostics: list[Any] | None = None,
) -> None:
    _install_fake_introspection(
        monkeypatch,
        diagnostics=diagnostics,
        pyproject_diagnostics=pyproject_diagnostics,
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop(env_name.replace("-", "_"), None)
    monkeypatch.setattr(
        env_config,
        "_resolve_for_eval",
        lambda env_ref, env_dir_path, *, messages_to_stderr: _resolved(env_name, env_path),
    )
    monkeypatch.setattr(env_config, "_env_config_instance", lambda raw=None: {"typed": True})


def test_env_doctor_success_on_tiny_local_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    env_path = _write_env_package(
        tmp_path,
        "tiny-env",
        "def load_environment(config):\n    return object()\n",
    )
    _install_doctor_stubs(monkeypatch, tmp_path, "tiny-env", env_path)
    monkeypatch.setattr(env_config, "_load_environment_with_typed_config", lambda *args: object())

    result = runner.invoke(app, ["--plain", "env", "doctor", "tiny-env"])

    assert result.exit_code == 0, result.output
    assert "Environment resolves" in result.output
    assert "Package imports" in result.output
    assert "Config surface renders" in result.output
    assert "PASS" in result.output


def test_env_doctor_fails_on_import_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    env_path = _write_env_package(tmp_path, "broken-env", "raise RuntimeError('boom')\n")
    _install_doctor_stubs(monkeypatch, tmp_path, "broken-env", env_path)

    result = runner.invoke(app, ["--plain", "env", "doctor", "broken-env"])

    assert result.exit_code == 1
    assert "Package imports" in result.output
    assert "boom" in result.output


def test_env_doctor_fails_on_missing_load_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    env_path = _write_env_package(tmp_path, "missing-loader", "VALUE = 1\n")
    _install_doctor_stubs(monkeypatch, tmp_path, "missing-loader", env_path)

    result = runner.invoke(app, ["--plain", "env", "doctor", "missing-loader"])

    assert result.exit_code == 1
    assert "load_environment exists" in result.output
    assert "missing or not callable" in result.output


def test_env_doctor_reports_bad_pyproject_config_section(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    env_path = _write_env_package(
        tmp_path,
        "bad-section",
        "def load_environment(config):\n    return object()\n",
    )
    (env_path / "pyproject.toml").write_text(
        '[project]\nname = "bad-section"\n\n[tool.verifiers.harness]\nmax_turns = 1\n'
    )
    _install_doctor_stubs(
        monkeypatch,
        tmp_path,
        "bad-section",
        env_path,
        pyproject_diagnostics=[
            FakeDiagnostic(
                "unsupported-section",
                "Unsupported config section",
                path="[tool.verifiers.harness]",
            )
        ],
    )
    monkeypatch.setattr(env_config, "_load_environment_with_typed_config", lambda *args: object())

    result = runner.invoke(app, ["--plain", "env", "doctor", "bad-section"])

    assert result.exit_code == 1
    assert "unsupported-section" in result.output
    assert "[tool.verifiers.harness]: Unsupported config section" in result.output


def test_env_doctor_reports_missing_required_key(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    env_path = _write_env_package(
        tmp_path,
        "missing-key",
        "def load_environment(config):\n    return object()\n",
    )
    _install_doctor_stubs(
        monkeypatch,
        tmp_path,
        "missing-key",
        env_path,
        diagnostics=[
            FakeDiagnostic(
                "missing-key",
                "Required key dataset is missing",
                path="env.taskset.dataset",
            )
        ],
    )
    monkeypatch.setattr(env_config, "_load_environment_with_typed_config", lambda *args: object())

    result = runner.invoke(app, ["--plain", "env", "doctor", "missing-key"])

    assert result.exit_code == 1
    assert "missing-key" in result.output
    assert "env.taskset.dataset: Required key dataset is missing" in result.output


def test_env_doctor_smoke_runs_small_eval_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    env_path = _write_env_package(
        tmp_path,
        "smoke-env",
        "def load_environment(config):\n    return object()\n",
    )
    _install_doctor_stubs(monkeypatch, tmp_path, "smoke-env", env_path)
    monkeypatch.setattr(env_config, "_load_environment_with_typed_config", lambda *args: object())
    custom_env_dir = tmp_path / "custom-envs"
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        env_config,
        "_run_smoke_eval",
        lambda env_name, env_dir_path: calls.append((env_name, env_dir_path)),
    )

    result = runner.invoke(
        app,
        [
            "--plain",
            "env",
            "doctor",
            "smoke-env",
            "--smoke",
            "--env-dir-path",
            str(custom_env_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls == [("smoke-env", str(custom_env_dir))]
    assert "Smoke eval" in result.output
