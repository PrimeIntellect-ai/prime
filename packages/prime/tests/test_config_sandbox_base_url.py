import json
from pathlib import Path
from typing import Any

import pytest
from prime_cli.core import Config
from prime_cli.main import app
from typer.testing import CliRunner

runner = CliRunner()

TEST_ENV = {
    "COLUMNS": "200",
    "LINES": "50",
    "PRIME_DISABLE_VERSION_CHECK": "1",
}


@pytest.fixture
def temp_home(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("PRIME_SANDBOX_BASE_URL", raising=False)
    monkeypatch.delenv("PRIME_SANDBOX_INGRESS_URL", raising=False)
    monkeypatch.setattr("prime_cli.main.check_for_update", lambda: (False, None))
    return tmp_path


def test_sandbox_base_url_is_saved_with_environment(temp_home: Path) -> None:
    config = Config()
    config.set_base_url("https://api.dev.example/api/v1")
    config.set_sandbox_base_url("https://sandbox.dev.example/api/v1")
    config.save_environment("dev")

    env_file = temp_home / ".prime" / "environments" / "dev.json"
    env_config = json.loads(env_file.read_text())

    assert config.sandbox_base_url == "https://sandbox.dev.example"
    assert config.view()["sandbox_base_url"] == "https://sandbox.dev.example"
    assert env_config["base_url"] == "https://api.dev.example"
    assert env_config["sandbox_base_url"] == "https://sandbox.dev.example"

    assert config.load_environment("production") is True
    assert config.sandbox_base_url is None

    assert config.load_environment("dev") is True
    assert config.base_url == "https://api.dev.example"
    assert config.sandbox_base_url == "https://sandbox.dev.example"


def test_sandbox_base_url_env_var_overrides_saved_config(
    temp_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = Config()
    config.set_sandbox_base_url("https://saved-sandbox.example/api/v1")

    monkeypatch.setenv("PRIME_SANDBOX_BASE_URL", "https://env-sandbox.example/api/v1")

    assert config.sandbox_base_url == "https://env-sandbox.example"
    assert config.view()["sandbox_base_url"] == "https://env-sandbox.example"


def test_sandbox_ingress_env_var_is_supported(
    temp_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = Config()
    config.set_sandbox_base_url("https://saved-sandbox.example/api/v1")

    monkeypatch.setenv("PRIME_SANDBOX_INGRESS_URL", "https://ingress-sandbox.example/api/v1")

    assert config.sandbox_base_url == "https://ingress-sandbox.example"


def test_set_sandbox_base_url_command_updates_config_view(temp_home: Path) -> None:
    set_result = runner.invoke(
        app,
        ["config", "set-sandbox-base-url", "https://sandbox.dev.example/api/v1"],
        env=TEST_ENV,
    )

    assert set_result.exit_code == 0, set_result.output

    view_result = runner.invoke(app, ["config", "view"], env=TEST_ENV)

    assert view_result.exit_code == 0, view_result.output
    assert "Sandbox Base URL" in view_result.output
    assert "https://sandbox.dev.example" in view_result.output


def test_set_sandbox_base_url_command_updates_active_environment_file(temp_home: Path) -> None:
    config = Config()
    config.set_base_url("https://api.dev.example")
    config.save_environment("dev")
    config.load_environment("dev")

    result = runner.invoke(
        app,
        ["config", "set-sandbox-base-url", "https://sandbox.dev.example/api/v1"],
        env=TEST_ENV,
    )

    assert result.exit_code == 0, result.output

    env_file = temp_home / ".prime" / "environments" / "dev.json"
    env_config = json.loads(env_file.read_text())

    assert env_config["sandbox_base_url"] == "https://sandbox.dev.example"

    reloaded = Config()
    assert reloaded.load_environment("production") is True
    assert reloaded.load_environment("dev") is True
    assert reloaded.sandbox_base_url == "https://sandbox.dev.example"
