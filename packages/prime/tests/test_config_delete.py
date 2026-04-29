from pathlib import Path
from typing import Any

import pytest
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
    monkeypatch.setattr("prime_cli.main.check_for_update", lambda: (False, None))
    return tmp_path


def test_delete_saved_environment_removes_file(temp_home: Path) -> None:
    save_result = runner.invoke(app, ["config", "save", "staging"], env=TEST_ENV)

    assert save_result.exit_code == 0, save_result.output
    env_file = temp_home / ".prime" / "environments" / "staging.json"
    assert env_file.exists()

    delete_result = runner.invoke(app, ["config", "delete", "staging"], env=TEST_ENV)

    assert delete_result.exit_code == 0, delete_result.output
    assert "Deleted environment 'staging'!" in delete_result.output
    assert not env_file.exists()

    list_result = runner.invoke(app, ["config", "envs"], env=TEST_ENV)

    assert list_result.exit_code == 0, list_result.output
    assert "staging" not in list_result.output


def test_delete_reserved_production_environment_fails(temp_home: Path) -> None:
    result = runner.invoke(app, ["config", "delete", "production"], env=TEST_ENV)

    assert result.exit_code == 1, result.output
    assert "Cannot delete built-in environment 'production'" in result.output


def test_delete_unknown_environment_fails(temp_home: Path) -> None:
    result = runner.invoke(app, ["config", "delete", "missing"], env=TEST_ENV)

    assert result.exit_code == 1, result.output
    assert "Unknown environment: missing" in result.output


def test_delete_current_environment_requires_switching_first(temp_home: Path) -> None:
    save_result = runner.invoke(app, ["config", "save", "staging"], env=TEST_ENV)
    assert save_result.exit_code == 0, save_result.output

    use_result = runner.invoke(app, ["config", "use", "staging"], env=TEST_ENV)
    assert use_result.exit_code == 0, use_result.output

    result = runner.invoke(app, ["config", "delete", "staging"], env=TEST_ENV)

    assert result.exit_code == 1, result.output
    assert "Cannot delete currently active environment 'staging'" in result.output
    assert "prime config use production" in result.output
    assert (temp_home / ".prime" / "environments" / "staging.json").exists()


def test_delete_current_environment_normalizes_name_before_guard(temp_home: Path) -> None:
    save_result = runner.invoke(app, ["config", "save", "staging"], env=TEST_ENV)
    assert save_result.exit_code == 0, save_result.output

    use_result = runner.invoke(app, ["config", "use", "staging"], env=TEST_ENV)
    assert use_result.exit_code == 0, use_result.output

    result = runner.invoke(app, ["config", "delete", "STAGING"], env=TEST_ENV)

    assert result.exit_code == 1, result.output
    assert "Cannot delete currently active environment 'STAGING'" in result.output
    assert "prime config use production" in result.output
    assert (temp_home / ".prime" / "environments" / "staging.json").exists()
