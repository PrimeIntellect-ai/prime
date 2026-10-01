import json
from pathlib import Path

import pytest
from prime_cli.commands.config import validate_team_id
from prime_cli.core import Config
from prime_cli.main import app
from typer.testing import CliRunner

runner = CliRunner()


class TestTeamIdValidation:
    """Test team ID validation logic."""

    def test_valid_team_id(self) -> None:
        """Test that valid team IDs pass validation."""
        valid_id = "cmf0ohr9s0026ilerf3w68s6n"
        assert validate_team_id(valid_id) is True

    def test_invalid_team_id_with_uppercase(self) -> None:
        """Test that team IDs with uppercase letters are invalid (CUID v1 is lowercase only)."""
        invalid_id = "CMF0OHR9S0026ILERF3W68S6N"
        assert validate_team_id(invalid_id) is False

    def test_invalid_team_id_mixed_case(self) -> None:
        """Test that team IDs with mixed case are invalid (CUID v1 is lowercase only)."""
        invalid_id = "CmF0OhR9s0026IlErF3w68S6n"
        assert validate_team_id(invalid_id) is False

    def test_invalid_team_id_not_starting_with_c(self) -> None:
        """Test that team IDs not starting with 'c' are invalid (CUID v1 requirement)."""
        invalid_id = "amf0ohr9s0026ilerf3w68s6n"
        assert validate_team_id(invalid_id) is False

    def test_empty_string_is_valid(self) -> None:
        """Test that empty string is valid (personal account)."""
        assert validate_team_id("") is True

    def test_invalid_team_id_too_short(self) -> None:
        """Test that team IDs shorter than 25 characters are invalid."""
        invalid_id = "cmf0ohr9s0026ilerf3w68s6"
        assert validate_team_id(invalid_id) is False

    def test_invalid_team_id_too_long(self) -> None:
        """Test that team IDs longer than 25 characters are invalid."""
        invalid_id = "cmf0ohr9s0026ilerf3w68s6nn"
        assert validate_team_id(invalid_id) is False

    def test_invalid_team_id_with_special_chars(self) -> None:
        """Test that team IDs with special characters are invalid."""
        invalid_id = "cmf0ohr9s0026ilerf3w68s-n"
        assert validate_team_id(invalid_id) is False

    def test_invalid_team_id_with_spaces(self) -> None:
        """Test that team IDs with spaces are invalid."""
        invalid_id = "cmf0ohr9s0026ilerf3w68s n"
        assert validate_team_id(invalid_id) is False

    def test_invalid_team_id_word(self) -> None:
        """Test that regular words are invalid."""
        invalid_id = "intertwine"
        assert validate_team_id(invalid_id) is False

    def test_invalid_team_id_with_underscore(self) -> None:
        """Test that team IDs with underscores are invalid."""
        invalid_id = "cmf0ohr9s0026ilerf3w68s_n"
        assert validate_team_id(invalid_id) is False


class TestRunsLegacySamples:
    """`prime config set-runs-legacy-samples` writes the key prime-runs reads."""

    @pytest.fixture
    def config_file(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("PRIME_DISABLE_VERSION_CHECK", "1")
        monkeypatch.delenv("PRIME_CONTEXT", raising=False)
        monkeypatch.delenv("PRIME_RUNS_LEGACY_SAMPLES", raising=False)
        return tmp_path / ".prime" / "config.json"

    def test_defaults_to_prime_traces(self, config_file: Path) -> None:
        assert Config().runs_legacy_samples is False

    @pytest.mark.parametrize("value,expected", [("true", True), ("false", False)])
    def test_set_persists_the_flag(self, config_file: Path, value: str, expected: bool) -> None:
        result = runner.invoke(app, ["config", "set-runs-legacy-samples", value])

        assert result.exit_code == 0, result.output
        assert json.loads(config_file.read_text())["runs_legacy_samples"] is expected
        # Survives the load/save round trip other config commands make.
        Config().set_api_key("key")
        assert Config().runs_legacy_samples is expected

    def test_rejects_other_values(self, config_file: Path) -> None:
        result = runner.invoke(app, ["config", "set-runs-legacy-samples", "maybe"])

        assert result.exit_code == 1
        assert "true" in result.output and "false" in result.output

    def test_view_shows_the_setting(self, config_file: Path) -> None:
        runner.invoke(app, ["config", "set-runs-legacy-samples", "true"])

        result = runner.invoke(app, ["config", "view"])

        assert result.exit_code == 0, result.output
        assert "Runs Legacy Samples" in result.output
