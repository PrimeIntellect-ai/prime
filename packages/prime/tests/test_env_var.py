import json
from typing import Any, Dict, Optional

import pytest
from prime_cli.main import app
from prime_cli.utils import strip_ansi
from typer.testing import CliRunner

runner = CliRunner()


@pytest.fixture
def mock_env_var_api(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock the API client for environment variable endpoints."""
    monkeypatch.setenv("PRIME_API_KEY", "test-key")

    sample_env_detail = {
        "id": "env-uuid-12345",
        "name": "test-env",
        "owner": {"name": "testuser", "type": "user"},
    }

    sample_variables = [
        {
            "id": "var-id-1234567890",
            "name": "DEBUG",
            "value": "true",
            "description": "Enable debug mode",
            "createdAt": "2026-01-15T10:00:00Z",
            "updatedAt": "2026-01-15T10:00:00Z",
        },
        {
            "id": "var-id-0987654321",
            "name": "LOG_LEVEL",
            "value": "info",
            "description": None,
            "createdAt": "2026-01-10T08:00:00Z",
            "updatedAt": "2026-01-10T08:00:00Z",
        },
    ]

    def mock_get(
        self: Any, endpoint: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        if "/@latest" in endpoint:
            return {"data": sample_env_detail}
        elif "/variables" in endpoint:
            if endpoint.endswith("/variables"):
                return {"data": sample_variables}
            var_id = endpoint.split("/")[-1]
            for v in sample_variables:
                if v["id"] == var_id or v["id"].startswith(var_id):
                    return {"data": v}
            return {"data": sample_variables[0]}
        return {"data": {}}

    def mock_post(
        self: Any, endpoint: str, json: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        if "/variables" in endpoint:
            return {
                "data": {
                    "id": "new-var-id-12345",
                    "name": json.get("name") if json else "NEW_VAR",
                    "value": json.get("value") if json else "value",
                    "description": json.get("description") if json else None,
                    "createdAt": "2026-02-01T12:00:00Z",
                    "updatedAt": "2026-02-01T12:00:00Z",
                }
            }
        return {"data": {}}

    def mock_patch(
        self: Any, endpoint: str, json: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        if "/variables/" in endpoint:
            return {
                "data": {
                    "id": "var-id-1234567890",
                    "name": json.get("name", "DEBUG") if json else "DEBUG",
                    "value": json.get("value", "true") if json else "true",
                    "description": json.get("description") if json else None,
                    "createdAt": "2026-01-15T10:00:00Z",
                    "updatedAt": "2026-02-01T12:00:00Z",
                }
            }
        return {"data": {}}

    def mock_delete(self: Any, endpoint: str) -> None:
        pass

    monkeypatch.setattr("prime_cli.core.APIClient.get", mock_get)
    monkeypatch.setattr("prime_cli.core.APIClient.post", mock_post)
    monkeypatch.setattr("prime_cli.core.APIClient.patch", mock_patch)
    monkeypatch.setattr("prime_cli.core.APIClient.delete", mock_delete)


class TestEnvVarList:
    """Tests for the env var list command."""

    def test_list_variables_table_output(self, mock_env_var_api: None) -> None:
        """Test listing variables with table output."""
        result = runner.invoke(
            app,
            ["env", "var", "list", "testuser/test-env"],
            env={"COLUMNS": "200", "LINES": "50"},
        )

        assert result.exit_code == 0, f"Failed: {result.output}"
        assert "Variables for testuser/test-env" in result.output
        assert "DEBUG" in result.output
        assert "LOG_LEVEL" in result.output
        assert "true" in result.output

    def test_list_variables_json_output(self, mock_env_var_api: None) -> None:
        """Test listing variables with JSON output."""
        result = runner.invoke(app, ["env", "var", "list", "testuser/test-env", "-o", "json"])

        assert result.exit_code == 0, f"Failed: {result.output}"
        output = json.loads(result.output)
        assert "variables" in output
        assert len(output["variables"]) == 2
        assert output["variables"][0]["name"] == "DEBUG"

    def test_list_empty_variables(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test listing when no variables exist."""
        monkeypatch.setenv("PRIME_API_KEY", "test-key")

        def mock_get(
            self: Any, endpoint: str, params: Optional[Dict[str, Any]] = None
        ) -> Dict[str, Any]:
            if "/@latest" in endpoint:
                return {"data": {"id": "env-uuid-12345"}}
            return {"data": []}

        monkeypatch.setattr("prime_cli.core.APIClient.get", mock_get)

        result = runner.invoke(app, ["env", "var", "list", "testuser/test-env"])

        assert result.exit_code == 0
        assert "No variables found" in result.output


class TestEnvVarCreate:
    """Tests for the env var create command."""

    def test_create_variable_success(self, mock_env_var_api: None) -> None:
        """Test creating a variable successfully."""
        result = runner.invoke(
            app,
            ["env", "var", "create", "testuser/test-env", "-n", "NEW_VAR", "-v", "value"],
        )

        assert result.exit_code == 0, f"Failed: {result.output}"
        assert "Created variable 'NEW_VAR'" in result.output
        assert "ID:" in result.output

    def test_create_variable_with_description(self, mock_env_var_api: None) -> None:
        """Test creating a variable with a description."""
        result = runner.invoke(
            app,
            [
                "env",
                "var",
                "create",
                "testuser/test-env",
                "-n",
                "NEW_VAR",
                "-v",
                "value",
                "-d",
                "My description",
            ],
        )

        assert result.exit_code == 0, f"Failed: {result.output}"
        assert "Created variable 'NEW_VAR'" in result.output

    def test_create_variable_json_output(self, mock_env_var_api: None) -> None:
        """Test creating a variable with JSON output."""
        result = runner.invoke(
            app,
            [
                "env",
                "var",
                "create",
                "testuser/test-env",
                "-n",
                "NEW_VAR",
                "-v",
                "value",
                "-o",
                "json",
            ],
        )

        assert result.exit_code == 0, f"Failed: {result.output}"
        output = json.loads(result.output)
        assert output["name"] == "NEW_VAR"
        assert "id" in output

    def test_create_variable_interactive_cancel_name(self) -> None:
        """Test that create can be cancelled during name prompt."""
        result = runner.invoke(
            app,
            ["env", "var", "create", "testuser/test-env"],
            input="\n",
        )

        assert result.exit_code == 0
        assert "Cancelled" in result.output

    def test_create_variable_interactive_cancel_value(self, mock_env_var_api: None) -> None:
        """Test that create can be cancelled during value prompt."""
        result = runner.invoke(
            app,
            ["env", "var", "create", "testuser/test-env", "-n", "NEW_VAR"],
            input="\n",
        )

        assert result.exit_code == 0
        assert "Cancelled" in result.output

    def test_create_variable_interactive(self, mock_env_var_api: None) -> None:
        """Test creating a variable interactively."""
        result = runner.invoke(
            app,
            ["env", "var", "create", "testuser/test-env"],
            input="NEW_VAR\nmy-value\n",
        )

        assert result.exit_code == 0, f"Failed: {result.output}"
        assert "Created variable 'NEW_VAR'" in result.output

    def test_create_variable_invalid_name(self, mock_env_var_api: None) -> None:
        """Test that create rejects invalid variable names."""
        result = runner.invoke(
            app,
            ["env", "var", "create", "testuser/test-env", "-n", "lowercase", "-v", "value"],
        )

        assert result.exit_code != 0
        assert "Invalid variable name" in result.output


class TestEnvVarUpdate:
    """Tests for the env var update command."""

    def test_update_variable_name(self, mock_env_var_api: None) -> None:
        """Test updating a variable's name."""
        result = runner.invoke(
            app,
            [
                "env",
                "var",
                "update",
                "var-id-1234567890",
                "--env",
                "testuser/test-env",
                "-n",
                "RENAMED_VAR",
            ],
        )

        assert result.exit_code == 0, f"Failed: {result.output}"
        assert "Updated variable" in result.output

    def test_update_variable_value(self, mock_env_var_api: None) -> None:
        """Test updating a variable's value."""
        result = runner.invoke(
            app,
            [
                "env",
                "var",
                "update",
                "var-id-1234567890",
                "--env",
                "testuser/test-env",
                "-v",
                "new-value",
            ],
        )

        assert result.exit_code == 0, f"Failed: {result.output}"
        assert "Updated variable" in result.output

    def test_update_variable_no_changes(self, mock_env_var_api: None) -> None:
        """Test that update fails when no changes are provided."""
        result = runner.invoke(
            app,
            ["env", "var", "update", "var-id-1234567890", "--env", "testuser/test-env"],
        )

        assert result.exit_code == 1
        assert "At least one of --name, --value, or --description is required" in result.output

    def test_update_variable_json_output(self, mock_env_var_api: None) -> None:
        """Test updating a variable with JSON output."""
        result = runner.invoke(
            app,
            [
                "env",
                "var",
                "update",
                "var-id-1234567890",
                "--env",
                "testuser/test-env",
                "-n",
                "NEW_NAME",
                "-o",
                "json",
            ],
        )

        assert result.exit_code == 0, f"Failed: {result.output}"
        output = json.loads(result.output)
        assert "id" in output


class TestEnvVarDelete:
    """Tests for the env var delete command."""

    def test_delete_variable_with_confirm(self, mock_env_var_api: None) -> None:
        """Test deleting a variable with confirmation."""
        result = runner.invoke(
            app,
            ["env", "var", "delete", "var-id-1234567890", "--env", "testuser/test-env", "-y"],
        )

        assert result.exit_code == 0, f"Failed: {result.output}"
        assert "Variable deleted" in result.output

    def test_delete_variable_cancelled(self, mock_env_var_api: None) -> None:
        """Test cancelling variable deletion."""
        result = runner.invoke(
            app,
            ["env", "var", "delete", "var-id-1234567890", "--env", "testuser/test-env"],
            input="n\n",
        )

        assert result.exit_code == 0
        assert "Cancelled" in result.output


class TestEnvVarHelp:
    """Tests for help output."""

    def test_env_var_help(self) -> None:
        """Test that env var help works."""
        result = runner.invoke(app, ["env", "var", "--help"])

        assert result.exit_code == 0
        assert "Manage environment variables" in result.output
        assert "list" in result.output
        assert "create" in result.output
        assert "update" in result.output
        assert "delete" in result.output

    def test_env_var_list_help(self) -> None:
        """Test that env var list help works."""
        result = runner.invoke(app, ["env", "var", "list", "--help"])

        assert result.exit_code == 0
        output = strip_ansi(result.output)
        assert "--output" in output

    def test_env_var_create_help(self) -> None:
        """Test that env var create help works."""
        result = runner.invoke(app, ["env", "var", "create", "--help"])

        assert result.exit_code == 0
        output = strip_ansi(result.output)
        assert "--name" in output
        assert "--value" in output
        assert "--description" in output
