import json
from typing import Any, Dict, Optional

import pytest
from prime_cli.main import app
from typer.testing import CliRunner

runner = CliRunner()


@pytest.fixture
def mock_env_inspect_api(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PRIME_API_KEY", "test-key")

    def mock_get(
        self: Any, endpoint: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        if endpoint.endswith("/@latest/inspect"):
            if params and params.get("path") == "README.md":
                return {
                    "data": {
                        "kind": "file",
                        "path": "README.md",
                        "version_id": "version-123",
                        "entry": {
                            "name": "README.md",
                            "path": "README.md",
                            "is_directory": False,
                            "size": 38,
                            "modified_at": None,
                            "content_hash": "hash-readme",
                        },
                        "content": "# literal [bold]README[/bold]",
                        "encoding": "utf-8",
                        "truncated": False,
                        "total_bytes": 38,
                    }
                }

            return {
                "data": {
                    "kind": "directory",
                    "path": "",
                    "version_id": "version-123",
                    "entries": [
                        {
                            "name": "src",
                            "path": "src",
                            "is_directory": True,
                            "size": None,
                            "modified_at": None,
                            "content_hash": None,
                        },
                        {
                            "name": "README.md",
                            "path": "README.md",
                            "is_directory": False,
                            "size": 38,
                            "modified_at": None,
                            "content_hash": "hash-readme",
                        },
                    ],
                    "content": None,
                    "encoding": None,
                    "truncated": False,
                    "total_bytes": None,
                }
            }

        if endpoint.endswith("/@0.2.0/inspect"):
            return {
                "data": {
                    "kind": "directory",
                    "path": "src",
                    "version_id": "version-200",
                    "entries": [
                        {
                            "name": "main.py",
                            "path": "src/main.py",
                            "is_directory": False,
                            "size": 21,
                            "modified_at": None,
                            "content_hash": "hash-main",
                        }
                    ],
                    "content": None,
                    "encoding": None,
                    "truncated": False,
                    "total_bytes": None,
                }
            }

        raise AssertionError(f"Unexpected endpoint: {endpoint} params={params}")

    monkeypatch.setattr("prime_cli.core.APIClient.get", mock_get)


class TestEnvInspect:
    def test_directory_listing_table_output(self, mock_env_inspect_api: None) -> None:
        result = runner.invoke(
            app,
            ["env", "inspect", "testuser/test-env"],
            env={"COLUMNS": "200", "LINES": "50"},
        )

        assert result.exit_code == 0, result.output
        assert "testuser/test-env@latest" in result.output
        assert "(path:" in result.output
        assert "README.md" in result.output
        assert "src" in result.output
        assert (
            "Inspect a file with: prime env inspect testuser/test-env@latest README.md"
            in result.output
        )

    def test_file_output_preserves_literal_content(self, mock_env_inspect_api: None) -> None:
        result = runner.invoke(
            app,
            ["env", "inspect", "testuser/test-env", "README.md"],
        )

        assert result.exit_code == 0, result.output
        assert "README.md" in result.output
        assert "# literal [bold]README[/bold]" in result.output

    def test_json_output(self, mock_env_inspect_api: None) -> None:
        result = runner.invoke(
            app,
            ["env", "inspect", "testuser/test-env", "--output", "json"],
        )

        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["kind"] == "directory"
        assert payload["entries"][0]["path"] == "src"

    def test_env_id_version_overrides_flag(self, mock_env_inspect_api: None) -> None:
        result = runner.invoke(
            app,
            ["env", "inspect", "testuser/test-env@0.2.0", "src", "--version", "latest"],
            env={"COLUMNS": "200", "LINES": "50"},
        )

        assert result.exit_code == 0, result.output
        assert "testuser/test-env@0.2.0" in result.output
        assert "(path: src)" in result.output
        assert "src/main.py" in result.output
