"""Tests for `prime rl models` — focused on price column rendering."""

import json
from typing import Any, Dict, List

import pytest
from prime_cli.main import app
from typer.testing import CliRunner


@pytest.fixture(autouse=True)
def _api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PRIME_API_KEY", "dummy")
    monkeypatch.setenv("PRIME_DISABLE_VERSION_CHECK", "1")


def _models_payload() -> Dict[str, Any]:
    return {
        "models": [
            {
                "name": "qwen/qwen3-8b",
                "atCapacity": False,
                "trainingPricePerMtok": 0.5,
                "inferenceInputPricePerMtok": 1.0,
                "inferenceOutputPricePerMtok": 3.0,
            },
            {
                "name": "openai/gpt-oss-20b",
                "atCapacity": True,
                "trainingPricePerMtok": None,
                "inferenceInputPricePerMtok": None,
                "inferenceOutputPricePerMtok": None,
            },
        ]
    }


def _mock_get_factory(calls: List[str]):
    def mock_get(self: Any, endpoint: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
        calls.append(endpoint)
        if endpoint == "/rft/models":
            return _models_payload()
        raise AssertionError(f"Unexpected endpoint: {endpoint}")

    return mock_get


def test_models_table_renders_pricing(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: List[str] = []
    monkeypatch.setattr("prime_cli.core.APIClient.get", _mock_get_factory(calls))

    result = CliRunner().invoke(app, ["rl", "models"], env={"COLUMNS": "200"})

    assert result.exit_code == 0, result.output
    assert "qwen/qwen3-8b" in result.output
    assert "$0.5" in result.output
    assert "$1" in result.output
    assert "$3" in result.output
    # Null pricing renders as a dash.
    assert "-" in result.output
    assert calls == ["/rft/models"]


def test_models_json_includes_pricing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("prime_cli.core.APIClient.get", _mock_get_factory([]))

    result = CliRunner().invoke(app, ["rl", "models", "--output", "json"])

    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["models"][0]["training_price_per_mtok"] == 0.5
    assert data["models"][0]["inference_input_price_per_mtok"] == 1.0
    assert data["models"][0]["inference_output_price_per_mtok"] == 3.0
    assert data["models"][1]["training_price_per_mtok"] is None


def test_models_handles_backend_without_pricing_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Older backends may not return the pricing fields at all."""

    def mock_get(self: Any, endpoint: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
        return {"models": [{"name": "qwen/qwen3-8b", "atCapacity": False}]}

    monkeypatch.setattr("prime_cli.core.APIClient.get", mock_get)

    result = CliRunner().invoke(app, ["rl", "models"], env={"COLUMNS": "200"})

    assert result.exit_code == 0, result.output
    assert "qwen/qwen3-8b" in result.output
