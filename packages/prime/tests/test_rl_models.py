"""Tests for `prime rl models` — focused on price column rendering."""

import json
from typing import Any, Dict, List

import pytest
from prime_cli.commands.rl import _model_name_sort_key
from prime_cli.main import app
from prime_cli.utils.formatters import strip_ansi
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

    result = CliRunner().invoke(app, ["train", "models", "--output", "json"])

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


def _promo_payload() -> Dict[str, Any]:
    return {
        "models": [
            {
                "name": "qwen/qwen3-8b",
                "atCapacity": False,
                "trainingPricePerMtok": 0.5,
                "inferenceInputPricePerMtok": 1.0,
                "inferenceOutputPricePerMtok": 3.0,
                "effectiveTrainingPricePerMtok": 0.0,
                "effectiveInferenceInputPricePerMtok": 0.0,
                "effectiveInferenceOutputPricePerMtok": 0.0,
                "promoLabel": "Free RFT week",
            },
        ]
    }


def test_models_table_renders_promo_arrow_and_caption(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def mock_get(self: Any, endpoint: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
        return _promo_payload()

    monkeypatch.setattr("prime_cli.core.APIClient.get", mock_get)

    result = CliRunner().invoke(app, ["rl", "models"], env={"COLUMNS": "200"})

    assert result.exit_code == 0, result.output
    plain = strip_ansi(result.output)
    # Discounted cells render as "original → effective".
    assert "→" in plain
    assert "FREE" in plain
    assert "$0.5" in plain
    assert "$1" in plain
    assert "$3" in plain
    # Promo label rendered once below the table.
    assert plain.count("Free RFT week") == 1


def test_models_table_no_promo_when_effective_equals_original(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "models": [
            {
                "name": "qwen/qwen3-8b",
                "atCapacity": False,
                "trainingPricePerMtok": 0.5,
                "inferenceInputPricePerMtok": 1.0,
                "inferenceOutputPricePerMtok": 3.0,
                "effectiveTrainingPricePerMtok": 0.5,
                "effectiveInferenceInputPricePerMtok": 1.0,
                "effectiveInferenceOutputPricePerMtok": 3.0,
                "promoLabel": None,
            }
        ]
    }

    def mock_get(self: Any, endpoint: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
        return payload

    monkeypatch.setattr("prime_cli.core.APIClient.get", mock_get)

    result = CliRunner().invoke(app, ["rl", "models"], env={"COLUMNS": "200"})

    assert result.exit_code == 0, result.output
    plain = strip_ansi(result.output)
    assert "FREE" not in plain
    assert "$0.5" in plain


def test_models_zero_original_with_promo_does_not_render_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "models": [
            {
                "name": "qwen/qwen3-8b",
                "atCapacity": False,
                "trainingPricePerMtok": 0.0,
                "inferenceInputPricePerMtok": 0.0,
                "inferenceOutputPricePerMtok": 0.0,
                "effectiveTrainingPricePerMtok": 0.0,
                "effectiveInferenceInputPricePerMtok": 0.0,
                "effectiveInferenceOutputPricePerMtok": 0.0,
                "promoLabel": None,
            }
        ]
    }

    def mock_get(self: Any, endpoint: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
        return payload

    monkeypatch.setattr("prime_cli.core.APIClient.get", mock_get)

    result = CliRunner().invoke(app, ["rl", "models"], env={"COLUMNS": "200"})

    assert result.exit_code == 0, result.output
    plain = strip_ansi(result.output)
    assert "FREE" not in plain


def test_models_promo_label_deduplicated_across_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "models": [
            {
                "name": "model-a",
                "atCapacity": False,
                "trainingPricePerMtok": 0.5,
                "inferenceInputPricePerMtok": 1.0,
                "inferenceOutputPricePerMtok": 3.0,
                "effectiveTrainingPricePerMtok": 0.0,
                "effectiveInferenceInputPricePerMtok": 0.0,
                "effectiveInferenceOutputPricePerMtok": 0.0,
                "promoLabel": "shared promo",
            },
            {
                "name": "model-b",
                "atCapacity": False,
                "trainingPricePerMtok": 0.2,
                "inferenceInputPricePerMtok": 0.4,
                "inferenceOutputPricePerMtok": 0.6,
                "effectiveTrainingPricePerMtok": 0.0,
                "effectiveInferenceInputPricePerMtok": 0.0,
                "effectiveInferenceOutputPricePerMtok": 0.0,
                "promoLabel": "shared promo",
            },
        ]
    }

    def mock_get(self: Any, endpoint: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
        return payload

    monkeypatch.setattr("prime_cli.core.APIClient.get", mock_get)

    result = CliRunner().invoke(app, ["rl", "models"], env={"COLUMNS": "200"})

    assert result.exit_code == 0, result.output
    plain = strip_ansi(result.output)
    assert plain.count("shared promo") == 1


def test_models_table_renders_promo_with_list_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Post-swap backend: legacy fields hold effective price, list_* hold list price."""
    payload = {
        "models": [
            {
                "name": "qwen/qwen3-8b",
                "atCapacity": False,
                "trainingPricePerMtok": 0.0,
                "inferenceInputPricePerMtok": 0.0,
                "inferenceOutputPricePerMtok": 0.0,
                "listTrainingPricePerMtok": 0.5,
                "listInferenceInputPricePerMtok": 1.0,
                "listInferenceOutputPricePerMtok": 3.0,
                "effectiveTrainingPricePerMtok": 0.0,
                "effectiveInferenceInputPricePerMtok": 0.0,
                "effectiveInferenceOutputPricePerMtok": 0.0,
                "promoLabel": "Free RFT week",
            },
        ]
    }

    def mock_get(self: Any, endpoint: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
        return payload

    monkeypatch.setattr("prime_cli.core.APIClient.get", mock_get)

    result = CliRunner().invoke(app, ["rl", "models"], env={"COLUMNS": "200"})

    assert result.exit_code == 0, result.output
    plain = strip_ansi(result.output)
    assert "→" in plain
    assert "FREE" in plain
    assert "$0.5" in plain
    assert "$1" in plain
    assert "$3" in plain
    assert plain.count("Free RFT week") == 1


def test_models_json_includes_effective_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    def mock_get(self: Any, endpoint: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
        return _promo_payload()

    monkeypatch.setattr("prime_cli.core.APIClient.get", mock_get)

    result = CliRunner().invoke(app, ["train", "models", "--output", "json"])

    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["models"][0]["effective_training_price_per_mtok"] == 0.0
    assert data["models"][0]["effective_inference_input_price_per_mtok"] == 0.0
    assert data["models"][0]["effective_inference_output_price_per_mtok"] == 0.0
    assert data["models"][0]["promo_label"] == "Free RFT week"


def test_model_name_sort_key_orders_parameter_counts_numerically() -> None:
    models = [
        "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "Qwen/Qwen3-4B-Instruct-2507",
        "Qwen/Qwen3-4B-Thinking-2507",
        "Qwen/Qwen3.5-0.8B",
        "Qwen/Qwen3.5-122B-A10B",
        "Qwen/Qwen3.5-2B",
        "Qwen/Qwen3.5-35B-A3B",
        "Qwen/Qwen3.5-397B-A17B",
        "Qwen/Qwen3.5-4B",
        "Qwen/Qwen3.5-9B",
        "meta-llama/Llama-3.2-1B-Instruct",
        "meta-llama/Llama-3.2-3B-Instruct",
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16",
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
    ]

    assert sorted(models, key=_model_name_sort_key) == [
        "Qwen/Qwen3-4B-Instruct-2507",
        "Qwen/Qwen3-4B-Thinking-2507",
        "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "Qwen/Qwen3.5-0.8B",
        "Qwen/Qwen3.5-2B",
        "Qwen/Qwen3.5-4B",
        "Qwen/Qwen3.5-9B",
        "Qwen/Qwen3.5-35B-A3B",
        "Qwen/Qwen3.5-122B-A10B",
        "Qwen/Qwen3.5-397B-A17B",
        "meta-llama/Llama-3.2-1B-Instruct",
        "meta-llama/Llama-3.2-3B-Instruct",
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16",
        "openai/gpt-oss-20b",
        "openai/gpt-oss-120b",
    ]


def test_model_name_sort_key_handles_active_params_case_insensitively() -> None:
    models = [
        "org/model-30B-A10b",
        "org/model-30b-a3B",
        "org/model-30B",
    ]

    assert sorted(models, key=_model_name_sort_key) == [
        "org/model-30b-a3B",
        "org/model-30B-A10b",
        "org/model-30B",
    ]
