"""Tests for `prime rl models` — focused on price column rendering."""

import json
from typing import Any, Dict, List

import pytest
from prime_cli.commands.rl import _model_name_sort_key
from prime_cli.core.client import NotFoundError
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


def _fft_models_payload() -> Dict[str, Any]:
    return {
        "models": [
            {
                "name": "meta-llama/Llama-3.1-8B-Instruct",
                "clusters": [
                    {
                        "clusterId": "cluster-a",
                        "clusterName": "athens",
                        "gpuType": "H200_141GB",
                        "cacheSyncedAt": "2026-06-01T10:15:00Z",
                    },
                    {
                        "clusterId": "cluster-b",
                        "clusterName": "berlin",
                        "gpuType": "H100_80GB",
                        "cacheSyncedAt": "2026-06-02T08:00:00Z",
                    },
                ],
            },
            {
                "name": "qwen/qwen3-8b",
                "clusters": [
                    {
                        "clusterId": "cluster-a",
                        "clusterName": "athens",
                        "gpuType": "H200_141GB",
                        "cacheSyncedAt": "2026-06-01T10:15:00Z",
                    }
                ],
            },
        ]
    }


def _mock_get_factory(
    calls: List[str],
    *,
    fft_payload: Dict[str, Any] | None = None,
):
    """Mock APIClient.get for the models command.

    By default returns an empty FFT list so tests that pre-date the FFT
    endpoint continue to exercise the LoRA-only rendering path. Pass
    ``fft_payload`` to opt into a populated FFT response.
    """

    def mock_get(self: Any, endpoint: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
        calls.append(endpoint)
        if endpoint == "/rft/models":
            return _models_payload()
        if endpoint == "/training/available-fft-models":
            return fft_payload if fft_payload is not None else {"models": []}
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
    # LoRA endpoint is always hit; FFT endpoint is polled every time so
    # the table shows up transparently when it starts returning data.
    assert calls == ["/rft/models", "/training/available-fft-models"]


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
        if endpoint == "/rft/models":
            return {"models": [{"name": "qwen/qwen3-8b", "atCapacity": False}]}
        if endpoint == "/training/available-fft-models":
            return {"models": []}
        raise AssertionError(f"Unexpected endpoint: {endpoint}")

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
        if endpoint == "/training/available-fft-models":
            return {"models": []}
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
    normalized = " ".join(plain.split())
    expected_footer = "Prices are per 1M tokens. All models support context windows of 64K tokens."
    assert expected_footer in normalized
    # Promo label rendered once below the table.
    assert plain.count("Free RFT week") == 1


def _lora_only_mock(payload: Dict[str, Any]):
    """Return a mock_get that serves ``payload`` for /rft/models and an
    empty FFT list for /training/available-fft-models — the shape most
    LoRA-focused tests want."""

    def mock_get(self: Any, endpoint: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
        if endpoint == "/rft/models":
            return payload
        if endpoint == "/training/available-fft-models":
            return {"models": []}
        raise AssertionError(f"Unexpected endpoint: {endpoint}")

    return mock_get


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

    monkeypatch.setattr("prime_cli.core.APIClient.get", _lora_only_mock(payload))

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

    monkeypatch.setattr("prime_cli.core.APIClient.get", _lora_only_mock(payload))

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

    monkeypatch.setattr("prime_cli.core.APIClient.get", _lora_only_mock(payload))

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

    monkeypatch.setattr("prime_cli.core.APIClient.get", _lora_only_mock(payload))

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
        if endpoint == "/training/available-fft-models":
            return {"models": []}
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


def test_models_command_renders_fft_section_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both LoRA and FFT tables render side by side when the FFT endpoint
    returns any results."""
    calls: List[str] = []
    monkeypatch.setattr(
        "prime_cli.core.APIClient.get",
        _mock_get_factory(calls, fft_payload=_fft_models_payload()),
    )

    result = CliRunner().invoke(app, ["train", "models"], env={"COLUMNS": "200"})

    assert result.exit_code == 0, result.output
    plain = strip_ansi(result.output)
    # LoRA table survives.
    assert "Hosted Training - Models" in plain
    assert "qwen/qwen3-8b" in plain
    # FFT table shows up too.
    assert "Full Finetuning" in plain
    assert "meta-llama/Llama-3.1-8B-Instruct" in plain
    # Model cached on two clusters + two GPU types.
    assert "H100_80GB" in plain
    assert "H200_141GB" in plain
    assert "athens" in plain
    assert "berlin" in plain
    # Both endpoints were hit.
    assert "/rft/models" in calls
    assert "/training/available-fft-models" in calls


def test_models_json_output_includes_available_fft_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "prime_cli.core.APIClient.get",
        _mock_get_factory([], fft_payload=_fft_models_payload()),
    )

    result = CliRunner().invoke(app, ["train", "models", "--output", "json"])

    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert [m["name"] for m in data["models"]] == [
        "qwen/qwen3-8b",
        "openai/gpt-oss-20b",
    ]
    fft = data["available_fft_models"]
    assert [m["name"] for m in fft] == [
        "meta-llama/Llama-3.1-8B-Instruct",
        "qwen/qwen3-8b",
    ]
    first = fft[0]
    assert [c["cluster_name"] for c in first["clusters"]] == ["athens", "berlin"]
    assert first["clusters"][0]["gpu_type"] == "H200_141GB"


def test_models_json_omits_fft_key_when_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """JSON output stays backwards compatible: no available_fft_models
    key when the FFT endpoint returns an empty list."""
    monkeypatch.setattr("prime_cli.core.APIClient.get", _mock_get_factory([]))

    result = CliRunner().invoke(app, ["train", "models", "--output", "json"])

    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert "models" in data
    assert "available_fft_models" not in data


def test_models_command_survives_fft_endpoint_404(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Older backends that haven't shipped the FFT endpoint yet should
    still get the LoRA listing rendered — the CLI must not crash."""

    def mock_get(self: Any, endpoint: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
        if endpoint == "/rft/models":
            return _models_payload()
        if endpoint == "/training/available-fft-models":
            raise NotFoundError("HTTP 404: available-fft-models not deployed on this backend")
        raise AssertionError(f"Unexpected endpoint: {endpoint}")

    monkeypatch.setattr("prime_cli.core.APIClient.get", mock_get)

    result = CliRunner().invoke(app, ["train", "models"], env={"COLUMNS": "200"})

    assert result.exit_code == 0, result.output
    plain = strip_ansi(result.output)
    assert "qwen/qwen3-8b" in plain
    assert "Full Finetuning" not in plain


def test_models_fft_only_suppresses_lora_section(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: List[str] = []

    def mock_get(self: Any, endpoint: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
        calls.append(endpoint)
        if endpoint == "/training/available-fft-models":
            return _fft_models_payload()
        raise AssertionError(f"Unexpected endpoint: {endpoint}")

    monkeypatch.setattr("prime_cli.core.APIClient.get", mock_get)

    result = CliRunner().invoke(app, ["train", "models", "--fft-only"], env={"COLUMNS": "200"})

    assert result.exit_code == 0, result.output
    plain = strip_ansi(result.output)
    assert "Full Finetuning" in plain
    assert "meta-llama/Llama-3.1-8B-Instruct" in plain
    # LoRA table title should not appear.
    assert "Hosted Training - Models" not in plain
    # Only the FFT endpoint was fetched.
    assert calls == ["/training/available-fft-models"]


def test_list_available_fft_models_returns_empty_on_404(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """API client method swallows 404 so `prime train models` can silently
    fall back to LoRA-only rendering on backends that haven't shipped the
    endpoint yet."""
    from prime_cli.api.training import HostedTrainingClient
    from prime_cli.core import APIClient

    def mock_get(self: Any, endpoint: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
        raise NotFoundError("HTTP 404: not found")

    monkeypatch.setattr("prime_cli.core.APIClient.get", mock_get)
    client = HostedTrainingClient(APIClient())
    assert client.list_available_fft_models(team_id=None) == []
