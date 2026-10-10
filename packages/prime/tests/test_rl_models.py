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


def _fft_models_payload() -> dict[str, Any]:
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
    calls: list[str],
    *,
    fft_payload: dict[str, Any] | None = None,
):
    """Mock APIClient.get for the models command.

    By default returns an empty FFT list so tests that pre-date the FFT
    endpoint continue to exercise the LoRA-only rendering path. Pass
    ``fft_payload`` to opt into a populated FFT response.
    """

    def mock_get(self: Any, endpoint: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
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


def _lora_only_mock(payload: dict[str, Any]):
    """Return a mock_get that serves ``payload`` for /rft/models and an
    empty FFT list for /training/available-fft-models — the shape most
    LoRA-focused tests want."""

    def mock_get(self: Any, endpoint: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
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
    calls: list[str] = []
    monkeypatch.setattr(
        "prime_cli.core.APIClient.get",
        _mock_get_factory(calls, fft_payload=_fft_models_payload()),
    )

    result = CliRunner().invoke(app, ["train", "models"], env={"COLUMNS": "200"})

    assert result.exit_code == 0, result.output
    plain = strip_ansi(result.output)
    # LoRA table survives.
    assert "LoRA" in plain
    assert "qwen/qwen3-8b" in plain
    # FFT table shows up too.
    assert "pre-cached" in plain
    assert "meta-llama/Llama-3.1-8B-Instruct" in plain
    # Model cached on two clusters → two GPU types collapse into the row.
    assert "H100_80GB" in plain
    assert "H200_141GB" in plain
    # Cluster names are intentionally not rendered in the table; users
    # dispatch by gpu_type, not by cluster.
    assert "athens" not in plain
    assert "berlin" not in plain
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
    # cache_synced_at is intentionally omitted from CLI output.
    assert "cache_synced_at" not in first["clusters"][0]


def test_models_json_omits_fft_key_when_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """JSON output stays backwards compatible: no available_fft_models
    key when the FFT endpoint returns an empty list."""
    monkeypatch.setattr("prime_cli.core.APIClient.get", _mock_get_factory([]))

    result = CliRunner().invoke(app, ["train", "models", "--output", "json"])

    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert "models" in data
    assert "available_fft_models" not in data


def test_models_json_fft_only_always_includes_fft_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With --fft-only, scripts specifically consume
    `.available_fft_models[]`. The key must be present even when the
    endpoint returned zero models, and the misleading `models: []` key
    must be absent since LoRA wasn't fetched."""

    def mock_get(self: Any, endpoint: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        if endpoint == "/training/available-fft-models":
            return {"models": []}
        raise AssertionError(f"Unexpected endpoint: {endpoint}")

    monkeypatch.setattr("prime_cli.core.APIClient.get", mock_get)

    result = CliRunner().invoke(app, ["train", "models", "--fft-only", "--output", "json"])

    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data == {"available_fft_models": []}


def test_models_json_fft_only_with_data(monkeypatch: pytest.MonkeyPatch) -> None:
    """--fft-only --output json with populated data: still no `models`
    key; `available_fft_models` carries the payload."""

    def mock_get(self: Any, endpoint: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        if endpoint == "/training/available-fft-models":
            return _fft_models_payload()
        raise AssertionError(f"Unexpected endpoint: {endpoint}")

    monkeypatch.setattr("prime_cli.core.APIClient.get", mock_get)

    result = CliRunner().invoke(app, ["train", "models", "--fft-only", "--output", "json"])

    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert "models" not in data
    assert [m["name"] for m in data["available_fft_models"]] == [
        "meta-llama/Llama-3.1-8B-Instruct",
        "qwen/qwen3-8b",
    ]


def test_models_command_survives_fft_endpoint_404(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Older backends that haven't shipped the FFT endpoint yet should
    still get the LoRA listing rendered — the CLI must not crash."""

    def mock_get(self: Any, endpoint: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
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
    assert "pre-cached" not in plain


def test_models_fft_only_suppresses_lora_section(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def mock_get(self: Any, endpoint: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        calls.append(endpoint)
        if endpoint == "/training/available-fft-models":
            return _fft_models_payload()
        raise AssertionError(f"Unexpected endpoint: {endpoint}")

    monkeypatch.setattr("prime_cli.core.APIClient.get", mock_get)

    result = CliRunner().invoke(app, ["train", "models", "--fft-only"], env={"COLUMNS": "200"})

    assert result.exit_code == 0, result.output
    plain = strip_ansi(result.output)
    assert "pre-cached" in plain
    assert "meta-llama/Llama-3.1-8B-Instruct" in plain
    # LoRA table title should not appear when --fft-only is set.
    assert "LoRA" not in plain
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

    def mock_get(self: Any, endpoint: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        raise NotFoundError("HTTP 404: not found")

    monkeypatch.setattr("prime_cli.core.APIClient.get", mock_get)
    client = HostedTrainingClient(APIClient())
    assert client.list_available_fft_models(team_id=None) == []


def test_list_available_fft_models_propagates_auth_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Auth failures must not be silently converted to an empty list —
    otherwise `prime train models --fft-only` on an expired key would
    exit 0 with a misleading 'No FFT models available' message."""
    from prime_cli.api.training import HostedTrainingClient
    from prime_cli.core import APIClient
    from prime_cli.core.client import UnauthorizedError

    def mock_get(self: Any, endpoint: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        raise UnauthorizedError("API key unauthorized.")

    monkeypatch.setattr("prime_cli.core.APIClient.get", mock_get)
    client = HostedTrainingClient(APIClient())
    with pytest.raises(UnauthorizedError):
        client.list_available_fft_models(team_id=None)


def test_models_fft_only_surfaces_auth_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """--fft-only skips the LoRA call, so an auth failure on the FFT
    endpoint is the only signal the caller has that their token is
    bad. It must exit non-zero with the auth error, not print the
    generic 'no models' fallback."""
    from prime_cli.core.client import UnauthorizedError

    def mock_get(self: Any, endpoint: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        if endpoint == "/training/available-fft-models":
            raise UnauthorizedError("API key unauthorized.")
        raise AssertionError(f"Unexpected endpoint: {endpoint}")

    monkeypatch.setattr("prime_cli.core.APIClient.get", mock_get)

    result = CliRunner().invoke(app, ["train", "models", "--fft-only"], env={"COLUMNS": "200"})

    assert result.exit_code != 0
    assert "unauthorized" in result.output.lower()
    assert "No FFT models available" not in result.output


def test_models_default_hides_fft_auth_error_after_lora_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """In default mode, if LoRA succeeds but the FFT call fails with
    anything other than 404, the LoRA table must still render — the FFT
    section is best-effort and shouldn't cascade the primary output."""
    from prime_cli.core.client import UnauthorizedError

    def mock_get(self: Any, endpoint: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        if endpoint == "/rft/models":
            return _models_payload()
        if endpoint == "/training/available-fft-models":
            raise UnauthorizedError("API key unauthorized.")
        raise AssertionError(f"Unexpected endpoint: {endpoint}")

    monkeypatch.setattr("prime_cli.core.APIClient.get", mock_get)

    result = CliRunner().invoke(app, ["train", "models"], env={"COLUMNS": "200"})

    assert result.exit_code == 0, result.output
    plain = strip_ansi(result.output)
    assert "qwen/qwen3-8b" in plain
    assert "pre-cached" not in plain


def test_list_available_fft_models_converts_pydantic_error_to_apierror(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A schema-drifted backend payload must surface as APIError, not
    raw pydantic ValidationError — otherwise the command's `except
    APIError` fallback would miss it and the LoRA table would fail to
    render alongside the FFT section (Bugbot finding on df0e6269)."""
    from prime_cli.api.training import HostedTrainingClient
    from prime_cli.core import APIClient, APIError

    def mock_get(self: Any, endpoint: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        # Missing required "clusters" list, wrong shape on "name".
        return {"models": [{"name": 12345, "clusters": "not-a-list"}]}

    monkeypatch.setattr("prime_cli.core.APIClient.get", mock_get)
    client = HostedTrainingClient(APIClient())
    with pytest.raises(APIError):
        client.list_available_fft_models(team_id=None)


def test_models_command_survives_fft_schema_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: even if the FFT endpoint returns an unparseable
    payload, `prime train models` should still emit the LoRA table
    rather than exiting with an unhandled traceback."""

    def mock_get(self: Any, endpoint: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        if endpoint == "/rft/models":
            return _models_payload()
        if endpoint == "/training/available-fft-models":
            return {"models": [{"name": 12345, "clusters": "not-a-list"}]}
        raise AssertionError(f"Unexpected endpoint: {endpoint}")

    monkeypatch.setattr("prime_cli.core.APIClient.get", mock_get)

    result = CliRunner().invoke(app, ["train", "models"], env={"COLUMNS": "200"})

    assert result.exit_code == 0, result.output
    plain = strip_ansi(result.output)
    assert "qwen/qwen3-8b" in plain
    assert "pre-cached" not in plain


def test_models_command_suppresses_lora_empty_banner_when_fft_populated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LoRA-empty + FFT-populated: the 'No models available for Hosted
    Training' banner would mislead readers into thinking the whole
    command failed. Only render the FFT section in that case."""

    def mock_get(self: Any, endpoint: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        if endpoint == "/rft/models":
            return {"models": []}
        if endpoint == "/training/available-fft-models":
            return _fft_models_payload()
        raise AssertionError(f"Unexpected endpoint: {endpoint}")

    monkeypatch.setattr("prime_cli.core.APIClient.get", mock_get)

    result = CliRunner().invoke(app, ["train", "models"], env={"COLUMNS": "200"})

    assert result.exit_code == 0, result.output
    plain = strip_ansi(result.output)
    # The misleading empty-LoRA banner must NOT appear.
    assert "No models available for Hosted Training" not in plain
    # FFT section still renders.
    assert "pre-cached" in plain
    assert "meta-llama/Llama-3.1-8B-Instruct" in plain


def test_models_command_shows_lora_empty_banner_when_both_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both sections empty: the LoRA fallback banner still surfaces so
    the user isn't left with a completely blank command output."""

    def mock_get(self: Any, endpoint: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        if endpoint == "/rft/models":
            return {"models": []}
        if endpoint == "/training/available-fft-models":
            return {"models": []}
        raise AssertionError(f"Unexpected endpoint: {endpoint}")

    monkeypatch.setattr("prime_cli.core.APIClient.get", mock_get)

    result = CliRunner().invoke(app, ["train", "models"], env={"COLUMNS": "200"})

    assert result.exit_code == 0, result.output
    plain = strip_ansi(result.output)
    assert "No models available for Hosted Training" in plain
    assert "pre-cached" not in plain


# --- on-demand beta enrollment (three-state) ------------------------------
#
# `onDemandBetaAccess` is tri-state: explicit False is a denial, explicit
# True is enrollment, and absent (older backend, 404, or a failed fetch) is
# *no signal*. Only an explicit False may ask the user to request access —
# an unavailable endpoint is not evidence the account lacks beta access.

_ACCESS_INSTRUCTION = "Contact Prime support to request access."


def _on_demand_entry(
    gpu_type: str = "H100_80GB",
    price: float = 2.0,
    *,
    available_now: bool = True,
    discount_label: str | None = None,
    is_beta: bool = False,
) -> dict[str, Any]:
    """One camelCase backend row of the discovery `onDemand` list."""
    entry: dict[str, Any] = {
        "gpuType": gpu_type,
        "pricePerGpuHour": price,
        "availableNow": available_now,
        "isBeta": is_beta,
    }
    if discount_label is not None:
        entry["discountLabel"] = discount_label
    return entry


def _discovery_mock(
    fft_payload: dict[str, Any] | Exception,
    *,
    lora_payload: dict[str, Any] | None = None,
):
    """mock_get serving `lora_payload` (default: no LoRA models) for
    /rft/models and `fft_payload` (a payload or an exception to raise) for
    the FFT discovery endpoint."""

    def mock_get(self: Any, endpoint: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        if endpoint == "/rft/models":
            return lora_payload if lora_payload is not None else {"models": []}
        if endpoint == "/training/available-fft-models":
            if isinstance(fft_payload, Exception):
                raise fft_payload
            return fft_payload
        raise AssertionError(f"Unexpected endpoint: {endpoint}")

    return mock_get


def test_models_default_unknown_enrollment_stays_neutral(monkeypatch: pytest.MonkeyPatch) -> None:
    """Field omitted (older backend): the empty state must not imply a
    beta denial — keep the legacy neutral fallback."""
    monkeypatch.setattr("prime_cli.core.APIClient.get", _discovery_mock({"models": []}))

    result = CliRunner().invoke(app, ["train", "models"], env={"COLUMNS": "200"})

    assert result.exit_code == 0, result.output
    plain = strip_ansi(result.output)
    assert "No models available for Hosted Training." in plain
    assert _ACCESS_INSTRUCTION not in plain


def test_models_default_explicit_false_directs_to_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "prime_cli.core.APIClient.get",
        _discovery_mock({"models": [], "onDemandBetaAccess": False}),
    )

    result = CliRunner().invoke(app, ["train", "models"], env={"COLUMNS": "200"})

    assert result.exit_code == 0, result.output
    plain = strip_ansi(result.output)
    assert "No models available for Hosted Training." in plain
    assert _ACCESS_INSTRUCTION in plain


def test_models_default_explicit_true_reports_missing_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Enrolled but empty: no denial, no missing-endpoint blame."""
    monkeypatch.setattr(
        "prime_cli.core.APIClient.get",
        _discovery_mock({"models": [], "onDemandBetaAccess": True}),
    )

    result = CliRunner().invoke(app, ["train", "models"], env={"COLUMNS": "200"})

    assert result.exit_code == 0, result.output
    plain = strip_ansi(result.output)
    assert "No models available for Hosted Training." in plain
    assert "No on-demand capacity is currently listed." in plain
    assert _ACCESS_INSTRUCTION not in plain
    assert "endpoint" not in plain


def test_models_fft_only_explicit_false_directs_to_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "prime_cli.core.APIClient.get",
        _discovery_mock({"models": [], "onDemandBetaAccess": False}),
    )

    result = CliRunner().invoke(app, ["train", "models", "--fft-only"], env={"COLUMNS": "200"})

    assert result.exit_code == 0, result.output
    plain = strip_ansi(result.output)
    assert "No FFT models available." in plain
    assert _ACCESS_INSTRUCTION in plain


def test_models_fft_only_explicit_true_reports_missing_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "prime_cli.core.APIClient.get",
        _discovery_mock({"models": [], "onDemandBetaAccess": True}),
    )

    result = CliRunner().invoke(app, ["train", "models", "--fft-only"], env={"COLUMNS": "200"})

    assert result.exit_code == 0, result.output
    plain = strip_ansi(result.output)
    assert "No FFT models available." in plain
    assert "No on-demand capacity is currently listed." in plain
    assert _ACCESS_INSTRUCTION not in plain
    assert "endpoint" not in plain


def test_models_fft_only_unknown_keeps_legacy_wording(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--fft-only on an older backend: the pre-on-demand neutral wording,
    never a guessed enrollment denial."""
    monkeypatch.setattr("prime_cli.core.APIClient.get", _discovery_mock({"models": []}))

    result = CliRunner().invoke(app, ["train", "models", "--fft-only"], env={"COLUMNS": "200"})

    assert result.exit_code == 0, result.output
    plain = strip_ansi(result.output)
    assert "No FFT models available." in plain
    assert "warm model cache" in plain
    assert _ACCESS_INSTRUCTION not in plain


def test_models_default_404_does_not_imply_missing_beta_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 404 on the discovery endpoint says the endpoint is absent, not
    that the account lacks enrollment — the LoRA table must render
    without the access-request instruction."""
    monkeypatch.setattr(
        "prime_cli.core.APIClient.get",
        _discovery_mock(NotFoundError("HTTP 404: not found"), lora_payload=_models_payload()),
    )

    result = CliRunner().invoke(app, ["train", "models"], env={"COLUMNS": "200"})

    assert result.exit_code == 0, result.output
    plain = strip_ansi(result.output)
    assert "qwen/qwen3-8b" in plain
    assert _ACCESS_INSTRUCTION not in plain


def test_models_default_server_error_does_not_imply_missing_beta_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from prime_cli.core.client import APIError

    monkeypatch.setattr(
        "prime_cli.core.APIClient.get",
        _discovery_mock(APIError("HTTP 500: server failure"), lora_payload=_models_payload()),
    )

    result = CliRunner().invoke(app, ["train", "models"], env={"COLUMNS": "200"})

    assert result.exit_code == 0, result.output
    plain = strip_ansi(result.output)
    assert "qwen/qwen3-8b" in plain
    assert _ACCESS_INSTRUCTION not in plain


def test_models_default_schema_drift_does_not_imply_missing_beta_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Malformed success: the payload explicitly says access is true, but
    the missing required price field makes parsing fail. The swallowed
    error must not surface as a denial."""
    malformed = {
        "models": [],
        "onDemand": [{"gpuType": "H100_80GB"}],
        "onDemandBetaAccess": True,
    }
    monkeypatch.setattr(
        "prime_cli.core.APIClient.get",
        _discovery_mock(malformed, lora_payload=_models_payload()),
    )

    result = CliRunner().invoke(app, ["train", "models"], env={"COLUMNS": "200"})

    assert result.exit_code == 0, result.output
    plain = strip_ansi(result.output)
    assert "qwen/qwen3-8b" in plain
    assert _ACCESS_INSTRUCTION not in plain


def test_models_fft_only_404_still_exits_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """--fft-only on a backend without the endpoint: the 404 is
    intentionally tolerated (empty response, unknown enrollment), so the
    command exits 0 with the neutral empty message."""
    monkeypatch.setattr(
        "prime_cli.core.APIClient.get", _discovery_mock(NotFoundError("HTTP 404: not found"))
    )

    result = CliRunner().invoke(app, ["train", "models", "--fft-only"], env={"COLUMNS": "200"})

    assert result.exit_code == 0, result.output
    plain = strip_ansi(result.output)
    assert "No FFT models available." in plain
    assert _ACCESS_INSTRUCTION not in plain


def test_models_fft_only_server_error_exits_nonzero(monkeypatch: pytest.MonkeyPatch) -> None:
    """--fft-only has no LoRA section to fall back on, so a server error
    must exit non-zero instead of printing an empty-table message."""
    from prime_cli.core.client import APIError

    monkeypatch.setattr(
        "prime_cli.core.APIClient.get", _discovery_mock(APIError("HTTP 500: server failure"))
    )

    result = CliRunner().invoke(app, ["train", "models", "--fft-only"], env={"COLUMNS": "200"})

    assert result.exit_code != 0
    assert "HTTP 500" in result.output


def test_get_available_fft_404_carries_unknown_enrollment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The client's 404 fallback must carry unknown (None) enrollment —
    not a False that reads as a denial downstream."""
    from prime_cli.api.training import HostedTrainingClient
    from prime_cli.core import APIClient

    monkeypatch.setattr("prime_cli.core.APIClient.get", _discovery_mock(NotFoundError("404")))
    response = HostedTrainingClient(APIClient()).get_available_fft()
    assert response.models == []
    assert response.on_demand == []
    assert response.on_demand_beta_access is None


# --- enrollment flag in JSON output ----------------------------------------


@pytest.mark.parametrize("beta_access", [False, True], ids=["false", "true"])
@pytest.mark.parametrize("mode", ["default", "fft-only"])
def test_models_json_retains_known_enrollment(
    monkeypatch: pytest.MonkeyPatch, mode: str, beta_access: bool
) -> None:
    """Known enrollment values survive in both JSON modes; scripts reading
    the flag get real data, not a dropped field."""
    monkeypatch.setattr(
        "prime_cli.core.APIClient.get",
        _discovery_mock({"models": [], "onDemandBetaAccess": beta_access}),
    )

    args = ["train", "models"]
    if mode == "fft-only":
        args.append("--fft-only")
    result = CliRunner().invoke(app, [*args, "--output", "json"])

    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["on_demand_beta_access"] is beta_access


@pytest.mark.parametrize("mode", ["default", "fft-only"])
def test_models_json_omits_unknown_enrollment(monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    """Unknown enrollment is omitted (not null, not false) so old-backend
    JSON stays byte-compatible."""
    monkeypatch.setattr("prime_cli.core.APIClient.get", _discovery_mock({"models": []}))

    args = ["train", "models"]
    if mode == "fft-only":
        args.append("--fft-only")
    result = CliRunner().invoke(app, [*args, "--output", "json"])

    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert "on_demand_beta_access" not in data


@pytest.mark.parametrize("mode", ["default", "fft-only"])
def test_models_json_retains_on_demand_rows(monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    """Populated on-demand rows survive both JSON modes with their
    snake_case keys."""
    fft_payload = {
        "models": [],
        "onDemand": [_on_demand_entry(price=2.5, available_now=False)],
        "onDemandBetaAccess": True,
    }
    monkeypatch.setattr("prime_cli.core.APIClient.get", _discovery_mock(fft_payload))

    args = ["train", "models"]
    if mode == "fft-only":
        args.append("--fft-only")
    result = CliRunner().invoke(app, [*args, "--output", "json"])

    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["on_demand"] == [
        {
            "gpu_type": "H100_80GB",
            "price_per_gpu_hour": 2.5,
            "discount_label": None,
            "available_now": False,
            "is_beta": False,
        }
    ]


# --- on-demand table rendering ---------------------------------------------


def test_on_demand_only_discovery_renders_no_empty_banner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No LoRA models, no warm FFT cache, but purchasable on-demand
    capacity: the misleading 'No models available' banner must not
    precede the on-demand table."""
    fft_payload = {"models": [], "onDemand": [_on_demand_entry()]}
    monkeypatch.setattr("prime_cli.core.APIClient.get", _discovery_mock(fft_payload))

    result = CliRunner().invoke(app, ["train", "models"], env={"COLUMNS": "200"})

    assert result.exit_code == 0, result.output
    plain = strip_ansi(result.output)
    assert "No models available for Hosted Training." not in plain
    assert "On-Demand Capacity" in plain
    assert "H100_80GB" in plain


def test_on_demand_table_renders_capacity_not_start_promise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The column is a coarse capacity hint (available/busy) with a queue
    caveat — never a `Starts: now` scheduling promise the endpoint cannot
    make: discovery only sees pool headroom, not the run's GPU count or
    topology."""
    entries = [
        _on_demand_entry("H100_80GB", 2.0, available_now=True),
        _on_demand_entry("H200_141GB", 3.0, available_now=False),
    ]
    monkeypatch.setattr(
        "prime_cli.core.APIClient.get", _discovery_mock({"models": [], "onDemand": entries})
    )

    result = CliRunner().invoke(app, ["train", "models"], env={"COLUMNS": "200"})

    assert result.exit_code == 0, result.output
    plain = strip_ansi(result.output)
    assert "Capacity" in plain
    assert "available" in plain
    assert "busy" in plain
    # The old immediate-start promise is gone.
    assert "Starts" not in plain
    assert "queued" not in plain
    normalized = " ".join(plain.split())
    assert "Runs may queue depending on GPU count and topology." in normalized
    # Rate formatting.
    assert "$2.00" in plain
    assert "$3.00" in plain


def test_on_demand_table_caption_rates_lock_at_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The rate is fixed at dispatch and billing starts at pod readiness —
    two different events. A queued run dispatched at $2 keeps $2 even if
    the live rate moves to $3 before its pods are ready (the backend
    reuses the persisted rate on queue promotion), so the caption must
    not say 'fixed when the run starts'."""
    monkeypatch.setattr(
        "prime_cli.core.APIClient.get",
        _discovery_mock({"models": [], "onDemand": [_on_demand_entry()]}),
    )

    result = CliRunner().invoke(app, ["train", "models"], env={"COLUMNS": "200"})

    assert result.exit_code == 0, result.output
    plain = strip_ansi(result.output)
    normalized = " ".join(plain.split())
    expected = (
        "Rate fixed at dispatch; billed per GPU-hour from when pods are "
        "ready, prorated to the second."
    )
    assert expected in normalized
    assert "fixed when the run starts" not in normalized


@pytest.mark.parametrize("is_beta", [False, True], ids=["stable", "beta"])
def test_on_demand_beta_badge_follows_is_beta(
    monkeypatch: pytest.MonkeyPatch, is_beta: bool
) -> None:
    """The (beta) badge is server-driven so it retires without a CLI
    release."""
    monkeypatch.setattr(
        "prime_cli.core.APIClient.get",
        _discovery_mock({"models": [], "onDemand": [_on_demand_entry(is_beta=is_beta)]}),
    )

    result = CliRunner().invoke(app, ["train", "models"], env={"COLUMNS": "200"})

    assert result.exit_code == 0, result.output
    plain = strip_ansi(result.output)
    if is_beta:
        assert "On-Demand Capacity (beta)" in plain
    else:
        assert "On-Demand Capacity" in plain
        assert "(beta)" not in plain


def test_on_demand_table_escapes_promo_markup(monkeypatch: pytest.MonkeyPatch) -> None:
    """A promo label containing Rich markup characters renders literally —
    a backend-supplied string must never style or inject console
    markup."""
    label = "[bold magenta]launch deal[/bold magenta]"
    monkeypatch.setattr(
        "prime_cli.core.APIClient.get",
        _discovery_mock({"models": [], "onDemand": [_on_demand_entry(discount_label=label)]}),
    )

    result = CliRunner().invoke(app, ["train", "models"], env={"COLUMNS": "400"})

    assert result.exit_code == 0, result.output
    plain = strip_ansi(result.output)
    # Rich may still wrap the caption to the table width, so compare on
    # whitespace-normalized text.
    assert " ".join(plain.split()).count(" ".join(label.split())) == 1
