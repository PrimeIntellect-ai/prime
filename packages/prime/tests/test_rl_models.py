"""Tests for `prime rl models` — focused on price column rendering."""

import json
import re
from typing import Any

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
    monkeypatch.setenv("PRIME_FRONTEND_URL", "https://frontend.test/")


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
    """Mock APIClient.get for the models command. Returns an empty model
    list unless ``fft_payload`` is given; any other endpoint (notably the
    retired shared-LoRA ``/rft/models``) fails the test."""

    def mock_get(self: Any, endpoint: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        calls.append(endpoint)
        if endpoint == "/training/available-fft-models":
            return fft_payload if fft_payload is not None else {"models": []}
        raise AssertionError(f"Unexpected endpoint: {endpoint}")

    return mock_get


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
    calls: list[str] = []
    monkeypatch.setattr(
        "prime_cli.core.APIClient.get",
        _mock_get_factory(calls, fft_payload=_fft_models_payload()),
    )

    result = CliRunner().invoke(app, ["train", "models"], env={"COLUMNS": "200"})

    assert result.exit_code == 0, result.output
    plain = strip_ansi(result.output)
    assert "LoRA" not in plain
    assert "pre-cached" in plain
    assert "meta-llama/Llama-3.1-8B-Instruct" in plain
    # Model cached on two clusters → two GPU types collapse into the row.
    assert "H100_80GB" in plain
    assert "H200_141GB" in plain
    # Cluster names are intentionally not rendered in the table; users
    # dispatch by gpu_type, not by cluster.
    assert "athens" not in plain
    assert "berlin" not in plain
    # Only the dedicated discovery endpoint is hit; shared LoRA is retired.
    assert calls == ["/training/available-fft-models"]


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
    assert "models" not in data
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


def test_models_json_always_includes_fft_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Scripts consume `.available_fft_models[]`, so the key is present even
    when the endpoint returned zero models."""
    monkeypatch.setattr("prime_cli.core.APIClient.get", _mock_get_factory([]))

    result = CliRunner().invoke(app, ["train", "models", "--output", "json"])

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {"available_fft_models": []}


def test_models_json_fft_only_always_includes_fft_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--fft-only is a deprecated no-op kept so existing scripts keep
    working; its JSON matches the default."""
    monkeypatch.setattr("prime_cli.core.APIClient.get", _mock_get_factory([]))

    result = CliRunner().invoke(app, ["train", "models", "--fft-only", "--output", "json"])

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {"available_fft_models": []}


def test_models_json_fft_only_with_data(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "prime_cli.core.APIClient.get",
        _mock_get_factory([], fft_payload=_fft_models_payload()),
    )

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
    """A backend without the discovery endpoint gets the neutral empty
    state, not a crash."""

    def mock_get(self: Any, endpoint: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        if endpoint == "/training/available-fft-models":
            raise NotFoundError("HTTP 404: available-fft-models not deployed on this backend")
        raise AssertionError(f"Unexpected endpoint: {endpoint}")

    monkeypatch.setattr("prime_cli.core.APIClient.get", mock_get)

    result = CliRunner().invoke(app, ["train", "models"], env={"COLUMNS": "200"})

    assert result.exit_code == 0, result.output
    plain = strip_ansi(result.output)
    assert "No models available for Hosted Training." in plain
    assert "pre-cached" not in plain


def test_models_fft_only_renders_like_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        "prime_cli.core.APIClient.get",
        _mock_get_factory(calls, fft_payload=_fft_models_payload()),
    )

    result = CliRunner().invoke(app, ["train", "models", "--fft-only"], env={"COLUMNS": "200"})

    assert result.exit_code == 0, result.output
    plain = strip_ansi(result.output)
    assert "pre-cached" in plain
    assert "meta-llama/Llama-3.1-8B-Instruct" in plain
    assert calls == ["/training/available-fft-models"]


def test_list_available_fft_models_returns_empty_on_404(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """API client method swallows 404 so `prime train models` renders the
    empty state on backends that haven't shipped the endpoint yet."""
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
    otherwise `prime train models` on an expired key would exit 0 with a
    misleading 'No models available' message."""
    from prime_cli.api.training import HostedTrainingClient
    from prime_cli.core import APIClient
    from prime_cli.core.client import UnauthorizedError

    def mock_get(self: Any, endpoint: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        raise UnauthorizedError("API key unauthorized.")

    monkeypatch.setattr("prime_cli.core.APIClient.get", mock_get)
    client = HostedTrainingClient(APIClient())
    with pytest.raises(UnauthorizedError):
        client.list_available_fft_models(team_id=None)


@pytest.mark.parametrize("mode", ["default", "fft-only"])
def test_models_surfaces_auth_error(monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    """The discovery endpoint is the only source, so an auth failure must
    exit non-zero with the auth error, not print the 'no models' fallback."""
    from prime_cli.core.client import UnauthorizedError

    def mock_get(self: Any, endpoint: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        if endpoint == "/training/available-fft-models":
            raise UnauthorizedError("API key unauthorized.")
        raise AssertionError(f"Unexpected endpoint: {endpoint}")

    monkeypatch.setattr("prime_cli.core.APIClient.get", mock_get)

    args = ["train", "models"] + (["--fft-only"] if mode == "fft-only" else [])
    result = CliRunner().invoke(app, args, env={"COLUMNS": "200"})

    assert result.exit_code != 0
    assert "unauthorized" in result.output.lower()
    assert "No models available" not in result.output


def test_list_available_fft_models_converts_pydantic_error_to_apierror(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A schema-drifted backend payload must surface as APIError, not raw
    pydantic ValidationError, so the command's `except APIError` reports
    it cleanly instead of with a traceback."""
    from prime_cli.api.training import HostedTrainingClient
    from prime_cli.core import APIClient, APIError

    def mock_get(self: Any, endpoint: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        # Missing required "clusters" list, wrong shape on "name".
        return {"models": [{"name": 12345, "clusters": "not-a-list"}]}

    monkeypatch.setattr("prime_cli.core.APIClient.get", mock_get)
    client = HostedTrainingClient(APIClient())
    with pytest.raises(APIError):
        client.list_available_fft_models(team_id=None)


def test_models_command_reports_fft_schema_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: an unparseable discovery payload exits non-zero with an
    error rather than an unhandled traceback."""
    monkeypatch.setattr(
        "prime_cli.core.APIClient.get",
        _mock_get_factory([], fft_payload={"models": [{"name": 12345, "clusters": "x"}]}),
    )

    result = CliRunner().invoke(app, ["train", "models"], env={"COLUMNS": "200"})

    assert result.exit_code == 1
    assert "Error" in result.output
    assert "Traceback" not in result.output


def test_models_command_shows_empty_banner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nothing to list: the fallback banner surfaces so the user isn't left
    with a completely blank command output."""
    monkeypatch.setattr("prime_cli.core.APIClient.get", _mock_get_factory([]))

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


def _discovery_mock(fft_payload: dict[str, Any] | Exception):
    """mock_get serving `fft_payload` (a payload or an exception to raise)
    for the discovery endpoint."""

    def mock_get(self: Any, endpoint: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
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
    assert "No models available for Hosted Training." in plain
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
    assert "No models available for Hosted Training." in plain
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
    assert "No models available for Hosted Training." in plain
    assert "warm model cache" in plain
    assert _ACCESS_INSTRUCTION not in plain


def test_models_default_404_does_not_imply_missing_beta_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 404 on the discovery endpoint says the endpoint is absent, not
    that the account lacks enrollment."""
    monkeypatch.setattr(
        "prime_cli.core.APIClient.get", _discovery_mock(NotFoundError("HTTP 404: not found"))
    )

    result = CliRunner().invoke(app, ["train", "models"], env={"COLUMNS": "200"})

    assert result.exit_code == 0, result.output
    plain = strip_ansi(result.output)
    assert "No models available for Hosted Training." in plain
    assert _ACCESS_INSTRUCTION not in plain


def test_models_default_schema_drift_does_not_imply_missing_beta_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Malformed success: the payload explicitly says access is true, but
    the missing required price field makes parsing fail. The error must
    not surface as a denial."""
    malformed = {
        "models": [],
        "onDemand": [{"gpuType": "H100_80GB"}],
        "onDemandBetaAccess": True,
    }
    monkeypatch.setattr("prime_cli.core.APIClient.get", _discovery_mock(malformed))

    result = CliRunner().invoke(app, ["train", "models"], env={"COLUMNS": "200"})

    assert result.exit_code == 1
    assert _ACCESS_INSTRUCTION not in strip_ansi(result.output)


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
    assert "No models available for Hosted Training." in plain
    assert _ACCESS_INSTRUCTION not in plain


@pytest.mark.parametrize("mode", ["default", "fft-only"])
def test_models_server_error_exits_nonzero(monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    """With no LoRA section to fall back on, a server error must exit
    non-zero instead of printing an empty-table message."""
    from prime_cli.core.client import APIError

    monkeypatch.setattr(
        "prime_cli.core.APIClient.get", _discovery_mock(APIError("HTTP 500: server failure"))
    )

    args = ["train", "models"] + (["--fft-only"] if mode == "fft-only" else [])
    result = CliRunner().invoke(app, args, env={"COLUMNS": "200"})

    assert result.exit_code != 0
    assert "HTTP 500" in result.output
    assert _ACCESS_INSTRUCTION not in result.output


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
            "at_capacity": False,
        }
    ]


# --- on-demand table rendering ---------------------------------------------


def test_on_demand_only_discovery_renders_no_empty_banner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No warm model cache, but purchasable on-demand capacity: the
    misleading 'No models available' banner must not
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
    """The column is a coarse capacity hint (Available/Busy) with a queue
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
    assert "Available" in plain
    assert "Busy" in plain
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
        "Rates are fixed at dispatch and billed per GPU-hour from pod ready, prorated per second."
    )
    assert expected in normalized
    assert "fixed when the run starts" not in normalized


def test_on_demand_footer_lines_are_not_wrapped_to_table_width(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Footer notes print under the table at console width, not as a caption
    that Rich wraps to the (much narrower) table width, and share one dim,
    non-italic style with the dedicated-capacity hint."""
    entries = [
        _on_demand_entry("H100_80GB", 2.0),
        {**_on_demand_entry("B200_180GB", 5.0, available_now=False), "atCapacity": True},
    ]
    monkeypatch.setattr(
        "prime_cli.core.APIClient.get", _discovery_mock({"models": [], "onDemand": entries})
    )

    _force_color_console(monkeypatch)

    result = CliRunner().invoke(app, ["train", "models"])

    assert result.exit_code == 0, result.output
    raw_lines = _OSC8.sub("", result.output).splitlines()
    lines = [strip_ansi(ln).rstrip() for ln in raw_lines]
    for note in (
        "Rates are fixed at dispatch and billed per GPU-hour from pod ready, prorated per second.",
        "Runs may queue depending on GPU count and topology.",
        _CAPACITY_HINT,
    ):
        raw = raw_lines[lines.index(note)]
        # Dim; never italic, which was the caption style.
        assert raw.startswith("\x1b[2m"), repr(raw)
        assert not _ITALIC.search(raw), repr(raw)


def test_capacity_hint_links_request_compute_page(monkeypatch: pytest.MonkeyPatch) -> None:
    """The link text itself is an underlined OSC 8 hyperlink to the configured
    frontend's Request Compute page; no raw URL or support email is printed."""
    entries = [{**_on_demand_entry("B200_180GB", 5.0, available_now=False), "atCapacity": True}]
    monkeypatch.setattr(
        "prime_cli.core.APIClient.get", _discovery_mock({"models": [], "onDemand": entries})
    )
    _force_color_console(monkeypatch)

    result = CliRunner().invoke(app, ["train", "models"])

    assert result.exit_code == 0, result.output
    assert _osc8_targets(result.output) == [_QUOTES_URL]
    assert "\x1b[2;4mRequest dedicated capacity" in result.output
    visible = strip_ansi(_OSC8.sub("", result.output))
    assert _QUOTES_URL not in visible
    assert "support@" not in result.output


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
    assert " ".join(plain.split()).count(" ".join(label.split())) == 1


# --- at-capacity GPU types (priced, no inventory) --------------------------

# The URL follows the configured frontend (PRIME_FRONTEND_URL, set above),
# so staging and dev point at their own Request Compute page.
_QUOTES_URL = "https://frontend.test/dashboard/quotes"
_CAPACITY_HINT = "Need a GPU type that's at capacity? Request dedicated capacity"
_OSC8 = re.compile(r"\x1b\]8;[^;]*;([^\x1b]*)\x1b\\")
_ITALIC = re.compile(r"\x1b\[(?:[0-9;]*;)?3(?:;[0-9;]*)?m")


def _osc8_targets(output: str) -> list[str]:
    return [target for target in _OSC8.findall(output) if target]


def _force_color_console(monkeypatch: pytest.MonkeyPatch) -> None:
    """rl.py builds its console at import, before CliRunner's env applies."""
    from prime_cli.utils import get_console

    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setattr(
        "prime_cli.commands.rl.console",
        get_console(force_terminal=True, color_system="standard", width=120),
    )


def _capacity_line(plain: str, gpu_type: str) -> str:
    return next(ln for ln in plain.splitlines() if gpu_type in ln)


@pytest.mark.parametrize("key", ["atCapacity", "at_capacity"])
def test_at_capacity_parses_both_casings(key: str) -> None:
    from prime_cli.api.training import OnDemandGpuTypeAvailability

    entry = OnDemandGpuTypeAvailability.model_validate({**_on_demand_entry(), key: True})
    assert entry.at_capacity is True


@pytest.mark.parametrize("payload", [{}, {"atCapacity": None}], ids=["absent", "null"])
def test_missing_at_capacity_means_available(payload: dict[str, Any]) -> None:
    from prime_cli.api.training import OnDemandGpuTypeAvailability

    entry = OnDemandGpuTypeAvailability.model_validate({**_on_demand_entry(), **payload})
    assert entry.at_capacity is False


def test_at_capacity_entries_render_last_with_one_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    entries = [
        {**_on_demand_entry("B200_180GB", 5.0, available_now=False), "atCapacity": True},
        _on_demand_entry("H100_80GB", 2.0, available_now=True),
        {**_on_demand_entry("GB200_186GB", 6.0, available_now=False), "atCapacity": True},
        _on_demand_entry("H200_141GB", 3.0, available_now=False),
    ]
    monkeypatch.setattr(
        "prime_cli.core.APIClient.get", _discovery_mock({"models": [], "onDemand": entries})
    )

    result = CliRunner().invoke(app, ["train", "models"], env={"COLUMNS": "200"})

    assert result.exit_code == 0, result.output
    plain = strip_ansi(result.output)
    order = [
        gpu
        for ln in plain.splitlines()
        for gpu in ("B200_180GB", "H100_80GB", "GB200_186GB", "H200_141GB")
        if gpu in ln
    ]
    # Sellable types first in backend order, then at-capacity in backend order.
    assert order == ["H100_80GB", "H200_141GB", "B200_180GB", "GB200_186GB"]
    assert "At capacity" in _capacity_line(plain, "B200_180GB")
    assert "At capacity" in _capacity_line(plain, "GB200_186GB")
    assert "Available" in _capacity_line(plain, "H100_80GB")
    assert "Busy" in _capacity_line(plain, "H200_141GB")
    # Priced types keep their rate even with no inventory.
    assert "$5.00" in _capacity_line(plain, "B200_180GB")
    assert " ".join(plain.split()).count(_CAPACITY_HINT) == 1
    assert "support@" not in plain


def test_no_capacity_hint_without_at_capacity_entries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "prime_cli.core.APIClient.get",
        _discovery_mock({"models": [], "onDemand": [_on_demand_entry()]}),
    )

    result = CliRunner().invoke(app, ["train", "models"], env={"COLUMNS": "200"})

    assert result.exit_code == 0, result.output
    plain = strip_ansi(result.output)
    assert "At capacity" not in plain
    assert "Request dedicated capacity" not in plain


def test_models_json_carries_at_capacity_sorted_last(monkeypatch: pytest.MonkeyPatch) -> None:
    entries = [
        {**_on_demand_entry("B200_180GB", 5.0), "atCapacity": True},
        _on_demand_entry("H100_80GB", 2.0),
    ]
    monkeypatch.setattr(
        "prime_cli.core.APIClient.get", _discovery_mock({"models": [], "onDemand": entries})
    )

    result = CliRunner().invoke(app, ["train", "models", "--output", "json"])

    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert [(o["gpu_type"], o["at_capacity"]) for o in data["on_demand"]] == [
        ("H100_80GB", False),
        ("B200_180GB", True),
    ]


@pytest.mark.parametrize(
    ("flags", "label", "absent"),
    [
        ({"availableNow": False, "atCapacity": True}, "At capacity", ("Available", "Busy")),
        ({"availableNow": False, "atCapacity": False}, "Busy", ("Available", "At capacity")),
        ({"availableNow": True, "atCapacity": False}, "Available", ("Busy", "At capacity")),
    ],
    ids=["at-capacity", "busy", "available"],
)
def test_platform_capacity_states_render_one_label(
    monkeypatch: pytest.MonkeyPatch, flags: dict[str, bool], label: str, absent: tuple[str, ...]
) -> None:
    """The platform contract: atCapacity always pairs with availableNow
    false, and only atCapacity reads as "At capacity"; a busy pool keeps
    the queue label."""
    entry = {"gpuType": "B300_262GB", "pricePerGpuHour": 9.0, "isBeta": False, **flags}
    monkeypatch.setattr(
        "prime_cli.core.APIClient.get",
        _discovery_mock({"models": [], "onDemand": [entry]}),
    )

    result = CliRunner().invoke(app, ["train", "models"], env={"COLUMNS": "200"})

    assert result.exit_code == 0, result.output
    plain = strip_ansi(result.output)
    row = _capacity_line(plain, "B300_262GB")
    assert label in row
    assert not any(other in row for other in absent)
    normalized = " ".join(plain.split())
    assert "Runs may queue depending on GPU count and topology." in normalized
    assert (_CAPACITY_HINT in normalized) is (label == "At capacity")
