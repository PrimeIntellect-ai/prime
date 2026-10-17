"""Tests for `prime rl models` — focused on price column rendering."""

import json
from typing import Any

import pytest
from prime_cli.commands.rl import _model_name_sort_key
from prime_cli.core.client import APIError, NotFoundError
from prime_cli.main import app
from prime_cli.utils.formatters import strip_ansi
from typer.testing import CliRunner


@pytest.fixture(autouse=True)
def _api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PRIME_API_KEY", "dummy")
    monkeypatch.setenv("PRIME_DISABLE_VERSION_CHECK", "1")


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


# ---------------------------------------------------------------------------
# `prime train gpus` — both capacity paths
#
# The command's own help, and two public docs pages, promise "every GPU
# type you can dispatch on". It used to read only
# `available-gpu-types`, which covers RESERVED capacity alone, so an
# on-demand-only account was told it had no GPU types and should contact
# support — while `prime train models` was simultaneously offering it
# purchasable capacity. That is the single worst message to show a
# self-serve customer, so it is worth pinning.
# ---------------------------------------------------------------------------


_UNSET = object()


def _gpus_mock_get(
    *,
    gpu_types: list[str],
    on_demand: list[dict[str, Any]],
    beta_access: Any = True,
):
    """`beta_access` models the three enrollment states the response can
    report: True (enrolled), False (not enrolled) and `_UNSET` for a
    backend that omits the field entirely."""

    def mock_get(self: Any, endpoint: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        if endpoint == "/training/available-gpu-types":
            return {"gpuTypes": gpu_types}
        if endpoint == "/training/available-fft-models":
            payload: dict[str, Any] = {"models": [], "onDemand": on_demand}
            if beta_access is not _UNSET:
                payload["onDemandBetaAccess"] = beta_access
            return payload
        raise AssertionError(f"Unexpected endpoint: {endpoint}")

    return mock_get


_OD_H200 = {
    "gpuType": "H200_141GB",
    "pricePerGpuHour": 4.9,
    "availableNow": True,
    "discountLabel": None,
    "isBeta": True,
}


def test_gpus_lists_on_demand_when_no_reserved_capacity(monkeypatch: pytest.MonkeyPatch) -> None:
    """An on-demand-only account must not be told it has no access."""
    monkeypatch.setattr(
        "prime_cli.core.client.APIClient.get",
        _gpus_mock_get(gpu_types=[], on_demand=[_OD_H200]),
    )
    result = CliRunner().invoke(app, ["train", "gpus"])
    assert result.exit_code == 0, result.output
    out = strip_ansi(result.output)
    assert "H200_141GB" in out
    assert "4.90" in out, f"the hourly rate belongs in the listing:\n{out}"
    assert "Contact support" not in out, (
        f"an account that can dispatch on-demand has access:\n{out}"
    )


def test_gpus_does_not_list_a_reserved_type_twice(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reserved wins over on-demand, so a held type is not purchasable."""
    monkeypatch.setattr(
        "prime_cli.core.client.APIClient.get",
        _gpus_mock_get(gpu_types=["H200_141GB"], on_demand=[_OD_H200]),
    )
    result = CliRunner().invoke(app, ["train", "gpus"])
    assert result.exit_code == 0, result.output
    out = strip_ansi(result.output)
    assert out.count("H200_141GB") == 1, f"listed twice:\n{out}"
    assert "On-demand capacity" not in out, (
        f"a reserved type must not also be offered for purchase:\n{out}"
    )


def test_gpus_asks_for_access_only_when_the_backend_denies_enrollment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit False is the one state where asking for access is right."""
    monkeypatch.setattr(
        "prime_cli.core.client.APIClient.get",
        _gpus_mock_get(gpu_types=[], on_demand=[], beta_access=False),
    )
    result = CliRunner().invoke(app, ["train", "gpus"])
    assert result.exit_code == 0, result.output
    out = strip_ansi(result.output)
    assert "No GPU types available" in out
    assert "Contact Prime support to request access" in out


def test_gpus_does_not_ask_an_enrolled_account_to_request_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`onDemandBetaAccess: true` with an empty pool is not a denial.

    Sending an already-onboarded customer to support for access they
    hold is the exact dead end this command was changed to stop
    producing.
    """
    monkeypatch.setattr(
        "prime_cli.core.client.APIClient.get",
        _gpus_mock_get(gpu_types=[], on_demand=[], beta_access=True),
    )
    result = CliRunner().invoke(app, ["train", "gpus"])
    assert result.exit_code == 0, result.output
    out = strip_ansi(result.output)
    assert "No on-demand capacity is currently listed" in out
    assert "request access" not in out, f"this account is enrolled:\n{out}"
    assert "Contact support" not in out, f"this account is enrolled:\n{out}"


def test_gpus_stays_neutral_when_the_backend_omits_enrollment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No signal is not evidence of missing access."""
    monkeypatch.setattr(
        "prime_cli.core.client.APIClient.get",
        _gpus_mock_get(gpu_types=[], on_demand=[], beta_access=_UNSET),
    )
    result = CliRunner().invoke(app, ["train", "gpus"])
    assert result.exit_code == 0, result.output
    out = strip_ansi(result.output)
    assert "No GPU types available" in out
    assert "request access" not in out, f"enrollment was never reported:\n{out}"


def test_gpus_marks_beta_on_demand_capacity(monkeypatch: pytest.MonkeyPatch) -> None:
    """Beta capacity must not read as a stable offering.

    The flag is server-controlled so the label retires without a CLI
    release; the models table and the dispatch result both derive it
    from the rows, and this listing is what a new customer reads first.
    """
    monkeypatch.setattr(
        "prime_cli.core.client.APIClient.get",
        _gpus_mock_get(gpu_types=[], on_demand=[_OD_H200]),
    )
    result = CliRunner().invoke(app, ["train", "gpus"])
    assert result.exit_code == 0, result.output
    out = strip_ansi(result.output)
    assert "On-demand capacity" in out
    assert "(beta)" in out, f"isBeta was set on the row:\n{out}"


def test_gpus_omits_the_beta_marker_once_capacity_is_stable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """And it disappears when the backend stops flagging it."""
    stable = {**_OD_H200, "isBeta": False}
    monkeypatch.setattr(
        "prime_cli.core.client.APIClient.get",
        _gpus_mock_get(gpu_types=[], on_demand=[stable]),
    )
    result = CliRunner().invoke(app, ["train", "gpus"])
    assert result.exit_code == 0, result.output
    out = strip_ansi(result.output)
    assert "On-demand capacity" in out
    assert "(beta)" not in out, f"the backend reported stable capacity:\n{out}"


def test_gpus_json_carries_both_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    """Machine-readable output needs the rate, not just the type name."""
    monkeypatch.setattr(
        "prime_cli.core.client.APIClient.get",
        _gpus_mock_get(gpu_types=["B200_180GB"], on_demand=[_OD_H200]),
    )
    result = CliRunner().invoke(app, ["train", "gpus", "--output", "json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(strip_ansi(result.output))
    assert payload["gpuTypes"] == ["B200_180GB"]
    assert payload["onDemand"][0]["gpuType"] == "H200_141GB"
    assert payload["onDemand"][0]["pricePerGpuHour"] == 4.9


def test_gpus_degrades_when_the_on_demand_endpoint_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reserved types must survive a failure of the newer endpoint.

    This listing worked off `available-gpu-types` alone before on-demand
    existed, so a transient 5xx or schema drift from the discovery
    endpoint must not take the caller's reserved types away with it.
    """

    def mock_get(self: Any, endpoint: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        if endpoint == "/training/available-gpu-types":
            return {"gpuTypes": ["H200_141GB"]}
        if endpoint == "/training/available-fft-models":
            raise APIError("500 Internal Server Error")
        raise AssertionError(f"Unexpected endpoint: {endpoint}")

    monkeypatch.setattr("prime_cli.core.client.APIClient.get", mock_get)
    result = CliRunner().invoke(app, ["train", "gpus"])
    assert result.exit_code == 0, result.output
    out = strip_ansi(result.output)
    assert "H200_141GB" in out, f"reserved types must still render:\n{out}"
    assert "Could not read on-demand capacity" in out
    assert "Contact support" not in out


def test_gpus_escapes_a_discount_label_containing_markup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bracket in a backend promo string must not abort the listing."""
    monkeypatch.setattr(
        "prime_cli.core.client.APIClient.get",
        _gpus_mock_get(
            gpu_types=[],
            on_demand=[{**_OD_H200, "discountLabel": "[/dim]launch [bold]50%"}],
        ),
    )
    result = CliRunner().invoke(app, ["train", "gpus"])
    assert result.exit_code == 0, result.output
    out = strip_ansi(result.output)
    assert "H200_141GB" in out
    assert "launch" in out, f"the label should render as text:\n{out}"


def test_gpus_does_not_call_a_failed_lookup_an_access_denial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A discovery failure is "we do not know", not "you have no access".

    Collapsing the two reinstates the original bug: an on-demand-only
    account whose endpoint 5xxs would be told to contact support for
    access it already has.
    """

    def mock_get(self: Any, endpoint: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        if endpoint == "/training/available-gpu-types":
            return {"gpuTypes": []}
        if endpoint == "/training/available-fft-models":
            raise APIError("503 Service Unavailable")
        raise AssertionError(f"Unexpected endpoint: {endpoint}")

    monkeypatch.setattr("prime_cli.core.client.APIClient.get", mock_get)
    result = CliRunner().invoke(app, ["train", "gpus"])
    assert result.exit_code == 0, result.output
    out = strip_ansi(result.output)
    assert "Contact support" not in out, (
        f"a failed lookup must not read as an access denial:\n{out}"
    )
    assert "could not be read" in out


def test_gpus_json_stdout_stays_parseable_when_discovery_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The degrade warning must not prefix human text onto the payload."""

    def mock_get(self: Any, endpoint: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        if endpoint == "/training/available-gpu-types":
            return {"gpuTypes": ["H200_141GB"]}
        if endpoint == "/training/available-fft-models":
            raise APIError("503 Service Unavailable")
        raise AssertionError(f"Unexpected endpoint: {endpoint}")

    monkeypatch.setattr("prime_cli.core.client.APIClient.get", mock_get)
    result = CliRunner().invoke(app, ["train", "gpus", "--output", "json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(strip_ansi(result.stdout))
    assert payload["gpuTypes"] == ["H200_141GB"]
    assert payload["onDemand"] == []
    assert "Could not read on-demand capacity" in strip_ansi(result.stderr)
