from __future__ import annotations

from typing import Any, Dict

import httpx
import pytest
from prime_cli.api.inference import InferenceAPIError, InferenceClient
from prime_cli.main import app
from typer.testing import CliRunner

TEST_ENV: Dict[str, str] = {
    "COLUMNS": "200",
    "LINES": "50",
    "NO_COLOR": "1",
    "PRIME_DISABLE_VERSION_CHECK": "1",
}


class NoKeyConfig:
    api_key = ""
    inference_url = "https://api.pinference.ai/api/v1"
    team_id = "team-1"


def test_inference_client_requires_api_key_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("prime_cli.api.inference.Config", lambda: NoKeyConfig())

    with pytest.raises(InferenceAPIError, match="No API key"):
        InferenceClient()


def test_inference_client_can_list_models_without_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("prime_cli.api.inference.Config", lambda: NoKeyConfig())
    created: Dict[str, Any] = {}

    class DummyHTTPClient:
        def get(self, url: str) -> httpx.Response:
            created["url"] = url
            request = httpx.Request("GET", url)
            return httpx.Response(200, request=request, json={"object": "list", "data": []})

    def fake_http_client(**kwargs: Any) -> DummyHTTPClient:
        created["headers"] = kwargs["headers"]
        return DummyHTTPClient()

    monkeypatch.setattr("prime_cli.api.inference.httpx.Client", fake_http_client)

    client = InferenceClient(require_auth=False)

    assert client.list_models() == {"object": "list", "data": []}
    assert created["url"] == "https://api.pinference.ai/api/v1/models"
    assert "Authorization" not in created["headers"]
    assert "X-Prime-Team-ID" not in created["headers"]


def test_models_command_uses_optional_auth_client(monkeypatch: pytest.MonkeyPatch) -> None:
    seen_kwargs: Dict[str, Any] = {}

    class DummyModelsClient:
        def __init__(self, **kwargs: Any) -> None:
            seen_kwargs.update(kwargs)

        def list_models(self) -> Dict[str, Any]:
            return {"object": "list", "data": [{"id": "qwen/qwen3-8b"}]}

    monkeypatch.setattr("prime_cli.commands.inference.InferenceClient", DummyModelsClient)

    result = CliRunner().invoke(
        app,
        ["inference", "models", "--output", "json"],
        env=TEST_ENV,
    )

    assert result.exit_code == 0, result.output
    assert seen_kwargs["require_auth"] is False
    assert '"id": "qwen/qwen3-8b"' in result.output


def _models_fixture() -> Dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {
                "id": "zeta/cheap",
                "pricing": {"input_usd_per_mtok": 0.10, "output_usd_per_mtok": 0.40},
            },
            {
                "id": "alpha/premium",
                "pricing": {"input_usd_per_mtok": 5.00, "output_usd_per_mtok": 15.00},
            },
            {
                "id": "mid/standard",
                "pricing": {"input_usd_per_mtok": 1.00, "output_usd_per_mtok": 2.00},
            },
            {"id": "no/pricing"},
        ],
    }


def _patch_models(monkeypatch: pytest.MonkeyPatch, payload: Any) -> None:
    class DummyClient:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def list_models(self) -> Any:
            return payload

    monkeypatch.setattr("prime_cli.commands.inference.InferenceClient", DummyClient)


def _ids_in_order(stdout: str, ids: list[str]) -> list[int]:
    return [stdout.find(i) for i in ids]


def test_models_default_table_omits_created_column(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_models(monkeypatch, {"object": "list", "data": [{"id": "x", "created": 1700000000}]})

    result = CliRunner().invoke(app, ["inference", "models"], env=TEST_ENV)

    assert result.exit_code == 0, result.output
    assert "created" not in result.output.lower()
    assert "x" in result.output


def test_models_search_filters_by_id(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_models(monkeypatch, _models_fixture())

    result = CliRunner().invoke(app, ["inference", "models", "--search", "ALPHA"], env=TEST_ENV)

    assert result.exit_code == 0, result.output
    assert "alpha/premium" in result.output
    assert "zeta/cheap" not in result.output
    assert "mid/standard" not in result.output


def test_models_sort_by_input_price_ascending(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_models(monkeypatch, _models_fixture())

    result = CliRunner().invoke(app, ["inference", "models", "--sort", "input"], env=TEST_ENV)

    assert result.exit_code == 0, result.output
    positions = _ids_in_order(
        result.output, ["zeta/cheap", "mid/standard", "alpha/premium", "no/pricing"]
    )
    assert all(p >= 0 for p in positions)
    assert positions == sorted(positions)


def test_models_sort_by_output_price_descending(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_models(monkeypatch, _models_fixture())

    result = CliRunner().invoke(
        app,
        ["inference", "models", "--sort", "output", "--order", "desc"],
        env=TEST_ENV,
    )

    assert result.exit_code == 0, result.output
    positions = _ids_in_order(
        result.output, ["alpha/premium", "mid/standard", "zeta/cheap", "no/pricing"]
    )
    assert all(p >= 0 for p in positions)
    assert positions == sorted(positions)


def test_models_sort_id_default_is_alphabetical(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_models(monkeypatch, _models_fixture())

    result = CliRunner().invoke(app, ["inference", "models"], env=TEST_ENV)

    assert result.exit_code == 0, result.output
    positions = _ids_in_order(
        result.output, ["alpha/premium", "mid/standard", "no/pricing", "zeta/cheap"]
    )
    assert all(p >= 0 for p in positions)
    assert positions == sorted(positions)


def test_models_invalid_sort_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_models(monkeypatch, _models_fixture())

    result = CliRunner().invoke(app, ["inference", "models", "--sort", "nope"], env=TEST_ENV)

    assert result.exit_code == 1
    assert "--sort must be one of" in result.output


def test_models_short_flags_work(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_models(monkeypatch, _models_fixture())

    result = CliRunner().invoke(
        app,
        ["inference", "models", "-q", "/", "-s", "input", "-d", "desc"],
        env=TEST_ENV,
    )

    assert result.exit_code == 0, result.output
    positions = _ids_in_order(
        result.output, ["alpha/premium", "mid/standard", "zeta/cheap", "no/pricing"]
    )
    assert all(p >= 0 for p in positions)
    assert positions == sorted(positions)


def test_models_json_output_applies_search_and_sort(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_models(monkeypatch, _models_fixture())

    result = CliRunner().invoke(
        app,
        [
            "inference",
            "models",
            "--output",
            "json",
            "--search",
            "/",
            "--sort",
            "input",
            "--order",
            "desc",
        ],
        env=TEST_ENV,
    )

    assert result.exit_code == 0, result.output
    positions = _ids_in_order(
        result.output, ["alpha/premium", "mid/standard", "zeta/cheap", "no/pricing"]
    )
    assert all(p >= 0 for p in positions)
    assert positions == sorted(positions)


def _catalog_fixture() -> Dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {
                "id": "anthropic/claude-haiku-4.5",
                "display_name": "Claude Haiku 4.5",
                "pricing": {
                    "input_usd_per_mtok": 1.0,
                    "output_usd_per_mtok": 5.0,
                    "cache_read_usd_per_mtok": 0.1,
                    "cache_write_usd_per_mtok": 1.25,
                },
                "specs": {
                    "context_window": 200000,
                    "max_output_tokens": 64000,
                    "modalities": {"input": ["text", "image", "file"], "output": ["text"]},
                    "supports_reasoning": True,
                },
            },
            {
                "id": "prime/hosted-model",
                "pricing": {
                    "input_usd_per_mtok": 0.11,
                    "output_usd_per_mtok": 0.44,
                    "cache_read_usd_per_mtok": 0.099,
                    "cache_write_usd_per_mtok": 0.11,
                },
            },
            {
                "id": "partial/cache-read-only",
                "pricing": {
                    "input_usd_per_mtok": 0.3,
                    "output_usd_per_mtok": 0.6,
                    "cache_read_usd_per_mtok": 0.03,
                },
            },
        ],
    }


def test_models_table_shows_catalog_columns(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_models(monkeypatch, _catalog_fixture())

    result = CliRunner().invoke(app, ["inference", "models"], env=TEST_ENV)

    assert result.exit_code == 0, result.output
    out = result.output
    assert "Claude Haiku 4.5" in out
    assert "context" in out
    assert "200k" in out
    assert "64k" in out
    assert "✓" in out
    # Modalities are upstream model data, not a productized gateway feature —
    # never rendered even when the endpoint serves them.
    assert "modalities" not in out
    assert "t+i+f" not in out
    # Cache read/write cell for the first model.
    assert "$0.1 / $1.25" in out
    # Second model has no specs -> em-dash cells, but cache pricing still shown.
    assert "prime/hosted-model" in out
    assert "$0.099 / $0.11" in out
    # Partial cache pricing marks the missing side explicitly.
    assert "$0.03 / —" in out
    # Legend explains the reasoning column.
    assert "reasoning ✓" in out


def test_models_table_folds_ids_in_narrow_terminals(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_models(monkeypatch, _catalog_fixture())

    result = CliRunner().invoke(app, ["inference", "models"], env={**TEST_ENV, "COLUMNS": "80"})

    assert result.exit_code == 0, result.output
    # The id column folds (wraps) instead of truncating, so the identifier's
    # tail stays visible and copyable even when catalog columns squeeze the
    # table (truncation would render 'anthrop…' and drop the tail).
    assert "u-4.5" in result.output


def test_models_table_omits_catalog_columns_for_legacy_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_models(monkeypatch, _models_fixture())

    result = CliRunner().invoke(app, ["inference", "models"], env=TEST_ENV)

    assert result.exit_code == 0, result.output
    out = result.output
    # No model carries catalog data -> slim table, unchanged from before.
    assert "name" not in out
    assert "context" not in out
    assert "reasoning" not in out
    assert "cache" not in out


def test_models_json_output_passes_catalog_fields_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_models(monkeypatch, _catalog_fixture())

    result = CliRunner().invoke(app, ["inference", "models", "--output", "json"], env=TEST_ENV)

    assert result.exit_code == 0, result.output
    assert '"display_name": "Claude Haiku 4.5"' in result.output
    assert '"context_window": 200000' in result.output
    assert '"cache_read_usd_per_mtok": 0.1' in result.output
