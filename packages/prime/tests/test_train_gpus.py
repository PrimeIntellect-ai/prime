"""Tests for `prime train gpus`.

The list is what a user copies into `--gpu-type`, so it spans both
placement paths the backend dispatches on: reserved capacity for types
the caller owns, the billed shared pool for everything else. It once
showed reserved types only, which contradicted `prime train models`
(RTX models and RTX on-demand capacity were listed there) while the
dispatch itself accepted `--gpu-type RTX_PRO_6000B_96GB` fine.
"""

from typing import Any

import pytest
from prime_cli.main import app
from prime_cli.utils.formatters import strip_ansi
from typer.testing import CliRunner

ENDPOINT = "/training/available-gpu-types"


@pytest.fixture(autouse=True)
def _api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PRIME_API_KEY", "dummy")
    monkeypatch.setenv("PRIME_DISABLE_VERSION_CHECK", "1")


def _mock_get_factory(payload: dict[str, Any]):
    def mock_get(self: Any, endpoint: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        if endpoint == ENDPOINT:
            return payload
        raise AssertionError(f"Unexpected endpoint: {endpoint}")

    return mock_get


def _invoke(monkeypatch: pytest.MonkeyPatch, payload: dict[str, Any], *args: str):
    monkeypatch.setattr("prime_cli.core.APIClient.get", _mock_get_factory(payload))
    return CliRunner().invoke(app, ["train", "gpus", *args], env={"COLUMNS": "200"})


def test_on_demand_types_are_listed_and_marked(monkeypatch: pytest.MonkeyPatch) -> None:
    result = _invoke(
        monkeypatch,
        {
            "gpuTypes": ["H100_80GB", "H200_141GB", "RTX_PRO_6000B_96GB"],
            "onDemandGpuTypes": ["RTX_PRO_6000B_96GB"],
        },
    )

    assert result.exit_code == 0, result.output
    plain = strip_ansi(result.output)
    assert "RTX_PRO_6000B_96GB" in plain
    # Only the pool type carries the billing marker; a reserved type is
    # already paid for and must not read as an hourly charge.
    on_demand_line = next(ln for ln in plain.splitlines() if "RTX_PRO_6000B_96GB" in ln)
    reserved_line = next(ln for ln in plain.splitlines() if "H100_80GB" in ln)
    assert "on-demand" in on_demand_line
    assert "on-demand" not in reserved_line
    assert "prime train models" in plain


def test_reserved_only_listing_mentions_no_billing(monkeypatch: pytest.MonkeyPatch) -> None:
    result = _invoke(
        monkeypatch,
        {"gpuTypes": ["H200_141GB"], "onDemandGpuTypes": []},
    )

    assert result.exit_code == 0, result.output
    plain = strip_ansi(result.output)
    assert "H200_141GB" in plain
    assert "on-demand" not in plain


def test_missing_on_demand_field_degrades_to_a_plain_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A backend predating `onDemandGpuTypes` must not break the command."""
    result = _invoke(monkeypatch, {"gpuTypes": ["H200_141GB"]})

    assert result.exit_code == 0, result.output
    assert "H200_141GB" in strip_ansi(result.output)


def test_json_output_carries_both_lists(monkeypatch: pytest.MonkeyPatch) -> None:
    import json

    result = _invoke(
        monkeypatch,
        {
            "gpuTypes": ["H200_141GB", "RTX_PRO_6000B_96GB"],
            "onDemandGpuTypes": ["RTX_PRO_6000B_96GB"],
        },
        "--output",
        "json",
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(strip_ansi(result.output))
    assert payload == {
        "gpuTypes": ["H200_141GB", "RTX_PRO_6000B_96GB"],
        "onDemandGpuTypes": ["RTX_PRO_6000B_96GB"],
    }


def test_empty_listing_keeps_the_support_banner(monkeypatch: pytest.MonkeyPatch) -> None:
    result = _invoke(monkeypatch, {"gpuTypes": [], "onDemandGpuTypes": []})

    assert result.exit_code == 0, result.output
    assert "No GPU types available" in strip_ansi(result.output)
