"""SFT dispatch tests for `prime train` (dedicated training path).

Covers the three layers of the SFT handoff:
  - detection: `[data]` + no `[trainer]`/`[orchestrator]` -> SFT, `--sft` flag
  - client-side fail-fast UX: missing `[data]`, RL blocks, online-eval blocks
  - payload building: `mode: "sft"` on the /training/runs request; RL
    payloads stay mode-free
"""

from pathlib import Path
from typing import Any

import toml
from prime_cli.api.training import build_payload_from_toml
from prime_cli.commands.rl import _is_sft, _looks_like_sft, _validate_sft_config
from prime_cli.main import app
from typer.testing import CliRunner

runner = CliRunner()

TEST_ENV = {"PRIME_DISABLE_VERSION_CHECK": "1", "PRIME_API_KEY": "test-key"}


def _sft_config(**overrides: Any) -> str:
    """A minimal valid SFT mega-TOML (prime-rl SFTConfig shape): a [data]
    block, no [trainer]/[orchestrator]. Same shape as the fft example."""
    cfg: dict[str, Any] = {
        "max_steps": 10,
        "model": {"name": "PrimeIntellect/Qwen3-0.6B"},
        "data": {"name": "PrimeIntellect/Reverse-Text-SFT", "seq_len": 1024},
        "deployment": {"type": "single_node", "num_train_gpus": 1},
    }
    for key, value in overrides.items():
        if value is None:
            cfg.pop(key, None)
        else:
            cfg[key] = value
    return toml.dumps(cfg)


def _write_config(tmp_path: Path, content: str) -> str:
    path = tmp_path / "sft.toml"
    path.write_text(content)
    return str(path)


def _capture_post(monkeypatch) -> dict[str, Any]:
    """Mock the API layer's post and capture the /training/runs payload."""
    captured: dict[str, Any] = {}

    def mock_post(self: Any, endpoint: str, json: dict[str, Any] | None = None) -> dict:
        captured["endpoint"] = endpoint
        captured["json"] = json
        return {"runId": "run-1", "tokenValue": "tok"}

    monkeypatch.setattr("prime_cli.client.APIClient.post", mock_post)
    return captured


# --- detection -------------------------------------------------------------


def test_sft_detection_requires_data_without_rl_blocks() -> None:
    assert _looks_like_sft({"data": {"name": "d"}}) is True
    assert _looks_like_sft({"data": {}, "deployment": {}}) is True
    # [trainer]/[orchestrator] mean the RL schema
    assert _looks_like_sft({"data": {}, "trainer": {}}) is False
    assert _looks_like_sft({"data": {}, "orchestrator": {}}) is False
    # no [data] means not SFT (LoRA/RL configs never carry one)
    assert _looks_like_sft({"trainer": {}, "orchestrator": {}}) is False
    assert _looks_like_sft({}) is False


def test_sft_flag_forces_the_sft_path() -> None:
    assert _is_sft({}, flag=True) is True
    assert _is_sft({"trainer": {}}, flag=True) is True  # flag forces; validation rejects
    assert _is_sft({"data": {}}, flag=False) is True
    assert _is_sft({}, flag=False) is False


# --- fail-fast validation ----------------------------------------------------


def _validate_exits(cfg: dict[str, Any]) -> Any:
    import pytest
    from typer import Exit

    with pytest.raises(Exit) as excinfo:
        _validate_sft_config(cfg, "sft.toml")
    return excinfo.value


def test_sft_validation_requires_data_block() -> None:
    _validate_exits({"model": {}})


def test_sft_validation_rejects_rl_blocks() -> None:
    _validate_exits({"data": {}, "trainer": {}, "orchestrator": {}})


def test_sft_validation_rejects_online_eval_blocks() -> None:
    for block in ("eval", "inference", "weight_broadcast"):
        _validate_exits({"data": {}, block: {}})


def test_sft_validation_accepts_trainer_only_config() -> None:
    # No exit: the bare trainer-only shape passes client-side checks.
    _validate_sft_config({"data": {}, "deployment": {}, "ckpt": {}}, "sft.toml")


# --- payload building --------------------------------------------------------


def test_build_payload_includes_sft_mode() -> None:
    payload = build_payload_from_toml({"data": {}}, name="run", mode="sft")
    assert payload["mode"] == "sft"
    assert payload["config"] == {"data": {}}


def test_build_payload_omits_mode_by_default() -> None:
    payload = build_payload_from_toml({"trainer": {}}, name="run")
    assert "mode" not in payload


# --- end-to-end CLI dispatch -------------------------------------------------


def test_train_sft_toml_auto_dispatches_with_sft_mode(tmp_path: Path, monkeypatch) -> None:
    config_path = _write_config(tmp_path, _sft_config())
    captured = _capture_post(monkeypatch)

    result = runner.invoke(app, ["train", config_path, "--yes"], env=TEST_ENV)

    assert result.exit_code == 0, result.output
    assert captured["endpoint"] == "/training/runs"
    payload = captured["json"]
    assert payload["mode"] == "sft"
    assert payload["config"]["data"]["name"] == "PrimeIntellect/Reverse-Text-SFT"
    assert "trainer" not in payload["config"] and "orchestrator" not in payload["config"]
    assert "Dispatched" in result.output


def test_train_sft_flag_dispatches_valid_sft_config(tmp_path: Path, monkeypatch) -> None:
    config_path = _write_config(tmp_path, _sft_config())
    captured = _capture_post(monkeypatch)

    result = runner.invoke(app, ["train", config_path, "--sft", "--yes"], env=TEST_ENV)

    assert result.exit_code == 0, result.output
    assert captured["json"]["mode"] == "sft"


def test_train_sft_json_output_keeps_run_id_parseable(tmp_path: Path, monkeypatch) -> None:
    config_path = _write_config(tmp_path, _sft_config())
    _capture_post(monkeypatch)

    result = runner.invoke(app, ["train", config_path, "--yes", "--output", "json"], env=TEST_ENV)

    assert result.exit_code == 0, result.output
    import json

    data = json.loads(result.stdout)
    assert data["run"]["runId"] == "run-1"


def test_train_rejects_sft_config_with_online_eval_blocks(tmp_path: Path, monkeypatch) -> None:
    _capture_post(monkeypatch)  # nothing should be posted
    config_path = _write_config(
        tmp_path, _sft_config(eval={"sources": {}}, inference={"model": {}})
    )

    result = runner.invoke(app, ["train", config_path, "--yes"], env=TEST_ENV)

    assert result.exit_code == 1
    assert "trainer-only" in result.output
    assert "[eval]" in result.output and "[inference]" in result.output


def test_train_rejects_sft_flag_without_data_block(tmp_path: Path, monkeypatch) -> None:
    _capture_post(monkeypatch)
    config_path = _write_config(
        tmp_path, toml.dumps({"model": {"name": "m"}, "deployment": {"type": "single_node"}})
    )

    result = runner.invoke(app, ["train", config_path, "--sft", "--yes"], env=TEST_ENV)

    assert result.exit_code == 1
    assert "--sft requires a [data] block" in result.output


def test_train_rejects_sft_flag_on_rl_config(tmp_path: Path, monkeypatch) -> None:
    _capture_post(monkeypatch)
    rl_cfg = toml.dumps(
        {
            "trainer": {"model": {"name": "m"}},
            "orchestrator": {"env": {}},
            "deployment": {"type": "single_node", "num_train_gpus": 1},
        }
    )
    config_path = _write_config(tmp_path, rl_cfg)

    result = runner.invoke(app, ["train", config_path, "--sft", "--yes"], env=TEST_ENV)

    assert result.exit_code == 1
    assert "RL mega-TOML" in result.output


def test_train_rejects_fft_flag_on_sft_config(tmp_path: Path, monkeypatch) -> None:
    _capture_post(monkeypatch)
    config_path = _write_config(tmp_path, _sft_config())

    result = runner.invoke(app, ["train", config_path, "--fft", "--yes"], env=TEST_ENV)

    assert result.exit_code == 1
    assert "Use --sft instead" in result.output


def test_train_rejects_sft_and_fft_flags_together(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, _sft_config())

    result = runner.invoke(app, ["train", config_path, "--sft", "--fft", "--yes"], env=TEST_ENV)

    assert result.exit_code == 1
    assert "mutually exclusive" in result.output


def test_train_fft_dispatch_stays_mode_free(tmp_path: Path, monkeypatch) -> None:
    """Regression: the RL full-FT payload must not grow a mode key."""
    captured = _capture_post(monkeypatch)
    rl_cfg = toml.dumps(
        {
            "trainer": {"model": {"name": "m"}},
            "orchestrator": {"env": {}},
            "deployment": {"type": "single_node", "num_train_gpus": 1},
        }
    )
    config_path = _write_config(tmp_path, rl_cfg)

    result = runner.invoke(app, ["train", config_path, "--yes"], env=TEST_ENV)

    assert result.exit_code == 0, result.output
    assert captured["json"]["config"]["trainer"]["model"]["name"] == "m"
    assert "mode" not in captured["json"]
