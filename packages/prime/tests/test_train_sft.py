"""SFT dispatch tests for `prime train` (dedicated training path).

Covers the SFT handoff:
  - detection: `[data]` + no `[trainer]`/`[orchestrator]` -> SFT, `--sft` flag
  - payload building: `mode: "sft"` on the /training/runs request; RL
    payloads stay mode-free

Config-shape validation (missing `[data]`, RL blocks, online-eval blocks)
lives server-side: the admission check returns a 400 synchronously, so the
CLI doesn't duplicate it.
"""

from pathlib import Path
from typing import Any

import toml
from prime_cli.api.training import build_payload_from_toml
from prime_cli.commands.rl import _is_sft, _looks_like_sft
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
    assert _is_sft({"trainer": {}}, flag=True) is True  # flag forces; server validates
    assert _is_sft({"data": {}}, flag=False) is True
    assert _is_sft({}, flag=False) is False


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

    result = runner.invoke(app, ["train", config_path, "--yes", "--json"], env=TEST_ENV)

    assert result.exit_code == 0, result.output
    import json

    data = json.loads(result.stdout)
    assert data["run"]["runId"] == "run-1"


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


# --- --volume forwarding -----------------------------------------------------
#
# main's dispatch auto-creates a missing --volume (ENG-6354 #975): the SFT
# --volume tests need list_volumes to report the volume as existing so the
# dispatch skips creation and posts the run payload directly.


def _existing_volume(monkeypatch, name: str) -> None:
    """Make the dispatch see `name` as an existing RUNNING volume."""

    class _V:
        pass

    v = _V()
    v.name = name
    v.size = "1Ti"
    monkeypatch.setattr(
        "prime_cli.api.training.HostedTrainingClient.list_volumes",
        lambda self, team_id=None: [v],
    )


def test_train_sft_forwards_volume_flag_to_payload(tmp_path: Path, monkeypatch) -> None:
    """--volume must reach the SFT dispatch: the volume carries both the
    run's outputs (runs/<runId>/) and the dataset read at
    /volume/<name>, so dropping it would silently launch an SFT run that
    can't see its dataset."""
    config_path = _write_config(tmp_path, _sft_config())
    captured = _capture_post(monkeypatch)
    _existing_volume(monkeypatch, "research")

    result = runner.invoke(
        app, ["train", config_path, "--volume", "research", "--yes"], env=TEST_ENV
    )

    assert result.exit_code == 0, result.output
    payload = captured["json"]
    assert payload["mode"] == "sft"
    assert payload["volume"] == "research"
    # Request-level key: never embedded in the prime-rl config.
    assert "volume" not in payload["config"]


def test_train_sft_forwards_toml_volume_to_payload(tmp_path: Path, monkeypatch) -> None:
    config_path = _write_config(tmp_path, _sft_config(volume="research"))
    captured = _capture_post(monkeypatch)
    _existing_volume(monkeypatch, "research")

    result = runner.invoke(app, ["train", config_path, "--yes"], env=TEST_ENV)

    assert result.exit_code == 0, result.output
    payload = captured["json"]
    assert payload["mode"] == "sft"
    assert payload["volume"] == "research"
    assert "volume" not in payload["config"]


def test_train_sft_volume_flag_overrides_toml(tmp_path: Path, monkeypatch) -> None:
    config_path = _write_config(tmp_path, _sft_config(volume="from-toml"))
    captured = _capture_post(monkeypatch)
    _existing_volume(monkeypatch, "from-flag")

    result = runner.invoke(
        app, ["train", config_path, "--volume", "from-flag", "--yes"], env=TEST_ENV
    )

    assert result.exit_code == 0, result.output
    payload = captured["json"]
    assert payload["mode"] == "sft"
    assert payload["volume"] == "from-flag"
    assert "volume" not in payload["config"]


def test_train_sft_without_volume_omits_the_key(tmp_path: Path, monkeypatch) -> None:
    """A fake-data SFT (no named volume) must dispatch without a volume
    key, exactly like the full-FT path."""
    config_path = _write_config(tmp_path, _sft_config(data={"name": "fake", "seq_len": 1024}))
    captured = _capture_post(monkeypatch)

    result = runner.invoke(app, ["train", config_path, "--yes"], env=TEST_ENV)

    assert result.exit_code == 0, result.output
    payload = captured["json"]
    assert payload["mode"] == "sft"
    assert "volume" not in payload


def test_train_sft_rejects_non_string_volume_before_post(tmp_path: Path, monkeypatch) -> None:
    """A top-level `volume` that is not a string must fail client-side,
    before anything is posted — mirrors the shared helper's guard."""
    captured = _capture_post(monkeypatch)
    for bad_volume in (42, ["research"]):
        config_path = _write_config(tmp_path, _sft_config(volume=bad_volume))

        # The guard message embeds the tmp config path; at Rich's 80-col
        # default (CI: no tty) the line wraps and can split the asserted
        # substrings. Widen the console so the message never wraps.
        result = runner.invoke(
            app, ["train", config_path, "--yes"], env={**TEST_ENV, "COLUMNS": "200"}
        )

        assert result.exit_code == 1, result.output
        assert "volume in" in result.output and "must be a string" in result.output
        assert "json" not in captured  # the guard fires before any POST


def test_train_sft_missing_volume_created_with_requested_size(tmp_path: Path, monkeypatch) -> None:
    """--volume-size must flow through the SFT dispatch too: the
    auto-created dataset volume honors it instead of the 1Ti default."""
    config_path = _write_config(tmp_path, _sft_config())
    captured = _capture_post(monkeypatch)

    state: list[Any] = []
    creates: list[tuple[str, str]] = []

    def create_volume(self: Any, name: str, size: str, team_id=None) -> Any:
        creates.append((name, size))

        class _V:
            pass

        created = _V()
        created.name = name
        created.size = size
        created.status = "PENDING"
        running = _V()
        running.name = name
        running.size = size
        running.status = "RUNNING"
        state.append(running)
        return created

    monkeypatch.setattr(
        "prime_cli.api.training.HostedTrainingClient.list_volumes",
        lambda self, team_id=None: list(state),
    )
    monkeypatch.setattr("prime_cli.api.training.HostedTrainingClient.create_volume", create_volume)
    monkeypatch.setattr("prime_cli.commands.rl.time.sleep", lambda s: None)

    result = runner.invoke(
        app,
        ["train", config_path, "--volume", "research", "--volume-size", "500Gi", "--yes"],
        env=TEST_ENV,
    )

    assert result.exit_code == 0, result.output
    assert creates == [("research", "500Gi")]
    payload = captured["json"]
    assert payload["mode"] == "sft"
    assert payload["volume"] == "research"
