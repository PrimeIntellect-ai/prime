from pathlib import Path

import pytest
import typer
from prime_cli.commands.rl import (
    RLConfig,
    _flatten_config_schema,
    generate_rl_config_template,
    load_config,
)


def test_load_config_warns_and_ignores_deprecated_trajectory_strategy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "rl.toml"
    config_path.write_text(
        'model = "PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT"\n'
        'trajectory_strategy = "interleaved"\n'
    )

    printed: list[str] = []

    def capture_print(*args: object, **_: object) -> None:
        printed.append(" ".join(str(arg) for arg in args))

    monkeypatch.setattr("prime_cli.commands.rl.console.print", capture_print)

    cfg = load_config(str(config_path))

    assert cfg.model == "PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT"
    assert "trajectory_strategy" not in cfg.model_dump()
    assert any("trajectory_strategy" in line and "deprecated" in line.lower() for line in printed)


def test_load_config_still_rejects_other_unknown_keys(tmp_path: Path) -> None:
    config_path = tmp_path / "rl.toml"
    config_path.write_text(
        'model = "PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT"\nunknown_field = 123\n'
    )

    with pytest.raises(typer.Exit):
        load_config(str(config_path))


def test_generate_rl_config_template_uses_broad_buffer_threshold_examples() -> None:
    template = generate_rl_config_template()

    assert "# easy_threshold = 1.0" in template
    assert "# hard_threshold = 0.0" in template


def test_flatten_config_schema_expands_optional_nested_models() -> None:
    schema = RLConfig.model_json_schema()
    rows = _flatten_config_schema(schema, schema.get("$defs", {}))
    paths = {path for path, _, _ in rows}

    assert "sampling.temp_scheduler.type" in paths
    assert "sampling.temp_scheduler.start_temperature" in paths
    assert "sampling.temp_scheduler" not in paths


def test_flatten_config_schema_preserves_optional_array_item_types() -> None:
    schema = RLConfig.model_json_schema()
    rows = {
        path: type_str
        for path, type_str, _ in _flatten_config_schema(schema, schema.get("$defs", {}))
    }

    assert rows["buffer.env_ratios"] == "list[number]"


def test_load_config_accepts_sampling_reasoning_effort(tmp_path: Path) -> None:
    config_path = tmp_path / "rl.toml"
    config_path.write_text('model = "openai/gpt-oss-20b"\n[sampling]\nreasoning_effort = "high"\n')

    cfg = load_config(str(config_path))

    assert cfg.sampling.reasoning_effort == "high"
    assert cfg.sampling.enable_thinking is None


def test_load_config_accepts_sampling_enable_thinking(tmp_path: Path) -> None:
    config_path = tmp_path / "rl.toml"
    config_path.write_text('model = "Qwen/Qwen3.5-35B-A3B"\n[sampling]\nenable_thinking = false\n')

    cfg = load_config(str(config_path))

    assert cfg.sampling.enable_thinking is False
    assert cfg.sampling.reasoning_effort is None


def test_load_config_rejects_both_reasoning_controls(tmp_path: Path) -> None:
    config_path = tmp_path / "rl.toml"
    config_path.write_text(
        'model = "openai/gpt-oss-20b"\n'
        "[sampling]\n"
        "enable_thinking = false\n"
        'reasoning_effort = "low"\n'
    )

    with pytest.raises(typer.Exit):
        load_config(str(config_path))


def test_load_config_rejects_top_level_reasoning_effort(tmp_path: Path) -> None:
    config_path = tmp_path / "rl.toml"
    config_path.write_text('model = "openai/gpt-oss-20b"\nreasoning_effort = "high"\n')

    with pytest.raises(typer.Exit):
        load_config(str(config_path))


def test_tailscale_config_disabled_by_default() -> None:
    cfg = RLConfig(model="dummy")
    assert cfg.tailscale.enabled is False
    assert cfg.tailscale.to_api_dict() is None


def test_tailscale_config_enabled_emits_payload(tmp_path: Path) -> None:
    config_path = tmp_path / "rl.toml"
    config_path.write_text(
        'model = "dummy"\n'
        "[tailscale]\n"
        "enabled = true\n"
        'auth_key = "tskey-auth-abc123"\n'
        'hostname_prefix = "rft"\n'
    )
    cfg = load_config(str(config_path))
    assert cfg.tailscale.enabled is True
    payload = cfg.tailscale.to_api_dict()
    # Match the file-wide convention: drop unset Optional fields rather than
    # sending explicit nulls.
    assert payload == {
        "enabled": True,
        "auth_key": "tskey-auth-abc123",
        "hostname_prefix": "rft",
    }


def test_tailscale_to_api_dict_includes_extra_args_when_set(tmp_path: Path) -> None:
    config_path = tmp_path / "rl.toml"
    config_path.write_text(
        'model = "dummy"\n'
        "[tailscale]\n"
        "enabled = true\n"
        'auth_key = "tskey-auth-abc"\n'
        'extra_args = "--advertise-tags=tag:rft-debug"\n'
    )
    cfg = load_config(str(config_path))
    payload = cfg.tailscale.to_api_dict()
    assert payload is not None
    assert payload["extra_args"] == "--advertise-tags=tag:rft-debug"


def test_tailscale_enabled_without_auth_key_rejected(tmp_path: Path) -> None:
    config_path = tmp_path / "rl.toml"
    config_path.write_text('model = "dummy"\n[tailscale]\nenabled = true\n')
    with pytest.raises(typer.Exit):
        load_config(str(config_path))


def test_tailscale_invalid_hostname_prefix_rejected(tmp_path: Path) -> None:
    config_path = tmp_path / "rl.toml"
    config_path.write_text(
        'model = "dummy"\n'
        "[tailscale]\n"
        "enabled = true\n"
        'auth_key = "tskey-auth-abc"\n'
        'hostname_prefix = "rft-"\n'
    )
    with pytest.raises(typer.Exit):
        load_config(str(config_path))


def test_tailscale_invalid_auth_key_format_rejected(tmp_path: Path) -> None:
    config_path = tmp_path / "rl.toml"
    config_path.write_text(
        'model = "dummy"\n[tailscale]\nenabled = true\nauth_key = "not-a-real-key"\n'
    )
    with pytest.raises(typer.Exit):
        load_config(str(config_path))


def test_tailscale_accepts_oauth_client_secret(tmp_path: Path) -> None:
    config_path = tmp_path / "rl.toml"
    config_path.write_text(
        'model = "dummy"\n[tailscale]\nenabled = true\nauth_key = "tskey-client-abc123"\n'
    )
    cfg = load_config(str(config_path))
    assert cfg.tailscale.auth_key == "tskey-client-abc123"
