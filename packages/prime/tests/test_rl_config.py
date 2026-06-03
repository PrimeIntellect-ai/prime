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


def test_generate_rl_config_template_keeps_default_surface_minimal() -> None:
    template = generate_rl_config_template()

    assert 'model = "Qwen/Qwen3.5-0.8B"' in template
    assert "# learning_rate = 3e-5 # optional; default is 1e-4" in template

    hidden_fields = [
        "oversampling_factor",
        "max_async_level",
        "lora_alpha",
        "repetition_penalty",
        "min_tokens",
        "temp_scheduler",
        "seed",
        "[wandb]",
        "[infrastructure]",
    ]
    for field in hidden_fields:
        assert field not in template


def test_generate_rl_config_template_sft_example_loads(tmp_path: Path) -> None:
    template = generate_rl_config_template()
    template = template.replace(
        'loss = "rl" # "rl" | "sft"; OPD is not yet supported on hosted runtimes',
        'loss = "sft" # "rl" | "sft"; OPD is not yet supported on hosted runtimes',
    )

    lines: list[str] = []
    in_teacher_example = False
    for line in template.splitlines():
        if line == "# Optional: SFT distillation teacher":
            in_teacher_example = True
            lines.append(line)
            continue
        if in_teacher_example and line == "":
            in_teacher_example = False
        if in_teacher_example and line.startswith("# ") and not line.startswith("# To use"):
            line = line[2:]
        lines.append(line)

    config_path = tmp_path / "sft-template.toml"
    config_path.write_text("\n".join(lines) + "\n")

    cfg = load_config(str(config_path))

    assert cfg.loss == "sft"
    assert cfg.teacher is not None
    assert cfg.teacher.model == "openai/gpt-oss-120b"
    assert cfg.teacher.client is None


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


def test_load_config_accepts_max_inflight_rollouts(tmp_path: Path) -> None:
    config_path = tmp_path / "rl.toml"
    config_path.write_text('model = "dummy"\nmax_inflight_rollouts = 96\n')

    cfg = load_config(str(config_path))

    assert cfg.max_inflight_rollouts == 96


def test_load_config_accepts_fractional_oversampling_factor(tmp_path: Path) -> None:
    config_path = tmp_path / "rl.toml"
    config_path.write_text('model = "dummy"\noversampling_factor = 0.375\n')

    cfg = load_config(str(config_path))

    assert cfg.oversampling_factor == 0.375


def test_load_config_rejects_max_inflight_and_oversampling(tmp_path: Path) -> None:
    config_path = tmp_path / "rl.toml"
    config_path.write_text(
        'model = "dummy"\n'
        "batch_size = 256\n"
        "rollouts_per_example = 8\n"
        "oversampling_factor = 0.377\n"
        "max_inflight_rollouts = 96\n"
    )

    with pytest.raises(typer.Exit):
        load_config(str(config_path))


def test_load_config_accepts_sft_teacher_without_client(tmp_path: Path) -> None:
    """Omitting [teacher.client] is allowed; the API defaults it to PI Inference."""
    config_path = tmp_path / "sft.toml"
    config_path.write_text(
        'model = "openai/gpt-oss-20b"\n'
        'loss = "sft"\n'
        "[teacher]\n"
        'model = "openai/gpt-oss-120b"\n'
        "[teacher.sampling]\n"
        "max_tokens = 2048\n"
        'reasoning_effort = "medium"\n'
    )

    cfg = load_config(str(config_path))

    assert cfg.loss == "sft"
    assert cfg.teacher is not None
    assert cfg.teacher.model == "openai/gpt-oss-120b"
    assert cfg.teacher.client is None
    assert cfg.teacher.to_api_dict() == {
        "model": {"name": "openai/gpt-oss-120b"},
        "sampling": {
            "max_tokens": 2048,
            "reasoning_effort": "medium",
        },
    }


def test_load_config_accepts_teacher_client(tmp_path: Path) -> None:
    config_path = tmp_path / "sft.toml"
    config_path.write_text(
        'model = "PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT"\n'
        'loss = "sft"\n'
        "[teacher]\n"
        'model = "qwen/qwen-2.5-7b-instruct"\n'
        "[teacher.client]\n"
        'base_url = "https://example.com/inference/api/v1"\n'
        'api_key_var = "PRIME_API_KEY"\n'
        "skip_model_check = true\n"
        "[teacher.client.headers_from_env]\n"
        'X-Prime-Team-ID = "PRIME_TEAM_ID"\n'
        "[teacher.sampling]\n"
        "temperature = 0.7\n"
        "max_tokens = 256\n"
    )

    cfg = load_config(str(config_path))

    assert cfg.teacher is not None
    assert cfg.teacher.client is not None
    assert cfg.teacher.client.base_url == "https://example.com/inference/api/v1"
    assert cfg.teacher.client.api_key_var == "PRIME_API_KEY"
    assert cfg.teacher.client.headers_from_env == {"X-Prime-Team-ID": "PRIME_TEAM_ID"}
    assert cfg.teacher.client.skip_model_check is True
    assert cfg.teacher.to_api_dict() == {
        "model": {"name": "qwen/qwen-2.5-7b-instruct"},
        "client": {
            "base_url": "https://example.com/inference/api/v1",
            "api_key_var": "PRIME_API_KEY",
            "headers_from_env": {"X-Prime-Team-ID": "PRIME_TEAM_ID"},
            "skip_model_check": True,
        },
        "sampling": {
            "temperature": 0.7,
            "max_tokens": 256,
        },
    }


def test_load_config_rejects_unknown_teacher_client_field(tmp_path: Path) -> None:
    config_path = tmp_path / "sft.toml"
    config_path.write_text(
        'model = "PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT"\n'
        'loss = "sft"\n'
        "[teacher]\n"
        'model = "qwen/qwen-2.5-7b-instruct"\n'
        "[teacher.client]\n"
        'base_url = "https://example.com/inference/api/v1"\n'
        'api_key_var = "PRIME_API_KEY"\n'
        'bogus = "value"\n'
    )

    with pytest.raises(typer.Exit):
        load_config(str(config_path))


def test_load_config_rejects_teacher_temp_scheduler(tmp_path: Path) -> None:
    config_path = tmp_path / "sft.toml"
    config_path.write_text(
        'model = "openai/gpt-oss-20b"\n'
        'loss = "sft"\n'
        "[teacher]\n"
        'model = "openai/gpt-oss-120b"\n'
        "[teacher.client]\n"
        'base_url = "https://api.pinference.ai/api/v1"\n'
        'api_key_var = "PRIME_API_KEY"\n'
        "[teacher.sampling.temp_scheduler]\n"
        'type = "linear"\n'
        "start_temperature = 1.0\n"
        "end_temperature = 0.1\n"
    )

    with pytest.raises(typer.Exit):
        load_config(str(config_path))


def test_to_api_dict_omits_client_when_unspecified(tmp_path: Path) -> None:
    """CLI must not duplicate the platform default — when the user omits
    [teacher.client], the API payload also omits ``client`` so the server's
    default kicks in without policy drift."""
    config_path = tmp_path / "sft.toml"
    config_path.write_text(
        'model = "openai/gpt-oss-20b"\nloss = "sft"\n[teacher]\nmodel = "openai/gpt-oss-120b"\n'
    )

    cfg = load_config(str(config_path))

    assert cfg.teacher is not None
    assert "client" not in cfg.teacher.to_api_dict()


def test_load_config_rejects_teacher_client_without_base_url(tmp_path: Path) -> None:
    config_path = tmp_path / "sft.toml"
    config_path.write_text(
        'model = "openai/gpt-oss-20b"\n'
        'loss = "sft"\n'
        "[teacher]\n"
        'model = "openai/gpt-oss-120b"\n'
        "[teacher.client]\n"
        'api_key_var = "PRIME_API_KEY"\n'
    )

    with pytest.raises(typer.Exit):
        load_config(str(config_path))


def test_load_config_rejects_teacher_client_without_api_key_var(tmp_path: Path) -> None:
    config_path = tmp_path / "sft.toml"
    config_path.write_text(
        'model = "openai/gpt-oss-20b"\n'
        'loss = "sft"\n'
        "[teacher]\n"
        'model = "openai/gpt-oss-120b"\n'
        "[teacher.client]\n"
        'base_url = "https://api.pinference.ai/api/v1"\n'
    )

    with pytest.raises(typer.Exit):
        load_config(str(config_path))


def test_load_config_rejects_sft_without_teacher(tmp_path: Path) -> None:
    config_path = tmp_path / "sft.toml"
    config_path.write_text('model = "openai/gpt-oss-20b"\nloss = "sft"\n')

    with pytest.raises(typer.Exit):
        load_config(str(config_path))


def test_load_config_rejects_opd_until_hosted_scoring_exists(tmp_path: Path) -> None:
    config_path = tmp_path / "opd.toml"
    config_path.write_text(
        'model = "openai/gpt-oss-20b"\nloss = "opd"\n[teacher]\nmodel = "openai/gpt-oss-120b"\n'
    )

    with pytest.raises(typer.Exit):
        load_config(str(config_path))


def test_load_config_rejects_max_inflight_below_rollouts_per_example(tmp_path: Path) -> None:
    config_path = tmp_path / "rl.toml"
    config_path.write_text('model = "dummy"\nrollouts_per_example = 8\nmax_inflight_rollouts = 4\n')

    with pytest.raises(typer.Exit):
        load_config(str(config_path))


def test_load_config_rejects_nonpositive_oversampling_factor(tmp_path: Path) -> None:
    config_path = tmp_path / "rl.toml"
    config_path.write_text('model = "dummy"\noversampling_factor = 0\n')

    with pytest.raises(typer.Exit):
        load_config(str(config_path))


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
        'model = "dummy"\n[tailscale]\nenabled = true\nauth_key = "tskey-auth-abc123"\n'
    )
    cfg = load_config(str(config_path))
    assert cfg.tailscale.enabled is True
    # Default hostname_prefix matches the platform default so the rename
    # actually lands on CLI-driven runs.
    assert cfg.tailscale.hostname_prefix == "prime-hosted-training"
    payload = cfg.tailscale.to_api_dict()
    assert payload == {
        "enabled": True,
        "auth_key": "tskey-auth-abc123",
        "hostname_prefix": "prime-hosted-training",
    }


def test_tailscale_extra_args_field_rejected(tmp_path: Path) -> None:
    """extra_args is intentionally not exposed — the platform always boots the
    sidecar with a locked-down arg set for tenant isolation."""
    config_path = tmp_path / "rl.toml"
    config_path.write_text(
        'model = "dummy"\n'
        "[tailscale]\n"
        "enabled = true\n"
        'auth_key = "tskey-auth-abc"\n'
        'extra_args = "--ssh"\n'
    )
    with pytest.raises(typer.Exit):
        load_config(str(config_path))


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


def test_tailscale_oauth_client_secret_rejected(tmp_path: Path) -> None:
    """Only pre-authenticated keys (tskey-auth-) are supported. OAuth client
    secrets (tskey-client-) are rejected because the platform validator
    only accepts auth keys."""
    config_path = tmp_path / "rl.toml"
    config_path.write_text(
        'model = "dummy"\n[tailscale]\nenabled = true\nauth_key = "tskey-client-abc123"\n'
    )
    with pytest.raises(typer.Exit):
        load_config(str(config_path))


def test_tailscale_auth_key_from_env(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("TAILSCALE_AUTH_KEY", "tskey-auth-fromenv")
    config_path = tmp_path / "rl.toml"
    config_path.write_text('model = "dummy"\n[tailscale]\nenabled = true\n')
    cfg = load_config(str(config_path))
    assert cfg.tailscale.auth_key == "tskey-auth-fromenv"


def test_tailscale_auth_key_in_config_overrides_env(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("TAILSCALE_AUTH_KEY", "tskey-auth-fromenv")
    config_path = tmp_path / "rl.toml"
    config_path.write_text(
        'model = "dummy"\n[tailscale]\nenabled = true\nauth_key = "tskey-auth-fromfile"\n'
    )
    cfg = load_config(str(config_path))
    assert cfg.tailscale.auth_key == "tskey-auth-fromfile"


def test_tailscale_env_auth_key_format_validated(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("TAILSCALE_AUTH_KEY", "not-a-real-key")
    config_path = tmp_path / "rl.toml"
    config_path.write_text('model = "dummy"\n[tailscale]\nenabled = true\n')
    with pytest.raises(typer.Exit):
        load_config(str(config_path))


def test_tailscale_no_auth_key_anywhere_rejected(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("TAILSCALE_AUTH_KEY", raising=False)
    config_path = tmp_path / "rl.toml"
    config_path.write_text('model = "dummy"\n[tailscale]\nenabled = true\n')
    with pytest.raises(typer.Exit):
        load_config(str(config_path))
