from pathlib import Path
from typing import Any

import pytest
import typer
from prime_cli.api.rl import RLClient
from prime_cli.commands.rl import (
    RLConfig,
    _flatten_config_schema,
    generate_rl_config_template,
    load_config,
)


def _rl_run_payload() -> dict[str, Any]:
    return {
        "id": "rft_test",
        "name": None,
        "userId": "user_test",
        "teamId": None,
        "rftClusterId": None,
        "status": "RUNNING",
        "rolloutsPerExample": 1,
        "seqLen": 2048,
        "maxSteps": 100,
        "maxTokens": None,
        "batchSize": 128,
        "baseModel": "openai/gpt-oss-20b",
        "environments": [],
        "runConfig": None,
        "evalConfig": None,
        "valConfig": None,
        "bufferConfig": None,
        "createdAt": "2026-01-01T00:00:00Z",
        "updatedAt": "2026-01-01T00:00:00Z",
    }


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


def test_load_config_rejects_legacy_sft_distill_config(tmp_path: Path) -> None:
    config_path = tmp_path / "rl.toml"
    config_path.write_text(
        'model = "PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT"\n'
        'loss_type = "sft"\n'
        "[teacher_rollout_model]\n"
        'base_url = ["https://api.example.com/v1"]\n'
        'api_key_var = "TEACHER_API_KEY"\n'
        'name = "teacher-model"\n'
    )

    with pytest.raises(typer.Exit):
        load_config(str(config_path))


@pytest.mark.parametrize(
    "config_body",
    [
        'loss_type = "sft"\n',
        (
            "[teacher_rollout_model]\n"
            'base_url = ["https://api.example.com/v1"]\n'
            'api_key_var = "TEACHER_API_KEY"\n'
            'name = "teacher-model"\n'
        ),
    ],
)
def test_load_config_rejects_legacy_sft_distill_fields(tmp_path: Path, config_body: str) -> None:
    config_path = tmp_path / "rl.toml"
    config_path.write_text('model = "PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT"\n' + config_body)

    with pytest.raises(typer.Exit):
        load_config(str(config_path))


def test_generate_rl_config_template_uses_broad_buffer_threshold_examples() -> None:
    template = generate_rl_config_template()

    assert "# easy_threshold = 1.0" in template
    assert "# hard_threshold = 0.0" in template


def test_generate_rl_config_template_keeps_checkpoint_id_top_level_with_sft(
    tmp_path: Path,
) -> None:
    template = generate_rl_config_template()
    config_text = template
    replacements = {
        '# checkpoint_id = "..."': 'checkpoint_id = "ckpt_test"',
        '# loss = "sft"': 'loss = "sft"',
        "# [teacher]": "[teacher]",
        '# model = "openai/gpt-oss-120b"': 'model = "openai/gpt-oss-120b"',
        "# save = false": "save = false",
        "# [teacher.sampling]": "[teacher.sampling]",
        "# max_tokens = 2048": "max_tokens = 2048",
        "# enable_thinking = false": "enable_thinking = false",
        '# reasoning_effort = "medium"': 'reasoning_effort = "medium"',
    }
    for old, new in replacements.items():
        config_text = config_text.replace(old, new, 1)

    config_path = tmp_path / "rl.toml"
    config_path.write_text(config_text)

    cfg = load_config(str(config_path))

    assert cfg.checkpoint_id == "ckpt_test"
    assert cfg.loss == "sft"
    assert cfg.teacher is not None
    assert cfg.teacher.sampling.max_tokens == 2048
    assert cfg.teacher.sampling.enable_thinking is False


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


def test_load_config_passes_reasoning_controls_to_chat_template_kwargs(tmp_path: Path) -> None:
    config_path = tmp_path / "rl.toml"
    config_path.write_text(
        'model = "openai/gpt-oss-20b"\n'
        "[sampling]\n"
        "enable_thinking = false\n"
        'reasoning_effort = "low"\n'
    )

    cfg = load_config(str(config_path))

    assert cfg.sampling.extra_body_to_api_dict() == {
        "chat_template_kwargs": {
            "enable_thinking": False,
            "reasoning_effort": "low",
        }
    }


def test_load_config_rejects_top_level_reasoning_effort(tmp_path: Path) -> None:
    config_path = tmp_path / "rl.toml"
    config_path.write_text('model = "openai/gpt-oss-20b"\nreasoning_effort = "high"\n')

    with pytest.raises(typer.Exit):
        load_config(str(config_path))


def test_load_config_accepts_sft_teacher_config_and_defaults_rollouts(tmp_path: Path) -> None:
    config_path = tmp_path / "rl.toml"
    config_path.write_text(
        'model = "openai/gpt-oss-20b"\n'
        'loss = "sft"\n'
        "[teacher]\n"
        'model = "openai/gpt-oss-120b"\n'
        "save = false\n"
        "[teacher.sampling]\n"
        "max_tokens = 2048\n"
        "enable_thinking = false\n"
        'reasoning_effort = "medium"\n'
        "[[env]]\n"
        'id = "primeintellect/wordle"\n'
    )

    cfg = load_config(str(config_path))

    assert cfg.loss == "sft"
    assert cfg.rollouts_per_example == 1
    assert cfg.teacher is not None
    assert cfg.teacher.to_api_dict() == {
        "model": "openai/gpt-oss-120b",
        "save": False,
        "sampling": {
            "max_tokens": 2048,
            "extra_body": {
                "chat_template_kwargs": {
                    "enable_thinking": False,
                    "reasoning_effort": "medium",
                }
            },
        },
    }


def test_load_config_preserves_explicit_sft_rollouts(tmp_path: Path) -> None:
    config_path = tmp_path / "rl.toml"
    config_path.write_text(
        'model = "openai/gpt-oss-20b"\n'
        'loss = "sft"\n'
        "rollouts_per_example = 4\n"
        "[teacher]\n"
        'model = "openai/gpt-oss-120b"\n'
    )

    cfg = load_config(str(config_path))

    assert cfg.rollouts_per_example == 4


def test_load_config_rejects_sft_without_teacher(tmp_path: Path) -> None:
    config_path = tmp_path / "rl.toml"
    config_path.write_text('model = "openai/gpt-oss-20b"\nloss = "sft"\n')

    with pytest.raises(typer.Exit):
        load_config(str(config_path))


def test_load_config_rejects_teacher_for_rl(tmp_path: Path) -> None:
    config_path = tmp_path / "rl.toml"
    config_path.write_text(
        'model = "openai/gpt-oss-20b"\n[teacher]\nmodel = "openai/gpt-oss-120b"\n'
    )

    with pytest.raises(typer.Exit):
        load_config(str(config_path))


def test_load_config_rejects_teacher_save_true(tmp_path: Path) -> None:
    config_path = tmp_path / "rl.toml"
    config_path.write_text(
        'model = "openai/gpt-oss-20b"\n'
        'loss = "sft"\n'
        "[teacher]\n"
        'model = "openai/gpt-oss-120b"\n'
        "save = true\n"
    )

    with pytest.raises(typer.Exit):
        load_config(str(config_path))


def test_load_config_accepts_arbitrary_teacher_api_fields(tmp_path: Path) -> None:
    config_path = tmp_path / "rl.toml"
    config_path.write_text(
        'model = "openai/gpt-oss-20b"\n'
        'loss = "sft"\n'
        "[teacher]\n"
        'model = "openai/gpt-oss-120b"\n'
        'base_url = ["https://api.openai.com/v1"]\n'
        'api_key_var = "OPENAI_API_KEY"\n'
    )

    cfg = load_config(str(config_path))

    assert cfg.teacher is not None
    assert cfg.teacher.to_api_dict()["base_url"] == ["https://api.openai.com/v1"]
    assert cfg.teacher.to_api_dict()["api_key_var"] == "OPENAI_API_KEY"


def test_rl_client_create_run_forwards_loss_and_teacher_payload() -> None:
    captured: dict[str, Any] = {}

    class DummyClient:
        def post(self, endpoint: str, json: dict[str, Any]) -> dict[str, Any]:
            captured["endpoint"] = endpoint
            captured["json"] = json
            return {"run": _rl_run_payload()}

    teacher = {
        "model": "openai/gpt-oss-120b",
        "save": False,
        "sampling": {
            "max_tokens": 2048,
            "extra_body": {
                "chat_template_kwargs": {
                    "reasoning_effort": "medium",
                }
            },
        },
    }

    RLClient(DummyClient()).create_run(
        model_name="openai/gpt-oss-20b",
        environments=[{"id": "primeintellect/wordle"}],
        rollouts_per_example=1,
        loss="sft",
        teacher=teacher,
    )

    assert captured["endpoint"] == "/rft/runs"
    assert captured["json"]["loss"] == "sft"
    assert captured["json"]["teacher"] == teacher
    assert "teacher_rollout_model" not in captured["json"]


def test_rl_client_create_run_merges_reasoning_controls_into_extra_body() -> None:
    captured: dict[str, Any] = {}

    class DummyClient:
        def post(self, endpoint: str, json: dict[str, Any]) -> dict[str, Any]:
            captured["json"] = json
            return {"run": _rl_run_payload()}

    RLClient(DummyClient()).create_run(
        model_name="openai/gpt-oss-20b",
        environments=[{"id": "primeintellect/wordle"}],
        extra_body={"provider": {"order": ["azure"]}},
        enable_thinking=False,
        reasoning_effort="medium",
    )

    assert captured["json"]["extra_body"] == {
        "provider": {"order": ["azure"]},
        "chat_template_kwargs": {
            "enable_thinking": False,
            "reasoning_effort": "medium",
        },
    }
    assert "enable_thinking" not in captured["json"]
    assert "reasoning_effort" not in captured["json"]


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
