from pathlib import Path

import pytest
import typer
from prime_cli.commands.rl import (
    _load_toml,
    _looks_like_lora,
    _warn_legacy_full_finetune_type,
)


def test_load_toml_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(typer.Exit) as exc_info:
        _load_toml(str(tmp_path / "missing.toml"))
    assert exc_info.value.exit_code == 1


def test_load_toml_rejects_invalid_toml(tmp_path: Path) -> None:
    config_path = tmp_path / "rl.toml"
    config_path.write_text("[model\nname = ")
    with pytest.raises(typer.Exit) as exc_info:
        _load_toml(str(config_path))
    assert exc_info.value.exit_code == 1


def test_load_toml_returns_raw_table(tmp_path: Path) -> None:
    config_path = tmp_path / "rl.toml"
    config_path.write_text('[model]\nname = "Qwen/Qwen3-0.6B"\n')
    assert _load_toml(str(config_path)) == {"model": {"name": "Qwen/Qwen3-0.6B"}}


def test_looks_like_lora_top_level_model_string() -> None:
    assert _looks_like_lora({"model": "Qwen/Qwen3-0.6B", "max_steps": 10}) is True


def test_looks_like_lora_top_level_env_array() -> None:
    assert _looks_like_lora({"env": [{"id": "primeintellect/reverse-text"}]}) is True


def test_looks_like_lora_prime_rl_config() -> None:
    cfg = {
        "model": {"name": "Qwen/Qwen3-0.6B"},
        "orchestrator": {"batch_size": 128},
        "trainer": {},
    }
    assert _looks_like_lora(cfg) is False


def test_warn_legacy_full_finetune_type_warns_on_leftover_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    printed: list[str] = []
    monkeypatch.setattr(
        "prime_cli.commands.rl.console.print",
        lambda *args, **_: printed.append(" ".join(str(a) for a in args)),
    )

    _warn_legacy_full_finetune_type({"type": "full_finetune"}, "rl.toml")

    assert any("type" in line and "no longer used" in line for line in printed)


def test_warn_legacy_full_finetune_type_silent_without_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    printed: list[str] = []
    monkeypatch.setattr(
        "prime_cli.commands.rl.console.print",
        lambda *args, **_: printed.append(" ".join(str(a) for a in args)),
    )

    _warn_legacy_full_finetune_type({"deployment": {"num_train_gpus": 1}}, "rl.toml")

    assert printed == []
