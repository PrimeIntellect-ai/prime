from pathlib import Path

import pytest
import typer
from prime_cli.commands.rl import generate_rl_config_template, load_config


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
    assert "narrow thresholds can skew plots and stall small-batch runs" in template
