from pathlib import Path

import pytest
from prime_cli.utils.env_vars import EnvParseError, parse_env_file


def test_parse_env_file_expands_braced_variable_references(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WANDB_API_KEY", "secret-123")
    env_file = tmp_path / "secrets.env"
    env_file.write_text("WANDB_API_KEY=${WANDB_API_KEY}\n")

    parsed = parse_env_file(env_file)

    assert parsed["WANDB_API_KEY"] == "secret-123"


def test_parse_env_file_expands_references_inside_quoted_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PRIME_TEST_PREFIX", "alpha")
    monkeypatch.setenv("PRIME_TEST_SUFFIX", "omega")
    env_file = tmp_path / "secrets.env"
    env_file.write_text('COMBINED="start-${PRIME_TEST_PREFIX}-${PRIME_TEST_SUFFIX}-end"\n')

    parsed = parse_env_file(env_file)

    assert parsed["COMBINED"] == "start-alpha-omega-end"


def test_parse_env_file_raises_for_missing_braced_variable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing_var = "PRIME_TEST_MISSING_ENV_VAR"
    monkeypatch.delenv(missing_var, raising=False)
    env_file = tmp_path / "secrets.env"
    env_file.write_text(f"WANDB_API_KEY=${{{missing_var}}}\n")

    with pytest.raises(EnvParseError, match=missing_var):
        parse_env_file(env_file)


def test_parse_env_file_keeps_unbraced_dollar_reference_literal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WANDB_API_KEY", "secret-123")
    env_file = tmp_path / "secrets.env"
    env_file.write_text("WANDB_API_KEY=$WANDB_API_KEY\n")

    parsed = parse_env_file(env_file)

    assert parsed["WANDB_API_KEY"] == "$WANDB_API_KEY"
