from __future__ import annotations

from pathlib import Path

import pytest
from prime import Config
from prime_cli import Config as CliConfig


@pytest.fixture
def temp_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("PRIME_TEAM_ID", raising=False)


def test_prime_exports_config() -> None:
    assert Config is CliConfig


def test_inference_headers_empty_without_team(temp_home: None) -> None:
    config = Config()

    assert config.inference_headers == {}


def test_inference_headers_use_active_team(temp_home: None) -> None:
    config = Config()
    config.set_team("cmf0ohr9s0026ilerf3w68s6n")

    assert config.inference_headers == {"X-Prime-Team-ID": "cmf0ohr9s0026ilerf3w68s6n"}


def test_inference_headers_respect_team_env_override(
    temp_home: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PRIME_TEAM_ID", "cmf0ohr9s0026ilerf3w68s6m")

    assert Config().inference_headers == {"X-Prime-Team-ID": "cmf0ohr9s0026ilerf3w68s6m"}
