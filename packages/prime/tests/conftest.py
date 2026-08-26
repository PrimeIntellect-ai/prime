import pytest


@pytest.fixture(autouse=True)
def isolated_config_discovery(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Keep config discovery away from the developer's machine.

    `Config()` walks up from the working directory looking for a project-local
    `.prime/config.json`; run from the repo checkout that walk would reach the
    developer's real ~/.prime — and tests that patch HOME would then read and
    *write* it. Start every test in a fresh tmp dir and ignore any
    PRIME_CONFIG_DIR from the developer's shell. Tests that need a different
    working directory still call `monkeypatch.chdir` themselves.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PRIME_CONFIG_DIR", raising=False)
