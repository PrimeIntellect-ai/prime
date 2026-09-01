from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolated_prime_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Keep tests hermetic: never read the developer's real ~/.prime.

    Config discovery also walks up from the working directory, so start each
    test in a fresh tmp dir and ignore PRIME_CONFIG_DIR from the shell.
    """
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PRIME_CONFIG_DIR", raising=False)
    monkeypatch.delenv("PRIME_API_KEY", raising=False)
    return tmp_path
