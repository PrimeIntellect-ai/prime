import importlib
import logging
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "module_name",
    [
        "prime_mcp.core.config",
        "prime_sandboxes.core.config",
        "prime_evals.core.config",
        "prime_tunnel.core.config",
    ],
)
def test_config_warns_and_uses_empty_config_for_invalid_file(
    module_name: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config_dir = tmp_path / ".prime"
    config_dir.mkdir()
    config_file = config_dir / "config.json"
    config_file.write_text("{invalid json", encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    caplog.set_level(logging.WARNING, logger=module_name)
    module = importlib.import_module(module_name)
    caplog.clear()

    config = module.Config()

    assert config.config == {}
    assert any(
        record.name == module_name
        and "Failed to load config file" in record.message
        and str(config_file) in record.message
        for record in caplog.records
    )
