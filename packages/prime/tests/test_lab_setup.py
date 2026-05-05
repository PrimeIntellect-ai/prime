from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from prime_cli.lab_agents import agent_adapter, agent_capability, known_agent_names
from prime_cli.lab_setup import (
    LabDoctorOptions,
    LabSetupOptions,
    LabSyncOptions,
    parse_lab_setup_args,
    parse_lab_sync_args,
    run_lab_doctor_service,
    run_lab_setup_service,
    run_lab_sync_service,
)


@pytest.fixture(autouse=True)
def fake_lab_asset_downloads(monkeypatch: Any) -> list[str]:
    urls: list[str] = []

    def fake_download_file(
        url: str,
        dest: Path,
        emit: Any,
        *,
        force: bool = False,
    ) -> None:
        urls.append(url)
        if dest.exists() and not force:
            emit(f"{dest.name} already exists\n")
            return
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.suffix == ".toml":
            dest.write_text('model = "openai/gpt-4.1-mini"\n', encoding="utf-8")
        else:
            dest.write_text(f"downloaded from {url}\n", encoding="utf-8")
        emit(f"Downloaded {dest}\n")

    monkeypatch.setattr("prime_cli.lab_setup._download_file", fake_download_file)
    return urls


def test_lab_setup_parses_selected_agents_and_all() -> None:
    selected = parse_lab_setup_args(["--agent", "droid,amp-code,claude-cli"])
    all_agents = parse_lab_sync_args(["--agents", "all"])

    assert selected.agents == ("factory-droid", "amp", "claude-code")
    assert all_agents.agents == known_agent_names()


def test_lab_setup_service_downloads_upstream_assets_without_agent_installs(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr("prime_cli.lab_agents.shutil.which", lambda _command: None)
    commands: list[list[str]] = []
    emitted: list[str] = []

    result = run_lab_setup_service(
        LabSetupOptions(skip_install=True, skip_agents_md=True, agents=("pi",)),
        workspace=tmp_path,
        emit=emitted.append,
        runner=lambda command, _cwd, _emit: commands.append(list(command)) or 0,
    )

    metadata = json.loads((tmp_path / ".prime" / "lab.json").read_text(encoding="utf-8"))
    home = tmp_path / "home"
    assert result.exit_code == 0
    assert commands == []
    assert metadata["choices"]["primary_agent"] == "pi"
    assert (home / ".prime" / "skills" / "create-environments" / "SKILL.md").is_file()
    assert (home / ".pi" / "skills" / "create-environments").exists()
    assert not (tmp_path / ".pi" / "skills").exists()
    assert (tmp_path / "configs" / "rl" / "gsm8k.toml").is_file()
    assert any("npm install -g pi-acp" in line for line in emitted)


def test_lab_setup_uses_existing_verifiers_sources(
    tmp_path: Path,
    monkeypatch: Any,
    fake_lab_asset_downloads: list[str],
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr("prime_cli.lab_agents.shutil.which", lambda _command: "/bin/tool")

    result = run_lab_setup_service(
        LabSetupOptions(skip_install=True, skip_agents_md=False, agents=("codex",)),
        workspace=tmp_path,
        emit=lambda _text: None,
    )

    assert result.exit_code == 0
    assert any(
        url.endswith(
            "/primeintellect-ai/verifiers/refs/heads/main/skills/create-environments/SKILL.md"
        )
        for url in fake_lab_asset_downloads
    )
    assert any(
        url.endswith("/primeintellect-ai/verifiers/refs/heads/main/configs/rl/gsm8k.toml")
        for url in fake_lab_asset_downloads
    )
    assert any(
        url.endswith("/primeintellect-ai/verifiers/refs/heads/main/assets/lab/AGENTS.md")
        for url in fake_lab_asset_downloads
    )


def test_lab_sync_all_scaffolds_amp_and_factory_surfaces(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr("prime_cli.lab_agents.shutil.which", lambda _command: "/bin/tool")

    result = run_lab_sync_service(
        LabSyncOptions(agents=("factory-droid", "amp"), skip_docs=True),
        workspace=tmp_path,
        emit=lambda _text: None,
    )

    home = tmp_path / "home"
    assert result.exit_code == 0
    assert (home / ".factory" / "skills" / "create-environments").exists()
    assert (home / ".config" / "amp" / "skills" / "create-environments").exists()
    assert not (tmp_path / ".factory" / "skills").exists()
    assert not (tmp_path / ".amp" / "skills").exists()
    assert (tmp_path / ".factory" / "mcp.json").is_file()
    assert (tmp_path / ".amp" / "settings.json").is_file()
    assert (tmp_path / ".prime" / "lab" / "templates" / "configs" / "rl" / "gsm8k.toml").is_file()


def test_lab_doctor_reports_missing_selected_agent_guidance(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr("prime_cli.lab_agents.shutil.which", lambda _command: None)
    run_lab_setup_service(
        LabSetupOptions(skip_install=True, skip_agents_md=True, agents=("amp",)),
        workspace=tmp_path,
        emit=lambda _text: None,
    )

    result = run_lab_doctor_service(LabDoctorOptions(), workspace=tmp_path)
    checks = {check.name: check for check in result.checks}

    assert checks["Amp Code native tools"].status == "WARN"
    assert "npm install -g @sourcegraph/amp@latest" in checks["Amp Code native tools"].remediation


def test_lab_sync_skips_user_owned_skill_conflicts(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr("prime_cli.lab_agents.shutil.which", lambda _command: "/bin/tool")
    user_skill = home / ".config" / "amp" / "skills" / "create-environments"
    user_skill.mkdir(parents=True)
    (user_skill / "SKILL.md").write_text("user skill\n", encoding="utf-8")
    emitted: list[str] = []

    result = run_lab_sync_service(
        LabSyncOptions(agents=("amp",), skip_docs=True),
        workspace=tmp_path,
        emit=emitted.append,
    )

    assert result.exit_code == 0
    assert (user_skill / "SKILL.md").read_text(encoding="utf-8") == "user skill\n"
    assert any("Skipped" in line and "create-environments" in line for line in emitted)


def test_lab_sync_removes_stale_managed_skill_links(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr("prime_cli.lab_agents.shutil.which", lambda _command: "/bin/tool")
    stale_source = home / ".prime" / "skills" / "old-lab-skill"
    stale_source.mkdir(parents=True)
    stale_target = home / ".config" / "amp" / "skills" / "old-lab-skill"
    stale_target.parent.mkdir(parents=True)
    stale_target.symlink_to(stale_source, target_is_directory=True)

    result = run_lab_sync_service(
        LabSyncOptions(agents=("amp",), skip_docs=True),
        workspace=tmp_path,
        emit=lambda _text: None,
    )

    assert result.exit_code == 0
    assert not stale_target.exists()


def test_lab_agent_metadata_includes_amp_and_factory() -> None:
    factory = agent_capability("droid")
    amp = agent_capability("amp-code")

    assert factory.name == "factory-droid"
    assert factory.requirements[0].install_command == ("npm", "install", "-g", "@factory/cli")
    assert agent_adapter("factory").prompt_command("hello") == ["droid", "exec", "hello"]
    assert amp.name == "amp"
    assert amp.requirements[0].install_command == (
        "npm",
        "install",
        "-g",
        "@sourcegraph/amp@latest",
    )
    assert agent_adapter("amp-code").prompt_command("hello") == ["amp", "--execute", "hello"]
