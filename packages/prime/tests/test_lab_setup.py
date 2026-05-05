from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from prime_cli import lab_setup
from prime_cli.lab_agents import known_agent_names
from prime_cli.lab_setup import (
    LabDoctorOptions,
    LabSetupOptions,
    LabSyncOptions,
    SkillSource,
    parse_lab_setup_args,
    parse_lab_sync_args,
    run_lab_doctor_service,
    run_lab_setup_service,
    run_lab_sync_service,
)


@pytest.fixture(autouse=True)
def fake_lab_asset_downloads(monkeypatch: Any) -> list[str]:
    urls: list[str] = []
    skill_names = (
        "create-environments",
        "browse-environments",
        "review-environments",
        "evaluate-environments",
        "optimize-with-environments",
        "train-with-environments",
        "brainstorm",
    )

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
    monkeypatch.setattr(
        "prime_cli.lab_setup._download_json",
        lambda _url: [{"name": name, "type": "dir"} for name in skill_names],
    )
    return urls


def test_lab_setup_parses_selected_agents_and_all() -> None:
    selected = parse_lab_setup_args(["--agent", "factory-droid,amp-code,claude-code"])
    all_agents = parse_lab_sync_args(["--agents", "all"])

    assert selected.agents == ("droid", "amp", "claude")
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


def test_lab_setup_manifest_tracks_skill_source(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr("prime_cli.lab_agents.shutil.which", lambda _command: "/bin/tool")

    result = run_lab_setup_service(
        LabSetupOptions(skip_install=True, skip_agents_md=True, agents=("codex",)),
        workspace=tmp_path,
        emit=lambda _text: None,
    )

    manifest = json.loads(
        (home / ".prime" / "skills" / ".prime-managed.json").read_text(encoding="utf-8")
    )
    assert result.exit_code == 0
    assert manifest["skills"]["create-environments"]["repo"] == "primeintellect-ai/verifiers"
    assert manifest["skills"]["create-environments"]["path"] == (
        "skills/create-environments/SKILL.md"
    )


def test_lab_setup_rejects_duplicate_skill_sources(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr("prime_cli.lab_agents.shutil.which", lambda _command: "/bin/tool")
    monkeypatch.setattr(
        "prime_cli.lab_setup.SKILL_SOURCES",
        (
            SkillSource(
                repo="primeintellect-ai/verifiers",
                ref="main",
            ),
            SkillSource(
                repo="primeintellect-ai/research-skills",
                ref="main",
            ),
        ),
    )
    emitted: list[str] = []

    result = run_lab_setup_service(
        LabSetupOptions(skip_install=True, skip_agents_md=True, agents=("codex",)),
        workspace=tmp_path,
        emit=emitted.append,
    )

    assert result.exit_code == 1
    assert any("defined by both" in line for line in emitted)


def test_lab_sync_all_scaffolds_amp_and_factory_skills(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr("prime_cli.lab_agents.shutil.which", lambda _command: "/bin/tool")

    result = run_lab_sync_service(
        LabSyncOptions(agents=("droid", "amp"), skip_docs=True),
        workspace=tmp_path,
        emit=lambda _text: None,
    )

    home = tmp_path / "home"
    assert result.exit_code == 0
    assert (home / ".factory" / "skills" / "create-environments").exists()
    assert (home / ".config" / "amp" / "skills" / "create-environments").exists()
    assert not (tmp_path / ".factory" / "skills").exists()
    assert not (tmp_path / ".amp" / "skills").exists()
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


def test_lab_doctor_fix_writes_standard_gitignore_patterns(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text(
        "# existing\ncustom.log\n.env.local\n/outputs-old/\n",
        encoding="utf-8",
    )

    result = run_lab_doctor_service(LabDoctorOptions(fix=True), workspace=tmp_path)

    gitignore = (tmp_path / ".gitignore").read_text(encoding="utf-8")
    gitignore_lines = set(gitignore.splitlines())
    assert result.exit_code == 1
    assert "custom.log" in gitignore
    assert ".env.local" in gitignore_lines
    assert ".env" in gitignore_lines
    assert "/outputs/" in gitignore_lines
    assert "/prime-rl/" in gitignore_lines
    assert "/environments/*/outputs/" in gitignore_lines
    assert "*.py[cod]" in gitignore_lines
    assert ".pytest_cache/" in gitignore_lines
    assert ".ruff_cache/" in gitignore_lines


def test_lab_doctor_validates_environment_table_refs(tmp_path: Path) -> None:
    (tmp_path / ".prime").mkdir()
    (tmp_path / ".prime" / "lab.json").write_text(
        json.dumps({"choices": {"agents": ["codex"], "primary_agent": "codex"}}),
        encoding="utf-8",
    )
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'lab'\n", encoding="utf-8")
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "gepa.toml").write_text(
        "[environment]\nid = 'missing-env'\n",
        encoding="utf-8",
    )
    (tmp_path / "environments" / "other-env").mkdir(parents=True)

    result = run_lab_doctor_service(LabDoctorOptions(), workspace=tmp_path)
    checks = {check.name: check for check in result.checks}

    assert checks["Config environment refs"].status == "WARN"
    assert "missing-env" in checks["Config environment refs"].message


def test_lab_setup_installs_prime_rl_envs_with_split_editable_args(tmp_path: Path) -> None:
    (tmp_path / "prime-rl" / ".venv" / "bin").mkdir(parents=True)
    (tmp_path / "prime-rl" / ".venv" / "bin" / "python").write_text("", encoding="utf-8")
    (tmp_path / "environments" / "foo").mkdir(parents=True)
    (tmp_path / "environments" / "foo" / "pyproject.toml").write_text(
        "[project]\nname = 'foo'\n",
        encoding="utf-8",
    )
    commands: list[list[str]] = []

    lab_setup._install_environments_to_prime_rl(
        tmp_path,
        emit=lambda _text: None,
        runner=lambda command, _cwd, _emit: commands.append(list(command)) or 0,
    )

    assert commands == [
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(tmp_path / "prime-rl" / ".venv" / "bin" / "python"),
            "-e",
            "environments/foo",
        ]
    ]


def test_workspace_agents_from_metadata_ignores_non_string_values(tmp_path: Path) -> None:
    (tmp_path / ".prime").mkdir()
    (tmp_path / ".prime" / "lab.json").write_text(
        json.dumps({"choices": {"agents": [None, 0, "", " amp "], "primary_agent": 0}}),
        encoding="utf-8",
    )

    assert lab_setup._workspace_agents_from_metadata(tmp_path) == ("amp",)


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


def test_lab_sync_removes_stale_global_managed_skills(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr("prime_cli.lab_agents.shutil.which", lambda _command: "/bin/tool")
    stale_skill = home / ".prime" / "skills" / "old-lab-skill"
    stale_skill.mkdir(parents=True)
    (stale_skill / "SKILL.md").write_text("old skill\n", encoding="utf-8")
    (home / ".prime" / "skills" / ".prime-managed.json").write_text(
        json.dumps({"skills": {"old-lab-skill": {"repo": "old/repo"}}}),
        encoding="utf-8",
    )
    emitted: list[str] = []

    result = run_lab_sync_service(
        LabSyncOptions(agents=("amp",), skip_docs=True),
        workspace=tmp_path,
        emit=emitted.append,
    )

    assert result.exit_code == 0
    assert not stale_skill.exists()
    assert any("Warning: removed stale managed skill" in line for line in emitted)


def test_lab_setup_accepts_amp_and_factory_aliases(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr("prime_cli.lab_agents.shutil.which", lambda _command: "/bin/tool")

    result = run_lab_sync_service(
        LabSyncOptions(agents=("droid", "amp"), skip_docs=True),
        workspace=tmp_path,
        emit=lambda _text: None,
    )

    metadata = json.loads((tmp_path / ".prime" / "lab.json").read_text(encoding="utf-8"))
    assert result.exit_code == 0
    assert metadata["choices"]["agents"] == ["droid", "amp"]
    assert (tmp_path / "home" / ".factory" / "skills" / "create-environments").exists()
    assert (tmp_path / "home" / ".config" / "amp" / "skills" / "create-environments").exists()
