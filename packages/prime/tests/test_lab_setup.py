from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.error import URLError

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

    def fake_download_json(url: str) -> list[dict[str, str]]:
        if "/contents/skills?" in url:
            return [{"name": name, "type": "dir"} for name in skill_names]
        if "/contents/skills/" in url:
            source_path = url.split("/contents/", 1)[1].split("?", 1)[0]
            if source_path.endswith("/references"):
                return [
                    {
                        "name": "notes.md",
                        "type": "file",
                        "path": f"{source_path}/notes.md",
                    }
                ]
            return [
                {
                    "name": "SKILL.md",
                    "type": "file",
                    "path": f"{source_path}/SKILL.md",
                },
                {
                    "name": "references",
                    "type": "dir",
                    "path": f"{source_path}/references",
                },
            ]
        return []

    monkeypatch.setattr("prime_cli.lab_setup._download_file", fake_download_file)
    monkeypatch.setattr("prime_cli.lab_setup._download_json", fake_download_json)
    return urls


def _is_pinned_ref(ref: str) -> bool:
    return len(ref) == 40 and all(char in "0123456789abcdef" for char in ref)


def test_lab_setup_parses_selected_agents_and_all() -> None:
    selected = parse_lab_setup_args(["--agent", "factory-droid,amp-code,claude-code"])
    all_agents = parse_lab_sync_args(["--agents", "all"])

    assert selected.agents == ("droid", "amp", "claude")
    assert all_agents.agents == known_agent_names()


def test_lab_setup_prompts_for_agent_when_interactive(
    monkeypatch: Any,
) -> None:
    class FakeStdin:
        def isatty(self) -> bool:
            return True

    answers = iter(("droid", "y", "amp-code,claude-code"))
    monkeypatch.setattr(lab_setup.sys, "stdin", FakeStdin())
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))

    options = parse_lab_setup_args([])

    assert options.agents == ("droid", "amp", "claude")


def test_lab_setup_non_interactive_requires_explicit_agent(
    monkeypatch: Any,
) -> None:
    class FakeStdin:
        def isatty(self) -> bool:
            return False

    monkeypatch.setattr(lab_setup.sys, "stdin", FakeStdin())

    with pytest.raises(ValueError, match="No --agent provided"):
        parse_lab_setup_args([])


def test_lab_setup_service_requires_configured_agent(tmp_path: Path) -> None:
    result = run_lab_setup_service(
        LabSetupOptions(skip_install=True, skip_agents_md=True),
        workspace=tmp_path,
        emit=lambda _text: None,
    )

    assert result.exit_code == 1


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
    assert metadata["setup_source"] == "prime lab setup"
    assert metadata["choices"]["primary_agent"] == "pi"
    assert (home / ".prime" / "skills" / "create-environments" / "SKILL.md").is_file()
    assert (
        home / ".prime" / "skills" / "create-environments" / "references" / "notes.md"
    ).is_file()
    assert (tmp_path / ".prime" / "skills" / "create-environments" / "SKILL.md").is_file()
    assert (
        tmp_path / ".prime" / "skills" / "create-environments" / "references" / "notes.md"
    ).is_file()
    assert (tmp_path / ".pi" / "skills" / "create-environments").exists()
    assert not (home / ".pi" / "agent" / "skills").exists()
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
    assert _is_pinned_ref(lab_setup.VERIFIERS_REF)
    assert _is_pinned_ref(lab_setup.PRIME_RL_REF)
    assert any(
        url.endswith(
            f"/primeintellect-ai/verifiers/{lab_setup.VERIFIERS_REF}/skills/create-environments/SKILL.md"
        )
        for url in fake_lab_asset_downloads
    )
    assert any(
        url.endswith(
            f"/primeintellect-ai/verifiers/{lab_setup.VERIFIERS_REF}/configs/rl/gsm8k.toml"
        )
        for url in fake_lab_asset_downloads
    )
    assert any(
        url.endswith(f"/primeintellect-ai/verifiers/{lab_setup.VERIFIERS_REF}/assets/lab/AGENTS.md")
        for url in fake_lab_asset_downloads
    )


def test_lab_setup_downloads_retry_transient_failures(monkeypatch: Any) -> None:
    calls: list[str] = []
    emitted: list[str] = []
    sleeps: list[float] = []

    class FakeResponse:
        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *_args: Any) -> None:
            return None

        def read(self) -> bytes:
            return b"ok"

    def fake_urlopen(url: str, *, timeout: int) -> FakeResponse:
        assert timeout == 60
        calls.append(url)
        if len(calls) == 1:
            raise URLError("temporary failure")
        return FakeResponse()

    monkeypatch.setattr(lab_setup, "urlopen", fake_urlopen)
    monkeypatch.setattr(lab_setup.time, "sleep", sleeps.append)

    assert lab_setup._read_url("https://example.test/file", emit=emitted.append) == b"ok"
    assert calls == ["https://example.test/file", "https://example.test/file"]
    assert sleeps == [lab_setup.DOWNLOAD_RETRY_DELAY_SECONDS]
    assert emitted == ["Download failed for https://example.test/file; retrying (2/3)\n"]


def test_lab_setup_installs_prime_rl_from_pinned_checkout(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(lab_setup, "_ensure_prime_rl_supported_platform", lambda: None)
    commands: list[tuple[list[str], Path]] = []

    def runner(command: Any, cwd: Path, _emit: Any) -> int:
        command_list = list(command)
        commands.append((command_list, cwd))
        if command_list[:3] == ["git", "clone", "--no-checkout"]:
            (tmp_path / "prime-rl" / "scripts").mkdir(parents=True)
            (tmp_path / "prime-rl" / "scripts" / "install.sh").write_text(
                "#!/usr/bin/env bash\n",
                encoding="utf-8",
            )
        return 0

    lab_setup._install_prime_rl(tmp_path, emit=lambda _text: None, runner=runner)

    assert commands == [
        (
            [
                "git",
                "clone",
                "--no-checkout",
                "https://github.com/primeintellect-ai/prime-rl.git",
                "prime-rl",
            ],
            tmp_path,
        ),
        (["git", "checkout", lab_setup.PRIME_RL_REF], tmp_path / "prime-rl"),
        (
            ["env", "SKIP_CLONE=1", "bash", "scripts/install.sh"],
            tmp_path / "prime-rl",
        ),
    ]


def test_lab_setup_prime_rl_fails_early_on_unsupported_platform(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    commands: list[list[str]] = []
    monkeypatch.setattr(lab_setup, "_prime_rl_platform", lambda: ("darwin", "arm64"))

    with pytest.raises(RuntimeError, match="prime-rl only supports Linux"):
        lab_setup._install_prime_rl(
            tmp_path,
            emit=lambda _text: None,
            runner=lambda command, _cwd, _emit: commands.append(list(command)) or 0,
        )

    assert commands == []


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
    assert manifest["skills"]["create-environments"]["path"] == "skills/create-environments"


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

    assert result.exit_code == 0
    assert (tmp_path / ".factory" / "skills" / "create-environments").exists()
    assert (tmp_path / ".agents" / "skills" / "create-environments").exists()
    assert (tmp_path / ".prime" / "skills" / "create-environments" / "SKILL.md").is_file()
    assert not (tmp_path / "home" / ".factory" / "skills").exists()
    assert not (tmp_path / "home" / ".config" / "agents" / "skills").exists()
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


def test_lab_sync_requires_agent_or_workspace_metadata(tmp_path: Path) -> None:
    result = run_lab_sync_service(
        LabSyncOptions(skip_docs=True),
        workspace=tmp_path,
        emit=lambda _text: None,
    )

    assert result.exit_code == 1


def test_lab_sync_no_agent_refreshes_shared_assets_without_metadata(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    result = run_lab_sync_service(
        LabSyncOptions(skip_docs=True, no_agent=True),
        workspace=tmp_path,
        emit=lambda _text: None,
    )

    assert result.exit_code == 0
    assert (tmp_path / ".prime" / "skills" / "create-environments" / "SKILL.md").is_file()
    assert not (tmp_path / ".agents" / "skills").exists()
    assert not (tmp_path / ".prime" / "lab.json").exists()


def test_lab_sync_replaces_workspace_skill_bundle_without_marker(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    skill_dir = tmp_path / ".prime" / "skills" / "create-environments"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("stale skill\n", encoding="utf-8")

    result = run_lab_sync_service(
        LabSyncOptions(skip_docs=True, no_agent=True),
        workspace=tmp_path,
        emit=lambda _text: None,
    )

    assert result.exit_code == 0
    assert "downloaded from" in (skill_dir / "SKILL.md").read_text(encoding="utf-8")


def test_lab_sync_rejects_agent_with_no_agent() -> None:
    with pytest.raises(ValueError, match="cannot be used together"):
        parse_lab_sync_args(["--agent", "codex", "--no-agent"])


def test_lab_sync_skips_user_owned_skill_conflicts(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr("prime_cli.lab_agents.shutil.which", lambda _command: "/bin/tool")
    user_skill = tmp_path / ".agents" / "skills" / "create-environments"
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
    stale_source = tmp_path / ".prime" / "skills" / "old-lab-skill"
    stale_source.mkdir(parents=True)
    stale_target = tmp_path / ".agents" / "skills" / "old-lab-skill"
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
    assert metadata["setup_source"] == "prime lab sync"
    assert metadata["choices"]["agents"] == ["droid", "amp"]
    assert (tmp_path / ".factory" / "skills" / "create-environments").exists()
    assert (tmp_path / ".agents" / "skills" / "create-environments").exists()
