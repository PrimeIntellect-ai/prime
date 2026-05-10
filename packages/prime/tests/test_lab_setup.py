from __future__ import annotations

import json
from io import StringIO
from pathlib import Path
from typing import Any
from urllib.error import URLError

import pytest
from prime_cli import lab_setup
from prime_cli.lab_agents import AgentCapability, known_agent_names
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
from rich.console import Console

AGENT_WHICH = "prime_lab_app.agent_capabilities.shutil.which"


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
    config_tree: dict[str, list[tuple[str, str]]] = {
        "configs": [
            ("endpoints.toml", "file"),
            ("eval", "dir"),
            ("gepa", "dir"),
            ("local", "dir"),
            ("rl", "dir"),
            ("zero3.yaml", "file"),
        ],
        "configs/eval": [
            ("debug.toml", "file"),
            ("minimal.toml", "file"),
            ("multi-env.toml", "file"),
            ("wordle.toml", "file"),
        ],
        "configs/gepa": [
            ("base.toml", "file"),
            ("wordle.toml", "file"),
        ],
        "configs/local": [("prime-rl", "dir")],
        "configs/local/prime-rl": [("wiki-search.toml", "file")],
        "configs/rl": [
            ("alphabet-sort.toml", "file"),
            ("gsm8k.toml", "file"),
            ("math-python.toml", "file"),
            ("reverse-text.toml", "file"),
            ("wiki-search.toml", "file"),
            ("wordle.toml", "file"),
        ],
    }

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

    def fake_download_json(url: str) -> Any:
        if "/git/trees/" in url:
            tree = [{"path": "skills", "type": "tree"}]
            for skill_name in skill_names:
                source_path = f"skills/{skill_name}"
                tree.extend(
                    [
                        {"path": source_path, "type": "tree"},
                        {"path": f"{source_path}/SKILL.md", "type": "blob"},
                        {"path": f"{source_path}/references", "type": "tree"},
                        {"path": f"{source_path}/references/notes.md", "type": "blob"},
                    ]
                )
            for source_path, entries in config_tree.items():
                tree.append({"path": source_path, "type": "tree"})
                for name, entry_type in entries:
                    tree.append(
                        {
                            "path": f"{source_path}/{name}",
                            "type": "tree" if entry_type == "dir" else "blob",
                        }
                    )
            return {"tree": tree}
        return []

    monkeypatch.setattr("prime_cli.lab_setup._download_file", fake_download_file)
    monkeypatch.setattr("prime_cli.lab_setup._download_json", fake_download_json)
    return urls


def _is_pinned_ref(ref: str) -> bool:
    return len(ref) == 40 and all(char in "0123456789abcdef" for char in ref)


def _render_emitted(items: list[Any]) -> str:
    console = Console(file=StringIO(), record=True, width=120)
    for item in items:
        lab_setup._emit_to_console(console, item)
    return console.export_text()


def test_lab_setup_parses_selected_agents_and_all() -> None:
    selected = parse_lab_setup_args(["--agent", "factory-droid,amp-code,claude-code,letta-code"])
    all_agents = parse_lab_sync_args(["--agents", "all"])

    assert selected.agents == ("droid", "amp", "claude", "letta")
    assert all_agents.agents == known_agent_names()


def test_lab_setup_no_interactive_uses_codex_default() -> None:
    options = parse_lab_setup_args(["--no-interactive"])

    assert options.agents == ("codex",)


def test_lab_setup_help_lists_supported_agents(capsys: Any) -> None:
    with pytest.raises(SystemExit) as exc_info:
        parse_lab_setup_args(["--help"])

    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    assert "Supported:" in output
    for agent in (*known_agent_names(), "all"):
        assert agent in output


def test_lab_setup_prompts_for_agent_when_interactive(monkeypatch: Any) -> None:
    class FakeStdin:
        def isatty(self) -> bool:
            return True

    answers = iter(("droid", "y", "amp-code,claude-code"))
    monkeypatch.setattr(lab_setup.sys, "stdin", FakeStdin())
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))

    options = parse_lab_setup_args([])

    assert options.agents == ("droid", "amp", "claude")


def test_lab_setup_interactive_agent_prompt_retries_invalid_input(
    monkeypatch: Any,
    capsys: Any,
) -> None:
    class FakeStdin:
        def isatty(self) -> bool:
            return True

    answers = iter(("codx", "droid", "y", "amp-code,codx", "amp-code,claude-code"))
    monkeypatch.setattr(lab_setup.sys, "stdin", FakeStdin())
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))

    options = parse_lab_setup_args([])

    assert options.agents == ("droid", "amp", "claude")
    output = capsys.readouterr().out
    assert "Unsupported coding agent 'codx'" in output


def test_lab_setup_non_interactive_requires_explicit_agent(monkeypatch: Any) -> None:
    class FakeStdin:
        def isatty(self) -> bool:
            return False

    monkeypatch.setattr(lab_setup.sys, "stdin", FakeStdin())

    with pytest.raises(ValueError, match="No --agent provided"):
        parse_lab_setup_args([])


def test_lab_sync_rejects_agent_with_no_agent() -> None:
    with pytest.raises(ValueError, match="--agent and --no-agent"):
        parse_lab_sync_args(["--agent", "codex", "--no-agent"])


def test_lab_setup_service_downloads_upstream_assets_without_agent_installs(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(AGENT_WHICH, lambda _command: None)
    commands: list[list[str]] = []
    emitted: list[Any] = []

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
    assert (tmp_path / "configs" / "eval" / "debug.toml").is_file()
    assert (tmp_path / "configs" / "zero3.yaml").is_file()
    assert (tmp_path / ".prime" / "lab" / "templates" / "configs" / "rl" / "gsm8k.toml").is_file()
    assert (tmp_path / ".prime" / "lab" / "docs" / "index.md").is_file()
    gitignore = (tmp_path / ".gitignore").read_text(encoding="utf-8")
    assert "/outputs/" in gitignore.splitlines()
    assert (tmp_path / ".pi" / "extensions" / "prime-lab" / "index.ts").is_file()
    output = _render_emitted(emitted)
    assert "pi-acp" not in output
    assert "Pi Coding Agent requires pi" in output


def test_lab_setup_service_emits_post_setup_call_to_action(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(AGENT_WHICH, lambda _command: "/bin/tool")
    emitted: list[Any] = []

    result = run_lab_setup_service(
        LabSetupOptions(skip_install=True, skip_agents_md=True, agents=("codex",)),
        workspace=tmp_path,
        emit=emitted.append,
    )

    output = _render_emitted(emitted)
    assert result.exit_code == 0
    assert "get started" in output
    assert "idea -> environment -> eval -> training" in output
    assert "ask codex" in output
    assert "I want to train a model for <my task domain>" in output
    assert "prime env init my-env" in output
    assert "prime eval run my-env -m openai/gpt-5-nano -n 5" in output
    assert "prime rl run configs/rl/qwen-3-5.toml" in output
    assert "prime gepa run my-env -m openai/gpt-5-nano" in output


def test_post_setup_call_to_action_uses_prime_rl_command_when_requested() -> None:
    output = _render_emitted(
        [lab_setup._post_setup_call_to_action(LabSetupOptions(prime_rl=True, agents=("pi",)))]
    )

    assert "ask pi" in output
    assert "uv run prime-rl configs/prime-rl/wiki-search.toml" in output
    assert "prime rl run configs/rl/qwen-3-5.toml" not in output


def test_lab_setup_refreshes_existing_workspace_guidance(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(AGENT_WHICH, lambda _command: "/bin/tool")
    existing_files = {
        tmp_path / "AGENTS.md": "workspace agents\n",
        tmp_path / "CLAUDE.md": "workspace claude\n",
        tmp_path / "environments" / "AGENTS.md": "environment agents\n",
    }
    for path, text in existing_files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    emitted: list[Any] = []

    result = run_lab_setup_service(
        LabSetupOptions(skip_install=True, skip_agents_md=False, agents=("codex",)),
        workspace=tmp_path,
        emit=emitted.append,
    )

    assert result.exit_code == 0
    for path in existing_files:
        assert path.read_text(encoding="utf-8").startswith("downloaded from ")
    assert "already exists" not in _render_emitted(emitted)


def test_lab_sync_refreshes_existing_workspace_guidance(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(AGENT_WHICH, lambda _command: "/bin/tool")
    existing_files = (
        tmp_path / "AGENTS.md",
        tmp_path / "CLAUDE.md",
        tmp_path / "environments" / "AGENTS.md",
    )
    for path in existing_files:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("stale guidance\n", encoding="utf-8")
    emitted: list[str] = []

    result = run_lab_sync_service(
        LabSyncOptions(agents=("codex",), skip_docs=False),
        workspace=tmp_path,
        emit=emitted.append,
    )

    assert result.exit_code == 0
    for path in existing_files:
        assert path.read_text(encoding="utf-8").startswith("downloaded from ")
    assert not any("already exists" in line for line in emitted)


def test_lab_setup_uses_existing_verifiers_sources(
    tmp_path: Path,
    monkeypatch: Any,
    fake_lab_asset_downloads: list[str],
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(AGENT_WHICH, lambda _command: "/bin/tool")

    result = run_lab_setup_service(
        LabSetupOptions(skip_install=True, skip_agents_md=False, agents=("codex",)),
        workspace=tmp_path,
        emit=lambda _text: None,
    )

    assert result.exit_code == 0
    assert _is_pinned_ref(lab_setup.VERIFIERS_REF)
    assert lab_setup.VERIFIERS_CONFIG_REF == "main"
    assert _is_pinned_ref(lab_setup.PRIME_RL_REF)
    assert any(
        url.endswith(
            f"/primeintellect-ai/verifiers/{lab_setup.VERIFIERS_REF}/skills/create-environments/SKILL.md"
        )
        for url in fake_lab_asset_downloads
    )
    assert any(
        url.endswith(
            f"/primeintellect-ai/verifiers/{lab_setup.VERIFIERS_CONFIG_REF}/configs/rl/gsm8k.toml"
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
    monkeypatch.setattr(AGENT_WHICH, lambda _command: "/bin/tool")

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
    monkeypatch.setattr(AGENT_WHICH, lambda _command: "/bin/tool")
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
    monkeypatch.setattr(AGENT_WHICH, lambda _command: "/bin/tool")

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
    assert (tmp_path / ".prime" / "lab" / "agent-mcp" / "amp.json").is_file()
    assert not (tmp_path / ".prime" / "lab" / "agent-mcp" / "droid.json").exists()
    assert (tmp_path / ".prime" / "lab" / "templates" / "configs" / "rl" / "gsm8k.toml").is_file()


def test_lab_sync_fully_refreshes_global_and_workspace_config_templates(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))

    global_templates = home / ".prime" / "lab" / "templates"
    workspace_templates = tmp_path / ".prime" / "lab" / "templates"
    for root in (global_templates, workspace_templates):
        (root / "configs" / "rl").mkdir(parents=True)
        (root / "configs" / "rl" / "gsm8k.toml").write_text("stale\n", encoding="utf-8")
        (root / "configs" / "old.toml").write_text("old\n", encoding="utf-8")

    result = run_lab_sync_service(
        LabSyncOptions(skip_docs=True, no_agent=True),
        workspace=tmp_path,
        emit=lambda _text: None,
    )

    assert result.exit_code == 0
    for root in (global_templates, workspace_templates):
        refreshed = root / "configs" / "rl" / "gsm8k.toml"
        assert refreshed.read_text(encoding="utf-8") == 'model = "openai/gpt-4.1-mini"\n'
        assert not (root / "configs" / "old.toml").exists()
        assert (root / "configs" / "eval" / "debug.toml").is_file()
        assert (root / "configs" / "eval" / "wordle.toml").is_file()
        assert (root / "configs" / "zero3.yaml").is_file()


def test_lab_doctor_reports_missing_selected_agent_guidance(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(AGENT_WHICH, lambda _command: None)
    run_lab_setup_service(
        LabSetupOptions(skip_install=True, skip_agents_md=True, agents=("amp",)),
        workspace=tmp_path,
        emit=lambda _text: None,
    )

    result = run_lab_doctor_service(LabDoctorOptions(), workspace=tmp_path)
    checks = {check.name: check for check in result.checks}

    assert checks["Amp Code native tools"].status == "WARN"
    assert "npm install -g @sourcegraph/amp@latest" in checks["Amp Code native tools"].remediation


def test_lab_doctor_warns_when_native_surface_has_no_paths(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    (tmp_path / ".prime").mkdir()
    (tmp_path / ".prime" / "lab.json").write_text(
        json.dumps({"choices": {"agents": ["future"], "primary_agent": "future"}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        lab_setup,
        "agent_capability",
        lambda agent: AgentCapability(
            name=agent,
            label="Future Agent",
            native_surface="mcp_config",
        ),
    )

    result = run_lab_doctor_service(LabDoctorOptions(), workspace=tmp_path)
    checks = {check.name: check for check in result.checks}

    assert checks["Future Agent native tools"].status == "WARN"
    assert "declares mcp_config but no path" in checks["Future Agent native tools"].message


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


def test_lab_doctor_allows_local_env_refs_when_environment_dir_empty(tmp_path: Path) -> None:
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "rl.toml").write_text(
        'model = "openai/gpt-oss-20b"\nenvironments = ["missing-env"]\n',
        encoding="utf-8",
    )
    (tmp_path / "environments").mkdir()

    result = run_lab_doctor_service(LabDoctorOptions(), workspace=tmp_path)
    checks = {check.name: check for check in result.checks}

    assert checks["Config environment refs"].status == "PASS"


def test_lab_doctor_warns_on_deprecated_config_fields(tmp_path: Path) -> None:
    (tmp_path / ".prime").mkdir()
    (tmp_path / ".prime" / "lab.json").write_text(
        json.dumps({"choices": {"agents": ["codex"], "primary_agent": "codex"}}),
        encoding="utf-8",
    )
    (tmp_path / "configs" / "rl").mkdir(parents=True)
    (tmp_path / "configs" / "rl" / "old.toml").write_text(
        'model = "Qwen/Qwen3.5-0.8B"\n'
        'trajectory_strategy = "interleaved"\n'
        'env_file = ["secrets.env"]\n'
        "oversampling_factor = 2.0\n"
        "max_async_level = 2\n"
        "max_off_policy_steps = 4\n",
        encoding="utf-8",
    )

    result = run_lab_doctor_service(LabDoctorOptions(), workspace=tmp_path)
    checks = {check.name: check for check in result.checks}

    assert checks["Config deprecated fields"].status == "WARN"
    assert "configs/rl/old.toml:trajectory_strategy" in checks["Config deprecated fields"].message
    assert "configs/rl/old.toml:env_file" in checks["Config deprecated fields"].message
    assert "configs/rl/old.toml:oversampling_factor" in checks["Config deprecated fields"].message
    assert "configs/rl/old.toml:max_async_level" in checks["Config deprecated fields"].message
    assert "configs/rl/old.toml:max_off_policy_steps" in checks["Config deprecated fields"].message


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
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(AGENT_WHICH, lambda _command: "/bin/tool")
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
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(AGENT_WHICH, lambda _command: "/bin/tool")
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
    monkeypatch.setattr(AGENT_WHICH, lambda _command: "/bin/tool")
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
    monkeypatch.setattr(AGENT_WHICH, lambda _command: "/bin/tool")

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


def test_lab_setup_supports_letta_project_skills(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(AGENT_WHICH, lambda _command: "/bin/letta")

    result = run_lab_setup_service(
        LabSetupOptions(skip_install=True, skip_agents_md=True, agents=("letta",)),
        workspace=tmp_path,
        emit=lambda _text: None,
    )

    metadata = json.loads((tmp_path / ".prime" / "lab.json").read_text(encoding="utf-8"))
    assert result.exit_code == 0
    assert metadata["choices"]["agents"] == ["letta"]
    assert (tmp_path / ".agents" / "skills" / "create-environments" / "SKILL.md").is_file()
