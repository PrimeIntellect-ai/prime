from __future__ import annotations

import json
import subprocess
from io import StringIO
from pathlib import Path
from typing import Any
from urllib.error import URLError

import pytest
from prime_cli import lab_setup
from prime_cli.commands.lab import app as lab_cli_app
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
from typer.testing import CliRunner

AGENT_WHICH = "prime_lab_app.agent_capabilities.shutil.which"
REAL_DOWNLOAD_FILE = lab_setup._download_file


def _git_init(path: Path) -> None:
    subprocess.run(["git", "init"], cwd=path, check=True, capture_output=True)


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
        quiet: bool = False,
    ) -> None:
        if dest.exists() and not force:
            if not quiet:
                emit(f"{dest.name} already exists\n")
            return
        urls.append(url)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.suffix == ".toml":
            dest.write_text('model = "openai/gpt-4.1-mini"\n', encoding="utf-8")
        else:
            dest.write_text(f"downloaded from {url}\n", encoding="utf-8")
        if not quiet:
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


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )


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


def test_lab_setup_rejects_prime_rl_flag() -> None:
    with pytest.raises(SystemExit) as exc_info:
        parse_lab_setup_args(["--prime-rl", "--no-interactive"])

    assert exc_info.value.code == 2


def test_lab_register_github_writes_hygiene_workflow(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(lab_cli_app, ["register-github"])

    workflow = tmp_path / ".github" / "workflows" / "prime-lab-hygiene.yml"
    content = workflow.read_text(encoding="utf-8")
    assert result.exit_code == 0
    assert "name: Prime Lab Hygiene" in content
    assert "actions/checkout@v4" in content
    assert "astral-sh/setup-uv@v5" in content
    assert "uvx --from prime prime lab hygiene" in content
    assert "pre-commit" not in content


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
    assert not (tmp_path / "configs" / "zero3.yaml").exists()
    assert not (tmp_path / "configs" / "endpoints.toml").exists()
    assert not (tmp_path / "configs" / "local").exists()
    assert (tmp_path / ".prime" / "lab" / "templates" / "configs" / "rl" / "gsm8k.toml").is_file()
    assert (tmp_path / ".prime" / "lab" / "docs" / "index.md").is_file()
    gitignore = (tmp_path / ".gitignore").read_text(encoding="utf-8")
    assert "/outputs/" in gitignore.splitlines()
    assert "/AGENTS.md" in gitignore.splitlines()
    assert "/CLAUDE.md" in gitignore.splitlines()
    assert "/CLAUDE.local.md" in gitignore.splitlines()
    assert "/.prime/" in gitignore.splitlines()
    assert (tmp_path / ".pi" / "extensions" / "prime-lab" / "index.ts").is_file()
    output = _render_emitted(emitted)
    assert "pi-acp" not in output
    assert "Pi Coding Agent requires pi" in output
    assert "Downloaded " not in output
    assert ".skills-staging-" not in output
    assert ".templates-staging-" not in output


def test_lab_setup_hygiene_preflight_nudges_tracked_guidance(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(AGENT_WHICH, lambda _command: "/bin/tool")
    (tmp_path / "AGENTS.md").write_text("tracked guidance\n", encoding="utf-8")
    _git_init(tmp_path)
    subprocess.run(["git", "add", "AGENTS.md"], cwd=tmp_path, check=True, capture_output=True)
    emitted: list[Any] = []

    result = run_lab_setup_service(
        LabSetupOptions(skip_install=True, skip_agents_md=True, agents=("codex",)),
        workspace=tmp_path,
        emit=emitted.append,
    )

    output = _render_emitted(emitted)
    assert result.exit_code == 0
    assert "untracked generated Lab files: AGENTS.md" in output
    assert _git(tmp_path, "ls-files", "--", "AGENTS.md").stdout == ""


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
    assert "prime eval run my-env -m openai/gpt-5.4-nano -n 5" in output
    assert "prime rl run configs/rl/qwen-3-5.toml" in output
    assert "prime gepa run my-env -m openai/gpt-5.4-nano" in output


def test_lab_setup_ignores_managed_guidance_and_skips_claude_local_for_codex(
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

    docs_index = (tmp_path / ".prime" / "lab" / "docs" / "index.md").read_text(encoding="utf-8")
    gitignore_lines = (tmp_path / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert result.exit_code == 0
    assert (tmp_path / "AGENTS.md").read_text(encoding="utf-8").startswith("downloaded from ")
    assert (tmp_path / "CLAUDE.md").read_text(encoding="utf-8").startswith("downloaded from ")
    assert not (tmp_path / "CLAUDE.local.md").exists()
    assert "- `AGENTS.md`" in docs_index
    assert "- `CLAUDE.md`" in docs_index
    assert "- `CLAUDE.local.md`" not in docs_index
    assert "- `environments/AGENTS.md`" in docs_index
    for pattern in (
        "/AGENTS.md",
        "/CLAUDE.md",
        "/CLAUDE.local.md",
        "/environments/AGENTS.md",
    ):
        assert pattern in gitignore_lines
    assert any("/assets/lab/CLAUDE.md" in url for url in fake_lab_asset_downloads)


def test_lab_setup_refreshes_existing_workspace_guidance(
    tmp_path: Path,
    monkeypatch: Any,
    fake_lab_asset_downloads: list[str],
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(AGENT_WHICH, lambda _command: "/bin/tool")
    existing_files = {
        tmp_path / "AGENTS.md": "workspace agents\n",
        tmp_path / "CLAUDE.md": "workspace claude\n",
        tmp_path / "CLAUDE.local.md": "workspace claude local guidance\n",
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
    for path in (
        tmp_path / "AGENTS.md",
        tmp_path / "CLAUDE.md",
        tmp_path / "environments" / "AGENTS.md",
    ):
        assert path.read_text(encoding="utf-8").startswith("downloaded from ")
    assert (tmp_path / "CLAUDE.local.md").read_text(encoding="utf-8") == (
        "workspace claude local guidance\n"
    )
    docs_index = (tmp_path / ".prime" / "lab" / "docs" / "index.md").read_text(encoding="utf-8")
    assert "- `CLAUDE.md`" in docs_index
    assert "- `CLAUDE.local.md`" not in docs_index
    assert any("/assets/lab/CLAUDE.md" in url for url in fake_lab_asset_downloads)
    assert "already exists" not in _render_emitted(emitted)


def test_lab_setup_refreshes_claude_guidance_for_claude_agent(
    tmp_path: Path,
    monkeypatch: Any,
    fake_lab_asset_downloads: list[str],
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(AGENT_WHICH, lambda _command: "/bin/tool")

    result = run_lab_setup_service(
        LabSetupOptions(skip_install=True, skip_agents_md=False, agents=("claude",)),
        workspace=tmp_path,
    )

    assert result.exit_code == 0
    assert (tmp_path / "CLAUDE.md").read_text(encoding="utf-8").startswith("downloaded from ")
    claude_local = tmp_path / "CLAUDE.local.md"
    assert claude_local.read_text(encoding="utf-8") == lab_setup.LOCAL_CLAUDE_GUIDANCE_TEMPLATE
    docs_index = (tmp_path / ".prime" / "lab" / "docs" / "index.md").read_text(encoding="utf-8")
    assert "- `CLAUDE.md`" in docs_index
    assert "- `CLAUDE.local.md`" in docs_index
    assert any("/assets/lab/CLAUDE.md" in url for url in fake_lab_asset_downloads)


def test_lab_setup_preserves_user_owned_claude_local_guidance(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(AGENT_WHICH, lambda _command: "/bin/tool")
    (tmp_path / "CLAUDE.local.md").write_text("claude local guidance\n", encoding="utf-8")

    result = run_lab_setup_service(
        LabSetupOptions(skip_install=True, skip_agents_md=False, agents=("claude",)),
        workspace=tmp_path,
    )

    assert result.exit_code == 0
    assert (tmp_path / "CLAUDE.local.md").read_text(encoding="utf-8") == ("claude local guidance\n")


def test_lab_sync_refreshes_existing_workspace_guidance(
    tmp_path: Path,
    monkeypatch: Any,
    fake_lab_asset_downloads: list[str],
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
    for path in (
        tmp_path / "AGENTS.md",
        tmp_path / "CLAUDE.md",
        tmp_path / "environments" / "AGENTS.md",
    ):
        assert path.read_text(encoding="utf-8").startswith("downloaded from ")
    assert not (tmp_path / "CLAUDE.local.md").exists()
    assert any("/assets/lab/CLAUDE.md" in url for url in fake_lab_asset_downloads)
    assert not any("already exists" in line for line in emitted)


def test_lab_sync_hygiene_preflight_applies_safe_fixes(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(AGENT_WHICH, lambda _command: "/bin/tool")
    emitted: list[Any] = []

    result = run_lab_sync_service(
        LabSyncOptions(no_agent=True, skip_docs=True),
        workspace=tmp_path,
        emit=emitted.append,
    )

    gitignore_lines = set((tmp_path / ".gitignore").read_text(encoding="utf-8").splitlines())
    output = _render_emitted(emitted)
    assert result.exit_code == 0
    assert (tmp_path / "configs").is_dir()
    assert (tmp_path / "environments").is_dir()
    assert "/AGENTS.md" in gitignore_lines
    assert "/.prime/" in gitignore_lines
    assert "Lab hygiene: added standard .gitignore entries" in output


def test_lab_sync_without_configured_agent_refreshes_shared_assets(
    tmp_path: Path,
    monkeypatch: Any,
    fake_lab_asset_downloads: list[str],
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(AGENT_WHICH, lambda _command: "/bin/tool")
    emitted: list[str] = []

    result = run_lab_sync_service(
        LabSyncOptions(skip_docs=False),
        workspace=tmp_path,
        emit=emitted.append,
    )

    assert result.exit_code == 0
    assert (tmp_path / ".prime" / "skills" / "create-environments" / "SKILL.md").is_file()
    assert (tmp_path / "AGENTS.md").read_text(encoding="utf-8").startswith("downloaded from ")
    assert (
        (tmp_path / "environments" / "AGENTS.md")
        .read_text(encoding="utf-8")
        .startswith("downloaded from ")
    )
    assert (tmp_path / "CLAUDE.md").read_text(encoding="utf-8").startswith("downloaded from ")
    assert not (tmp_path / "CLAUDE.local.md").exists()
    assert "- `CLAUDE.md`" in (tmp_path / ".prime" / "lab" / "docs" / "index.md").read_text(
        encoding="utf-8"
    )
    assert (tmp_path / ".prime" / "lab" / "templates" / "configs" / "rl" / "gsm8k.toml").is_file()
    assert not (tmp_path / ".agents" / "skills").exists()
    assert any("/assets/lab/CLAUDE.md" in url for url in fake_lab_asset_downloads)
    assert any("no configured agent; pass --agent to configure one" in line for line in emitted)


def test_lab_sync_no_agent_keeps_stored_claude_guidance(
    tmp_path: Path,
    monkeypatch: Any,
    fake_lab_asset_downloads: list[str],
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(AGENT_WHICH, lambda _command: "/bin/tool")
    (tmp_path / ".prime").mkdir()
    (tmp_path / ".prime" / "lab.json").write_text(
        json.dumps({"choices": {"agents": ["claude"], "primary_agent": "claude"}}),
        encoding="utf-8",
    )
    for path in (
        tmp_path / "AGENTS.md",
        tmp_path / "CLAUDE.md",
        tmp_path / "environments" / "AGENTS.md",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("stale guidance\n", encoding="utf-8")
    emitted: list[str] = []

    result = run_lab_sync_service(
        LabSyncOptions(skip_docs=False, no_agent=True),
        workspace=tmp_path,
        emit=emitted.append,
    )

    assert result.exit_code == 0
    assert (tmp_path / "CLAUDE.md").read_text(encoding="utf-8").startswith("downloaded from ")
    assert (tmp_path / "CLAUDE.local.md").read_text(
        encoding="utf-8"
    ) == lab_setup.LOCAL_CLAUDE_GUIDANCE_TEMPLATE
    docs_index = (tmp_path / ".prime" / "lab" / "docs" / "index.md").read_text(encoding="utf-8")
    assert "- `CLAUDE.md`" in docs_index
    assert "- `CLAUDE.local.md`" in docs_index
    assert any("/assets/lab/CLAUDE.md" in url for url in fake_lab_asset_downloads)
    assert not (tmp_path / ".claude" / "skills").exists()
    assert any("Skipped coding-agent skill roots (--no-agent)" in line for line in emitted)


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
    assert (
        sum(
            url.endswith(
                f"/primeintellect-ai/verifiers/{lab_setup.VERIFIERS_CONFIG_REF}"
                "/configs/rl/gsm8k.toml"
            )
            for url in fake_lab_asset_downloads
        )
        == 1
    )
    assert any(
        url.endswith(f"/primeintellect-ai/verifiers/{lab_setup.VERIFIERS_REF}/assets/lab/AGENTS.md")
        for url in fake_lab_asset_downloads
    )
    assert not any("/configs/endpoints.toml" in url for url in fake_lab_asset_downloads)
    assert not any("/configs/local/" in url for url in fake_lab_asset_downloads)
    assert not any("/configs/zero3.yaml" in url for url in fake_lab_asset_downloads)


def test_lab_config_downloader_targets_known_config_folders(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    calls: list[tuple[str, Path, bool]] = []

    def fake_download_repo_directory(
        _repo: str,
        _ref: str,
        source_path: str,
        dest: Path,
        _emit: Any,
        *,
        force: bool = True,
        missing_ok: bool = False,
    ) -> None:
        assert missing_ok is True
        calls.append((source_path, dest, force))

    monkeypatch.setattr(lab_setup, "_download_repo_directory", fake_download_repo_directory)

    lab_setup._download_lab_config_folders(
        tmp_path / "configs",
        emit=lambda _text: None,
        force=False,
    )

    assert calls == [
        ("configs/rl", tmp_path / "configs" / "rl", False),
        ("configs/gepa", tmp_path / "configs" / "gepa", False),
        ("configs/eval", tmp_path / "configs" / "eval", False),
        ("configs/sft", tmp_path / "configs" / "sft", False),
        ("configs/opd", tmp_path / "configs" / "opd", False),
        ("configs/fft", tmp_path / "configs" / "fft", False),
    ]


def test_copy_setup_configs_downloads_missing_cached_template_folders(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    template_configs = home / ".prime" / "lab" / "templates" / "configs"
    (template_configs / "rl").mkdir(parents=True)
    (template_configs / "rl" / "gsm8k.toml").write_text("cached\n", encoding="utf-8")
    workspace = tmp_path / "workspace"
    calls: list[tuple[Path, bool, tuple[str, ...]]] = []

    def fake_repo_tree_entries(_repo: str, _ref: str) -> tuple[tuple[str, str], ...]:
        return (
            ("configs/rl/gsm8k.toml", "blob"),
            ("configs/gepa/base.toml", "blob"),
            ("configs/eval/debug.toml", "blob"),
        )

    def fake_download_lab_config_folders(
        dest: Path,
        _emit: Any,
        *,
        force: bool,
        folders: tuple[str, ...] = lab_setup.LAB_CONFIG_FOLDERS,
    ) -> None:
        calls.append((dest, force, folders))
        for folder in folders:
            (dest / folder).mkdir(parents=True, exist_ok=True)
            (dest / folder / "downloaded.toml").write_text("downloaded\n", encoding="utf-8")

    monkeypatch.setattr(lab_setup, "_repo_tree_entries", fake_repo_tree_entries)
    monkeypatch.setattr(lab_setup, "_download_lab_config_folders", fake_download_lab_config_folders)

    lab_setup._copy_setup_configs(workspace, emit=lambda _text: None)

    assert (workspace / "configs" / "rl" / "gsm8k.toml").read_text(encoding="utf-8") == "cached\n"
    assert (workspace / "configs" / "gepa" / "downloaded.toml").is_file()
    assert (workspace / "configs" / "eval" / "downloaded.toml").is_file()
    assert calls == [(template_configs, False, ("gepa", "eval"))]


def test_download_file_quiet_suppresses_per_file_status(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    emitted: list[str] = []

    def fake_read_url(_url: str, *, emit: Any | None = None) -> bytes:
        if emit is not None:
            emit("retry detail\n")
        return b"ok"

    monkeypatch.setattr(lab_setup, "_download_file", REAL_DOWNLOAD_FILE)
    monkeypatch.setattr(lab_setup, "_read_url", fake_read_url)

    lab_setup._download_file(
        "https://example.test/file",
        tmp_path / "file",
        emitted.append,
        quiet=True,
    )
    lab_setup._download_file(
        "https://example.test/file",
        tmp_path / "file",
        emitted.append,
        quiet=True,
    )

    assert (tmp_path / "file").read_bytes() == b"ok"
    assert emitted == ["retry detail\n"]


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
        assert not (root / "configs" / "zero3.yaml").exists()
        assert not (root / "configs" / "endpoints.toml").exists()
        assert not (root / "configs" / "local").exists()


def test_lab_doctor_accepts_current_template_config_names(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    global_configs = home / ".prime" / "lab" / "templates" / "configs" / "rl"
    workspace_configs = tmp_path / ".prime" / "lab" / "templates" / "configs" / "rl"
    global_configs.mkdir(parents=True)
    workspace_configs.mkdir(parents=True)
    (global_configs / "qwen.toml").write_text('model = "qwen"\n', encoding="utf-8")
    (workspace_configs / "qwen.toml").write_text('model = "qwen"\n', encoding="utf-8")

    result = run_lab_doctor_service(LabDoctorOptions(), workspace=tmp_path)
    checks = {check.name: check for check in result.checks}

    assert checks["Global Lab template cache"].status == "PASS"
    assert checks["Lab templates"].status == "PASS"


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
    assert (tmp_path / "configs").is_dir()
    assert (tmp_path / "environments").is_dir()
    assert "custom.log" in gitignore
    assert ".env.local" in gitignore_lines
    assert ".env" in gitignore_lines
    assert "/AGENTS.md" in gitignore_lines
    assert "/CLAUDE.md" in gitignore_lines
    assert "/CLAUDE.local.md" in gitignore_lines
    assert "/.prime/" in gitignore_lines
    assert "/.agents/skills/" in gitignore_lines
    assert "/outputs/" in gitignore_lines
    assert "/prime-rl/" in gitignore_lines
    assert "/environments/AGENTS.md" in gitignore_lines
    assert "/environments/*/outputs/" in gitignore_lines
    assert "*.py[cod]" in gitignore_lines
    assert ".pytest_cache/" in gitignore_lines
    assert ".ruff_cache/" in gitignore_lines


def test_lab_doctor_warns_about_tracked_lab_guidance(tmp_path: Path) -> None:
    _git(tmp_path, "init")
    guidance_paths = ("AGENTS.md", "CLAUDE.md", "CLAUDE.local.md", "environments/AGENTS.md")
    for relative_path in guidance_paths:
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("guidance\n", encoding="utf-8")
    _git(tmp_path, "add", *guidance_paths)

    result = run_lab_doctor_service(LabDoctorOptions(), workspace=tmp_path)
    checks = {check.name: check for check in result.checks}

    check = checks["Tracked Lab git hygiene"]
    assert check.status == "FAIL"
    assert "AGENTS.md" in check.message
    assert "CLAUDE.md" in check.message
    assert "CLAUDE.local.md" in check.message
    assert "environments/AGENTS.md" in check.message
    assert "git rm --cached" in check.remediation


def test_lab_doctor_fix_untracks_lab_guidance(tmp_path: Path) -> None:
    _git(tmp_path, "init")
    guidance_paths = ("AGENTS.md", "CLAUDE.md", "CLAUDE.local.md", "environments/AGENTS.md")
    for relative_path in guidance_paths:
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("guidance\n", encoding="utf-8")
    _git(tmp_path, "add", *guidance_paths)

    result = run_lab_doctor_service(LabDoctorOptions(fix=True), workspace=tmp_path)
    checks = {check.name: check for check in result.checks}
    tracked = _git(tmp_path, "ls-files", "--", *guidance_paths)

    assert checks["Tracked Lab git hygiene"].status == "PASS"
    assert tracked.stdout == ""
    assert (tmp_path / "AGENTS.md").is_file()
    assert (tmp_path / "CLAUDE.md").is_file()
    assert (tmp_path / "CLAUDE.local.md").is_file()
    assert (tmp_path / "environments" / "AGENTS.md").is_file()
    assert result.exit_code == 1


def test_lab_doctor_fix_untracks_staged_and_modified_lab_guidance(tmp_path: Path) -> None:
    _git(tmp_path, "init")
    guidance_paths = ("AGENTS.md", "CLAUDE.md", "CLAUDE.local.md", "environments/AGENTS.md")
    for relative_path in guidance_paths:
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("committed guidance\n", encoding="utf-8")
    _git(tmp_path, "add", *guidance_paths)
    _git(
        tmp_path,
        "-c",
        "user.email=lab@example.com",
        "-c",
        "user.name=Prime Lab",
        "commit",
        "-m",
        "track guidance",
    )
    for relative_path in guidance_paths:
        (tmp_path / relative_path).write_text("staged guidance\n", encoding="utf-8")
    _git(tmp_path, "add", *guidance_paths)
    for relative_path in guidance_paths:
        (tmp_path / relative_path).write_text("working guidance\n", encoding="utf-8")

    result = run_lab_doctor_service(LabDoctorOptions(fix=True), workspace=tmp_path)
    checks = {check.name: check for check in result.checks}
    tracked = _git(tmp_path, "ls-files", "--", *guidance_paths)

    assert checks["Tracked Lab git hygiene"].status == "PASS"
    assert tracked.stdout == ""
    for relative_path in guidance_paths:
        assert (tmp_path / relative_path).read_text(encoding="utf-8") == "working guidance\n"
    assert result.exit_code == 1


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
