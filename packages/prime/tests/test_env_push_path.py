import json
import subprocess
from types import SimpleNamespace

import pytest
import typer
from prime_cli.commands import env as env_command
from prime_cli.commands.env import (
    _environment_push_metadata,
    _environment_ref,
    _environment_resolve_data,
    _resolve_push_environment_path,
    _run_env_push_lab_hygiene_preflight,
)
from typer.testing import CliRunner


def _git_init(path):
    subprocess.run(["git", "init"], cwd=path, check=True, capture_output=True)


def test_defaults_to_current_directory_without_env_id(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    resolved = _resolve_push_environment_path(path=None, env_id=None)

    assert resolved == tmp_path.resolve()


def test_uses_environments_parent_when_env_id_is_provided(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    resolved = _resolve_push_environment_path(path=None, env_id="my-env")

    assert resolved == (tmp_path / "environments" / "my_env").resolve()


def test_appends_env_id_to_custom_parent_path(tmp_path):
    custom_parent = tmp_path / "custom"

    resolved = _resolve_push_environment_path(path=str(custom_parent), env_id="env-name")

    assert resolved == (custom_parent / "env_name").resolve()


def test_uses_final_segment_for_owner_prefixed_env_id(tmp_path):
    custom_parent = tmp_path / "custom"

    resolved = _resolve_push_environment_path(path=str(custom_parent), env_id="owner/env-name")

    assert resolved == (custom_parent / "env_name").resolve()


def test_respects_explicit_path_without_env_id(tmp_path):
    custom_path = tmp_path / "single-env-dir"

    resolved = _resolve_push_environment_path(path=str(custom_path), env_id=None)

    assert resolved == custom_path.resolve()


def test_environment_resolve_data_uses_configured_team_id() -> None:
    resolve_data = _environment_resolve_data(
        "demo-env",
        visibility=None,
        owner=None,
        team=None,
        configured_team="cmf0ohr9s0026ilerf3w68s6n",
    )

    assert resolve_data == {
        "name": "demo-env",
        "team_id": "cmf0ohr9s0026ilerf3w68s6n",
    }


def test_environment_resolve_data_treats_configured_slug_as_team_slug() -> None:
    resolve_data = _environment_resolve_data(
        "demo-env",
        visibility=None,
        owner=None,
        team=None,
        configured_team=" my-team ",
    )

    assert resolve_data == {"name": "demo-env", "team_slug": "my-team"}


def test_environment_resolve_data_explicit_team_overrides_configured_team() -> None:
    resolve_data = _environment_resolve_data(
        "demo-env",
        visibility="PUBLIC",
        owner=None,
        team="research",
        configured_team="cmf0ohr9s0026ilerf3w68s6n",
    )

    assert resolve_data == {
        "name": "demo-env",
        "visibility": "PUBLIC",
        "team_slug": "research",
    }


def test_environment_resolve_data_owner_overrides_team_context() -> None:
    resolve_data = _environment_resolve_data(
        "demo-env",
        visibility=None,
        owner="upstream-owner",
        team="research",
        configured_team="cmf0ohr9s0026ilerf3w68s6n",
    )

    assert resolve_data == {"name": "demo-env", "owner_slug": "upstream-owner"}


def test_env_init_runs_lab_hygiene_preflight_inside_lab_workspace(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".prime").mkdir()
    (tmp_path / ".prime" / "lab.json").write_text(
        json.dumps({"choices": {"agents": ["codex"], "primary_agent": "codex"}}),
        encoding="utf-8",
    )

    class DummyPlugin:
        init_module = "verifiers.v1.cli.init"

        def build_module_command(self, module, args):
            return ["verifiers-init", module, *args]

    monkeypatch.setattr(
        "prime_cli.commands.env.load_verifiers_prime_plugin",
        lambda: DummyPlugin(),
    )

    def fake_run(command, *args, **kwargs):
        if command and command[0] == "git":
            return SimpleNamespace(returncode=1, stdout="")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(
        "prime_cli.commands.env.subprocess.run",
        fake_run,
    )

    result = CliRunner().invoke(env_command.app, ["init", "demo"])

    gitignore_lines = set((tmp_path / ".gitignore").read_text(encoding="utf-8").splitlines())
    assert result.exit_code == 0
    assert "/AGENTS.md" in gitignore_lines
    assert "/.prime/" in gitignore_lines


def test_env_push_blocks_tracked_generated_lab_outputs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".prime").mkdir()
    (tmp_path / ".prime" / "lab.json").write_text(
        json.dumps({"choices": {"agents": ["codex"], "primary_agent": "codex"}}),
        encoding="utf-8",
    )
    env_dir = tmp_path / "environments" / "demo"
    output = env_dir / "outputs" / "run.jsonl"
    output.parent.mkdir(parents=True)
    output.write_text("{}\n", encoding="utf-8")
    _git_init(tmp_path)
    subprocess.run(
        ["git", "add", "environments/demo/outputs/run.jsonl"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )

    with pytest.raises(typer.Exit) as exc_info:
        _run_env_push_lab_hygiene_preflight(env_dir)

    assert exc_info.value.exit_code == 1


def test_env_push_allows_explicit_path_outside_current_lab_workspace(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".prime").mkdir()
    (tmp_path / ".prime" / "lab.json").write_text(
        json.dumps({"choices": {"agents": ["codex"], "primary_agent": "codex"}}),
        encoding="utf-8",
    )
    output = tmp_path / "environments" / "demo" / "outputs" / "run.jsonl"
    output.parent.mkdir(parents=True)
    output.write_text("{}\n", encoding="utf-8")
    external_env = tmp_path.parent / "outside-env"
    external_env.mkdir()
    _git_init(tmp_path)
    subprocess.run(
        ["git", "add", "environments/demo/outputs/run.jsonl"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )

    _run_env_push_lab_hygiene_preflight(external_env)


def test_push_metadata_replaces_existing_version() -> None:
    metadata = _environment_push_metadata(
        {
            "environment_id": "old-id",
            "owner": "base",
            "name": "math-env",
            "version": "0.1.0",
        },
        environment_id="new-id",
        owner="research",
        name="math-env",
        version="0.2.0",
        pushed_at="2026-05-05T12:00:00",
        wheel_sha256="abc123",
    )

    assert metadata["environment_id"] == "new-id"
    assert metadata["owner"] == "research"
    assert metadata["name"] == "math-env"
    assert metadata["version"] == "0.2.0"
    assert metadata["forked_from"] == {
        "environment_id": "old-id",
        "owner": "base",
        "name": "math-env",
        "version": "0.1.0",
    }


def test_push_metadata_preserves_existing_version_when_project_version_is_missing() -> None:
    metadata = _environment_push_metadata(
        {
            "environment_id": "env-id",
            "owner": "research",
            "name": "math-env",
            "version": "0.2.0",
        },
        environment_id="env-id",
        owner="research",
        name="math-env",
        version=None,
        pushed_at="2026-05-05T12:15:00",
        wheel_sha256="def456",
    )

    assert metadata["environment_id"] == "env-id"
    assert metadata["version"] == "0.2.0"


def test_push_metadata_preserves_existing_forked_from_without_new_upstream_change() -> None:
    origin = {
        "environment_id": "old-id",
        "owner": "base",
        "name": "math-env",
        "version": "0.1.0",
    }

    metadata = _environment_push_metadata(
        {
            "environment_id": "new-id",
            "owner": "research",
            "name": "math-env",
            "version": "0.2.0",
            "origin": origin,
            "fork_chain": [origin],
            "forked_from": origin,
        },
        environment_id="new-id",
        owner="research",
        name="math-env",
        version="0.3.0",
        pushed_at="2026-05-05T12:30:00",
        wheel_sha256="def456",
    )

    assert metadata["environment_id"] == "new-id"
    assert metadata["owner"] == "research"
    assert metadata["name"] == "math-env"
    assert metadata["version"] == "0.3.0"
    assert metadata["origin"] == origin
    assert metadata["fork_chain"] == [origin]
    assert metadata["forked_from"] == origin


def test_push_metadata_clears_malformed_stale_fork_provenance() -> None:
    metadata = _environment_push_metadata(
        {
            "environment_id": "new-id",
            "owner": "research",
            "name": "math-env",
            "origin": {"owner": "", "name": ""},
            "fork_chain": [{"owner": "", "name": ""}],
            "forked_from": {"owner": "", "name": ""},
        },
        environment_id="new-id",
        owner="research",
        name="math-env",
        version=None,
        pushed_at="2026-05-05T12:45:00",
        wheel_sha256="def456",
    )

    assert "origin" not in metadata
    assert "fork_chain" not in metadata
    assert "forked_from" not in metadata


def test_push_metadata_preserves_existing_valid_forked_from() -> None:
    forked_from = {"owner": "base", "name": "old-env", "environment_id": "base-id"}
    metadata = _environment_push_metadata(
        {
            "environment_id": "new-id",
            "owner": "research",
            "name": "math-env",
            "forked_from": forked_from,
        },
        environment_id="new-id",
        owner="research",
        name="math-env",
        version=None,
        pushed_at="2026-05-05T12:45:00",
        wheel_sha256="def456",
    )

    assert metadata["forked_from"] == forked_from


def test_environment_ref_preserves_zero_identifiers() -> None:
    assert _environment_ref("owner", "env", environment_id=0, version=0) == {
        "owner": "owner",
        "name": "env",
        "environment_id": "0",
        "version": "0",
    }
