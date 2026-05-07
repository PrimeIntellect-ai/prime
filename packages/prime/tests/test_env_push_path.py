from prime_cli.commands.env import (
    _environment_push_metadata,
    _environment_ref,
    _resolve_push_environment_path,
)


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


def test_push_metadata_clears_stale_forked_from_without_new_upstream_change() -> None:
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
    assert "forked_from" not in metadata


def test_environment_ref_preserves_zero_identifiers() -> None:
    assert _environment_ref("owner", "env", environment_id=0, version=0) == {
        "owner": "owner",
        "name": "env",
        "environment_id": "0",
        "version": "0",
    }
