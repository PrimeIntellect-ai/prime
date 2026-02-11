from prime_cli.commands.env import _resolve_push_environment_path


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
