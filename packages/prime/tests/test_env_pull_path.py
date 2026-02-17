from prime_cli.commands.env import _resolve_pull_environment_path


def test_defaults_to_cwd_when_environments_dir_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    resolved = _resolve_pull_environment_path(target=None, env_name="my-env")

    assert resolved == tmp_path / "my_env"


def test_defaults_to_environments_dir_when_present(tmp_path, monkeypatch):
    (tmp_path / "environments").mkdir()
    monkeypatch.chdir(tmp_path)

    resolved = _resolve_pull_environment_path(target=None, env_name="my-env")

    assert resolved == tmp_path / "environments" / "my_env"


def test_respects_explicit_target_path(tmp_path):
    explicit = tmp_path / "custom-target"

    resolved = _resolve_pull_environment_path(target=str(explicit), env_name="my-env")

    assert resolved == explicit
