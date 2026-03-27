import json

from prime_cli.commands.env import compute_content_hash
from prime_cli.verifiers_bridge import _compute_local_content_hash, _resolve_local_env_display


def _create_env(env_dir):
    env_dir.mkdir(parents=True)
    (env_dir / "pyproject.toml").write_text('[project]\nname = "simpleqa"\nversion = "0.1.0"\n')
    (env_dir / "README.md").write_text("# simpleqa\n")
    (env_dir / "simpleqa.py").write_text("def load_environment():\n    return None\n")
    package_dir = env_dir / "prompts"
    package_dir.mkdir()
    (package_dir / "system.txt").write_text("answer carefully\n")


def test_compute_local_content_hash_matches_env_push_hash(tmp_path):
    env_dir = tmp_path / "simpleqa"
    _create_env(env_dir)

    assert _compute_local_content_hash(env_dir) == compute_content_hash(env_dir)


def test_resolve_local_env_display_recognizes_in_sync_metadata_connected_env(tmp_path, monkeypatch):
    env_dir = tmp_path / "simpleqa"
    _create_env(env_dir)

    prime_dir = env_dir / ".prime"
    prime_dir.mkdir()
    (prime_dir / ".env-metadata.json").write_text(
        json.dumps({"owner": "alice", "name": "simpleqa", "environment_id": "env-123"})
    )

    remote_hash = compute_content_hash(env_dir)
    monkeypatch.setattr(
        "prime_cli.verifiers_bridge._fetch_remote_env_details",
        lambda client, owner_slug, env_name, version="latest": {
            "latest_version": {"content_hash": remote_hash}
        },
    )

    env_display_id, platform_slug, platform_url, recommend_push, push_reason = (
        _resolve_local_env_display("simpleqa", env_dir, object())
    )

    assert env_display_id == "alice/simpleqa"
    assert platform_slug == "alice/simpleqa"
    assert platform_url is not None
    assert recommend_push is False
    assert push_reason is None
