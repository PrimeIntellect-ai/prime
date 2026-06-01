import importlib.util
import py_compile
import sys
import tarfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "packages" / "prime" / "scripts" / "release_e2e.py"
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "release-e2e.yml"
OLD_WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "prime-release-e2e.yml"


def load_release_e2e_module():
    spec = importlib.util.spec_from_file_location("release_e2e", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["release_e2e"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_env_int_treats_empty_environment_value_as_unset(monkeypatch):
    release_e2e = load_release_e2e_module()

    monkeypatch.delenv("PRIME_E2E_TEST_INT", raising=False)
    assert release_e2e.env_int("PRIME_E2E_TEST_INT", "120") == 120

    monkeypatch.setenv("PRIME_E2E_TEST_INT", "")
    assert release_e2e.env_int("PRIME_E2E_TEST_INT", "120") == 120

    monkeypatch.setenv("PRIME_E2E_TEST_INT", "240")
    assert release_e2e.env_int("PRIME_E2E_TEST_INT", "120") == 240


def test_source_archive_excludes_generated_secret_and_symlink_paths(tmp_path):
    release_e2e = load_release_e2e_module()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "safe.py").write_text("print('ok')\n")
    (repo_root / ".env").write_text("SECRET=1\n")
    (repo_root / "service-account-prod.json").write_text("{}\n")
    (repo_root / "build").mkdir()
    (repo_root / "build" / "artifact.py").write_text("print('skip')\n")
    (repo_root / ".venv").mkdir()
    (repo_root / ".venv" / "installed.py").write_text("print('skip')\n")
    (repo_root / "safe-link.py").symlink_to(repo_root / "safe.py")

    archive = release_e2e.create_source_archive(repo_root)
    with tarfile.open(archive) as tar:
        names = set(tar.getnames())

    assert "prime/safe.py" in names
    assert "prime/.env" not in names
    assert "prime/service-account-prod.json" not in names
    assert "prime/build/artifact.py" not in names
    assert "prime/.venv/installed.py" not in names
    assert "prime/safe-link.py" not in names


def test_remote_script_compiles_and_keeps_cleanup_best_effort(tmp_path):
    release_e2e = load_release_e2e_module()
    script = release_e2e.remote_script(
        release_e2e.RemoteConfig(
            model="deepseek/deepseek-chat",
            hosted_mode="submit",
            env_prefix="prime-e2e",
            hosted_timeout_minutes=120,
            cleanup_remote_env=True,
            run_suffix="test",
        )
    )
    remote_path = tmp_path / "remote.py"
    remote_path.write_text(script)

    py_compile.compile(str(remote_path), doraise=True)
    assert "def best_effort_cancel_hosted_evals" in script
    assert "Warning: failed to delete temporary environment" in script
    assert "line.rsplit(\":\", 1)[-1]" in script


def test_release_e2e_workflow_uses_standard_name_and_safe_inputs():
    workflow = WORKFLOW_PATH.read_text()

    assert not OLD_WORKFLOW_PATH.exists()
    assert workflow.startswith("name: Release E2E Tests")
    assert '"packages/prime-tunnel/**"' in workflow
    assert '".github/workflows/release-e2e.yml"' in workflow
    assert 'HOSTED_MODE: ${{ inputs.hosted_mode || \'submit\' }}' in workflow
    assert 'MODEL: ${{ inputs.model || \'deepseek/deepseek-chat\' }}' in workflow
    assert 'HOSTED_MODE="${{ inputs.hosted_mode' not in workflow
    assert 'MODEL="${{ inputs.model' not in workflow
    assert 'REGION="${{ inputs.sandbox_region' not in workflow
