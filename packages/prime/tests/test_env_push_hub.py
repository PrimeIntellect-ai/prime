"""`prime env push` against a fake Hub API: what it sends, and how it handles errors."""

import hashlib
import json
import tarfile
from pathlib import Path

import httpx
import pytest
from fake_hub import write_wheel
from prime_cli.commands import env as env_command
from prime_cli.core.client import APIClient
from prime_cli.main import app
from prime_cli.utils import hub as hub_module
from prime_cli.utils import package_push, strip_ansi
from typer.testing import CliRunner

runner = CliRunner()


@pytest.fixture
def package(tmp_path, monkeypatch) -> Path:
    root = tmp_path / "math_env"
    (root / "math_env").mkdir(parents=True)
    (root / "pyproject.toml").write_text(
        '[project]\nname = "math-env"\nversion = "0.2.0"\n'
        'description = "Math"\ndependencies = ["verifiers>=0.2.0"]\n'
    )
    (root / "README.md").write_text("# Math\n")
    (root / "math_env" / "__init__.py").write_text(
        "import verifiers.v1 as vf\n\n"
        "class MathTaskset(vf.Taskset):\n    pass\n\n"
        "class MathJudge(vf.ReferenceJudge):\n    pass\n\n"
        '__all__ = ["MathTaskset", "MathJudge"]\n'
    )

    def build(package_dir: Path) -> Path:
        project = env_command.tomllib.loads((package_dir / "pyproject.toml").read_text())["project"]
        return write_wheel(
            package_dir / "dist",
            project["name"],
            project["version"],
            project.get("dependencies", []),
        )

    monkeypatch.setattr(env_command, "build_wheel", build)
    return root


def _push(*args: str):
    return runner.invoke(app, ["env", "push", *args], env={"COLUMNS": "200"})


def test_push_creates_the_package_and_publishes_a_release(hub, package):
    result = _push("--path", str(package), "--visibility", "PUBLIC")

    output = strip_ansi(result.output)
    assert result.exit_code == 0, output
    assert "Exports: taskset MathTaskset, judge MathJudge" in output
    assert "Successfully pushed alice/math-env@0.2.0" in output

    create = next(c for c in hub.calls if c[:2] == ("POST", "/api/v1/hub/packages"))
    assert create[2] == {
        "owner": "alice",
        "name": "math-env",
        "visibility": "public",
        "description": "Math",
    }
    (release,) = hub.releases
    assert release["version"] == "0.2.0"
    assert "framework" not in release
    assert release["components"] == [
        {"kind": "taskset", "name": "MathTaskset"},
        {"kind": "judge", "name": "MathJudge"},
    ]
    assert release["metadata"]["requires_dist"] == ["verifiers>=0.2.0"]
    artifacts = {a["type"]: a for a in release["artifacts"]}
    assert artifacts["source"]["filename"] == "math_env-0.2.0.tar.gz"
    assert artifacts["wheel"]["filename"] == "math_env-0.2.0-py3-none-any.whl"
    for artifact in artifacts.values():
        uploaded = hub.uploads[artifact["filename"]]
        assert hashlib.sha256(uploaded).hexdigest() == artifact["sha256"]
        assert len(uploaded) == artifact["size"]

    finalize = next(c for c in hub.calls if c[1].endswith("/finalize"))
    assert finalize[1] == "/api/v1/hub/packages/alice/math-env/releases/0.2.0/finalize"
    assert finalize[3] == {"content_hash": artifacts["source"]["sha256"]}

    # The source archive is flat and holds the package's files.
    source = package / "source.tar.gz"
    source.write_bytes(hub.uploads["math_env-0.2.0.tar.gz"])
    with tarfile.open(source) as archive:
        assert sorted(archive.getnames()) == ["README.md", "math_env/__init__.py", "pyproject.toml"]

    metadata = json.loads((package / ".prime" / ".env-metadata.json").read_text())
    assert metadata["environment_id"] == "pkg-alice-math-env"
    assert metadata["owner"] == "alice"
    assert metadata["name"] == "math-env"
    assert metadata["version"] == "0.2.0"


def test_push_to_an_existing_package_under_its_old_spelling(hub, package):
    hub.packages["alice/math_env"] = hub._package("alice", "math_env")

    result = _push("--path", str(package), "--visibility", "public")

    output = strip_ansi(result.output)
    assert result.exit_code == 0, output
    assert not any(c[:2] == ("POST", "/api/v1/hub/packages") for c in hub.calls)
    assert "Successfully pushed alice/math_env@0.2.0" in output
    assert "--visibility applies to new environments" in output


def test_configured_team_id_pushes_under_the_team_slug(hub, package, monkeypatch):
    monkeypatch.setenv("PRIME_TEAM_ID", hub.teams[0]["teamId"])

    result = _push("--path", str(package))

    assert result.exit_code == 0, strip_ansi(result.output)
    assert "research/math-env" in hub.packages


def test_explicit_owner_wins(hub, package, monkeypatch):
    monkeypatch.setenv("PRIME_TEAM_ID", hub.teams[0]["teamId"])
    hub.packages["upstream/math-env"] = hub._package("upstream", "math-env")

    result = _push("--path", str(package), "--owner", "upstream", "--team", "research")

    assert result.exit_code == 0, strip_ansi(result.output)
    paths = [c[1] for c in hub.calls]
    assert "/api/v1/user/teams" not in paths
    assert "/api/v1/hub/packages/upstream/math-env/releases" in paths


def test_a_published_version_is_refused_with_a_tip(hub, package):
    hub.release_error = (409, "version_exists", "0.2.0 is already published")

    result = _push("--path", str(package))

    output = strip_ansi(result.output)
    assert result.exit_code == 1
    assert "0.2.0 is already published (version_exists)" in output
    assert "--auto-bump" in output


def test_finalize_uploads_again_once_if_storage_has_nothing(hub, package):
    hub.finalize_errors = [(409, "artifact_missing", "not uploaded")]

    result = _push("--path", str(package))

    assert result.exit_code == 0, strip_ansi(result.output)
    assert len([c for c in hub.calls if c[1].endswith("/finalize")]) == 2


def test_v0_packages_declare_no_components(hub, package):
    result = _push("--path", str(package), "--runtime", "v0")

    output = strip_ansi(result.output)
    assert result.exit_code == 0, output
    (release,) = hub.releases
    assert release["components"] == []
    assert release["framework"] == "v0"
    assert "Exports:" not in output


def test_name_must_match_the_distribution(hub, package):
    result = _push("--path", str(package), "--name", "other-env")

    output = strip_ansi(result.output)
    assert result.exit_code == 1
    assert "can't be pushed as 'other-env'" in output
    assert hub.calls == []


def test_missing_username_is_chosen_before_pushing(hub, package):
    hub.slug = None

    result = runner.invoke(
        app, ["env", "push", "--path", str(package)], input="bob\n", env={"COLUMNS": "200"}
    )

    assert result.exit_code == 0, strip_ansi(result.output)
    assert ("PATCH", "/api/v1/user/slug", {"slug": "bob"}, {}) in hub.calls
    # Asked on stderr, so stdout stays the command's own output.
    assert "Enter your desired username" in result.stderr
    assert "missing a username" not in result.stdout
    assert "bob/math-env" in hub.packages


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("math-env", ["math_env", "math-env"]),
        ("math_env", ["math_env", "math-env"]),
        ("Math_Env", ["Math_Env", "math_env", "math-env"]),
        ("gsm8k", ["gsm8k"]),
    ],
)
def test_hub_names_end_with_the_normalized_name(name, expected):
    assert package_push.hub_names(name) == expected


def test_resolve_owner_rejects_a_configured_team_you_are_not_in(hub):
    api = APIClient()

    with pytest.raises(package_push.PushError, match="isn't one of your teams"):
        package_push.resolve_owner(api, owner=None, team=None, configured_team="c" + "x" * 24)

    assert (
        package_push.resolve_owner(api, owner=None, team=None, configured_team=" my-team ")
        == "my-team"
    )


def test_a_failed_upload_is_reported_without_a_traceback(hub, package, monkeypatch):
    def put(url, content, headers, timeout):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(hub_module.httpx, "put", put)

    result = _push("--path", str(package))

    output = strip_ansi(result.output)
    assert result.exit_code == 1
    assert "Uploading math_env-0.2.0.tar.gz failed: ConnectError: connection refused" in output
    assert not any(c[1].endswith("/finalize") for c in hub.calls)
