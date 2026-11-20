"""`prime env push` against a fake Hub API: what it sends, and how it handles errors."""

import hashlib
import json
import tarfile
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import pytest
from prime_cli.commands import env as env_command
from prime_cli.core.client import APIClient
from prime_cli.main import app
from prime_cli.utils import hub as hub_module
from prime_cli.utils import package_push, strip_ansi
from typer.testing import CliRunner

runner = CliRunner()
API = "https://api.test"


class FakeHub:
    """The Hub write path, keeping what each call received."""

    def __init__(self) -> None:
        self.packages: Dict[str, Dict[str, Any]] = {}
        self.calls: List[tuple] = []
        self.releases: List[Dict[str, Any]] = []
        self.uploads: Dict[str, bytes] = {}
        self.finalize_errors: List[tuple] = []
        self.release_error: Optional[tuple] = None
        self.slug: Optional[str] = "alice"
        self.teams = [{"teamId": "c" + "t" * 24, "slug": "research", "name": "Research"}]

    @staticmethod
    def _error(status: int, code: str, message: str) -> httpx.Response:
        return httpx.Response(
            status, json={"error": {"code": code, "message": message, "details": {}}}
        )

    def _package(self, owner: str, name: str) -> Dict[str, Any]:
        return {
            "id": f"pkg-{owner}-{name}",
            "owner": {"kind": "user", "slug": owner},
            "name": name,
            "visibility": "private",
        }

    def handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        body = json.loads(request.content) if request.content else None
        self.calls.append((request.method, path, body, dict(request.url.params)))
        if path == "/api/v1/user/whoami":
            return httpx.Response(200, json={"data": {"id": "u1", "slug": self.slug}})
        if path == "/api/v1/user/teams":
            return httpx.Response(200, json={"data": self.teams, "total_count": len(self.teams)})
        if path == "/api/v1/user/slug":
            self.slug = body["slug"]
            return httpx.Response(200, json={"data": {"slug": self.slug}})
        prefix = "/api/v1/hub"
        assert path.startswith(prefix), path
        path = path[len(prefix) :]
        parts = path.strip("/").split("/")
        if request.method == "POST" and path == "/packages":
            key = f"{body['owner']}/{body['name']}"
            if key in self.packages:
                return self._error(409, "package_exists", "exists")
            self.packages[key] = self._package(body["owner"], body["name"])
            if body.get("visibility"):
                self.packages[key]["visibility"] = body["visibility"]
            return httpx.Response(201, json=self.packages[key])
        owner, name = parts[1], parts[2]
        key = f"{owner}/{name}"
        if request.method == "GET" and len(parts) == 3:
            matches = [p for k, p in self.packages.items() if k.lower() == key.lower()]
            if not matches:
                return self._error(404, "package_not_found", "nope")
            return httpx.Response(200, json=matches[0])
        if request.method == "POST" and parts[3:] == ["releases"]:
            if self.release_error:
                return self._error(*self.release_error)
            self.releases.append(body)
            source = next(a for a in body["artifacts"] if a["type"] == "source")
            return httpx.Response(
                201,
                json={
                    "release_id": "rel-1",
                    "version": body["version"],
                    "content_hash": source["sha256"],
                    "uploads": [
                        {
                            "type": a["type"],
                            "filename": a["filename"],
                            "url": f"https://storage.test/{a['filename']}",
                            "headers": {"Content-Type": "application/octet-stream"},
                            "expires_at": "2026-11-20T00:00:00Z",
                        }
                        for a in body["artifacts"]
                    ],
                },
            )
        if request.method == "POST" and parts[-1] == "finalize":
            if self.finalize_errors:
                return self._error(*self.finalize_errors.pop(0))
            release = self.releases[-1]
            return httpx.Response(
                200,
                json={
                    "id": "rel-1",
                    "version": parts[4],
                    "content_hash": request.url.params["content_hash"],
                    "components": [
                        {**c, "source": "declared"} for c in release.get("components", [])
                    ],
                },
            )
        raise AssertionError(f"unexpected {request.method} {path}")


def _wheel(dist: Path, distribution: str, version: str, requires: List[str]) -> Path:
    dist.mkdir(parents=True, exist_ok=True)
    wheel = dist / f"{distribution.replace('-', '_')}-{version}-py3-none-any.whl"
    metadata = "Metadata-Version: 2.1\nName: {}\nVersion: {}\n".format(distribution, version)
    metadata += "".join(f"Requires-Dist: {r}\n" for r in requires)
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(f"{distribution.replace('-', '_')}-{version}.dist-info/METADATA", metadata)
    return wheel


@pytest.fixture
def hub(monkeypatch, tmp_path) -> FakeHub:
    fake = FakeHub()
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("PRIME_API_KEY", "test-key")
    monkeypatch.setenv("PRIME_API_BASE_URL", API)
    monkeypatch.delenv("PRIME_TEAM_ID", raising=False)

    original_init = APIClient.__init__

    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.client = httpx.Client(
            transport=httpx.MockTransport(fake.handle), headers=self.client.headers
        )

    monkeypatch.setattr(APIClient, "__init__", init)

    def put(url, content, headers, timeout):
        fake.uploads[url.rsplit("/", 1)[-1]] = content.read()
        return httpx.Response(200, request=httpx.Request("PUT", url))

    monkeypatch.setattr(hub_module.httpx, "put", put)
    return fake


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
        project = env_command.toml.load(package_dir / "pyproject.toml")["project"]
        return _wheel(
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
