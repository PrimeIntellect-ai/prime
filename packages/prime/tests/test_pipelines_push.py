"""`prime pipelines push` against the fake Hub (the `hub` fixture)."""

import json
from pathlib import Path

import pytest
from fake_hub import FakeHub, write_wheel
from prime_cli.commands import pipelines as pipelines_command
from prime_cli.main import app
from prime_cli.utils import strip_ansi
from typer.testing import CliRunner

runner = CliRunner()


@pytest.fixture
def pipeline(tmp_path, monkeypatch) -> Path:
    root = tmp_path / "pipeline"
    (root / "math_loop").mkdir(parents=True)
    (root / "pyproject.toml").write_text(
        '[project]\nname = "math-loop"\nversion = "1.0.0"\ndescription = "Loop"\n'
        'dependencies = ["verifiers>=0.2.0"]\n'
    )
    (root / "math_loop" / "__init__.py").write_text(
        "import verifiers.v1 as vf\n\n"
        "class LoopTaskset(vf.Taskset):\n    pass\n\n"
        '__all__ = ["LoopTaskset", "Loop"]\n'
    )

    def build(package_dir: Path) -> Path:
        return write_wheel(package_dir / "dist", "math-loop", "1.0.0", ["verifiers>=0.2.0"])

    monkeypatch.setattr(pipelines_command, "build_wheel", build)
    return root


def _push(*args: str):
    return runner.invoke(app, ["pipelines", "push", *args], env={"COLUMNS": "200"})


def test_push_declares_a_pipeline_and_prints_the_reference(hub: FakeHub, pipeline):
    result = _push("--path", str(pipeline))

    output = strip_ansi(result.output)
    assert result.exit_code == 0, output
    assert "Pushed alice/math-loop@1.0.0" in output
    assert "prime pipelines run alice/math-loop@1.0.0" in output
    (release,) = hub.releases
    assert release["components"] == [
        {"kind": "taskset", "name": "LoopTaskset"},
        {"kind": "pipeline"},
    ]
    assert release["metadata"] == {"description": "Loop", "requires_dist": ["verifiers>=0.2.0"]}
    assert {a["type"] for a in release["artifacts"]} == {"source", "wheel"}
    create = next(c for c in hub.calls if c[:2] == ("POST", "/api/v1/hub/packages"))
    assert create[2] == {"owner": "alice", "name": "math-loop", "description": "Loop"}


def test_push_json(hub: FakeHub, pipeline):
    result = _push("--path", str(pipeline), "--team", "research", "--output", "json")

    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["reference"] == "research/math-loop@1.0.0"
    assert data["created_package"] is True
    assert [c["kind"] for c in data["components"]] == ["taskset", "pipeline"]


def test_a_published_version_is_refused(hub: FakeHub, pipeline):
    hub.release_error = (409, "version_exists", "1.0.0 is already published")

    result = _push("--path", str(pipeline))

    output = strip_ansi(result.output)
    assert result.exit_code == 1
    assert "1.0.0 is already published (version_exists)" in output
    assert "Bump the version in pyproject.toml." in output
    assert "--auto-bump" not in output


def test_a_folder_without_a_project_is_refused(hub: FakeHub, tmp_path):
    result = _push("--path", str(tmp_path))

    assert result.exit_code == 1
    assert "pyproject.toml" in strip_ansi(result.output)
    assert hub.calls == []


def test_json_stays_clean_when_a_username_is_chosen_first(hub: FakeHub, pipeline):
    hub.slug = None

    result = runner.invoke(
        app,
        ["pipelines", "push", "--path", str(pipeline), "--output", "json"],
        input="bob\n",
        env={"COLUMNS": "200"},
    )

    assert result.exit_code == 0, result.output
    # CliRunner echoes the typed answer onto stdout, as a terminal shows typing; the
    # prompt and the messages around it go to stderr.
    echoed, document = result.stdout.split("\n", 1)
    assert echoed.strip() == "bob"
    assert json.loads(document)["reference"] == "bob/math-loop@1.0.0"
    assert "Enter your desired username" in result.stderr
    assert "Username set to bob" in result.stderr
