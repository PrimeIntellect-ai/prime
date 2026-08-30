from pathlib import Path
from typing import Iterable, Optional

from prime_cli.utils.environment_runtime import (
    MAX_SOURCE_FILE_BYTES,
    VERIFIERS_V0,
    VERIFIERS_V1,
    classify_runtime_from_metadata,
    detect_environment_runtime,
    effective_requirement_strings,
)

PYPROJECT = "[project]\nname = 'demo'\nversion = '0.1.0'\n"


def _detect(
    env_path: Path,
    requires_dist: Optional[Iterable[str]] = None,
    dependencies: Optional[Iterable[str]] = None,
) -> Optional[str]:
    files = [path for path in env_path.rglob("*") if path.is_file()]
    return detect_environment_runtime(
        env_path, files, requires_dist=requires_dist, dependencies=dependencies
    )


def _write(env_path: Path, rel_path: str, content: str) -> None:
    file_path = env_path / rel_path
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content)


def test_v1_pin_floor_classifies_v1(tmp_path):
    _write(tmp_path, "pyproject.toml", PYPROJECT)
    _write(tmp_path, "demo.py", "print('no markers here')\n")

    assert _detect(tmp_path, requires_dist=["verifiers>=0.2.0"]) == VERIFIERS_V1


def test_old_pin_classifies_v0(tmp_path):
    _write(tmp_path, "pyproject.toml", PYPROJECT)
    _write(tmp_path, "demo.py", "print('no markers here')\n")

    assert _detect(tmp_path, requires_dist=["verifiers>=0.1.2"]) == VERIFIERS_V0
    assert _detect(tmp_path, requires_dist=["verifiers"]) == VERIFIERS_V0


def test_no_verifiers_requirement_omits_hint(tmp_path):
    _write(tmp_path, "pyproject.toml", PYPROJECT)
    _write(tmp_path, "demo.py", "print('no markers here')\n")

    assert _detect(tmp_path, requires_dist=["httpx>=0.25"]) is None


def test_url_pin_gives_no_opinion(tmp_path):
    _write(tmp_path, "pyproject.toml", PYPROJECT)
    _write(tmp_path, "demo.py", "print('no markers here')\n")

    requires = ["verifiers @ git+https://github.com/PrimeIntellect-ai/verifiers.git"]
    assert _detect(tmp_path, requires_dist=requires) is None


def test_load_environment_beats_v1_pin(tmp_path):
    _write(tmp_path, "pyproject.toml", PYPROJECT)
    _write(tmp_path, "demo.py", "def load_environment(**kwargs):\n    return None\n")

    assert _detect(tmp_path, requires_dist=["verifiers>=0.2.0"]) == VERIFIERS_V0


def test_v1_import_beats_old_pin(tmp_path):
    _write(tmp_path, "pyproject.toml", PYPROJECT)
    _write(tmp_path, "demo.py", "from verifiers.v1 import Taskset\n")

    assert _detect(tmp_path, requires_dist=["verifiers>=0.1.0"]) == VERIFIERS_V1


def test_v1_marker_beats_load_environment_shim(tmp_path):
    _write(tmp_path, "pyproject.toml", PYPROJECT)
    _write(
        tmp_path,
        "demo.py",
        "import verifiers.v1 as v1\n\ndef load_environment(**kwargs):\n    return None\n",
    )

    assert _detect(tmp_path) == VERIFIERS_V1


def test_vf_alias_marker_classifies_v1(tmp_path):
    _write(tmp_path, "pyproject.toml", PYPROJECT)
    _write(
        tmp_path,
        "demo/__init__.py",
        "import verifiers as vf\n\nclass Demo(vf.Taskset):\n    pass\n",
    )

    assert _detect(tmp_path) == VERIFIERS_V1


def test_project_tags_count_as_weak_v0_evidence(tmp_path):
    _write(tmp_path, "pyproject.toml", PYPROJECT + "tags = ['math']\n")
    _write(tmp_path, "demo.py", "print('no markers here')\n")

    assert _detect(tmp_path) == VERIFIERS_V0


def test_markers_under_tests_directory_are_ignored(tmp_path):
    _write(tmp_path, "pyproject.toml", PYPROJECT)
    _write(tmp_path, "demo.py", "print('no markers here')\n")
    _write(tmp_path, "tests/test_demo.py", "from verifiers.v1 import Taskset\n")

    assert _detect(tmp_path) is None


def test_oversized_files_are_skipped(tmp_path):
    _write(tmp_path, "pyproject.toml", PYPROJECT)
    padding = "#" * (MAX_SOURCE_FILE_BYTES + 1) + "\n"
    _write(tmp_path, "demo.py", padding + "from verifiers.v1 import Taskset\n")

    assert _detect(tmp_path) is None


def test_unconditional_requirement_beats_marker_guarded_one():
    requires = [
        'verifiers>=0.2.0; python_version >= "3.11"',
        "verifiers>=0.1.0",
    ]
    assert classify_runtime_from_metadata(requires) == VERIFIERS_V0


def test_repeated_requirements_take_the_highest_floor():
    assert classify_runtime_from_metadata(["verifiers>=0.1", "verifiers>=0.2"]) == VERIFIERS_V1


def test_effective_requirement_strings_prefers_requires_dist():
    assert effective_requirement_strings(["verifiers>=0.2.0"], ["verifiers>=0.1.0"]) == [
        "verifiers>=0.2.0"
    ]
    assert effective_requirement_strings([], ["verifiers>=0.1.0"]) == ["verifiers>=0.1.0"]
    assert effective_requirement_strings(None, None) == []
