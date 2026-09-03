from pathlib import Path
from typing import Dict, Iterable, Optional

import pytest
from prime_cli.utils.environment_runtime import (
    MAX_SOURCE_FILE_BYTES,
    VERIFIERS_V0,
    VERIFIERS_V1,
    classify_runtime_from_metadata,
    detect_environment_runtime,
)

NO_MARKERS = "print('no markers here')\n"
V1_IMPORT = "from verifiers.v1 import Taskset\n"


def _detect(
    env_path: Path,
    files: Dict[str, str],
    requires_dist: Optional[Iterable[str]] = None,
    legacy_tags: bool = False,
) -> Optional[str]:
    for rel_path, content in files.items():
        file_path = env_path / rel_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
    return detect_environment_runtime(
        env_path,
        [path for path in env_path.rglob("*") if path.is_file()],
        requires_dist=requires_dist,
        legacy_tags=legacy_tags,
    )


@pytest.mark.parametrize(
    ("requirements", "expected"),
    [
        (["verifiers>=0.2.0"], VERIFIERS_V1),
        (["verifiers>=0.1.2"], VERIFIERS_V0),
        (["verifiers"], VERIFIERS_V0),
        (["httpx>=0.25"], None),
        (["verifiers @ git+https://github.com/PrimeIntellect-ai/verifiers.git"], None),
        # Unconditional entries beat marker-guarded ones.
        (['verifiers>=0.2.0; python_version >= "3.11"', "verifiers>=0.1.0"], VERIFIERS_V0),
        # Repeated entries intersect, so the highest floor wins.
        (["verifiers>=0.1", "verifiers>=0.2"], VERIFIERS_V1),
    ],
)
def test_classify_from_metadata(requirements, expected):
    assert classify_runtime_from_metadata(requirements) == expected


def test_no_markers_fall_back_to_the_pin(tmp_path):
    files = {"demo.py": NO_MARKERS}
    assert _detect(tmp_path, files, requires_dist=["verifiers>=0.2.0"]) == VERIFIERS_V1
    assert _detect(tmp_path, files, requires_dist=["httpx>=0.25"]) is None


def test_load_environment_beats_v1_pin(tmp_path):
    files = {"demo.py": "def load_environment(**kwargs):\n    return None\n"}
    assert _detect(tmp_path, files, requires_dist=["verifiers>=0.2.0"]) == VERIFIERS_V0


def test_v1_import_beats_old_pin(tmp_path):
    assert _detect(tmp_path, {"demo.py": V1_IMPORT}, requires_dist=["verifiers>=0.1.0"]) == (
        VERIFIERS_V1
    )


def test_v1_marker_beats_load_environment_shim(tmp_path):
    shim = "import verifiers.v1 as v1\n\ndef load_environment(**kwargs):\n    return None\n"
    files = {"demo.py": shim}
    assert _detect(tmp_path, files) == VERIFIERS_V1


def test_vf_alias_marker_classifies_v1(tmp_path):
    files = {"demo/__init__.py": "import verifiers as vf\n\nclass Demo(vf.Taskset):\n    pass\n"}
    assert _detect(tmp_path, files) == VERIFIERS_V1


def test_project_tags_count_as_weak_v0_evidence(tmp_path):
    assert _detect(tmp_path, {"demo.py": NO_MARKERS}, legacy_tags=True) == VERIFIERS_V0
    assert _detect(tmp_path, {"demo.py": V1_IMPORT}, legacy_tags=True) == VERIFIERS_V1


def test_markers_under_tests_directory_are_ignored(tmp_path):
    files = {"demo.py": NO_MARKERS, "tests/test_demo.py": V1_IMPORT}
    assert _detect(tmp_path, files) is None


def test_oversized_files_are_skipped(tmp_path):
    padding = "#" * (MAX_SOURCE_FILE_BYTES + 1) + "\n"
    assert _detect(tmp_path, {"demo.py": padding + V1_IMPORT}) is None
