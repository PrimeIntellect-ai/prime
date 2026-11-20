from pathlib import Path
from textwrap import dedent

import pytest
from prime_cli.utils.hub_components import detect_components, kind_of_verifiers_name


def _write(root: Path, files: dict) -> Path:
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(dedent(content))
    return root


def _declared(root: Path, name: str):
    detected = detect_components(root, name)
    return [(item["kind"], item["name"]) for item in detected.components], detected


@pytest.mark.parametrize(
    ("name", "kind"),
    [
        ("Taskset", "taskset"),
        ("GSM8KTaskset", "taskset"),
        ("ACPHarness", "harness"),
        ("SingleAgentEnv", "env"),
        ("ReferenceJudge", "judge"),
        ("HarnessConfig", None),
        ("TasksetConfig", None),
        ("Task", None),
        ("TaskData", None),
    ],
)
def test_kind_of_verifiers_name(name, kind):
    assert kind_of_verifiers_name(name) == kind


def test_package_reexporting_classes_from_its_modules(tmp_path):
    _write(
        tmp_path,
        {
            "kuhn_poker/__init__.py": """
                from kuhn_poker.taskset import KuhnPokerEnv, KuhnPokerTaskset

                __all__ = ["KuhnPokerEnv", "KuhnPokerTaskset"]
            """,
            "kuhn_poker/taskset.py": """
                import verifiers.v1 as vf

                class KuhnPokerEnvConfig(vf.BaseConfig):
                    pass

                class KuhnPokerTask(vf.Task):
                    pass

                class KuhnPokerTaskset(vf.Taskset[KuhnPokerTask, vf.TasksetConfig]):
                    pass

                class KuhnPokerEnv(vf.Env[KuhnPokerEnvConfig]):
                    pass
            """,
        },
    )

    declared, detected = _declared(tmp_path, "kuhn-poker")

    # The Hub's order: taskset, harness, env, judge.
    assert declared == [("taskset", "KuhnPokerTaskset"), ("env", "KuhnPokerEnv")]
    assert detected.found_exports
    assert detected.ambiguous == {}


def test_relative_imports_aliases_and_local_base_classes(tmp_path):
    _write(
        tmp_path,
        {
            "src/my_env/__init__.py": """
                from .judges import Strict as StrictJudge, Lenient
                from .taskset import MyTaskset

                __all__ = ["MyTaskset", "StrictJudge", "Lenient", "helper"]

                def helper():
                    pass
            """,
            "src/my_env/base.py": """
                from verifiers import v1 as vf

                class BaseJudge(vf.ReferenceJudge):
                    pass
            """,
            "src/my_env/judges.py": """
                from .base import BaseJudge

                class Strict(BaseJudge):
                    pass

                class Lenient(BaseJudge):
                    pass
            """,
            "src/my_env/taskset/__init__.py": """
                import verifiers.v1 as vf

                class MyTaskset(vf.Taskset):
                    pass
            """,
        },
    )

    declared, _detected = _declared(tmp_path, "my-env")

    assert declared == [
        ("taskset", "MyTaskset"),
        ("judge", "StrictJudge"),
        ("judge", "Lenient"),
    ]


def test_names_reexported_from_verifiers_count_by_their_own_name(tmp_path):
    _write(
        tmp_path,
        {
            "web_search/__init__.py": """
                from verifiers.v1.harnesses.codex import CodexHarness, CodexHarnessConfig
                from verifiers.v1.tasksets.textarena import TextArenaEnv
                from web_search.taskset import WebSearchTaskset

                __all__ = [
                    "CodexHarness",
                    "CodexHarnessConfig",
                    "TextArenaEnv",
                    "WebSearchTaskset",
                ]
            """,
            "web_search/taskset.py": """
                import verifiers.v1

                class WebSearchTaskset(verifiers.v1.Taskset):
                    pass
            """,
        },
    )

    declared, _detected = _declared(tmp_path, "web_search")

    assert declared == [
        ("taskset", "WebSearchTaskset"),
        ("harness", "CodexHarness"),
        ("env", "TextArenaEnv"),
    ]


def test_single_module_package(tmp_path):
    _write(
        tmp_path,
        {
            "reverse_text.py": """
                import verifiers.v1 as vf

                class ReverseTextTaskset(vf.Taskset):
                    pass

                __all__ = ["ReverseTextTaskset"]
            """,
        },
    )

    declared, _detected = _declared(tmp_path, "reverse-text")

    assert declared == [("taskset", "ReverseTextTaskset")]


def test_two_of_a_kind_is_left_out_and_reported(tmp_path):
    _write(
        tmp_path,
        {
            "two/__init__.py": """
                import verifiers.v1 as vf

                class EasyTaskset(vf.Taskset):
                    pass

                class HardTaskset(vf.Taskset):
                    pass

                class TwoEnv(vf.SingleAgentEnv):
                    pass

                __all__ = ["EasyTaskset", "HardTaskset"]
                __all__ += ["TwoEnv"]
            """,
        },
    )

    declared, detected = _declared(tmp_path, "two")

    assert declared == [("env", "TwoEnv")]
    assert detected.ambiguous == {"taskset": ["EasyTaskset", "HardTaskset"]}


@pytest.mark.parametrize(
    "files",
    [
        {},
        {"pkg/__init__.py": "import verifiers.v1 as vf\nclass T(vf.Taskset): pass\n"},
        {"pkg/__init__.py": "__all__ = names()\n"},
        {"pkg/__init__.py": "def broken(:\n"},
    ],
    ids=["no module", "no __all__", "computed __all__", "syntax error"],
)
def test_nothing_declared_when_exports_cant_be_read(tmp_path, files):
    _write(tmp_path, files)

    declared, detected = _declared(tmp_path, "pkg")

    assert declared == []
    assert not detected.found_exports


def test_classes_from_other_libraries_and_cycles_are_ignored(tmp_path):
    _write(
        tmp_path,
        {
            "pkg/__init__.py": """
                from other_lib import ThirdPartyTaskset
                from pkg.a import Loop

                __all__ = ["ThirdPartyTaskset", "Loop", "Missing"]
            """,
            "pkg/a.py": """
                from pkg.b import Back

                class Loop(Back):
                    pass
            """,
            "pkg/b.py": """
                from pkg.a import Loop as Back
            """,
        },
    )

    declared, detected = _declared(tmp_path, "pkg")

    assert declared == []
    assert detected.found_exports
