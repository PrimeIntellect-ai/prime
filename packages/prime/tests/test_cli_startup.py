import json
import os
import subprocess
import sys
from pathlib import Path

MODULE_PROBE = (
    "prime_cli.lab_setup",
    "prime_lab_app",
    "prime_lab_app.app",
    "textual",
    "verifiers",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _pythonpath(repo: Path) -> str:
    paths = [
        repo / "packages" / "prime" / "src",
        repo / "packages" / "prime-evals" / "src",
        repo / "packages" / "prime-sandboxes" / "src",
        repo / "packages" / "prime-tunnel" / "src",
    ]
    existing = os.environ.get("PYTHONPATH")
    parts = [str(path) for path in paths]
    if existing:
        parts.append(existing)
    return os.pathsep.join(parts)


def _run_probe(script: str) -> dict[str, bool]:
    repo = _repo_root()
    env = {
        **os.environ,
        "PYTHONPATH": _pythonpath(repo),
        "PRIME_DISABLE_VERSION_CHECK": "1",
    }
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


def _probe_expression() -> str:
    names = ", ".join(repr(name) for name in MODULE_PROBE)
    return f"print(json.dumps({{name: name in sys.modules for name in ({names})}}))"


def test_prime_main_import_does_not_load_lab_setup_or_heavy_lab_runtime() -> None:
    loaded = _run_probe(f"import json, sys\nimport prime_cli.main\n{_probe_expression()}\n")

    assert loaded == {
        "prime_cli.lab_setup": False,
        "prime_lab_app": False,
        "prime_lab_app.app": False,
        "textual": False,
        "verifiers": False,
    }


def test_prime_lab_setup_help_does_not_load_textual_or_verifiers() -> None:
    loaded = _run_probe(
        "import json, sys\n"
        "sys.argv = ['prime', 'lab', 'setup', '--help']\n"
        "from prime_cli.main import run\n"
        "try:\n"
        "    run()\n"
        "except SystemExit:\n"
        "    pass\n"
        f"{_probe_expression()}\n"
    )

    assert loaded == {
        "prime_cli.lab_setup": False,
        "prime_lab_app": False,
        "prime_lab_app.app": False,
        "textual": False,
        "verifiers": False,
    }
