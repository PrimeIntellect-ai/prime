import os
import subprocess
import sys
from pathlib import Path


def test_help_works_without_fcntl() -> None:
    project_root = Path(__file__).resolve().parents[3]
    pythonpath = os.pathsep.join(
        [
            str(project_root / "packages" / "prime" / "src"),
            str(project_root / "packages" / "prime-tunnel" / "src"),
            str(project_root / "packages" / "prime-evals" / "src"),
            str(project_root / "packages" / "prime-sandboxes" / "src"),
        ]
    )
    env = {
        **os.environ,
        "PRIME_DISABLE_VERSION_CHECK": "1",
        "PYTHONPATH": pythonpath,
        "PYTHONIOENCODING": "utf-8",
    }
    script = """
import builtins
import sys

real_import = builtins.__import__

def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "fcntl":
        raise ModuleNotFoundError("No module named 'fcntl'")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = fake_import

from prime_cli.main import app
from typer.testing import CliRunner

result = CliRunner().invoke(app, ["--help"], env={"PRIME_DISABLE_VERSION_CHECK": "1"})
if result.exit_code != 0:
    sys.stderr.write(result.output)
    raise SystemExit(result.exit_code)
if "Prime Intellect CLI" not in result.output:
    sys.stderr.write("missing help banner\\n")
    raise SystemExit(1)
sys.stdout.write("ok\\n")
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=False,
        env=env,
        check=False,
    )

    stdout = result.stdout.decode("utf-8", errors="replace")
    stderr = result.stderr.decode("utf-8", errors="replace")

    assert result.returncode == 0, stderr or stdout
    assert stdout.strip() == "ok"
