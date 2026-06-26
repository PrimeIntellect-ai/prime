import os
import subprocess
import sys
from pathlib import Path


def test_prime_help_does_not_import_tunnel_package() -> None:
    project_root = Path(__file__).resolve().parents[3]
    pythonpath = os.pathsep.join(
        [
            str(project_root / "packages" / "prime" / "src"),
            str(project_root / "packages" / "prime-tunnel" / "src"),
            str(project_root / "packages" / "prime-evals" / "src"),
            str(project_root / "packages" / "prime-mcp-server" / "src"),
            str(project_root / "packages" / "prime-sandboxes" / "src"),
        ]
    )
    env = {
        **os.environ,
        "PRIME_DISABLE_VERSION_CHECK": "1",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONPATH": pythonpath,
    }
    script = """
import builtins
import sys

real_import = builtins.__import__


def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "prime_tunnel" or name.startswith("prime_tunnel."):
        raise RuntimeError("prime_tunnel imported during CLI startup")
    return real_import(name, globals, locals, fromlist, level)


builtins.__import__ = fake_import

from prime_cli.main import app
from click.testing import CliRunner

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
        text=True,
        env=env,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    assert result.stdout.strip() == "ok"
