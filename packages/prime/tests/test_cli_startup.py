import json
import os
import subprocess
import sys
from pathlib import Path

MODULE_PROBE = (
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
        repo / "packages" / "prime-traces" / "src",
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


def test_prime_main_import_does_not_load_heavy_runtime() -> None:
    loaded = _run_probe(f"import json, sys\nimport prime_cli.main\n{_probe_expression()}\n")

    assert loaded == {
        "textual": False,
        "verifiers": False,
    }


def test_update_notice_renders_remote_version_literally(monkeypatch) -> None:
    from prime_cli import main
    from typer.testing import CliRunner

    class EmptyModelsClient:
        def __init__(self, **_kwargs):
            pass

        def list_models(self):
            return {"data": []}

    monkeypatch.setattr(main, "check_for_update", lambda: (True, "[/INFO]"))
    monkeypatch.setattr("prime_cli.commands.inference.InferenceClient", EmptyModelsClient)

    result = CliRunner().invoke(
        main.app,
        ["inference", "models"],
        env={"PRIME_DISABLE_VERSION_CHECK": "0"},
    )

    assert result.exit_code == 0, result.output
    assert "[/INFO]" in result.output


def test_upgrade_diagnostic_renders_subprocess_output_literally(monkeypatch, capsys) -> None:
    from prime_cli.commands import upgrade as upgrade_command

    monkeypatch.setattr(upgrade_command.shutil, "which", lambda _name: "/usr/bin/tool")
    monkeypatch.setattr(
        upgrade_command.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            args=["uv"], returncode=1, stderr="backend failed [/INFO]"
        ),
    )

    assert upgrade_command._run_upgrade("uv_tool") is False
    assert "backend failed [/INFO]" in capsys.readouterr().out


def test_upgrade_diagnostic_renders_runner_exception_literally(monkeypatch, capsys) -> None:
    from prime_cli.commands import upgrade as upgrade_command

    monkeypatch.setattr(upgrade_command.shutil, "which", lambda _name: "/usr/bin/tool")

    def raise_runner(*_args, **_kwargs):
        raise RuntimeError("runner failed [/INFO]")

    monkeypatch.setattr(upgrade_command.subprocess, "run", raise_runner)

    assert upgrade_command._run_upgrade("uv_tool") is False
    assert "runner failed [/INFO]" in capsys.readouterr().out
