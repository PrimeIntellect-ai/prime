"""Opt-in end-to-end tests for the Prime CLI eval path.

Run with a demo user key:
    PRIME_E2E_API_KEY=... uv run pytest packages/prime/tests/test_eval_cli_e2e.py -m integration

The test intentionally exercises the real CLI, Prime Inference preflight, hub
environment install, evaluation execution, and default result upload path.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ENVIRONMENT = "single_turn_math"
DEFAULT_MODEL = "deepseek/deepseek-chat"

pytestmark = pytest.mark.integration


def _demo_api_key() -> str:
    e2e_api_key = os.environ.get("PRIME_E2E_API_KEY")
    api_key = e2e_api_key if e2e_api_key else os.environ.get("PRIME_API_KEY")
    if api_key is not None and api_key != "":
        return api_key
    pytest.skip("Set PRIME_E2E_API_KEY to run the real Prime CLI eval E2E test.")
    raise AssertionError("pytest.skip should stop execution")


def _run_prime(
    args: list[str],
    *,
    tmp_path: Path,
    api_key: str,
    timeout: int,
) -> subprocess.CompletedProcess[str]:
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)

    result = subprocess.run(
        ["uv", "run", "--project", str(REPO_ROOT), "prime", *args],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=timeout,
        env={
            **os.environ,
            "HOME": str(home),
            "PRIME_API_KEY": api_key,
            "PYTHONUNBUFFERED": "1",
        },
    )

    print(f"command: prime {' '.join(args)}")
    print(f"stdout: {result.stdout}")
    print(f"stderr: {result.stderr}")
    return result


def test_prime_eval_run_e2e_with_demo_key(tmp_path: Path) -> None:
    """Install a hub environment and run a one-rollout eval through the real CLI."""
    api_key = _demo_api_key()
    environment = os.environ.get("PRIME_E2E_ENVIRONMENT", DEFAULT_ENVIRONMENT)
    model = os.environ.get("PRIME_E2E_MODEL", DEFAULT_MODEL)

    install = _run_prime(
        ["env", "install", environment, "--with", "pip"],
        tmp_path=tmp_path,
        api_key=api_key,
        timeout=300,
    )
    assert install.returncode == 0, (
        f"Environment install failed:\n{install.stderr}\n{install.stdout}"
    )

    eval_run = _run_prime(
        [
            "eval",
            "run",
            environment,
            "-m",
            model,
            "-n",
            "1",
            "-r",
            "1",
        ],
        tmp_path=tmp_path,
        api_key=api_key,
        timeout=900,
    )
    output = f"{eval_run.stdout}\n{eval_run.stderr}"

    assert eval_run.returncode == 0, f"Eval run failed:\n{eval_run.stderr}\n{eval_run.stdout}"
    assert "Eval job_id:" in output
    assert "Failed to push results to hub" not in output
    assert "Skipped uploading evaluation results" not in output
