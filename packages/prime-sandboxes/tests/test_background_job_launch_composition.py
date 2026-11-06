"""Background-job launch scripts execute caller commands verbatim under real sh."""

import os
import subprocess
import time
from typing import Any, cast

import pytest

from prime_sandboxes.core.client import APIClient
from prime_sandboxes.models import CommandResponse
from prime_sandboxes.sandbox import AsyncSandboxClient, SandboxClient

_OK = CommandResponse(stdout="", stderr="", exit_code=0)

pytestmark = pytest.mark.skipif(os.name != "posix", reason="launches a real POSIX sh")


def _run_launch_script(script: str, job: Any) -> None:
    """Execute the captured launch script under the native POSIX shell and assert the
    background job it starts wrote its stdout log, stderr log, and exit file."""
    result = subprocess.run(["/bin/sh", "-c", script], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    paths = [
        job.stdout_log_file,
        job.stderr_log_file,
        job.exit_file,
    ]
    try:
        deadline = time.monotonic() + 5
        while not os.path.exists(job.exit_file):
            assert time.monotonic() < deadline, "background job never wrote its exit file"
            time.sleep(0.05)
        with open(job.exit_file) as f:
            assert f.read().strip() == "0"
        with open(job.stdout_log_file) as f:
            assert f.read() == "heredoc body\n"
        with open(job.stderr_log_file) as f:
            assert f.read() == ""
    finally:
        for path in paths:
            if os.path.exists(path):
                os.unlink(path)
        launch_dir = f"/tmp/job_{job.job_id}.launch"
        if os.path.isdir(launch_dir):
            for name in os.listdir(launch_dir):
                os.unlink(os.path.join(launch_dir, name))
            os.rmdir(launch_dir)


def _heredoc_command() -> str:
    # A valid shell command whose final heredoc delimiter has no trailing newline.
    return "cat <<'EOF'\nheredoc body\nEOF"


class TestSyncLaunchComposition:
    def test_launches_heredoc_command_without_trailing_newline(self):
        client = SandboxClient(APIClient(api_key="test-key"))
        scripts = []

        def execute(_sandbox_id, command, **_kwargs):
            scripts.append(command)
            return _OK

        cast(Any, client).execute_command = execute
        job = client.start_background_job("sb", _heredoc_command())
        _run_launch_script(scripts[0], job)

    def test_launches_command_with_trailing_newline(self):
        client = SandboxClient(APIClient(api_key="test-key"))
        scripts = []

        def execute(_sandbox_id, command, **_kwargs):
            scripts.append(command)
            return _OK

        cast(Any, client).execute_command = execute
        job = client.start_background_job("sb", _heredoc_command() + "\n")
        _run_launch_script(scripts[0], job)


class TestAsyncLaunchComposition:
    @pytest.mark.asyncio
    async def test_launches_heredoc_command_without_trailing_newline(self):
        client = AsyncSandboxClient(APIClient(api_key="test-key"))
        scripts = []

        async def execute(_sandbox_id, command, **_kwargs):
            scripts.append(command)
            return _OK

        cast(Any, client).execute_command = execute
        job = await client.start_background_job("sb", _heredoc_command())
        _run_launch_script(scripts[0], job)

    @pytest.mark.asyncio
    async def test_launches_command_with_trailing_newline(self):
        client = AsyncSandboxClient(APIClient(api_key="test-key"))
        scripts = []

        async def execute(_sandbox_id, command, **_kwargs):
            scripts.append(command)
            return _OK

        cast(Any, client).execute_command = execute
        job = await client.start_background_job("sb", _heredoc_command() + "\n")
        _run_launch_script(scripts[0], job)
