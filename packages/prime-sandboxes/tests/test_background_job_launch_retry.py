"""Background-job launch retries are guarded against duplicate execution."""

from typing import Any, cast

import pytest

from prime_sandboxes.core.client import APIClient, APIError
from prime_sandboxes.exceptions import CommandTimeoutError
from prime_sandboxes.models import CommandResponse
from prime_sandboxes.sandbox import AsyncSandboxClient, SandboxClient

_OK = CommandResponse(stdout="", stderr="", exit_code=0)
_DISK_FULL = CommandResponse(
    stdout="", stderr="sh: can't create /tmp/job_x.launch/pid: No space left on device", exit_code=1
)


def _timeout():
    return CommandTimeoutError("sb", "nohup ...", 30)


async def _no_sleep(_):
    return None


class TestSyncLaunchRetry:
    def test_retries_and_succeeds(self, monkeypatch):
        monkeypatch.setattr("prime_sandboxes.sandbox.time.sleep", lambda _: None)
        client = SandboxClient(APIClient(api_key="test-key"))
        commands = []
        users = []
        cwds = []

        def execute(_sandbox_id, command, **_kwargs):
            commands.append(command)
            users.append(_kwargs.get("user"))
            cwds.append(_kwargs.get("working_dir"))
            if len(commands) == 1:
                raise _timeout()
            return _OK

        cast(Any, client).execute_command = execute
        job = client.start_background_job("sb", "rm -rf x", working_dir="/srv/app", user="ubuntu")
        assert users == ["ubuntu", "ubuntu"]
        assert cwds == ["/", "/"]
        assert "cd /srv/app || exit 1; rm -rf x" in commands[0]
        assert len(commands) == 2
        assert commands[0] == commands[1]
        assert commands[0].startswith(f"mkdir /tmp/job_{job.job_id}.launch || ")
        assert job.job_id
        client.start_background_job("sb", "ls", working_dir="app", user="ubuntu")
        assert cwds[-1] is None
        assert "cd app || exit 1; ls" in commands[-1]

    def test_gives_up_after_max_attempts(self, monkeypatch):
        monkeypatch.setattr("prime_sandboxes.sandbox.time.sleep", lambda _: None)
        client = SandboxClient(APIClient(api_key="test-key"))
        calls = {"n": 0}

        def execute(*_a, **_k):
            calls["n"] += 1
            raise _timeout()

        cast(Any, client).execute_command = execute
        with pytest.raises(CommandTimeoutError):
            client.start_background_job("sb", "rm -rf x", user="ubuntu")
        assert calls["n"] == 3

    def test_raises_when_job_does_not_start(self):
        client = SandboxClient(APIClient(api_key="test-key"))
        cast(Any, client).execute_command = lambda *_a, **_k: _DISK_FULL
        with pytest.raises(APIError, match="No space left on device"):
            client.start_background_job("sb", "ls")


class TestAsyncLaunchRetry:
    @pytest.mark.asyncio
    async def test_retries_and_succeeds(self, monkeypatch):
        monkeypatch.setattr("prime_sandboxes.sandbox.asyncio.sleep", _no_sleep)
        client = AsyncSandboxClient(APIClient(api_key="test-key"))
        commands = []
        users = []
        cwds = []

        async def execute(_sandbox_id, command, **_kwargs):
            commands.append(command)
            users.append(_kwargs.get("user"))
            cwds.append(_kwargs.get("working_dir"))
            if len(commands) == 1:
                raise _timeout()
            return _OK

        cast(Any, client).execute_command = execute
        job = await client.start_background_job(
            "sb", "rm -rf x", working_dir="/srv/app", user="ubuntu"
        )
        assert users == ["ubuntu", "ubuntu"]
        assert cwds == ["/", "/"]
        assert "cd /srv/app || exit 1; rm -rf x" in commands[0]
        assert len(commands) == 2
        assert commands[0] == commands[1]
        assert commands[0].startswith(f"mkdir /tmp/job_{job.job_id}.launch || ")
        assert job.job_id

    @pytest.mark.asyncio
    async def test_gives_up_after_max_attempts(self, monkeypatch):
        monkeypatch.setattr("prime_sandboxes.sandbox.asyncio.sleep", _no_sleep)
        client = AsyncSandboxClient(APIClient(api_key="test-key"))
        calls = {"n": 0}

        async def execute(*_a, **_k):
            calls["n"] += 1
            raise _timeout()

        cast(Any, client).execute_command = execute
        with pytest.raises(CommandTimeoutError):
            await client.start_background_job("sb", "rm -rf x", user="ubuntu")
        assert calls["n"] == 3

    @pytest.mark.asyncio
    async def test_raises_when_job_does_not_start(self):
        client = AsyncSandboxClient(APIClient(api_key="test-key"))

        async def execute(*_a, **_k):
            return _DISK_FULL

        cast(Any, client).execute_command = execute
        with pytest.raises(APIError, match="No space left on device"):
            await client.start_background_job("sb", "ls")


def _unavailable():
    from connectrpc.code import Code
    from connectrpc.errors import ConnectError

    error = APIError("Connect RPC failed (unavailable): Failed to route request to sandbox")
    error.__cause__ = ConnectError(Code.UNAVAILABLE, "Failed to route request to sandbox")
    return error


_RUNNING = CommandResponse(stdout="", stderr="", exit_code=0)


class TestTransientExecRetry:
    def test_sync_launch_and_status_retry_transient_faults(self, monkeypatch):
        monkeypatch.setattr("prime_sandboxes.sandbox.time.sleep", lambda _: None)
        client = SandboxClient(APIClient(api_key="test-key"))
        calls = {"n": 0}

        def execute(*_a, **_k):
            calls["n"] += 1
            if calls["n"] % 2:
                raise _unavailable()
            return _OK

        cast(Any, client).execute_command = execute
        job = client.start_background_job("sb", "ls")
        assert calls["n"] == 2
        assert not client.get_background_job_status("sb", job).completed
        assert calls["n"] == 4

    def test_sync_non_transient_api_error_is_not_retried(self):
        client = SandboxClient(APIClient(api_key="test-key"))
        calls = {"n": 0}

        def execute(*_a, **_k):
            calls["n"] += 1
            raise APIError("Command stream ended without exit code")

        cast(Any, client).execute_command = execute
        with pytest.raises(APIError, match="without exit code"):
            client.start_background_job("sb", "ls")
        assert calls["n"] == 1

    @pytest.mark.asyncio
    async def test_async_launch_and_status_retry_transient_faults(self, monkeypatch):
        monkeypatch.setattr("prime_sandboxes.sandbox.asyncio.sleep", _no_sleep)
        client = AsyncSandboxClient(APIClient(api_key="test-key"))
        calls = {"n": 0}

        async def execute(*_a, **_k):
            calls["n"] += 1
            if calls["n"] % 2:
                raise _unavailable()
            return _RUNNING

        cast(Any, client).execute_command = execute
        job = await client.start_background_job("sb", "ls")
        assert calls["n"] == 2
        assert not (await client.get_background_job_status("sb", job)).completed
        assert calls["n"] == 4
