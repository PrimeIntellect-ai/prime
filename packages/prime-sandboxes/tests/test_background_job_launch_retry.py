"""start_background_job retries a transient timeout on the fire-and-forget launch."""

from typing import Any, cast

import pytest

from prime_sandboxes.core.client import APIClient
from prime_sandboxes.exceptions import CommandTimeoutError
from prime_sandboxes.models import CommandResponse
from prime_sandboxes.sandbox import (
    AsyncSandboxClient,
    SandboxClient,
    _BG_LAUNCH_ATTEMPTS,
)

_OK = CommandResponse(stdout="", stderr="", exit_code=0)


def _timeout():
    return CommandTimeoutError("sb", "nohup ...", 30)


async def _no_sleep(_):
    return None


class TestSyncLaunchRetry:
    def test_retries_and_succeeds(self, monkeypatch):
        monkeypatch.setattr("prime_sandboxes.sandbox.time.sleep", lambda _: None)
        client = SandboxClient(APIClient(api_key="test-key"))
        calls = {"n": 0}

        def execute(*_a, **_k):
            calls["n"] += 1
            if calls["n"] < _BG_LAUNCH_ATTEMPTS:
                raise _timeout()
            return _OK

        cast(Any, client).execute_command = execute
        job = client.start_background_job("sb", "rm -rf x")
        assert calls["n"] == _BG_LAUNCH_ATTEMPTS
        assert job.job_id

    def test_gives_up_after_max_attempts(self, monkeypatch):
        monkeypatch.setattr("prime_sandboxes.sandbox.time.sleep", lambda _: None)
        client = SandboxClient(APIClient(api_key="test-key"))
        calls = {"n": 0}

        def execute(*_a, **_k):
            calls["n"] += 1
            raise _timeout()

        cast(Any, client).execute_command = execute
        with pytest.raises(CommandTimeoutError):
            client.start_background_job("sb", "rm -rf x")
        assert calls["n"] == _BG_LAUNCH_ATTEMPTS


class TestAsyncLaunchRetry:
    @pytest.mark.asyncio
    async def test_retries_and_succeeds(self, monkeypatch):
        monkeypatch.setattr("prime_sandboxes.sandbox.asyncio.sleep", _no_sleep)
        client = AsyncSandboxClient(APIClient(api_key="test-key"))
        calls = {"n": 0}

        async def execute(*_a, **_k):
            calls["n"] += 1
            if calls["n"] < _BG_LAUNCH_ATTEMPTS:
                raise _timeout()
            return _OK

        cast(Any, client).execute_command = execute
        job = await client.start_background_job("sb", "rm -rf x")
        assert calls["n"] == _BG_LAUNCH_ATTEMPTS
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
            await client.start_background_job("sb", "rm -rf x")
        assert calls["n"] == _BG_LAUNCH_ATTEMPTS
