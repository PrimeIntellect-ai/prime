"""Background-job launcher execs honor start_background_job's working_dir."""

from typing import Any, List, cast

import pytest

from prime_sandboxes.core.client import APIClient
from prime_sandboxes.models import CommandResponse
from prime_sandboxes.sandbox import AsyncSandboxClient, SandboxClient

_OK = CommandResponse(stdout="", stderr="", exit_code=0)


def test_sync_launcher_exec_passes_working_dir():
    client = SandboxClient(APIClient(api_key="test-key"))
    seen: List[Any] = []

    def execute(_sandbox_id, _command, **kwargs):
        seen.append(kwargs.get("working_dir", "<absent>"))
        return _OK

    cast(Any, client).execute_command = execute
    client.start_background_job("sb", "pwd", working_dir="/")
    client.start_background_job("sb", "pwd")
    assert seen == ["/", None]


@pytest.mark.asyncio
async def test_async_launcher_exec_passes_working_dir():
    client = AsyncSandboxClient(APIClient(api_key="test-key"))
    seen: List[Any] = []

    async def execute(_sandbox_id, _command, **kwargs):
        seen.append(kwargs.get("working_dir", "<absent>"))
        return _OK

    cast(Any, client).execute_command = execute
    await client.start_background_job("sb", "pwd", working_dir="/")
    await client.start_background_job("sb", "pwd")
    assert seen == ["/", None]
