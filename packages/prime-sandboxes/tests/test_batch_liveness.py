"""Lost jobs must be terminal through both public and coalesced batch paths."""

import asyncio
import re
import threading
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from prime_sandboxes.core import APIClient, APIError
from prime_sandboxes.models import CommandResponse
from prime_sandboxes.sandbox import AsyncSandboxClient, SandboxClient, _canonical_background_job


def response(stdout, exit_code=0):
    return CommandResponse(
        stdout=stdout, stderr="probe failure" if exit_code else "", exit_code=exit_code
    )


def rows(command, status=""):
    ids = re.findall(r"printf '%s %s\\n' ([0-9a-f]{8})", command)
    assert ids, command
    return "".join(f"{job_id} {status}\n" for job_id in ids)


async def async_client(jobs):
    client = AsyncSandboxClient(api_key="test-key")
    await client.client.aclose()
    client.client = SimpleNamespace(
        request=AsyncMock(return_value=platform_body(jobs)), aclose=AsyncMock()
    )
    return client


def platform_body(jobs):
    return {
        "statuses": [
            dict(sandbox_id=j.sandbox_id, job_id=j.job_id, completed=False, exit_code=None)
            for j in jobs
        ],
        "errors": [],
    }


@pytest.mark.parametrize("coalesced", [False, True])
def test_sync_lost_job_is_terminal(coalesced):
    client = SandboxClient(APIClient(api_key="test-key"))
    client.client.client.close()
    jobs = [_canonical_background_job("sandbox-a", "deadbeef")]
    client.client = SimpleNamespace(request=Mock(return_value=platform_body(jobs)))
    client.execute_command = Mock(
        return_value=CommandResponse(stdout="deadbeef lost\n", stderr="", exit_code=0)
    )
    with pytest.raises(APIError, match="exited without recording"):
        if coalesced:
            client._background_job_status_batcher.get(("sandbox-a", "deadbeef"))
        else:
            client.get_background_job_statuses(jobs)


@pytest.mark.asyncio
@pytest.mark.parametrize("coalesced", [False, True])
async def test_async_lost_job_is_terminal(coalesced):
    client = AsyncSandboxClient(api_key="test-key")
    await client.client.aclose()
    jobs = [_canonical_background_job("sandbox-a", "deadbeef")]
    client.client = SimpleNamespace(
        request=AsyncMock(return_value=platform_body(jobs)), aclose=AsyncMock()
    )
    client.execute_command = AsyncMock(
        return_value=CommandResponse(stdout="deadbeef lost\n", stderr="", exit_code=0)
    )
    try:
        with pytest.raises(APIError, match="exited without recording"):
            if coalesced:
                await client._background_job_status_batcher.get(("sandbox-a", "deadbeef"))
            else:
                await client.get_background_job_statuses(jobs)
    finally:
        await client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stdout,code",
    [
        ("", 0),
        ("garbage\n", 0),
        ("deadbeef pending\n", 0),
        ("deadbeef \ndeadbeef \n", 0),
        ("ffffffff \n", 0),
        ("deadbeef \n", 1),
        ("deadbeef 0\nextra\n", 0),
    ],
)
async def test_probe_failure_never_becomes_running(stdout, code):
    jobs = [_canonical_background_job("a", "deadbeef")]
    client = await async_client(jobs)
    client.execute_command = AsyncMock(return_value=response(stdout, code))
    try:
        with pytest.raises(APIError):
            await client.get_background_job_statuses(jobs)
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_long_live_jobs_stay_live_and_one_guest_uses_one_probe():
    jobs = [_canonical_background_job("a", f"{n:08x}") for n in range(100)]
    client = await async_client(jobs)
    client.execute_command = AsyncMock(
        side_effect=lambda _id, command, **kw: response(rows(command))
    )
    try:
        for _ in range(20):
            statuses = await client.get_background_job_statuses(jobs)
            assert len(statuses) == 100
            assert all(not s.completed and s.exit_code is None for s in statuses)
        assert client.client.request.call_count == 20
        assert client.execute_command.call_count == 20
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_completion_race_preserves_exit_code_and_order():
    jobs = [_canonical_background_job("a", i) for i in ["deadbeef", "feedface"]]
    client = await async_client(jobs)
    client.execute_command = AsyncMock(return_value=response("feedface 7\ndeadbeef 0\n"))
    try:
        statuses = await client.get_background_job_statuses(jobs)
        assert [s.job_id for s in statuses] == [j.job_id for j in jobs]
        assert [s.exit_code for s in statuses] == [0, 7]
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_completed_batch_has_zero_extra_probes():
    jobs = [_canonical_background_job(str(n), f"{n:08x}") for n in range(100)]
    client = await async_client(jobs)
    body = platform_body(jobs)
    for status in body["statuses"]:
        status.update(completed=True, exit_code=0)
    client.client.request.return_value = body
    client.execute_command = AsyncMock(side_effect=AssertionError("unnecessary probe"))
    try:
        assert all(s.completed for s in await client.get_background_job_statuses(jobs))
        client.execute_command.assert_not_called()
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_probe_failure_is_isolated_from_other_guests_and_provider_errors():
    jobs = [
        _canonical_background_job(s, j)
        for s, j in [("a", "deadbeef"), ("b", "feedface"), ("c", "cafebabe")]
    ]
    client = await async_client(jobs)
    body = platform_body(jobs[:2])
    body["errors"] = [
        dict(sandbox_id="c", job_id="cafebabe", code="RUNTIME_ERROR", message="provider fault")
    ]
    client.client.request.return_value = body

    async def execute(sandbox_id, command, **kwargs):
        if sandbox_id == "a":
            raise APIError("transient RPC fault")
        return response(rows(command))

    client.execute_command = AsyncMock(side_effect=execute)
    try:
        results = await asyncio.gather(
            *(client._background_job_status_batcher.get((j.sandbox_id, j.job_id)) for j in jobs),
            return_exceptions=True,
        )
        assert isinstance(results[0], APIError) and "transient RPC" in str(results[0])
        assert not results[1].completed
        assert isinstance(results[2], APIError) and "provider fault" in str(results[2])
        assert client.execute_command.call_count == 2
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_async_probe_concurrency_is_client_wide_across_simultaneous_batches():
    jobs = [_canonical_background_job(str(n), f"{n:08x}") for n in range(200)]
    client = await async_client(jobs)
    client.client.request.side_effect = lambda *a, **kw: {
        "statuses": [dict(**j, completed=False, exit_code=None) for j in kw["json"]["jobs"]],
        "errors": [],
    }
    active = peak = 0

    async def execute(sandbox_id, command, **kwargs):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        try:
            await asyncio.sleep(0.005)
            return response(rows(command))
        finally:
            active -= 1

    client.execute_command = AsyncMock(side_effect=execute)
    try:
        batches = await asyncio.gather(
            client.get_background_job_statuses(jobs[:100]),
            client.get_background_job_statuses(jobs[100:]),
        )
        assert all(not status.completed for batch in batches for status in batch)
        assert peak == 20 and active == 0
        assert client.execute_command.call_count == 200
    finally:
        await client.aclose()


def test_sync_probe_concurrency_and_fault_isolation():
    from concurrent.futures import ThreadPoolExecutor

    client = SandboxClient(APIClient(api_key="test-key"))
    client.client.client.close()
    client.client = SimpleNamespace(
        request=Mock(
            side_effect=lambda *a, **kw: {
                "statuses": [
                    dict(**j, completed=False, exit_code=None) for j in kw["json"]["jobs"]
                ],
                "errors": [],
            }
        )
    )
    jobs = [_canonical_background_job(str(n), f"{n:08x}") for n in range(200)]
    active = peak = 0
    lock = threading.Lock()

    def execute(sandbox_id, command, **kwargs):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        try:
            time.sleep(0.005)
            return response(rows(command))
        finally:
            with lock:
                active -= 1

    client.execute_command = Mock(side_effect=execute)
    with ThreadPoolExecutor(2) as pool:
        batches = list(pool.map(client.get_background_job_statuses, [jobs[:100], jobs[100:]]))
    assert all(not s.completed for b in batches for s in b)
    assert 1 < peak <= 20 and active == 0
    assert client.execute_command.call_count == 200


@pytest.mark.asyncio
async def test_async_cancellation_drains_active_and_queued_probes():
    jobs = [_canonical_background_job(str(n), f"{n:08x}") for n in range(100)]
    client = await async_client(jobs)
    active = 0
    full = asyncio.Event()

    async def execute(*args, **kwargs):
        nonlocal active
        active += 1
        if active == 20:
            full.set()
        try:
            await asyncio.Event().wait()
        finally:
            active -= 1

    client.execute_command = AsyncMock(side_effect=execute)
    task = asyncio.create_task(client.get_background_job_statuses(jobs))
    try:
        await asyncio.wait_for(full.wait(), 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert active == 0
        assert client.execute_command.call_count == 20
        assert client._background_job_probe_slots._value == 20
        # Reuse the same client after cancellation: no stranded lease/slot.
        client.execute_command = AsyncMock(side_effect=lambda _id, cmd, **kw: response(rows(cmd)))
        assert len(await client.get_background_job_statuses(jobs)) == 100
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_async_probe_stage_deadline_includes_waiting_for_slots():
    jobs = [_canonical_background_job(str(n), f"{n:08x}") for n in range(100)]
    client = await async_client(jobs)
    client.execute_command = AsyncMock(side_effect=lambda *a, **kw: asyncio.sleep(100))

    # An async side effect must await; a returned coroutine is merely data.
    async def slow(*args, **kwargs):
        await asyncio.sleep(100)

    client.execute_command.side_effect = slow
    started = time.monotonic()
    try:
        with pytest.raises(APIError, match="deadline exceeded"):
            await client.get_background_job_statuses(jobs, timeout=0.02)
        assert time.monotonic() - started < 0.5
        assert client._background_job_probe_slots._value == 20
    finally:
        await client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ["id", "exit", "duplicate"])
async def test_unsafe_or_custom_handles_fail_before_any_rpc(mutation):
    jobs = [_canonical_background_job("a", "deadbeef")]
    client = await async_client(jobs)
    if mutation == "id":
        jobs[0].job_id = "deadbeef; echo injected"
    elif mutation == "exit":
        jobs[0].exit_file = "/tmp/custom; echo injected"
    else:
        jobs.append(jobs[0])
    try:
        with pytest.raises(ValueError):
            await client.get_background_job_statuses(jobs)
        client.client.request.assert_not_called()
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_coalesced_cancel_preserves_other_waiter_and_delete_waits_for_probe():
    jobs = [_canonical_background_job("a", "deadbeef"), _canonical_background_job("b", "feedface")]
    client = await async_client(jobs)
    entered = asyncio.Event()
    release = asyncio.Event()
    deleted = asyncio.Event()
    cancelled_probes = []

    async def request(method, path, **kwargs):
        if method == "DELETE":
            deleted.set()
            return {}
        return platform_body(jobs)

    async def execute(sandbox_id, command, **kwargs):
        entered.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            cancelled_probes.append(sandbox_id)
            raise
        return response(rows(command))

    client.client.request.side_effect = request
    client.execute_command = AsyncMock(side_effect=execute)
    batcher = client._background_job_status_batcher
    first = asyncio.create_task(batcher.get(("a", "deadbeef")))
    second = asyncio.create_task(batcher.get(("b", "feedface")))
    try:
        await asyncio.wait_for(entered.wait(), 1)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        deleting = asyncio.create_task(client.delete("a"))
        while "a" not in client._operation_leases._draining:
            await asyncio.sleep(0)
        assert not deleted.is_set()
        assert not cancelled_probes
        release.set()
        assert not (await second).completed
        await deleting
        assert deleted.is_set()
        assert not client._operation_leases._active
    finally:
        release.set()
        await client.aclose()


@pytest.mark.asyncio
async def test_last_coalesced_waiter_cancellation_drains_probe_cleanup():
    jobs = [_canonical_background_job("a", "deadbeef")]
    client = await async_client(jobs)
    entered = asyncio.Event()
    cleaning = asyncio.Event()
    finish = asyncio.Event()

    async def execute(*args, **kwargs):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaning.set()
            await finish.wait()

    client.execute_command = AsyncMock(side_effect=execute)
    task = asyncio.create_task(client._background_job_status_batcher.get(("a", "deadbeef")))
    try:
        await asyncio.wait_for(entered.wait(), 1)
        task.cancel()
        await asyncio.wait_for(cleaning.wait(), 1)
        assert not task.done()
        assert client._operation_leases._active
        finish.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not client._operation_leases._active
        assert client._background_job_probe_slots._value == 20
    finally:
        finish.set()
        await client.aclose()


@pytest.mark.asyncio
async def test_async_live_and_lost_jobs_are_isolated_in_one_sandbox():
    client = AsyncSandboxClient(api_key="test-key")
    await client.client.aclose()
    jobs = [_canonical_background_job("sandbox-a", i) for i in ["deadbeef", "feedface"]]
    client.client = SimpleNamespace(
        request=AsyncMock(return_value=platform_body(jobs)), aclose=AsyncMock()
    )
    client.execute_command = AsyncMock(
        return_value=CommandResponse(stdout="deadbeef lost\nfeedface \n", stderr="", exit_code=0)
    )
    try:
        results = await asyncio.gather(
            *(client._background_job_status_batcher.get((j.sandbox_id, j.job_id)) for j in jobs),
            return_exceptions=True,
        )
        assert isinstance(results[0], APIError)
        assert not results[1].completed
        assert client.execute_command.call_count == 1
    finally:
        await client.aclose()
