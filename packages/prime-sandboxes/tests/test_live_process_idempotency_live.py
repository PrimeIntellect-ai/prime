"""Live idempotency semantics of VM sandbox live-process RPCs.

Opt-in E2E tests against a real stack. The hermetic suite proves the client
sends the idempotency keys (session_uuid / input_uuid / signal_uuid); these
tests prove sandboxd actually deduplicates on them: a retried Start attaches
instead of respawning, duplicate SendInput/SendSignal applies land once, and a
Start replay after exit returns the retained EndEvent.

Enable with PRIME_LIVE_VM_SMOKE=1; PRIME_VM_IMAGE picks the sandbox image.
"""

import asyncio
import contextlib
import os
import time
import uuid
from unittest import mock

import pytest
import pytest_asyncio
from connectrpc.client import ConnectClient

import prime_sandboxes.sandbox as sandbox_module
from prime_sandboxes import AsyncSandboxClient, CreateSandboxRequest
from prime_sandboxes._connectrpc import GOOGLE_PROTOBUF_BINARY_CODEC
from prime_sandboxes.rpc_command_session import (
    COMMAND_SESSION_SEND_INPUT_RPC_METHOD,
    COMMAND_SESSION_SEND_SIGNAL_RPC_METHOD,
    COMMAND_SESSION_START_RPC_METHOD,
    build_command_session_send_input_request,
    build_command_session_send_signal_request,
    build_command_session_start_request,
    parse_command_session_start_event,
)

pytestmark = [
    pytest.mark.skipif(
        os.environ.get("PRIME_LIVE_VM_SMOKE") != "1",
        reason="Live VM idempotency smoke is opt-in.",
    ),
    pytest.mark.asyncio(loop_scope="module"),
]


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def client():
    # No home-dir isolation patch (unlike the platform VM E2E conftest): these
    # tests are driven by PRIME_API_KEY / PRIME_API_BASE_URL / PRIME_TEAM_ID,
    # which override ~/.prime config; patching Path.home would only hide a
    # missing env var by silently falling back to an empty config.
    async with AsyncSandboxClient() as client:
        yield client


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def vm(client: AsyncSandboxClient):
    """One shared RUNNING VM for the whole module, to bound cost."""
    sandbox = await client.create(
        CreateSandboxRequest(
            name=f"live-idem-{uuid.uuid4().hex[:8]}",
            docker_image=os.environ.get("PRIME_VM_IMAGE", "python:3.11-slim"),
            vm=True,
            cpu_cores=1,
            memory_gb=2,
            disk_size_gb=10,
            timeout_minutes=60,
        )
    )
    try:
        await client.wait_for_creation(sandbox.id, max_attempts=120)
        yield sandbox.id
    finally:
        await client.delete(sandbox.id)


async def _open_process_recording_session_uuid(client, sandbox_id, command):
    """Open a live process and capture its session_uuid.

    open_process mints the session_uuid in its body and never exposes it;
    recording the mint is the only way a test can rebuild byte-identical
    requests that address the live session.
    """
    minted: list[str] = []
    real_mint = sandbox_module._canonical_uuid_key

    def recording_mint() -> str:
        minted.append(real_mint())
        return minted[-1]

    with mock.patch.object(sandbox_module, "_canonical_uuid_key", recording_mint):
        process = await client.open_process(sandbox_id, command)
    return process, minted[0]


async def _reissue_start(client, sandbox_id, request, *, stop_after_start=False):
    """Re-issue a built StartRequest on its own stream, returning parsed events.

    This is the byte-identical retry a pre-StartEvent drop would send; the
    stream setup mirrors open_process (auth cache + gateway URL are client
    internals with no public single-RPC surface).
    """
    auth = await client._auth_cache.get_or_refresh(sandbox_id)
    base_url = f"{auth['gateway_url'].rstrip('/')}/{auth['user_ns']}/{auth['job_id']}"
    rpc_client = ConnectClient(base_url, codec=GOOGLE_PROTOBUF_BINARY_CODEC, send_compression=None)
    stream = rpc_client.execute_server_stream(
        request=request,
        method=COMMAND_SESSION_START_RPC_METHOD,
        headers={"Authorization": f"Bearer {auth['token']}"},
        timeout_ms=60_000,
    )
    events = []
    try:
        async for response in stream:
            event = parse_command_session_start_event(response)
            if event is not None:
                events.append(event)
            if stop_after_start and event is not None and event[0] == "start":
                break
    finally:
        # ConnectClient.close() only flags the client; aclose() is what tears
        # down a stream abandoned by stop_after_start deterministically.
        with contextlib.suppress(BaseException):
            await stream.aclose()
        await rpc_client.close()
    return events


async def _line_count(client, sandbox_id, path) -> int:
    result = await client.execute_command(sandbox_id, f"cat {path} 2>/dev/null | wc -l")
    return int(result.stdout.strip())


async def _settled_line_count(client, sandbox_id, path) -> int:
    """Wait for the first line, then give a duplicate apply time to land."""
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline and await _line_count(client, sandbox_id, path) < 1:
        await asyncio.sleep(0.5)
    await asyncio.sleep(2)
    return await _line_count(client, sandbox_id, path)


async def test_start_retry_attaches_not_respawns(client, vm):
    """A retried Start against a running session attaches instead of respawning."""
    marker = f"/tmp/live-a-{uuid.uuid4().hex[:8]}"
    command = f"echo spawned >> {marker}; exec cat"
    process, session_uuid = await _open_process_recording_session_uuid(client, vm, command)
    try:
        pid = process.pid
        request = build_command_session_start_request(
            command=command, working_dir=None, env=None, stdin=True, session_uuid=session_uuid
        )
        events = await _reissue_start(client, vm, request, stop_after_start=True)
        assert events == [("start", pid)], "retried Start spawned a new process"
        probe = await client.execute_command(
            vm, f"wc -l < {marker}; test -d /proc/{pid} && echo alive"
        )
        assert probe.stdout.split() == ["1", "alive"], "retried Start respawned or killed cat"
    finally:
        await process.aclose()


async def test_input_uuid_duplicate_single_write(client, vm):
    """A byte-identical SendInput resend (same input_uuid) is applied once."""
    out = f"/tmp/live-b-{uuid.uuid4().hex[:8]}"
    process, session_uuid = await _open_process_recording_session_uuid(client, vm, f"cat >> {out}")
    try:
        request = build_command_session_send_input_request(
            session_uuid=session_uuid,
            data=b"once\n",
            input_uuid=str(uuid.uuid4()),
        )
        # _execute_process_control_rpc is the SDK's retry executor; sending one
        # built request twice reproduces a retried write byte-for-byte.
        for _ in range(2):
            await client._execute_process_control_rpc(
                vm,
                request,
                COMMAND_SESSION_SEND_INPUT_RPC_METHOD,
                sandbox_module._PROCESS_INPUT_TIMEOUT_MS,
                "stdin",
            )
        assert await _settled_line_count(client, vm, out) == 1
    finally:
        await process.aclose()


async def test_signal_uuid_duplicate_single_delivery(client, vm):
    """A duplicated SendSignal (same signal_uuid) is delivered once."""
    trap_log = f"/tmp/live-c-{uuid.uuid4().hex[:8]}"
    process, session_uuid = await _open_process_recording_session_uuid(
        client, vm, f"trap 'echo term >> {trap_log}' TERM; while true; do sleep 0.2; done"
    )
    try:
        request = build_command_session_send_signal_request(
            session_uuid=session_uuid,
            signal="terminate",
            signal_uuid=str(uuid.uuid4()),
        )
        # Same seam as above: the SDK's own retries reuse one signal_uuid, so a
        # forced duplicate must resend the identical built request.
        for _ in range(2):
            await client._execute_process_control_rpc(
                vm,
                request,
                COMMAND_SESSION_SEND_SIGNAL_RPC_METHOD,
                sandbox_module._PROCESS_SIGNAL_TIMEOUT_MS,
                "signal",
            )
        assert await _settled_line_count(client, vm, trap_log) == 1
    finally:
        # kill() first: the command traps TERM without exiting, so aclose()'s
        # terminate-then-wait escalation would stall on its grace period.
        await process.kill()
        await process.aclose()


async def test_replay_after_exit(client, vm):
    """A Start replay after exit returns the retained EndEvent, runs nothing."""
    marker = f"/tmp/live-d-{uuid.uuid4().hex[:8]}"
    command = f"echo ran >> {marker}; exit 7"
    process, session_uuid = await _open_process_recording_session_uuid(client, vm, command)
    try:
        assert await asyncio.wait_for(process.wait(), timeout=30) == 7
    finally:
        await process.aclose()

    request = build_command_session_start_request(
        command=command, working_dir=None, env=None, stdin=True, session_uuid=session_uuid
    )
    events = await _reissue_start(client, vm, request)
    end_codes = [value for kind, value in events if kind == "end"]
    assert end_codes == [7], f"replay did not return the retained exit: {events}"
    assert await _line_count(client, vm, marker) == 1, "replayed Start re-ran the command"
