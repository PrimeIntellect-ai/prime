"""A cancelled auth refresh must never strand later sandbox operations."""

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest

from prime_sandboxes.sandbox import AsyncSandboxAuthCache


@pytest.mark.asyncio
async def test_repeated_cancellation_releases_refresh_waiters_while_cache_lock_is_busy(tmp_path):
    entered = asyncio.Event()
    blocked = asyncio.Event()
    auth = {
        "gateway_url": "https://gateway.example.com",
        "user_ns": "ns",
        "job_id": "job",
        "token": "test-token",
        "expires_at": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
    }
    calls = 0

    async def request(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            entered.set()
            await blocked.wait()
        return auth

    cache = AsyncSandboxAuthCache(tmp_path / "auth.json", AsyncMock(request=request))
    owner = asyncio.create_task(cache.get_or_refresh("sandbox"))
    await entered.wait()
    waiter = asyncio.create_task(cache.get_or_refresh("sandbox"))
    await asyncio.sleep(0)
    await cache._lock.acquire()
    try:
        owner.cancel()
        await asyncio.sleep(0)
        owner.cancel()
        with pytest.raises(asyncio.CancelledError):
            await owner
    finally:
        cache._lock.release()
    try:
        assert await asyncio.wait_for(waiter, timeout=2) == auth
        assert calls == 2
        assert await cache.get_or_refresh("sandbox") == auth
        assert calls == 2
    finally:
        waiter.cancel()
        await asyncio.gather(waiter, return_exceptions=True)
