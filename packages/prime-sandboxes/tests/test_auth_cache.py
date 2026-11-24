import asyncio
import json
from unittest.mock import patch

from prime_sandboxes import sandbox as sb

AUTH = {"token": "t", "expires_at": "2999-01-01T00:00:00Z"}


def test_bulk_set_throttles_writes_and_flush_persists_latest(tmp_path):
    path = tmp_path / "sandbox_auth_cache.json"
    cache = sb.AsyncSandboxAuthCache(path, client=None)

    async def bulk():
        for i in range(1000):
            await cache.set(f"sb{i}", AUTH)

    with patch.object(sb, "_write_auth_cache", wraps=sb._write_auth_cache) as write:
        asyncio.run(bulk())
        assert write.call_count < 5  # was one full-file rewrite per token
        cache._file.flush()  # what atexit runs
    assert len(json.loads(path.read_text())) == 1000


def test_clear_writes_immediately(tmp_path):
    path = tmp_path / "sandbox_auth_cache.json"
    cache = sb.SandboxAuthCache(path, client=None)
    cache.set("a", AUTH)
    cache.set("b", AUTH)  # throttled
    cache.clear()
    assert json.loads(path.read_text()) == {}
