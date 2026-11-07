"""The shared deny-list retry classifier."""

import httpx
import pytest
from connectrpc.code import Code
from connectrpc.errors import ConnectError

from prime_sandboxes.core.client import APIError, is_retryable
from prime_sandboxes.exceptions import CommandTimeoutError, SandboxNotRunningError

_REQ = httpx.Request("GET", "http://test")


def _status(code: int, body: str = "") -> httpx.HTTPStatusError:
    return httpx.HTTPStatusError("x", request=_REQ, response=httpx.Response(code, text=body))


def _wrapped(cause: BaseException, outer: BaseException) -> BaseException:
    outer.__cause__ = cause
    return outer


@pytest.mark.parametrize(
    "exc, idempotent, expected",
    [
        # Transport faults retry by default on idempotent calls.
        (httpx.ReadError("reset", request=_REQ), True, True),
        (httpx.ReadTimeout("slow", request=_REQ), True, True),
        (_status(500), True, True),
        (_status(524), True, True),
        (_status(408), True, True),
        (ConnectError(Code.UNAVAILABLE, "Failed to route request to sandbox"), True, True),
        (ConnectError(Code.INTERNAL, "Request failed"), True, True),
        (ConnectError(Code.UNKNOWN, "x"), True, True),
        # Definitive answers never retry.
        (_status(400), True, False),
        (_status(404), True, False),
        (_status(409), True, False),
        (_status(501), True, False),
        (_status(502, '{"error": "sandbox_not_found"}'), True, False),
        (ConnectError(Code.NOT_FOUND, "gone"), True, False),
        (ConnectError(Code.INVALID_ARGUMENT, "bad"), True, False),
        (ConnectError(Code.OUT_OF_RANGE, "evicted"), True, False),
        (httpx.UnsupportedProtocol("ftp", request=_REQ), True, False),
        (ValueError("not transport"), True, False),
        # Not idempotent: only failures that prove the request was never processed.
        (httpx.ConnectError("refused", request=_REQ), False, True),
        (httpx.ConnectTimeout("connect", request=_REQ), False, True),
        (_status(429), False, True),
        (httpx.ReadError("reset", request=_REQ), False, False),
        (_status(503), False, False),
        (ConnectError(Code.UNAVAILABLE, "x"), False, False),
        # Plain APIError wrappers are transparent; SDK outcomes are final.
        (_wrapped(ConnectError(Code.UNAVAILABLE, "x"), APIError("Connect RPC failed")), True, True),
        (APIError("Command stream ended without exit code"), True, False),
        (
            _wrapped(
                ConnectError(Code.DEADLINE_EXCEEDED, "x"), CommandTimeoutError("sb", "ls", 30)
            ),
            True,
            False,
        ),
        (_wrapped(ConnectError(Code.NOT_FOUND, "x"), SandboxNotRunningError("sb")), True, False),
    ],
)
def test_is_retryable(exc, idempotent, expected):
    assert is_retryable(exc, idempotent=idempotent) is expected
