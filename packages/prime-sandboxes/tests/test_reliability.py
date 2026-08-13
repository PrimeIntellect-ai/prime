"""Transient-fault classification used across the sandbox RPC surface."""

from connectrpc.code import Code
from connectrpc.errors import ConnectError

from prime_sandboxes.core import APIError
from prime_sandboxes.exceptions import CommandTimeoutError
from prime_sandboxes._reliability import is_transient_rpc_error


def test_transient_connect_codes():
    for code in (Code.DEADLINE_EXCEEDED, Code.UNAVAILABLE, Code.ABORTED, Code.INTERNAL):
        assert is_transient_rpc_error(ConnectError(code, "boom"))


def test_transient_stream_break_messages():
    # Both production stream-break variants: UNAVAILABLE "... timed out" and INTERNAL "Error
    # reading content". Each classifies by code and, for robustness, by message alone.
    assert is_transient_rpc_error(
        ConnectError(Code.UNAVAILABLE, "error reading a body from connection: timed out")
    )
    assert is_transient_rpc_error(ConnectError(Code.INTERNAL, "Error reading content"))
    assert is_transient_rpc_error(ConnectError(Code.UNKNOWN, "Error reading content"))


def test_command_timeout_is_transient():
    assert is_transient_rpc_error(CommandTimeoutError("sb", "cmd", 30))


def test_permanent_faults_not_transient():
    assert not is_transient_rpc_error(ConnectError(Code.NOT_FOUND, "no such sandbox"))
    assert not is_transient_rpc_error(APIError("HTTP 404: Sandbox not found"))
    assert not is_transient_rpc_error(APIError("Sandbox is no longer present"))
    assert not is_transient_rpc_error(ValueError("bad arg"))
