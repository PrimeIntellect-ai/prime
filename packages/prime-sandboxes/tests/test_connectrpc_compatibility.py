"""Regression coverage for the Connect RPC package and protobuf runtime boundary."""

import pytest

from prime_sandboxes._connectrpc import (
    GOOGLE_PROTOBUF_BINARY_CODEC,
    _validate_connectrpc_runtime,
)
from prime_sandboxes.rpc_command_session import build_command_session_start_request


def test_google_protobuf_command_request_round_trips() -> None:
    request = build_command_session_start_request("echo ready", None, None)

    payload = GOOGLE_PROTOBUF_BINARY_CODEC.encode(request)
    decoded = GOOGLE_PROTOBUF_BINARY_CODEC.decode(payload, type(request))

    assert decoded == request


def test_legacy_and_current_connect_distributions_are_rejected() -> None:
    with pytest.raises(RuntimeError, match="Both provide the 'connectrpc' Python package"):
        _validate_connectrpc_runtime("0.11.1", "0.9.0", "0.11.1")


def test_stale_connect_module_files_are_rejected() -> None:
    with pytest.raises(RuntimeError, match="does not match the installed distribution"):
        _validate_connectrpc_runtime("0.11.1", None, "0.9.0")
