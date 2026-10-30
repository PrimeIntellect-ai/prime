"""Regression coverage for the Connect RPC package and protobuf runtime boundary."""

import pytest

from prime_sandboxes._connectrpc import (
    GOOGLE_PROTOBUF_BINARY_CODEC,
    _reject_legacy_connect_python,
)
from prime_sandboxes.rpc_command_session import build_command_session_start_request


def test_google_protobuf_command_request_round_trips() -> None:
    request = build_command_session_start_request(command="echo ready", working_dir=None, env=None)

    payload = GOOGLE_PROTOBUF_BINARY_CODEC.encode(request)
    decoded = GOOGLE_PROTOBUF_BINARY_CODEC.decode(payload, type(request))

    assert decoded == request


def test_legacy_connect_distribution_is_rejected() -> None:
    with pytest.raises(RuntimeError, match="Legacy connect-python=0.9.0"):
        _reject_legacy_connect_python("0.9.0")


@pytest.mark.parametrize("user", [None, "root", "ubuntu"])
def test_command_user_round_trips(user):
    request = build_command_session_start_request(
        command="id", working_dir=None, env=None, user=user
    )
    decoded = GOOGLE_PROTOBUF_BINARY_CODEC.decode(
        GOOGLE_PROTOBUF_BINARY_CODEC.encode(request), type(request)
    )
    assert decoded.command.HasField("user") is (user is not None)
    assert decoded.command.user == (user or "")


def test_empty_command_user_is_rejected():
    with pytest.raises(ValueError, match="non-empty guest username"):
        build_command_session_start_request(command="id", working_dir=None, env=None, user="")
