"""Smoke check run against the built prime-sandboxes wheel in a clean environment."""

from importlib.metadata import version

from prime_sandboxes._connectrpc import GOOGLE_PROTOBUF_BINARY_CODEC
from prime_sandboxes.rpc_command_session import build_command_session_start_request

request = build_command_session_start_request("echo ready", None, None)
payload = GOOGLE_PROTOBUF_BINARY_CODEC.encode(request)
decoded = GOOGLE_PROTOBUF_BINARY_CODEC.decode(payload, type(request))

assert decoded == request
assert version("connectrpc").startswith("0.11.")
