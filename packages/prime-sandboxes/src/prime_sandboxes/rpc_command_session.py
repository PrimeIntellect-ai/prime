"""Command session Connect RPC helpers."""

from typing import Dict, List, Optional, Protocol, cast

from connectrpc.method import IdempotencyLevel, MethodInfo
from google.protobuf.message import Message

from ._proto.command_session import command_session_pb2


class _CommandSpecLike(Protocol):
    cwd: str


class _CommandSpecFactory(Protocol):
    def __call__(self, *, cmd: str, args: List[str], envs: Dict[str, str]) -> _CommandSpecLike: ...


class _CommandSessionStartRequestFactory(Protocol):
    def __call__(self, *, command: _CommandSpecLike, stdin: bool) -> Message: ...


class _CommandSessionDataEventLike(Protocol):
    stdout: bytes
    stderr: bytes
    pty: bytes

    def WhichOneof(self, field_name: str) -> str | None: ...


class _CommandSessionEndEventLike(Protocol):
    exit_code: int


class _CommandSessionEventLike(Protocol):
    data: _CommandSessionDataEventLike
    end: _CommandSessionEndEventLike

    def WhichOneof(self, field_name: str) -> str | None: ...


class _CommandSessionStartResponseLike(Protocol):
    event: _CommandSessionEventLike

    def HasField(self, field_name: str) -> bool: ...


_COMMAND_SESSION_START_REQUEST_TYPE = cast(
    type[Message], getattr(command_session_pb2, "StartRequest")
)
_COMMAND_SESSION_START_RESPONSE_TYPE = cast(
    type[Message], getattr(command_session_pb2, "StartResponse")
)
_COMMAND_SESSION_START_REQUEST_FACTORY = cast(
    _CommandSessionStartRequestFactory, _COMMAND_SESSION_START_REQUEST_TYPE
)
_COMMAND_SPEC_FACTORY = cast(_CommandSpecFactory, getattr(command_session_pb2, "CommandSpec"))


COMMAND_SESSION_START_RPC_METHOD = MethodInfo(
    name="Start",
    service_name="command_session.CommandSession",
    input=_COMMAND_SESSION_START_REQUEST_TYPE,
    output=_COMMAND_SESSION_START_RESPONSE_TYPE,
    idempotency_level=IdempotencyLevel.UNKNOWN,
)


def build_command_session_start_request(
    command: str,
    working_dir: Optional[str],
    env: Optional[Dict[str, str]],
) -> Message:
    command_spec = _COMMAND_SPEC_FACTORY(
        cmd="/bin/bash",
        args=["-l", "-c", command],
        envs=env or {},
    )
    if working_dir is not None:
        command_spec.cwd = working_dir

    return _COMMAND_SESSION_START_REQUEST_FACTORY(command=command_spec, stdin=False)


def collect_command_session_start_event(
    response: Message,
    stdout_parts: List[str],
    stderr_parts: List[str],
) -> Optional[int]:
    start_response = cast(_CommandSessionStartResponseLike, response)
    if not start_response.HasField("event"):
        return None

    event = start_response.event
    event_kind = event.WhichOneof("event")

    if event_kind == "data":
        data_kind = event.data.WhichOneof("output")
        if data_kind == "stdout" and event.data.stdout:
            stdout_parts.append(event.data.stdout.decode("utf-8", errors="replace"))
        elif data_kind == "stderr" and event.data.stderr:
            stderr_parts.append(event.data.stderr.decode("utf-8", errors="replace"))
        elif data_kind == "pty" and event.data.pty:
            stdout_parts.append(event.data.pty.decode("utf-8", errors="replace"))
    elif event_kind == "end":
        return int(event.end.exit_code)

    return None
