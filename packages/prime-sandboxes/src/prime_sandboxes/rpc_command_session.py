"""Command session Connect RPC helpers."""

from typing import Dict, List, Literal, Optional, Protocol, cast

from connectrpc.method import IdempotencyLevel, MethodInfo
from google.protobuf.message import Message

from ._proto.command_session import command_session_pb2


class _CommandSpecLike(Protocol):
    cwd: str


class _CommandSpecFactory(Protocol):
    def __call__(self, *, cmd: str, args: List[str], envs: Dict[str, str]) -> _CommandSpecLike: ...


class _CommandSessionStartRequestFactory(Protocol):
    def __call__(self, *, command: _CommandSpecLike, stdin: bool) -> Message: ...


class _CommandSessionSelectorFactory(Protocol):
    def __call__(self, *, pid: int) -> Message: ...


class _CommandInputFactory(Protocol):
    def __call__(self, *, stdin: bytes) -> Message: ...


class _CommandSessionSendInputRequestFactory(Protocol):
    def __call__(self, *, session: Message, input: Message) -> Message: ...


class _CommandSessionSendSignalRequestFactory(Protocol):
    def __call__(self, *, session: Message, signal: int) -> Message: ...


class _CommandSessionDataEventLike(Protocol):
    stdout: bytes
    stderr: bytes
    pty: bytes

    def WhichOneof(self, field_name: str) -> str | None: ...


class _CommandSessionEndEventLike(Protocol):
    exit_code: int


class _CommandSessionEventLike(Protocol):
    start: "_CommandSessionStartEventLike"
    data: _CommandSessionDataEventLike
    end: _CommandSessionEndEventLike

    def WhichOneof(self, field_name: str) -> str | None: ...


class _CommandSessionStartEventLike(Protocol):
    pid: int


class _CommandSessionStartResponseLike(Protocol):
    event: _CommandSessionEventLike

    def HasField(self, field_name: str) -> bool: ...


_COMMAND_SESSION_START_REQUEST_TYPE = cast(
    type[Message], getattr(command_session_pb2, "StartRequest")
)
_COMMAND_SESSION_START_RESPONSE_TYPE = cast(
    type[Message], getattr(command_session_pb2, "StartResponse")
)
_COMMAND_SESSION_SEND_INPUT_REQUEST_TYPE = cast(
    type[Message], getattr(command_session_pb2, "SendInputRequest")
)
_COMMAND_SESSION_SEND_INPUT_RESPONSE_TYPE = cast(
    type[Message], getattr(command_session_pb2, "SendInputResponse")
)
_COMMAND_SESSION_SEND_SIGNAL_REQUEST_TYPE = cast(
    type[Message], getattr(command_session_pb2, "SendSignalRequest")
)
_COMMAND_SESSION_SEND_SIGNAL_RESPONSE_TYPE = cast(
    type[Message], getattr(command_session_pb2, "SendSignalResponse")
)
_COMMAND_SESSION_CONNECT_REQUEST_TYPE = cast(
    type[Message], getattr(command_session_pb2, "ConnectRequest")
)
_COMMAND_SESSION_CONNECT_RESPONSE_TYPE = cast(
    type[Message], getattr(command_session_pb2, "ConnectResponse")
)
_COMMAND_SESSION_START_REQUEST_FACTORY = cast(
    _CommandSessionStartRequestFactory, _COMMAND_SESSION_START_REQUEST_TYPE
)
_COMMAND_SPEC_FACTORY = cast(_CommandSpecFactory, getattr(command_session_pb2, "CommandSpec"))
_COMMAND_SESSION_SELECTOR_FACTORY = cast(
    _CommandSessionSelectorFactory,
    getattr(command_session_pb2, "CommandSessionSelector"),
)
_COMMAND_INPUT_FACTORY = cast(_CommandInputFactory, getattr(command_session_pb2, "CommandInput"))
_COMMAND_SESSION_SEND_INPUT_REQUEST_FACTORY = cast(
    _CommandSessionSendInputRequestFactory, _COMMAND_SESSION_SEND_INPUT_REQUEST_TYPE
)
_COMMAND_SESSION_SEND_SIGNAL_REQUEST_FACTORY = cast(
    _CommandSessionSendSignalRequestFactory, _COMMAND_SESSION_SEND_SIGNAL_REQUEST_TYPE
)


COMMAND_SESSION_START_RPC_METHOD = MethodInfo(
    name="Start",
    service_name="command_session.CommandSession",
    input=_COMMAND_SESSION_START_REQUEST_TYPE,
    output=_COMMAND_SESSION_START_RESPONSE_TYPE,
    idempotency_level=IdempotencyLevel.UNKNOWN,
)

COMMAND_SESSION_SEND_INPUT_RPC_METHOD = MethodInfo(
    name="SendInput",
    service_name="command_session.CommandSession",
    input=_COMMAND_SESSION_SEND_INPUT_REQUEST_TYPE,
    output=_COMMAND_SESSION_SEND_INPUT_RESPONSE_TYPE,
    idempotency_level=IdempotencyLevel.UNKNOWN,
)

COMMAND_SESSION_SEND_SIGNAL_RPC_METHOD = MethodInfo(
    name="SendSignal",
    service_name="command_session.CommandSession",
    input=_COMMAND_SESSION_SEND_SIGNAL_REQUEST_TYPE,
    output=_COMMAND_SESSION_SEND_SIGNAL_RESPONSE_TYPE,
    idempotency_level=IdempotencyLevel.UNKNOWN,
)

# Re-attach to an already-running session's output stream (server-streaming), selected by pid.
# Used to resume a live process after its Start stream drops on a transient link fault.
COMMAND_SESSION_CONNECT_RPC_METHOD = MethodInfo(
    name="Connect",
    service_name="command_session.CommandSession",
    input=_COMMAND_SESSION_CONNECT_REQUEST_TYPE,
    output=_COMMAND_SESSION_CONNECT_RESPONSE_TYPE,
    idempotency_level=IdempotencyLevel.NO_SIDE_EFFECTS,
)


def build_command_session_start_request(
    command: str,
    working_dir: Optional[str],
    env: Optional[Dict[str, str]],
    *,
    stdin: bool = False,
) -> Message:
    command_spec = _COMMAND_SPEC_FACTORY(
        cmd="/bin/bash",
        args=["-c", command],
        envs=env or {},
    )
    if working_dir is not None:
        command_spec.cwd = working_dir

    return _COMMAND_SESSION_START_REQUEST_FACTORY(command=command_spec, stdin=stdin)


def build_command_session_connect_request(pid: int) -> Message:
    return _COMMAND_SESSION_CONNECT_REQUEST_TYPE(
        session=_COMMAND_SESSION_SELECTOR_FACTORY(pid=pid)
    )


def build_command_session_send_input_request(pid: int, data: bytes) -> Message:
    return _COMMAND_SESSION_SEND_INPUT_REQUEST_FACTORY(
        session=_COMMAND_SESSION_SELECTOR_FACTORY(pid=pid),
        input=_COMMAND_INPUT_FACTORY(stdin=data),
    )


def build_command_session_send_signal_request(
    pid: int, signal: Literal["terminate", "kill"]
) -> Message:
    signal_value = getattr(
        command_session_pb2,
        "SIGNAL_SIGTERM" if signal == "terminate" else "SIGNAL_SIGKILL",
    )
    return _COMMAND_SESSION_SEND_SIGNAL_REQUEST_FACTORY(
        session=_COMMAND_SESSION_SELECTOR_FACTORY(pid=pid),
        signal=signal_value,
    )


def parse_command_session_start_event(
    response: Message,
) -> (
    tuple[Literal["start"], int]
    | tuple[Literal["stdout", "stderr"], bytes]
    | tuple[Literal["end"], int]
    | None
):
    start_response = cast(_CommandSessionStartResponseLike, response)
    if not start_response.HasField("event"):
        return None

    event = start_response.event
    event_kind = event.WhichOneof("event")
    if event_kind == "start":
        return "start", int(event.start.pid)
    if event_kind == "data":
        data_kind = event.data.WhichOneof("output")
        if data_kind == "stdout":
            return "stdout", bytes(event.data.stdout)
        if data_kind == "stderr":
            return "stderr", bytes(event.data.stderr)
        if data_kind == "pty":
            return "stdout", bytes(event.data.pty)
    if event_kind == "end":
        return "end", int(event.end.exit_code)
    return None


def collect_command_session_start_event(
    response: Message,
    stdout_parts: List[str],
    stderr_parts: List[str],
) -> Optional[int]:
    event = parse_command_session_start_event(response)
    if event is None:
        return None
    kind, value = event
    if kind == "stdout":
        assert isinstance(value, bytes)
        if value:
            stdout_parts.append(value.decode("utf-8", errors="replace"))
    elif kind == "stderr":
        assert isinstance(value, bytes)
        if value:
            stderr_parts.append(value.decode("utf-8", errors="replace"))
    elif kind == "end":
        assert isinstance(value, int)
        return value

    return None
