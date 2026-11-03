"""Command-session RPC helpers."""

from typing import Dict, List, Literal, Optional, Protocol, Sequence, cast

from connectrpc.errors import ConnectError
from connectrpc.method import IdempotencyLevel, MethodInfo
from google.protobuf.message import Message

from ._proto.command_session import command_session_pb2
from .core.client import PERMANENT_CONNECT_CODES


class _CommandSpecLike(Protocol):
    cwd: str
    user: str


class _CommandSpecFactory(Protocol):
    def __call__(self, *, cmd: str, args: List[str], envs: Dict[str, str]) -> _CommandSpecLike: ...


class _CommandSessionStartRequestFactory(Protocol):
    def __call__(
        self,
        *,
        command: _CommandSpecLike,
        stdin: bool,
        session_uuid: str | None = None,
        data_offset: int | None = None,
    ) -> Message: ...


class _CommandSessionSelectorFactory(Protocol):
    def __call__(self, *, session_uuid: str) -> Message: ...


class _CommandInputFactory(Protocol):
    def __call__(self, *, stdin: bytes) -> Message: ...


class _CommandSessionSendInputRequestFactory(Protocol):
    def __call__(self, *, session: Message, input: Message, input_uuid: str) -> Message: ...


class _CommandSessionSendSignalRequestFactory(Protocol):
    def __call__(self, *, session: Message, signal: int, signal_uuid: str) -> Message: ...


class _CommandSessionConnectRequestFactory(Protocol):
    def __call__(self, *, session: Message, data_offset: int | None = None) -> Message: ...


class _CommandSessionInfoLike(Protocol):
    pid: int
    session_uuid: str
    command: _CommandSpecLike


class _CommandSessionListResponseLike(Protocol):
    sessions: Sequence[_CommandSessionInfoLike]


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
    data_offset: int

    def HasField(self, field_name: str) -> bool: ...


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
_COMMAND_SESSION_LIST_REQUEST_TYPE = cast(
    type[Message], getattr(command_session_pb2, "ListRequest")
)
_COMMAND_SESSION_LIST_RESPONSE_TYPE = cast(
    type[Message], getattr(command_session_pb2, "ListResponse")
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
_COMMAND_SESSION_CONNECT_REQUEST_FACTORY = cast(
    _CommandSessionConnectRequestFactory, _COMMAND_SESSION_CONNECT_REQUEST_TYPE
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

# Re-attach to an already-running session's output stream by its session selector.
COMMAND_SESSION_CONNECT_RPC_METHOD = MethodInfo(
    name="Connect",
    service_name="command_session.CommandSession",
    input=_COMMAND_SESSION_CONNECT_REQUEST_TYPE,
    output=_COMMAND_SESSION_CONNECT_RESPONSE_TYPE,
    idempotency_level=IdempotencyLevel.NO_SIDE_EFFECTS,
)

# Live-process introspection: pid, session_uuid, and command for each running
# process. Permanent public API; exited sessions are not listed.
COMMAND_SESSION_LIST_RPC_METHOD = MethodInfo(
    name="List",
    service_name="command_session.CommandSession",
    input=_COMMAND_SESSION_LIST_REQUEST_TYPE,
    output=_COMMAND_SESSION_LIST_RESPONSE_TYPE,
    idempotency_level=IdempotencyLevel.NO_SIDE_EFFECTS,
)


def is_recoverable_stream_fault(error: BaseException | None) -> bool:
    """Whether a dropped command-session stream may be re-attached (None: clean EOF).

    Re-attach (Connect, or a create-or-attach Start resending the identical
    request) is idempotent, so only a definitive Connect answer is fatal. Unlike
    is_retryable, a non-RPC fault (a clean EOF, a local stream error) also
    re-attaches: the process outlives the client's view of it.
    """
    return not (isinstance(error, ConnectError) and error.code in PERMANENT_CONNECT_CODES)


def build_command_session_start_request(
    *,
    command: str,
    working_dir: Optional[str],
    env: Optional[Dict[str, str]],
    stdin: bool = False,
    session_uuid: str | None = None,
    user: str | None = None,
    data_offset: int | None = None,
) -> Message:
    command_spec = _COMMAND_SPEC_FACTORY(
        cmd="/bin/bash",
        args=["-c", command],
        envs=env or {},
    )
    if working_dir is not None:
        command_spec.cwd = working_dir

    if user is not None:
        if not user:
            raise ValueError("user must be a non-empty guest username")
        command_spec.user = user

    return _COMMAND_SESSION_START_REQUEST_FACTORY(
        command=command_spec,
        stdin=stdin,
        session_uuid=session_uuid,
        data_offset=data_offset,
    )


def build_command_session_list_request() -> Message:
    return _COMMAND_SESSION_LIST_REQUEST_TYPE()


def build_command_session_connect_request(
    *, session_uuid: str, data_offset: int | None = None
) -> Message:
    return _COMMAND_SESSION_CONNECT_REQUEST_FACTORY(
        session=_COMMAND_SESSION_SELECTOR_FACTORY(session_uuid=session_uuid),
        data_offset=data_offset,
    )


def build_command_session_send_input_request(
    *, session_uuid: str, data: bytes, input_uuid: str
) -> Message:
    return _COMMAND_SESSION_SEND_INPUT_REQUEST_FACTORY(
        session=_COMMAND_SESSION_SELECTOR_FACTORY(session_uuid=session_uuid),
        input=_COMMAND_INPUT_FACTORY(stdin=data),
        input_uuid=input_uuid,
    )


def build_command_session_send_signal_request(
    *, session_uuid: str, signal: Literal["terminate", "kill"], signal_uuid: str
) -> Message:
    signal_value = getattr(
        command_session_pb2,
        "SIGNAL_SIGTERM" if signal == "terminate" else "SIGNAL_SIGKILL",
    )
    return _COMMAND_SESSION_SEND_SIGNAL_REQUEST_FACTORY(
        session=_COMMAND_SESSION_SELECTOR_FACTORY(session_uuid=session_uuid),
        signal=signal_value,
        signal_uuid=signal_uuid,
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


def start_event_data_offset(response: Message) -> Optional[int]:
    """The StartEvent's data_offset echo; None when absent (a server without output replay)."""
    start = cast(_CommandSessionStartResponseLike, response).event.start
    return int(start.data_offset) if start.HasField("data_offset") else None


def collect_command_session_start_event(
    response: Message,
    stdout_parts: bytearray,
    stderr_parts: bytearray,
) -> Optional[int]:
    event = parse_command_session_start_event(response)
    if event is None:
        return None
    kind, value = event
    if kind == "stdout":
        assert isinstance(value, bytes)
        if value:
            stdout_parts.extend(value)
    elif kind == "stderr":
        assert isinstance(value, bytes)
        if value:
            stderr_parts.extend(value)
    elif kind == "end":
        assert isinstance(value, int)
        return value

    return None
