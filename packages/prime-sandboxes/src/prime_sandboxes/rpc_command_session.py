"""Command-session RPC helpers."""

from typing import Dict, List, Literal, Optional, Protocol, Sequence, cast

from connectrpc.code import Code
from connectrpc.errors import ConnectError
from connectrpc.method import IdempotencyLevel, MethodInfo
from google.protobuf.message import Message

from ._connectrpc import PYQWEST_BODY_READ_ERROR_MARKERS
from ._proto.command_session import command_session_pb2


class _CommandSpecLike(Protocol):
    cwd: str


class _CommandSpecFactory(Protocol):
    def __call__(self, *, cmd: str, args: List[str], envs: Dict[str, str]) -> _CommandSpecLike: ...


class _CommandSessionStartRequestFactory(Protocol):
    def __call__(
        self, *, command: _CommandSpecLike, stdin: bool, session_uuid: str | None = None
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
    def __call__(self, *, session: Message) -> Message: ...


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


# The two fault predicates below classify command-session RPC failures for
# retry, with deliberately opposite polarity. Stream re-attach (Connect, or a
# create-or-attach Start resending the identical request) is idempotent, so
# is_recoverable_stream_fault is a deny-list: retry everything except the codes
# command_session.proto promises as definitive answers. A unary control RPC's
# unknown fault may itself be a definitive answer, so is_transient_control_fault
# is an allow-list: fail fast on everything except known link faults.

# Stream faults recovery cannot fix, per command_session.proto's code promises:
# NOT_FOUND (the session is gone or its retention expired) and
# FAILED_PRECONDITION (a Start reusing the session_uuid with a different spec —
# a guard for a future non-identical retry; today's reconnect resends the
# identical request, so the server cannot answer it with a spec conflict).
_STREAM_FATAL_CODES = frozenset({Code.NOT_FOUND, Code.FAILED_PRECONDITION})

# Link faults a unary control RPC may retry; pyqwest body-read faults surface
# as ConnectError INTERNAL and are matched by message marker instead.
_TRANSIENT_CONTROL_CODES = frozenset({Code.DEADLINE_EXCEEDED, Code.UNAVAILABLE})


def is_recoverable_stream_fault(error: BaseException | None) -> bool:
    """Whether a dropped command-session stream may be re-attached (None: clean EOF)."""
    return not (isinstance(error, ConnectError) and error.code in _STREAM_FATAL_CODES)


def is_transient_control_fault(error: ConnectError) -> bool:
    """Whether a unary control-RPC fault is a link hiccup rather than a definitive answer."""
    if error.code in _TRANSIENT_CONTROL_CODES:
        return True
    message = (error.message or "").lower()
    return error.code == Code.INTERNAL and any(
        marker in message for marker in PYQWEST_BODY_READ_ERROR_MARKERS
    )


def build_command_session_start_request(
    *,
    command: str,
    working_dir: Optional[str],
    env: Optional[Dict[str, str]],
    stdin: bool = False,
    session_uuid: str | None = None,
) -> Message:
    command_spec = _COMMAND_SPEC_FACTORY(
        cmd="/bin/bash",
        args=["-c", command],
        envs=env or {},
    )
    if working_dir is not None:
        command_spec.cwd = working_dir

    return _COMMAND_SESSION_START_REQUEST_FACTORY(
        command=command_spec,
        stdin=stdin,
        session_uuid=session_uuid,
    )


def build_command_session_list_request() -> Message:
    return _COMMAND_SESSION_LIST_REQUEST_TYPE()


def build_command_session_connect_request(*, session_uuid: str) -> Message:
    return _COMMAND_SESSION_CONNECT_REQUEST_FACTORY(
        session=_COMMAND_SESSION_SELECTOR_FACTORY(session_uuid=session_uuid)
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
