"""Live process handles for VM sandboxes."""

import asyncio
import contextlib
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, Literal

from connectrpc.client import ConnectClient
from connectrpc.errors import ConnectError
from google.protobuf.message import Message

from .core import APIError
from .rpc_command_session import parse_command_session_start_event

_EOF = object()
_EXIT_WAIT_SECONDS = 5

_WriteStdin = Callable[[int, bytes], Awaitable[None]]
_SendSignal = Callable[[int, Literal["terminate", "kill"]], Awaitable[None]]


class _AsyncProcessStream(AsyncIterator[bytes]):
    """One byte stream produced by an :class:`AsyncSandboxProcess`."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[bytes | BaseException | object] = asyncio.Queue()
        self._closed = False

    def __aiter__(self) -> "_AsyncProcessStream":
        return self

    async def __anext__(self) -> bytes:
        item = await self._queue.get()
        if item is _EOF:
            raise StopAsyncIteration
        if isinstance(item, BaseException):
            raise item
        assert isinstance(item, bytes)
        return item

    def feed(self, data: bytes) -> None:
        if data and not self._closed:
            self._queue.put_nowait(data)

    def fail(self, error: BaseException) -> None:
        if not self._closed:
            self._queue.put_nowait(error)

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._queue.put_nowait(_EOF)


class AsyncSandboxProcess:
    """A live command running in a VM sandbox.

    ``stdout`` and ``stderr`` are independent async byte iterators. Stdin stays
    open until the process exits; the VM command-session protocol currently has
    no stdin-EOF operation, so callers should use their application's graceful
    shutdown message or ``terminate()``/``kill()``.
    """

    def __init__(
        self,
        stream_client: ConnectClient,
        stream: AsyncIterator[Message],
        write_stdin: _WriteStdin,
        send_signal: _SendSignal,
        transport: Any = None,
    ) -> None:
        self.stdout = _AsyncProcessStream()
        self.stderr = _AsyncProcessStream()
        self._stream_client = stream_client
        self._stream = stream
        self._transport = transport
        self._write_stdin = write_stdin
        self._send_process_signal = send_signal
        self._remote_exited = False
        self._signals_sent: set[Literal["terminate", "kill"]] = set()
        self._closed = False
        self._close_lock = asyncio.Lock()
        loop = asyncio.get_running_loop()
        self._started: asyncio.Future[int] = loop.create_future()
        self._exit: asyncio.Future[int] = loop.create_future()
        # Retrieving a future's exception in a callback prevents an un-awaited
        # failed process from producing a noisy "exception was never retrieved"
        # warning; awaiting the future still raises the same exception.
        self._exit.add_done_callback(
            lambda future: future.exception() if not future.cancelled() else None
        )
        self._pump_task = asyncio.create_task(self._pump())

    @classmethod
    async def _create(
        cls,
        stream_client: ConnectClient,
        stream: AsyncIterator[Message],
        write_stdin: _WriteStdin,
        send_signal: _SendSignal,
        transport: Any = None,
    ) -> "AsyncSandboxProcess":
        process = cls(stream_client, stream, write_stdin, send_signal, transport)
        try:
            await asyncio.shield(process._started)
        except asyncio.CancelledError:
            # The Start RPC may already have created the remote process even if
            # its first event has not reached us. Briefly retain the stream so a
            # reported PID can be signalled instead of leaking on cancellation.
            with contextlib.suppress(BaseException):
                await asyncio.wait_for(asyncio.shield(process._started), timeout=5)
            await process.aclose()
            raise
        except BaseException:
            await process.aclose()
            raise
        return process

    @property
    def pid(self) -> int:
        if not self._started.done() or self._started.cancelled():
            raise RuntimeError("process has not started")
        return self._started.result()

    @property
    def returncode(self) -> int | None:
        if not self._exit.done() or self._exit.cancelled():
            return None
        try:
            return self._exit.result()
        except BaseException:
            return None

    async def write_stdin(self, data: bytes) -> None:
        """Write bytes to the process's standard input."""
        if not data:
            return
        if self._closed or self._remote_exited:
            raise BrokenPipeError("process has exited")
        await self._write_stdin(self.pid, data)

    async def wait(self) -> int:
        """Wait for the process to exit and return its exit code."""
        return await asyncio.shield(self._exit)

    async def terminate(self) -> None:
        """Send SIGTERM to the process."""
        await self._send_signal("terminate")

    async def kill(self) -> None:
        """Send SIGKILL to the process."""
        await self._send_signal("kill")

    async def _send_signal(self, signal: Literal["terminate", "kill"]) -> None:
        if self._closed or self._remote_exited:
            return
        await self._send_process_signal(self.pid, signal)
        self._signals_sent.add(signal)

    async def aclose(self) -> None:
        """Stop the process if needed and release its transport."""
        async with self._close_lock:
            if self._closed:
                return

            started = (
                self._started.done()
                and not self._started.cancelled()
                and self._started.exception() is None
            )
            if started and not self._remote_exited:
                if "kill" in self._signals_sent:
                    await self._wait_for_exit_event()
                else:
                    terminate_sent = "terminate" in self._signals_sent
                    if not terminate_sent:
                        try:
                            await self.terminate()
                            terminate_sent = True
                        except Exception:
                            pass
                    if terminate_sent:
                        await self._wait_for_exit_event()

                    if not self._remote_exited:
                        try:
                            await self.kill()
                        except Exception:
                            pass
                        else:
                            await self._wait_for_exit_event()

            if not self._pump_task.done():
                self._pump_task.cancel()
            with contextlib.suppress(BaseException):
                await self._pump_task
            if not self._exit.done():
                self._exit.set_exception(
                    APIError("Process closed before its exit status was observed")
                )
            await self._stream_client.close()
            # Dropping the dedicated transport tears down this process's
            # connection outright, so its gateway stream slot is released
            # even when the stream itself is wedged.
            if self._transport is not None:
                with contextlib.suppress(Exception):
                    await self._transport.aclose()
            self._closed = True

    async def _wait_for_exit_event(self) -> bool:
        if self._remote_exited:
            return True
        if self._exit.done():
            return False
        try:
            await asyncio.wait_for(
                asyncio.shield(self._exit),
                timeout=_EXIT_WAIT_SECONDS,
            )
        except Exception:
            return False
        return self._remote_exited

    async def _pump(self) -> None:
        ended = False
        try:
            async for response in self._stream:
                event = parse_command_session_start_event(response)
                if event is None:
                    continue
                kind, value = event
                if kind == "start":
                    if not self._started.done():
                        self._started.set_result(value)
                elif kind == "stdout":
                    self.stdout.feed(value)
                elif kind == "stderr":
                    self.stderr.feed(value)
                elif kind == "end":
                    ended = True
                    self._remote_exited = True
                    if not self._started.done():
                        raise APIError("Process exited before reporting its PID")
                    if not self._exit.done():
                        self._exit.set_result(value)
                    break
            if not ended:
                raise APIError("Process stream ended without an exit event")
        except asyncio.CancelledError:
            raise
        except BaseException as error:
            if isinstance(error, ConnectError):
                error = APIError(f"process stream RPC failed ({error.code.value}): {error.message}")
            if not self._started.done():
                self._started.set_exception(error)
            elif not self._exit.done():
                self._exit.set_exception(error)
                self.stdout.fail(error)
                self.stderr.fail(error)
        finally:
            self.stdout.close()
            self.stderr.close()
            close_stream = getattr(self._stream, "aclose", None)
            if close_stream is not None:
                with contextlib.suppress(BaseException):
                    await close_stream()
            await self._stream_client.close()
