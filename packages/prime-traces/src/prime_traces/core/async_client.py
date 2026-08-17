"""Async HTTP client for the Prime Traces service.

The asyncio counterpart of ``core.client.TracesAPIClient``. It shares that
module's configuration, response mapping and retry classification, so the two
transports cannot disagree about what a response means or when a replay is
safe; only the awaiting differs.

Two kinds of work are handed to worker threads rather than run on the event
loop: gzipping an upload body (a batch is tens of MiB, and compressing one
inline would stall every other task for as long as it takes) and writing a
streamed download to disk. Everything else here is I/O the loop can await.
"""

import asyncio
import gzip as gzip_module
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, Optional

import httpx

from ..exceptions import AmbiguousDeleteError, APIError, RetryableAPIError
from ..models import LineFormat
from .client import (
    AMBIGUOUS_TRANSPORT_ERRORS,
    DEFAULT_TIMEOUT,
    IDEMPOTENT_RETRY_ATTEMPTS,
    BaseTracesAPIClient,
    is_ambiguous_delete_failure,
    raise_for_response,
    retry_delay,
)


async def retry_sleep(seconds: float) -> None:
    """Wait out a retry delay.

    A named indirection rather than a bare ``asyncio.sleep`` call so tests can
    record the delays this SDK chooses without patching ``asyncio`` for every
    other coroutine in the process.
    """
    await asyncio.sleep(seconds)


async def _run_async_cleanup_safely(operation: Callable[[], Awaitable[None]]) -> None:
    """Finish async resource cleanup before propagating cancellation.

    A caller may cancel a task again while its first cancellation is already
    unwinding. Run cleanup in a shielded task and wait through repeated
    cancellation requests so an HTTP response cannot remain checked out from
    the connection pool.
    """
    operation_task = asyncio.create_task(operation())
    try:
        await asyncio.shield(operation_task)
    except asyncio.CancelledError:
        while not operation_task.done():
            try:
                await asyncio.shield(operation_task)
            except asyncio.CancelledError:
                continue
            except BaseException:
                break

        if not operation_task.cancelled():
            try:
                operation_task.result()
            except BaseException:
                # Cancellation remains caller-visible. Retrieving a concurrent
                # close error also prevents an unobserved-task warning.
                pass
        raise


class AsyncTracesAPIClient(BaseTracesAPIClient):
    """Thin asynchronous client for the Prime Traces REST API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        team_id: Optional[str] = None,
        user_agent: Optional[str] = None,
        timeout: Optional[httpx.Timeout] = None,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ):
        super().__init__(api_key=api_key, base_url=base_url, team_id=team_id)
        self.client = httpx.AsyncClient(
            headers=self._default_headers(user_agent),
            follow_redirects=True,
            timeout=timeout or DEFAULT_TIMEOUT,
            transport=transport,
        )

    # -- write path ---------------------------------------------------------

    async def upload_batch(
        self,
        body: bytes,
        idempotency_key: str,
        *,
        line_format: LineFormat = LineFormat.TRACE,
        schema_version: int = 1,
        context: Optional[Dict[str, str]] = None,
        compress: bool = True,
    ) -> Dict[str, Any]:
        """POST one content-addressed JSONL request. See the sync client."""
        self._check_auth()

        # mtime=0 keeps the compressed bytes reproducible across retries.
        compressed = (
            await asyncio.to_thread(gzip_module.compress, body, mtime=0) if compress else None
        )
        files = self._multipart_files(
            body, schema_version=schema_version, context=context, compressed=compressed
        )

        try:
            response = await self.client.post(
                self._url("/traces"),
                headers=self._upload_headers(idempotency_key, line_format),
                files=files,
            )
        except Exception as exc:
            raise self._wrap_transport_errors(exc) from exc

        return self._upload_result(response)

    # -- read path ----------------------------------------------------------

    async def _idempotent_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]],
    ) -> httpx.Response:
        """Send with bounded retries when replay is known to be safe.

        Same policy as the sync client: GET retries every transient failure,
        DELETE retries only failures that could not have reached the service
        plus explicit service refusals, and anything that may hide a completed
        deletion surfaces as ``AmbiguousDeleteError``.
        """
        is_delete = method.upper() == "DELETE"
        for attempt in range(IDEMPOTENT_RETRY_ATTEMPTS):
            last = attempt == IDEMPOTENT_RETRY_ATTEMPTS - 1
            try:
                response = await self.client.request(method, self._url(endpoint), params=params)
            except Exception as exc:
                error = self._wrap_transport_errors(exc)
                if error is exc:
                    raise
                if is_delete and isinstance(exc, AMBIGUOUS_TRANSPORT_ERRORS):
                    raise AmbiguousDeleteError(f"Delete outcome is unknown: {error}") from exc
                if last:
                    raise error from exc
                await retry_sleep(retry_delay(error, attempt))
                continue
            try:
                raise_for_response(response)
            except RetryableAPIError as exc:
                if is_delete and is_ambiguous_delete_failure(exc):
                    raise AmbiguousDeleteError(
                        f"Delete outcome is unknown after HTTP {exc.status_code}: {exc}",
                        status_code=exc.status_code,
                        code=exc.code,
                    ) from exc
                if last:
                    raise
                await retry_sleep(retry_delay(exc, attempt))
                continue
            return response
        raise AssertionError("unreachable")  # pragma: no cover

    async def get_json(
        self, endpoint: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        self._check_auth()
        response = await self._idempotent_request("GET", endpoint, params)
        result = response.json()
        if not isinstance(result, dict):
            raise APIError("API response was not a dictionary")
        return result

    async def delete(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> None:
        """Send a DELETE and discard the body — 202 carries none.

        See the sync client: an ambiguous failure that may hide a completed
        deletion is returned to the caller rather than replayed.
        """
        self._check_auth()
        await self._idempotent_request("DELETE", endpoint, params)

    async def stream_bytes(
        self, endpoint: str, params: Optional[Dict[str, Any]] = None
    ) -> AsyncIterator[bytes]:
        """Stream a response body in chunks (a raw trace can be 64 MiB).

        As in the sync client, transient failures are retried only until the
        first body byte has been yielded: a mid-stream retry would silently
        restart the body under the consumer.
        """
        self._check_auth()
        for attempt in range(IDEMPOTENT_RETRY_ATTEMPTS):
            last = attempt == IDEMPOTENT_RETRY_ATTEMPTS - 1
            yielded = False
            try:
                # Own the response directly instead of using ``client.stream``
                # so its close can be protected from repeated cancellation.
                request = self.client.build_request("GET", self._url(endpoint), params=params)
                response = await self.client.send(request, stream=True)
                try:
                    if not response.is_success:
                        await response.aread()
                        raise_for_response(response)
                    async for chunk in response.aiter_bytes():
                        yielded = True
                        yield chunk
                finally:
                    await _run_async_cleanup_safely(response.aclose)
                return
            except APIError as exc:
                # Already typed by raise_for_response.
                if yielded or last or not isinstance(exc, RetryableAPIError):
                    raise
                await retry_sleep(retry_delay(exc, attempt))
            except Exception as exc:
                error = self._wrap_transport_errors(exc)
                if error is exc:
                    raise
                if yielded or last:
                    raise error from exc
                await retry_sleep(retry_delay(error, attempt))

    async def aclose(self) -> None:
        await self.client.aclose()

    async def __aenter__(self) -> "AsyncTracesAPIClient":
        return self

    async def __aexit__(self, *exc_info) -> None:
        await self.aclose()
