"""Typed clients for the dev-sandboxes API (/api/v1/dev-sandboxes).

This package is a thin typed client of the platform backend only — it never
runs a local lifecycle engine. Long actions (create/exec/delete) are admitted
as operations; the clients poll ``GET /operations/{id}`` until terminal.
"""

import asyncio
import time
from typing import AsyncIterator, Callable, Iterator, List, Optional

from .core import APIClient, APIError, APIStatusError, AsyncAPIClient
from .errors import DevSandboxError
from .models import (
    CreateEnvironmentRequest,
    CreateEnvironmentResponse,
    DeleteEnvironmentResponse,
    DevEnvironment,
    EnvironmentListResponse,
    ExecCommandRequest,
    ExecCommandResponse,
    Operation,
    Template,
)

__all__ = [
    "DevSandboxClient",
    "AsyncDevSandboxClient",
    "DevSandboxError",
]

#: Default poll interval for operation status while a long action runs.
DEFAULT_POLL_INTERVAL_SECONDS = 5.0

#: Endpoint prefixes under /api/v1.
_ENVIRONMENTS_ENDPOINT = "/dev-sandboxes"
_TEMPLATES_ENDPOINT = "/dev-sandboxes/templates"


def _translate_api_error(error: APIError) -> DevSandboxError:
    """Map a transport-level failure onto the structured taxonomy.

    Only ``APIStatusError`` (a real HTTP error response) carries a payload to
    interpret; other failures surface as transport admission failures.
    """
    if isinstance(error, APIStatusError):
        return DevSandboxError.from_api_error(error)
    return DevSandboxError(code="INVALID_INPUT", stage="api", message=str(error))


class DevSandboxClient:
    """Synchronous typed client for the dev-sandboxes API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        client: Optional[APIClient] = None,
        poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
    ):
        self.api = client or APIClient(api_key=api_key)
        self.poll_interval_seconds = poll_interval_seconds

    # -- environments -----------------------------------------------------

    def create(
        self,
        template: str,
        ref: str,
        ttl_minutes: int = 120,
        team_id: Optional[str] = None,
    ) -> CreateEnvironmentResponse:
        """Admit a new dev environment; returns environment + operation ids."""
        body = CreateEnvironmentRequest(
            template=template,
            ref=ref,
            ttl_minutes=ttl_minutes,
            team_id=team_id,
        )
        try:
            data = self.api.post(_ENVIRONMENTS_ENDPOINT, json=body.model_dump())
        except APIError as e:
            raise _translate_api_error(e) from e
        return CreateEnvironmentResponse.model_validate(data)

    def list(
        self,
        page: int = 1,
        per_page: int = 50,
    ) -> EnvironmentListResponse:
        """List dev environments (paginated)."""
        try:
            data = self.api.get(
                _ENVIRONMENTS_ENDPOINT,
                params={"page": page, "per_page": per_page},
            )
        except APIError as e:
            raise _translate_api_error(e) from e
        return EnvironmentListResponse.model_validate(data)

    def get(self, environment_id: str) -> DevEnvironment:
        """Fetch one dev environment's status."""
        try:
            data = self.api.get(f"{_ENVIRONMENTS_ENDPOINT}/{environment_id}")
        except APIError as e:
            raise _translate_api_error(e) from e
        return DevEnvironment.model_validate(data)

    def delete(self, environment_id: str) -> DeleteEnvironmentResponse:
        """Admit deletion of a dev environment; returns the operation id."""
        try:
            data = self.api.delete(f"{_ENVIRONMENTS_ENDPOINT}/{environment_id}")
        except APIError as e:
            raise _translate_api_error(e) from e
        return DeleteEnvironmentResponse.model_validate(data)

    def exec(
        self,
        environment_id: str,
        command: str,
        timeout_seconds: int = 300,
    ) -> ExecCommandResponse:
        """Admit a command execution (READY environments only)."""
        body = ExecCommandRequest(command=command, timeout_seconds=timeout_seconds)
        try:
            data = self.api.post(
                f"{_ENVIRONMENTS_ENDPOINT}/{environment_id}/exec",
                json=body.model_dump(),
            )
        except APIError as e:
            raise _translate_api_error(e) from e
        return ExecCommandResponse.model_validate(data)

    # -- operations -------------------------------------------------------

    def get_operation(self, operation_id: str) -> Operation:
        """Fetch one operation's status."""
        try:
            data = self.api.get(f"{_ENVIRONMENTS_ENDPOINT}/operations/{operation_id}")
        except APIError as e:
            raise _translate_api_error(e) from e
        return Operation.model_validate(data)

    def iter_operation(
        self,
        operation_id: str,
        poll_interval_seconds: Optional[float] = None,
        timeout_seconds: Optional[float] = None,
    ) -> Iterator[Operation]:
        """Yield operation snapshots until the operation reaches a terminal state.

        The first snapshot is fetched immediately; ``on_update``-style
        consumers can diff consecutive snapshots to stream progress.
        """
        interval = (
            poll_interval_seconds
            if poll_interval_seconds is not None
            else self.poll_interval_seconds
        )
        deadline = time.monotonic() + timeout_seconds if timeout_seconds else None
        while True:
            operation = self.get_operation(operation_id)
            yield operation
            if operation.is_terminal:
                return
            if deadline is not None and time.monotonic() + interval > deadline:
                raise DevSandboxError(
                    code="PROVISION_TIMEOUT",
                    stage="poll",
                    message=f"Timed out waiting for operation {operation_id}",
                )
            time.sleep(interval)

    def wait_for_operation(
        self,
        operation_id: str,
        poll_interval_seconds: Optional[float] = None,
        timeout_seconds: Optional[float] = None,
        on_update: Optional[Callable[[Operation], None]] = None,
    ) -> Operation:
        """Poll an operation until terminal and return the final snapshot."""
        operation: Optional[Operation] = None
        for snapshot in self.iter_operation(
            operation_id,
            poll_interval_seconds=poll_interval_seconds,
            timeout_seconds=timeout_seconds,
        ):
            operation = snapshot
            if on_update is not None:
                on_update(snapshot)
        assert operation is not None
        return operation

    # -- templates --------------------------------------------------------

    def list_templates(self) -> List[Template]:
        """List the read-only environment templates."""
        try:
            data = self.api.get(_TEMPLATES_ENDPOINT)
        except APIError as e:
            raise _translate_api_error(e) from e
        templates = data.get("templates", data) if isinstance(data, dict) else data
        if isinstance(templates, dict):
            templates = templates.get("templates", [])
        return [Template.model_validate(item) for item in templates]


class AsyncDevSandboxClient:
    """Asynchronous typed client for the dev-sandboxes API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        client: Optional[AsyncAPIClient] = None,
        poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
    ):
        self.api = client or AsyncAPIClient(api_key=api_key)
        self.poll_interval_seconds = poll_interval_seconds

    # -- environments -----------------------------------------------------

    async def create(
        self,
        template: str,
        ref: str,
        ttl_minutes: int = 120,
        team_id: Optional[str] = None,
    ) -> CreateEnvironmentResponse:
        body = CreateEnvironmentRequest(
            template=template,
            ref=ref,
            ttl_minutes=ttl_minutes,
            team_id=team_id,
        )
        try:
            data = await self.api.post(_ENVIRONMENTS_ENDPOINT, json=body.model_dump())
        except APIError as e:
            raise _translate_api_error(e) from e
        return CreateEnvironmentResponse.model_validate(data)

    async def list(
        self,
        page: int = 1,
        per_page: int = 50,
    ) -> EnvironmentListResponse:
        try:
            data = await self.api.get(
                _ENVIRONMENTS_ENDPOINT,
                params={"page": page, "per_page": per_page},
            )
        except APIError as e:
            raise _translate_api_error(e) from e
        return EnvironmentListResponse.model_validate(data)

    async def get(self, environment_id: str) -> DevEnvironment:
        try:
            data = await self.api.get(f"{_ENVIRONMENTS_ENDPOINT}/{environment_id}")
        except APIError as e:
            raise _translate_api_error(e) from e
        return DevEnvironment.model_validate(data)

    async def delete(self, environment_id: str) -> DeleteEnvironmentResponse:
        try:
            data = await self.api.delete(f"{_ENVIRONMENTS_ENDPOINT}/{environment_id}")
        except APIError as e:
            raise _translate_api_error(e) from e
        return DeleteEnvironmentResponse.model_validate(data)

    async def exec(
        self,
        environment_id: str,
        command: str,
        timeout_seconds: int = 300,
    ) -> ExecCommandResponse:
        body = ExecCommandRequest(command=command, timeout_seconds=timeout_seconds)
        try:
            data = await self.api.post(
                f"{_ENVIRONMENTS_ENDPOINT}/{environment_id}/exec",
                json=body.model_dump(),
            )
        except APIError as e:
            raise _translate_api_error(e) from e
        return ExecCommandResponse.model_validate(data)

    # -- operations -------------------------------------------------------

    async def get_operation(self, operation_id: str) -> Operation:
        try:
            data = await self.api.get(f"{_ENVIRONMENTS_ENDPOINT}/operations/{operation_id}")
        except APIError as e:
            raise _translate_api_error(e) from e
        return Operation.model_validate(data)

    async def iter_operation(
        self,
        operation_id: str,
        poll_interval_seconds: Optional[float] = None,
        timeout_seconds: Optional[float] = None,
    ) -> AsyncIterator[Operation]:
        interval = (
            poll_interval_seconds
            if poll_interval_seconds is not None
            else self.poll_interval_seconds
        )
        deadline = time.monotonic() + timeout_seconds if timeout_seconds else None
        while True:
            operation = await self.get_operation(operation_id)
            yield operation
            if operation.is_terminal:
                return
            if deadline is not None and time.monotonic() + interval > deadline:
                raise DevSandboxError(
                    code="PROVISION_TIMEOUT",
                    stage="poll",
                    message=f"Timed out waiting for operation {operation_id}",
                )
            await asyncio.sleep(interval)

    async def wait_for_operation(
        self,
        operation_id: str,
        poll_interval_seconds: Optional[float] = None,
        timeout_seconds: Optional[float] = None,
        on_update: Optional[Callable[[Operation], None]] = None,
    ) -> Operation:
        operation: Optional[Operation] = None
        async for snapshot in self.iter_operation(
            operation_id,
            poll_interval_seconds=poll_interval_seconds,
            timeout_seconds=timeout_seconds,
        ):
            operation = snapshot
            if on_update is not None:
                on_update(snapshot)
        assert operation is not None
        return operation

    # -- templates --------------------------------------------------------

    async def list_templates(self) -> List[Template]:
        try:
            data = await self.api.get(_TEMPLATES_ENDPOINT)
        except APIError as e:
            raise _translate_api_error(e) from e
        templates = data.get("templates", data) if isinstance(data, dict) else data
        if isinstance(templates, dict):
            templates = templates.get("templates", [])
        return [Template.model_validate(item) for item in templates]
