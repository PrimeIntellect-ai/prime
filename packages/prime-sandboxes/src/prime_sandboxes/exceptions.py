"""Custom exceptions for Prime Sandboxes SDK."""

from typing import Optional

from .core.client import APIError


class SandboxFileNotFoundError(APIError):
    """Raised when a file is not found in the sandbox (404)."""

    pass


class SandboxFileTooLargeError(APIError):
    """Raised when a file exceeds the sandbox read size limit (413).

    Read a bounded window via read_file(offset=..., length=...) or use
    download_file() for the full content.
    """

    pass


class SandboxDiskFullError(APIError):
    """Raised when the sandbox filesystem has no space left (507)."""

    pass


class BatchStatusUnsupportedError(APIError):
    """Raised when batch status is unsupported for the requested sandbox runtime."""

    pass


class SandboxNotRunningError(RuntimeError):
    """Raised when an operation requires a RUNNING sandbox but it is not running."""

    def __init__(
        self,
        sandbox_id: str,
        status: str | None = None,
        error_type: str | None = None,
        command: str | None = None,
        message: str | None = None,
    ):
        self.sandbox_id = sandbox_id
        self.status = status
        self.error_type = error_type
        self.command = command

        if message:
            msg = message
        elif error_type:
            msg = f"Sandbox {sandbox_id} failed ({error_type})"
        elif status:
            msg = f"Sandbox {sandbox_id} is not running (status={status})"
        else:
            msg = f"Sandbox {sandbox_id} is not running"
        super().__init__(msg)


class VolumeSyncFailedError(RuntimeError):
    """Raised when a volume sync ends FAILED instead of DONE.

    failure is the server's failure code (BUSY, PUBLISH_FAILED, MERGE_FAILED,
    DEADLINE, NODE_ERROR or ROLLBACK_FAILED); None when the server gave none.
    """

    def __init__(self, sandbox_id: str, name: str, failure: Optional[str] = None):
        self.sandbox_id = sandbox_id
        self.volume_name = name
        self.failure = failure
        super().__init__(
            f"Volume sync of '{name}' in sandbox {sandbox_id} failed: {failure or 'unknown'}"
        )


class VolumeSyncSupersededError(RuntimeError):
    """Raised when the polled sync is no longer the one this call started.

    Another request began a new sync after this one ended, so this sync's
    outcome can no longer be read; check the volume and sync again.
    """

    def __init__(
        self, sandbox_id: str, name: str, sync_id: Optional[str], current_sync_id: Optional[str]
    ):
        self.sandbox_id = sandbox_id
        self.volume_name = name
        self.sync_id = sync_id
        self.current_sync_id = current_sync_id
        super().__init__(
            f"Volume sync {sync_id} of '{name}' in sandbox {sandbox_id} was replaced by "
            f"sync {current_sync_id} before its result was read"
        )


class CommandTimeoutError(RuntimeError):
    """Raised when a command execution times out."""

    def __init__(self, sandbox_id: str, command: str, timeout: int):
        msg = f"Command '{command}' timed out after {timeout}s in sandbox {sandbox_id}"
        super().__init__(msg)


class UploadTimeoutError(RuntimeError):
    """Raised when a file upload times out."""

    def __init__(self, sandbox_id: str, file_path: str, timeout: int):
        msg = f"Upload to '{file_path}' timed out after {timeout}s in sandbox {sandbox_id}"
        super().__init__(msg)


class DownloadTimeoutError(RuntimeError):
    """Raised when a file download times out."""

    def __init__(self, sandbox_id: str, file_path: str, timeout: int):
        msg = f"Download from '{file_path}' timed out after {timeout}s in sandbox {sandbox_id}"
        super().__init__(msg)


class SandboxOOMError(SandboxNotRunningError):
    """Raised when sandbox fails due to out-of-memory."""

    pass


class SandboxTimeoutError(SandboxNotRunningError):
    """Raised when sandbox times out."""

    pass


class SandboxImagePullError(SandboxNotRunningError):
    """Raised when Docker image cannot be pulled."""

    pass


class ProcessOutputLostError(APIError):
    """Raised when a command's complete output cannot be recovered safely."""
