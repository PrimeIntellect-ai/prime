"""Custom exceptions for Prime Sandboxes SDK."""


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


class SandboxUnresponsiveError(CommandTimeoutError):
    """Raised when sandbox appears running but commands time out unexpectedly."""

    def __init__(
        self,
        sandbox_id: str,
        command: str,
        message: str,
        sandbox_status: str | None = None,
    ):
        self.sandbox_id = sandbox_id
        self.command = command
        self.sandbox_status = sandbox_status
        RuntimeError.__init__(self, message)
