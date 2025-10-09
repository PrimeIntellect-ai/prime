"""Custom exceptions for Prime Sandboxes SDK."""


class SandboxNotRunningError(RuntimeError):
    """Raised when an operation requires a RUNNING sandbox but it is not running."""

    def __init__(self, sandbox_id: str, status: str | None = None):
        if status:
            msg = f"Sandbox {sandbox_id} is not running (status={status})"
        else:
            msg = f"Sandbox {sandbox_id} is not running"
        super().__init__(msg)


class CommandTimeoutError(RuntimeError):
    """Raised when a command execution times out."""

    def __init__(self, sandbox_id: str, command: str, timeout: int):
        msg = f"Command '{command}' timed out after {timeout}s in sandbox {sandbox_id}"
        super().__init__(msg)
