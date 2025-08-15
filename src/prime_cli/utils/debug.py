"""Debug logging utility for prime-cli."""



class DebugLogger:
    """Simple debug logger that respects the global debug flag."""

    def __init__(self, debug_enabled: bool = False):
        self.debug_enabled = debug_enabled

    def log(self, message: str) -> None:
        """Log a debug message if debug is enabled."""
        if self.debug_enabled:
            print(f"DEBUG: {message}")

    def log_hex(self, label: str, data: bytes) -> None:
        """Log hex representation of data if debug is enabled."""
        if self.debug_enabled:
            print(f"DEBUG: {label} (hex): {data.hex()}")

    def log_ascii(self, label: str, data: bytes) -> None:
        """Log ASCII representation of data if debug is enabled."""
        if self.debug_enabled:
            print(f"DEBUG: {label} (ascii): {data.decode('utf-8', errors='replace')}")


# Global debug logger instance
_debug_logger = DebugLogger()


def set_debug_enabled(enabled: bool) -> None:
    """Set the global debug flag."""
    global _debug_logger
    _debug_logger.debug_enabled = enabled


def debug_log(message: str) -> None:
    """Log a debug message using the global debug logger."""
    _debug_logger.log(message)


def debug_log_hex(label: str, data: bytes) -> None:
    """Log hex representation of data using the global debug logger."""
    _debug_logger.log_hex(label, data)


def debug_log_ascii(label: str, data: bytes) -> None:
    """Log ASCII representation of data using the global debug logger."""
    _debug_logger.log_ascii(label, data)


def is_debug_enabled() -> bool:
    """Check if debug logging is enabled."""
    return _debug_logger.debug_enabled
