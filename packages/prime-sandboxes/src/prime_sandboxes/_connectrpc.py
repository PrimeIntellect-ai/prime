"""Connect RPC runtime compatibility for the SDK's Google protobuf messages."""

import asyncio
from importlib.metadata import PackageNotFoundError, version


def _legacy_connect_python_version() -> str | None:
    try:
        return version("connect-python")
    except PackageNotFoundError:
        return None


def _reject_legacy_connect_python(legacy_version: str | None) -> None:
    """Reject an upgrade that left the old distribution installed."""
    if legacy_version is not None:
        raise RuntimeError(
            f"Legacy connect-python={legacy_version} is still installed. It shares the "
            "'connectrpc' module path with the required connectrpc distribution. Uninstall "
            "connect-python and reinstall prime-sandboxes."
        )


_reject_legacy_connect_python(_legacy_connect_python_version())

from connectrpc.code import Code  # noqa: E402
from connectrpc.compat import google_protobuf_binary_codec  # noqa: E402
from connectrpc.errors import ConnectError  # noqa: E402

GOOGLE_PROTOBUF_BINARY_CODEC = google_protobuf_binary_codec()


def raise_if_cancelled(error: BaseException) -> None:
    """Re-raise the CancelledError connectrpc wrapped as ConnectError(CANCELED).

    connectrpc's async client converts a cancelled awaiting task into
    ConnectError(CANCELED) with the CancelledError as __cause__. Treating that
    as an RPC failure lets retry loops swallow the cancellation. A CANCELED the
    server sent has no such cause and stays an error.
    """
    if (
        isinstance(error, ConnectError)
        and error.code == Code.CANCELED
        and isinstance(error.__cause__, asyncio.CancelledError)
    ):
        raise error.__cause__


# Response-body read faults raised inside pyqwest's Rust core (string set
# verified against the installed pyqwest 0.8.0 extension binary). They reach
# fault classifiers inside ConnectError INTERNAL messages minted upstream
# (e.g. the gateway); a client-local pyqwest fault on a unary RPC surfaces as
# UNAVAILABLE, already retried by code. Classifiers match the markers
# case-insensitively as substrings to tell this transport-level class apart
# from a real server INTERNAL. "read cancelled" subsumes the longer sibling
# under substring matching; both stay listed as the verified source strings.
PYQWEST_BODY_READ_ERROR_MARKERS = frozenset(
    {
        "error reading content",
        "error reading full content",
        "response body read cancelled",
        "read cancelled",
    }
)
