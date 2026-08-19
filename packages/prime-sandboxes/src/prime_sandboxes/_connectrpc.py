"""Connect RPC runtime compatibility for the SDK's Google protobuf messages."""

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

from connectrpc.compat import google_protobuf_binary_codec  # noqa: E402

GOOGLE_PROTOBUF_BINARY_CODEC = google_protobuf_binary_codec()
