"""Connect RPC runtime compatibility for the SDK's Google protobuf messages."""

from importlib.metadata import PackageNotFoundError, version


def _distribution_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _validate_connectrpc_runtime(
    connectrpc_version: str | None,
    legacy_version: str | None,
    imported_version: str | None,
) -> None:
    """Reject ambiguous or partially upgraded Connect RPC installations."""
    if legacy_version is not None:
        current = connectrpc_version or "not installed"
        raise RuntimeError(
            "Conflicting Connect RPC distributions are installed: "
            f"connectrpc={current} and connect-python={legacy_version}. Both provide the "
            "'connectrpc' Python package. Uninstall connect-python and reinstall "
            "prime-sandboxes."
        )
    if connectrpc_version is None:
        raise RuntimeError(
            "prime-sandboxes requires connectrpc>=0.11.1,<0.12. Reinstall "
            "prime-sandboxes to restore its RPC dependency."
        )
    if imported_version != connectrpc_version:
        raise RuntimeError(
            "The imported Connect RPC runtime does not match the installed distribution "
            f"(imported={imported_version or 'unknown'}, installed={connectrpc_version}). "
            "Recreate the environment to remove stale package files."
        )


_CONNECTRPC_VERSION = _distribution_version("connectrpc")
_LEGACY_CONNECT_PYTHON_VERSION = _distribution_version("connect-python")

# Import only after checking the distribution metadata. The legacy and current
# distributions both install the same module path, so importing first can load
# an arbitrary mixture of files in a partially upgraded environment.
if _LEGACY_CONNECT_PYTHON_VERSION is not None or _CONNECTRPC_VERSION is None:
    _validate_connectrpc_runtime(
        _CONNECTRPC_VERSION,
        _LEGACY_CONNECT_PYTHON_VERSION,
        imported_version=None,
    )

from connectrpc import __version__ as _IMPORTED_CONNECTRPC_VERSION  # noqa: E402
from connectrpc.compat import google_protobuf_binary_codec  # noqa: E402

_validate_connectrpc_runtime(
    _CONNECTRPC_VERSION,
    _LEGACY_CONNECT_PYTHON_VERSION,
    _IMPORTED_CONNECTRPC_VERSION,
)

GOOGLE_PROTOBUF_BINARY_CODEC = google_protobuf_binary_codec()
