"""CLI-specific API client wrapper that auto-adds User-Agent header."""

import sys
from typing import Optional

from prime_intellect_core import APIClient as CoreAPIClient

# Re-export exceptions and other classes from prime_intellect_core
from prime_intellect_core import (
    APIError,
    APITimeoutError,
    PaymentRequiredError,
    UnauthorizedError,
)
from prime_intellect_core import AsyncAPIClient as CoreAsyncAPIClient

__all__ = [
    "APIClient",
    "AsyncAPIClient",
    "APIError",
    "APITimeoutError",
    "PaymentRequiredError",
    "UnauthorizedError",
]


def _build_user_agent() -> str:
    """Build User-Agent string for prime-cli"""
    from prime_cli import __version__

    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    return f"prime-cli/{__version__} python/{python_version}"


class APIClient(CoreAPIClient):
    """APIClient wrapper that automatically adds prime-cli User-Agent"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        require_auth: bool = True,
        user_agent: Optional[str] = None,
    ):
        # Auto-set user_agent to prime-cli if not provided
        if user_agent is None:
            user_agent = _build_user_agent()
        super().__init__(api_key=api_key, require_auth=require_auth, user_agent=user_agent)


class AsyncAPIClient(CoreAsyncAPIClient):
    """AsyncAPIClient wrapper that automatically adds prime-cli User-Agent"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        require_auth: bool = True,
        user_agent: Optional[str] = None,
    ):
        # Auto-set user_agent to prime-cli if not provided
        if user_agent is None:
            user_agent = _build_user_agent()
        super().__init__(api_key=api_key, require_auth=require_auth, user_agent=user_agent)
