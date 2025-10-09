"""Prime Intellect API clients."""

# Import submodules to make them available as attributes
# Re-export from prime_core for backwards compatibility
from prime_core import (
    APIClient,
    APIError,
    APITimeoutError,
    AsyncAPIClient,
    PaymentRequiredError,
    UnauthorizedError,
)

from . import availability, client, inference, pods

__all__ = [
    "APIClient",
    "APIError",
    "APITimeoutError",
    "AsyncAPIClient",
    "PaymentRequiredError",
    "UnauthorizedError",
    "availability",
    "client",
    "inference",
    "pods",
]
