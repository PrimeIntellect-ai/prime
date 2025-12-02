from prime_mcp.client import make_prime_request
from prime_mcp.mcp import mcp
from prime_mcp.tools import availability, pods, ssh

__version__ = "0.1.1"

__all__ = [
    "mcp",
    "make_prime_request",
    "availability",
    "pods",
    "ssh",
]
