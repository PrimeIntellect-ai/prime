import pytest
from prime_cli.commands.env import _is_trusted_download_url

BASE = "https://api.primeintellect.ai"


@pytest.mark.parametrize(
    "url,trusted",
    [
        # Same host as the API base.
        ("https://api.primeintellect.ai/api/v1/environmentshub/x/y/@abc/download", True),
        # Sibling host on the same registrable domain (real tracked_package_url host).
        ("https://hub.primeintellect.ai/api/v1/environmentshub/x/y/@abc/download", True),
        # Apex domain.
        ("https://primeintellect.ai/download", True),
        # Presigned object storage carries its own auth; must not receive the token.
        ("https://storage.googleapis.com/bucket/obj?X-Goog-Signature=abc", False),
        # Arbitrary attacker host.
        ("https://evil.example.com/download", False),
        # Lookalike suffix must not match.
        ("https://api.primeintellect.ai.evil.com/download", False),
        # Prefix lookalike (no dot boundary) must not match.
        ("https://evilprimeintellect.ai/download", False),
        # userinfo trick resolves to evil.com as the real host.
        ("https://api.primeintellect.ai@evil.com/download", False),
        # Non-HTTPS is never trusted, even on the right host.
        ("http://api.primeintellect.ai/download", False),
        # Garbage.
        ("not a url", False),
        ("", False),
    ],
)
def test_is_trusted_download_url(url: str, trusted: bool) -> None:
    assert _is_trusted_download_url(url, BASE) is trusted


def test_is_trusted_download_url_respects_configured_base() -> None:
    # A self-hosted/dev base URL trusts its own registrable domain, not prod.
    base = "https://api.staging.example.org"
    assert _is_trusted_download_url("https://hub.staging.example.org/d", base) is True
    assert _is_trusted_download_url("https://api.primeintellect.ai/d", base) is False
