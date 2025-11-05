"""Shared pytest configuration and fixtures for sandbox tests"""

import pytest

from prime_sandboxes import APIClient, SandboxClient


@pytest.fixture(scope="session")
def sandbox_client():
    """Create a shared sandbox client for all tests"""
    client = APIClient()
    return SandboxClient(client)
