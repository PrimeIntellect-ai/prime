"""Shared pytest configuration and fixtures for sandbox tests"""

import uuid

import pytest

from prime_sandboxes import APIClient, SandboxClient


@pytest.fixture(scope="session")
def sandbox_client():
    """Create a shared sandbox client for all tests"""
    client = APIClient()
    return SandboxClient(client)


@pytest.fixture
def unique_id():
    """Generate a unique ID for each test to avoid collisions in parallel runs"""
    return str(uuid.uuid4())[:8]
