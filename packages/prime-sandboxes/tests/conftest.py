"""Shared pytest configuration and fixtures for sandbox tests"""

import os
import uuid
from unittest.mock import patch

import pytest

from prime_sandboxes import APIClient, SandboxClient


@pytest.fixture(scope="session")
def sandbox_client(tmp_path_factory):
    """Create a shared sandbox client for live tests with isolated config per worker"""
    api_key = os.environ.get("PRIME_API_KEY")
    if not api_key:
        pytest.skip("live sandbox tests require PRIME_API_KEY")

    # Create a unique config directory for this test worker to avoid file collisions
    worker_id = os.environ.get("PYTEST_XDIST_WORKER", "master")
    config_dir = tmp_path_factory.mktemp(f"prime_config_{worker_id}", numbered=False)

    # Isolate the client config without leaking Path.home() to the rest of pytest.
    with patch("pathlib.Path.home", return_value=config_dir):
        client = APIClient(api_key=api_key)
    yield SandboxClient(client)


@pytest.fixture
def unique_id():
    """Generate a unique ID for each test to avoid collisions in parallel runs"""
    return str(uuid.uuid4())[:8]
