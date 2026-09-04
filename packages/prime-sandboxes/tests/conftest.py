"""Shared pytest configuration, fixtures, and gateway RPC test doubles."""

import os
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from prime_sandboxes import APIClient, SandboxClient
from prime_sandboxes._proto.command_session import command_session_pb2 as pb

_EV = pb.CommandSessionEvent


def _auth_payload():
    """Gateway auth payload returned by the fake auth cache."""
    return {
        "gateway_url": "https://gateway.example.com",
        "user_ns": "ns",
        "job_id": "job",
        "token": "tok",
        "expires_at": (datetime.now(timezone.utc) + timedelta(minutes=30)).isoformat(),
    }


class _AsyncFakeCache:
    """Always-VM async auth-cache double for gateway RPC tests."""

    async def get_or_refresh(self, _sandbox_id: str):
        return _auth_payload()

    async def is_vm(self, _sandbox_id: str) -> bool:
        return True


def _start_event(pid, response_type=pb.StartResponse):
    return response_type(event=_EV(start=_EV.StartEvent(pid=pid)))


def _stdout_event(data, response_type=pb.StartResponse):
    return response_type(event=_EV(data=_EV.DataEvent(stdout=data)))


def _end_event(code, response_type=pb.StartResponse):
    return response_type(event=_EV(end=_EV.EndEvent(exit_code=code)))


@pytest.fixture(scope="session")
def sandbox_client(tmp_path_factory):
    """Create a shared sandbox client for all tests with isolated config per worker"""
    # Create a unique config directory for this test worker to avoid file collisions
    worker_id = os.environ.get("PYTEST_XDIST_WORKER", "master")
    config_dir = tmp_path_factory.mktemp(f"prime_config_{worker_id}", numbered=False)

    # Patch Path.home() to return our isolated config directory for this worker
    with patch("pathlib.Path.home", return_value=config_dir):
        client = APIClient()
        yield SandboxClient(client)


@pytest.fixture
def unique_id():
    """Generate a unique ID for each test to avoid collisions in parallel runs"""
    return str(uuid.uuid4())[:8]
