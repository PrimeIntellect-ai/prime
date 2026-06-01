import os

import pytest

from prime_sandboxes import (
    APIClient,
    CreateSandboxRequest,
    SandboxClient,
)


@pytest.mark.skipif(
    not os.environ.get("PRIME_API_KEY"), reason="live sandbox tests require PRIME_API_KEY"
)
def test_command_timeout(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    sandbox = None

    try:
        client = APIClient(api_key=os.environ["PRIME_API_KEY"])
        sandbox_client = SandboxClient(client)

        print("\nCreating sandbox...")
        sandbox = sandbox_client.create(
            CreateSandboxRequest(
                name="test-sandbox",
                docker_image="python:3.11-slim",
                cpu_cores=1,
                memory_gb=2,
                timeout_minutes=60,
            )
        )
        print(f"✓ Created: {sandbox.id}")

        print("\nWaiting for sandbox to be ready...")
        sandbox_client.wait_for_creation(sandbox.id, max_attempts=120)
        print("✓ Sandbox is running!")

        print("\nExecuting command...")
        command = "sleep 100 && echo 'done'"
        result = sandbox_client.execute_command(
            sandbox_id=sandbox.id, command=command, timeout=60 * 2
        )
        print(f"✓ Executed command: {command} with result: {result}")

        assert result.exit_code == 0
        assert result.stdout.strip() == "done"
        assert result.stderr.strip() == ""
    finally:
        if sandbox and sandbox.id:
            print("\nDeleting sandbox...")
            sandbox_client.delete(sandbox.id)
            print(f"✓ Deleted sandbox {sandbox.id}")
