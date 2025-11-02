from prime_sandboxes import (
    APIClient,
    CreateSandboxRequest,
    SandboxClient,
)


def test_command_timeout():
    try:
        client = APIClient()
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
        sandbox_client.wait_for_creation(sandbox.id)
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
