#!/usr/bin/env python3
"""
Basic usage example for prime-sandboxes SDK

This demonstrates the standalone SDK without any CLI dependencies.
"""

from prime_sandboxes import (
    APIClient,
    APIError,
    CreateSandboxRequest,
    SandboxClient,
)


def main():
    """Basic sandbox lifecycle example"""
    try:
        # Initialize client from prime login / active config, or from a process-scoped
        # PRIME_API_KEY override. Standalone scripts may also pass api_key directly.
        client = APIClient()
        sandbox_client = SandboxClient(client)

        print("Creating sandbox...")
        sandbox = sandbox_client.create(
            CreateSandboxRequest(
                name="basic-example",
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

        print("\nExecuting commands...")
        result = sandbox_client.execute_command(sandbox.id, "python --version")
        print(f"Python version: {result.stdout.strip()}")

        result = sandbox_client.execute_command(
            sandbox.id, "python -c 'print(\"Hello from sandbox!\")'"
        )
        print(f"Output: {result.stdout.strip()}")

        print("\nCleaning up...")
        sandbox_client.delete(sandbox.id)
        print("✓ Deleted")

    except APIError as e:
        print(f"✗ API Error: {e}")
        print("  Run 'prime login' or pass a process-scoped PRIME_API_KEY for this script")
    except Exception as e:
        print(f"✗ Error: {e}")


if __name__ == "__main__":
    main()
