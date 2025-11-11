#!/usr/bin/env python3
"""
Simple Sandbox API Demo - shows auth and basic usage

Works with both:
- prime-sandboxes (lightweight SDK):
    pip install prime-sandboxes
    from prime_sandboxes import APIClient, SandboxClient, CreateSandboxRequest

- prime (full CLI + SDK):
    pip install prime
    from prime import APIClient, SandboxClient, CreateSandboxRequest
"""

# Using the modern package structure (works with both prime and prime-sandboxes)
from prime_core import APIClient, APIError
from prime_sandboxes import CreateSandboxRequest, SandboxClient


def main() -> None:
    """Simple sandbox demo"""
    try:
        # 1. Authentication - uses API key from config or environment
        # Run 'prime login' first to set up your API key
        client = APIClient()  # Automatically loads API key from ~/.prime/config.json
        sandbox_client = SandboxClient(client)

        # 2. Create a sandbox
        request = CreateSandboxRequest(
            name="demo-sandbox",
            docker_image="python:3.11-slim",
            start_command="tail -f /dev/null",  # Keep container running indefinitely
            cpu_cores=1,
            memory_gb=2,
            timeout_minutes=120,  # 2 hours to avoid timeout during demo
            environment_vars={"ENV": "demo", "DEBUG": "false"},
            secrets={"API_KEY": "demo-secret-key-12345"},
        )

        print("Creating sandbox...")
        sandbox = sandbox_client.create(request)
        print(f"✅ Created: {sandbox.name} ({sandbox.id})")

        # 3. Wait for sandbox to be running

        print("\nWaiting for sandbox to be running...")
        sandbox_client.wait_for_creation(sandbox.id, max_attempts=60)
        print("✅ Sandbox is running!")

        # 4. Execute commands in the sandbox
        print("\nExecuting commands...")

        # Test basic commands that definitely work
        result = sandbox_client.execute_command(sandbox.id, "whoami")
        print(f"Current user: {result.stdout.strip()}")

        result = sandbox_client.execute_command(sandbox.id, "pwd")
        print(f"Working directory: {result.stdout.strip()}")

        result = sandbox_client.execute_command(sandbox.id, "python --version")
        print(f"Python version: {result.stdout.strip()}")

        # List files in working directory
        result = sandbox_client.execute_command(sandbox.id, "ls -la")
        print(f"Files in working directory:\n{result.stdout}")

        # Test inline Python execution (no file creation needed)
        result = sandbox_client.execute_command(
            sandbox.id, "python -c 'print(\"Hello from sandbox!\")'"
        )
        print(f"Python hello: {result.stdout.strip()}")

        result = sandbox_client.execute_command(
            sandbox.id, "python -c 'print(f\"2 + 2 = {2 + 2}\")'"
        )
        print(f"Math result: {result.stdout.strip()}")

        # Check environment
        result = sandbox_client.execute_command(sandbox.id, "env | grep SANDBOX")
        print(f"Sandbox environment variables:\n{result.stdout}")

        # 5. List all sandboxes
        print("\nYour sandboxes:")
        sandbox_list = sandbox_client.list()
        for sb in sandbox_list.sandboxes:
            print(f"  {sb.name}: {sb.status}")

        # 6. Get logs
        print(f"\nLogs for {sandbox.name}:")
        logs = sandbox_client.get_logs(sandbox.id)
        print(logs)

        # 7. Clean up
        print(f"\nDeleting {sandbox.name}...")
        sandbox_client.delete(sandbox.id)
        print("✅ Deleted")

    except APIError as e:
        print(f"❌ API Error: {e}")
        print("💡 Make sure you're logged in: run 'prime login' first")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
