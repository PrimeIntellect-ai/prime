#!/usr/bin/env python3
"""
Simple Sandbox API Demo - shows auth and basic usage
"""

from prime_cli.api.client import APIClient, APIError
from prime_cli.api.sandbox import CreateSandboxRequest, SandboxClient


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
            start_command="python -c 'print(\"Hello World!\"); import time; time.sleep(60)'",
            cpu_cores=1,
            memory_gb=2,
        )

        print("Creating sandbox...")
        sandbox = sandbox_client.create(request)
        print(f"✅ Created: {sandbox.name} ({sandbox.id})")

        # 3. List all sandboxes
        print("\nYour sandboxes:")
        sandbox_list = sandbox_client.list()
        for sb in sandbox_list.sandboxes:
            print(f"  {sb.name}: {sb.status}")

        # 4. Get logs
        print(f"\nLogs for {sandbox.name}:")
        logs = sandbox_client.get_logs(sandbox.id)
        print(logs)

        # 5. Clean up
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
