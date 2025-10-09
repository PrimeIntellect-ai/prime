#!/usr/bin/env python3
"""
Async Sandbox API Demo - shows the improved async developer experience
"""

import asyncio

from prime_core import APIError
from prime_sandboxes import AsyncSandboxClient, CreateSandboxRequest


async def main() -> None:
    """Async sandbox demo showing much cleaner async code"""
    try:
        # 1. Authentication - uses API key from config or environment
        # Run 'prime login' first to set up your API key
        async with AsyncSandboxClient() as sandbox_client:
            # 2. Create multiple sandboxes concurrently
            print("Creating multiple sandboxes concurrently...")

            sandbox_requests = [
                CreateSandboxRequest(
                    name="async-demo-python-1",
                    docker_image="python:3.11-slim",
                    start_command="tail -f /dev/null",
                    cpu_cores=1,
                    memory_gb=1,
                    timeout_minutes=120,
                ),
                CreateSandboxRequest(
                    name="async-demo-node-2",
                    docker_image="node:20-slim",
                    start_command="tail -f /dev/null",
                    cpu_cores=1,
                    memory_gb=1,
                    timeout_minutes=120,
                ),
            ]

            # Create sandboxes concurrently
            sandboxes = await asyncio.gather(
                *[sandbox_client.create(request) for request in sandbox_requests]
            )

            print(f"✅ Created {len(sandboxes)} sandboxes concurrently:")
            for sandbox in sandboxes:
                print(f"  - {sandbox.name} ({sandbox.id})")

            # 3. Wait for all sandboxes to be running concurrently
            print("\nWaiting for all sandboxes to be running...")
            await asyncio.gather(
                *[
                    sandbox_client.wait_for_creation(sandbox.id, max_attempts=60)
                    for sandbox in sandboxes
                ]
            )
            print("✅ All sandboxes are running!")

            # 4. Execute commands in parallel across sandboxes
            print("\nExecuting commands in parallel...")

            commands = []
            if len(sandboxes) >= 1:
                commands.extend(
                    [
                        (sandboxes[0].id, "python --version"),
                        (
                            sandboxes[0].id,
                            "python -c 'print(\"Hello from Python sandbox!\")'",
                        ),
                    ]
                )
            if len(sandboxes) >= 2:
                commands.extend(
                    [
                        (sandboxes[1].id, "node --version"),
                        (sandboxes[1].id, "echo 'Hello from Node sandbox!'"),
                    ]
                )

            # Execute all commands concurrently
            results = await asyncio.gather(
                *[
                    sandbox_client.execute_command(sandbox_id, command)
                    for sandbox_id, command in commands
                ]
            )

            for (sandbox_id, command), result in zip(commands, results):
                sandbox_name = next(s.name for s in sandboxes if s.id == sandbox_id)
                print(f"[{sandbox_name}] {command}: {result.stdout.strip()}")

            # 5. Get logs from all sandboxes concurrently
            print("\nFetching logs from all sandboxes...")
            all_logs = await asyncio.gather(
                *[sandbox_client.get_logs(sandbox.id) for sandbox in sandboxes]
            )

            for sandbox, logs in zip(sandboxes, all_logs):
                print(f"\n[{sandbox.name}] Logs:")
                print(logs[:200] + "..." if len(logs) > 200 else logs)

            # 6. List all sandboxes
            print("\nListing all sandboxes...")
            sandbox_list = await sandbox_client.list()
            print(f"Total sandboxes: {sandbox_list.total}")

            # 7. Clean up all sandboxes concurrently
            print(f"\nDeleting {len(sandboxes)} sandboxes concurrently...")
            await asyncio.gather(
                *[sandbox_client.delete(sandbox.id) for sandbox in sandboxes]
            )
            print("✅ All sandboxes deleted!")

    except APIError as e:
        print(f"❌ API Error: {e}")
        print("💡 Make sure you're logged in: run 'prime login' first")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
