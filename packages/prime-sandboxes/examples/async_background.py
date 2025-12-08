#!/usr/bin/env python3
"""Async sandbox example showing background job usage."""

import asyncio
from typing import Optional

from prime_sandboxes import (
    APIError,
    AsyncSandboxClient,
    BackgroundJobStatus,
    CreateSandboxRequest,
)


async def run_background_count_example() -> None:
    """Create a sandbox, run a counting job asynchronously, and poll status."""
    sandbox_client = AsyncSandboxClient()
    sandbox_id: Optional[str] = None

    try:
        sandbox_id = await create_ready_sandbox(sandbox_client)
        result = await execute_with_status_updates(sandbox_client, sandbox_id)
        print("\nBackground job finished!")
        print(f"exit_code: {result.exit_code}")
        print("stdout:\n" + (result.stdout or "").strip())
    finally:
        await cleanup_sandbox(sandbox_client, sandbox_id)


async def create_ready_sandbox(sandbox_client: AsyncSandboxClient) -> str:
    """Provision a sandbox and wait until it is running."""
    print("Creating sandbox for async background example...")
    sandbox = await sandbox_client.create(
        CreateSandboxRequest(
            name="async-background-example",
            docker_image="python:3.11-slim",
            cpu_cores=1,
            memory_gb=2,
            timeout_minutes=60,
        )
    )
    print(f"✓ Created sandbox {sandbox.id}")

    print("Waiting for sandbox to be ready...")
    await sandbox_client.wait_for_creation(sandbox.id)
    print("✓ Sandbox is running")
    return sandbox.id


async def execute_with_status_updates(
    sandbox_client: AsyncSandboxClient, sandbox_id: str
) -> BackgroundJobStatus:
    """Start a background job and poll for completion with status updates."""
    # This command counts up once per second to simulate a long-running task.
    background_command = "for i in $(seq 1 15); do echo Background counter: $i; sleep 1; done"

    print("\nStarting async background job...")
    job = await sandbox_client.start_background_job(sandbox_id, background_command)
    print(f"✓ Job started: {job.job_id}")

    checks = 0
    while True:
        status = await sandbox_client.get_background_job(sandbox_id, job)
        if status.completed:
            return status
        checks += 1
        print(f"[status] Still running... check #{checks}")
        await asyncio.sleep(3)


async def cleanup_sandbox(sandbox_client: AsyncSandboxClient, sandbox_id: Optional[str]) -> None:
    """Delete sandbox if it exists."""
    if not sandbox_id:
        return
    try:
        print("\nCleaning up sandbox...")
        await sandbox_client.delete(sandbox_id)
        print("✓ Sandbox deleted")
    except APIError as exc:
        print(f"⚠ Could not delete sandbox: {exc}")


async def main() -> None:
    """Entrypoint for the async example."""
    try:
        await run_background_count_example()
    except APIError as exc:
        print(f"✗ API error: {exc}")
        print("  Make sure PRIME_API_KEY is set or run 'prime login' before executing.")
    except Exception as exc:
        print(f"✗ Unexpected error: {exc}")


if __name__ == "__main__":
    asyncio.run(main())
