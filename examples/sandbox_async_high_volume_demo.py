#!/usr/bin/env python3

import asyncio
import logging
import os
import time

import httpx
from prime_sandboxes import AsyncSandboxClient, CreateSandboxRequest
from tqdm.asyncio import tqdm

# Configure logging based on environment variable
# Usage: LOG_LEVEL=DEBUG uv run examples/sandbox_async_high_volume_demo.py
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=logging.WARNING,  # Set root logger to WARNING to suppress httpx logs
    format="%(levelname)s - %(message)s",
)

# Enable specified log level only for prime packages
for logger_name in ["prime_sandboxes", "prime_core"]:
    logging.getLogger(logger_name).setLevel(getattr(logging, log_level, logging.INFO))


async def execute_commands(
    sandbox_id: str,
    commands: list[str],
    auth: dict,
    pbar: tqdm,
    semaphore: asyncio.Semaphore,
) -> tuple[int, list[float], int]:
    gateway_url = auth["gateway_url"].rstrip("/")
    url = f"{gateway_url}/{auth['user_ns']}/{auth['job_id']}/exec"
    headers = {"Authorization": f"Bearer {auth['token']}"}

    successful = 0
    failed = 0
    execution_times = []

    async with httpx.AsyncClient(timeout=30.0) as http_client:
        tasks = []
        for command in commands:
            task = execute_single(
                http_client, url, headers, command, sandbox_id, pbar, semaphore
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, tuple):
                success, exec_time = result
                if success:
                    successful += 1
                    execution_times.append(exec_time)
                else:
                    failed += 1
            elif isinstance(result, Exception):
                print(f"Error in sandbox {sandbox_id}: {result}")
                failed += 1

    return successful, execution_times, failed


async def execute_single(
    http_client: httpx.AsyncClient,
    url: str,
    headers: dict,
    command: str,
    sandbox_id: str,
    pbar: tqdm,
    semaphore: asyncio.Semaphore,
) -> tuple[bool, float]:
    async with semaphore:
        payload: dict = {
            "command": command,
            "working_dir": None,
            "env": {},
            "sandbox_id": sandbox_id,
        }

        start = time.time()
        try:
            response = await http_client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            pbar.update(1)
            return True, time.time() - start
        except httpx.HTTPStatusError as e:
            # HTTP error (4xx, 5xx)
            error_detail = f"HTTP {e.response.status_code}"
            try:
                error_body = e.response.json()
                error_detail += f" - {error_body}"
            except Exception:
                error_detail += f" - {e.response.text[:200]}"
            print(
                f"Error executing command '{command}' on sandbox {sandbox_id}: {error_detail}"
            )
            pbar.update(1)
            return False, 0
        except httpx.TimeoutException as e:
            print(
                f"Error executing command '{command}' on sandbox {sandbox_id}: Timeout - {e}"
            )
            pbar.update(1)
            return False, 0
        except httpx.RequestError as e:
            # Connection errors, etc.
            print(
                f"Error executing command '{command}' on sandbox {sandbox_id}: Request failed - {type(e).__name__}: {e}"
            )
            pbar.update(1)
            return False, 0
        except Exception as e:
            # Catch-all for any other errors
            print(
                f"Error executing command '{command}' on sandbox {sandbox_id}: Unexpected error - {type(e).__name__}: {e}"
            )
            pbar.update(1)
            return False, 0


async def main() -> None:
    async with AsyncSandboxClient() as client:
        num_sandboxes = 50
        commands_per_sandbox = 100
        total_commands = num_sandboxes * commands_per_sandbox

        print(f"Sandboxes: {num_sandboxes}")
        print(f"Commands per sandbox: {commands_per_sandbox}")
        print(f"Total commands: {total_commands}")
        print("Target rate: 2000 req/min\n")

        # Create sandboxes
        print("Creating sandboxes...")
        create_start = time.time()
        create_tasks = []
        for i in range(num_sandboxes):
            create_tasks.append(
                client.create(
                    CreateSandboxRequest(
                        name=f"sandbox-{i}",
                        docker_image="python:3.11-slim",
                        start_command="tail -f /dev/null",
                        cpu_cores=1,
                        memory_gb=1,
                        timeout_minutes=10,
                    )
                )
            )
        sandboxes = await asyncio.gather(*create_tasks)
        create_duration = time.time() - create_start
        print(f"Created {len(sandboxes)} sandboxes in {create_duration:.2f}s\n")

        # Wait for sandboxes with status updates
        print("Waiting for sandboxes to be ready...")
        wait_start = time.time()
        sandbox_ids = [s.id for s in sandboxes]

        # Define status callback for progress updates
        def status_callback(
            elapsed_time: float, state_counts: dict, attempt: int
        ) -> None:
            status_str = ", ".join(
                [f"{state}: {count}" for state, count in sorted(state_counts.items())]
            )
            print(f"  [{elapsed_time:.1f}s] {status_str}")

        # Wait for all sandboxes to be ready (with status updates)
        await client.bulk_wait_for_creation(
            sandbox_ids, status_callback=status_callback
        )

        wait_duration = time.time() - wait_start
        total_setup_time = time.time() - create_start
        print(
            f"All sandboxes ready in {wait_duration:.2f}s (total setup: {total_setup_time:.2f}s)\n"
        )

        # Get auth for all sandboxes
        print("Authenticating...")
        auth_tasks = []
        for sandbox in sandboxes:
            auth_tasks.append(
                client.client.request("POST", f"/sandbox/{sandbox.id}/auth")
            )
        auth_responses = await asyncio.gather(*auth_tasks)
        auth_map = {sandboxes[i].id: auth_responses[i] for i in range(len(sandboxes))}

        # Generate commands
        sandbox_commands = {}
        for sandbox in sandboxes:
            cmds = []
            for i in range(commands_per_sandbox):
                if i % 2 == 0:
                    cmds.append(f"echo {i}")
                else:
                    cmds.append(f"python -c 'print({i}*2)'")
            sandbox_commands[sandbox.id] = cmds

        # Rate limiting: 33 concurrent requests for ~2000/min
        semaphore = asyncio.Semaphore(33)

        print(f"Executing {total_commands} commands\n")

        start_time = time.time()

        with tqdm(total=total_commands, desc="Progress", unit="cmd") as pbar:
            tasks = []
            for sandbox in sandboxes:
                tasks.append(
                    execute_commands(
                        sandbox.id,
                        sandbox_commands[sandbox.id],
                        auth_map[sandbox.id],
                        pbar,
                        semaphore,
                    )
                )

            results = await asyncio.gather(*tasks)

        duration = time.time() - start_time

        # Statistics
        total_successful = sum(r[0] for r in results)
        total_failed = sum(r[2] for r in results)
        all_exec_times = []
        for _, exec_times, _ in results:
            all_exec_times.extend(exec_times)

        avg_exec_time = (
            sum(all_exec_times) / len(all_exec_times) if all_exec_times else 0
        )

        print("\n" + "=" * 60)
        print("OVERALL RESULTS")
        print("=" * 60)
        print("\nSetup Phase:")
        print(f"  Sandbox creation API calls: {create_duration:.2f}s")
        print(f"  Waiting for sandboxes to be ready: {wait_duration:.2f}s")
        print(f"  Total setup time: {total_setup_time:.2f}s")
        print(f"  Avg time per sandbox: {total_setup_time / num_sandboxes:.2f}s")

        print("\nExecution Phase:")
        print(f"  Successful: {total_successful}/{total_commands}")
        print(f"  Failed: {total_failed}/{total_commands}")
        print(f"  Duration: {duration:.2f}s")
        rate_per_sec = total_successful / duration
        rate_per_min = rate_per_sec * 60
        print(f"  Rate: {rate_per_sec:.1f} req/s ({rate_per_min:.0f} req/min)")
        print(f"  Avg latency: {avg_exec_time * 1000:.1f}ms")
        print(f"  Success rate: {total_successful / total_commands * 100:.1f}%")

        print(f"\nTotal End-to-End Time: {duration + total_setup_time:.2f}s")

        # Per-sandbox analysis
        print("\n" + "=" * 60)
        print("PER-SANDBOX ANALYSIS")
        print("=" * 60)
        sandboxes_with_failures = []
        for i, (successful, exec_times, failed) in enumerate(results):
            if failed > 0:
                sandbox_id = sandboxes[i].id
                success_rate = (successful / (successful + failed)) * 100
                sandboxes_with_failures.append(
                    (sandbox_id, successful, failed, success_rate)
                )

        if sandboxes_with_failures:
            print(f"Found {len(sandboxes_with_failures)} sandboxes with failures:\n")
            for sandbox_id, successful, failed, success_rate in sandboxes_with_failures[
                :10
            ]:
                print(
                    f"  {sandbox_id}: {successful} ok, {failed} failed ({success_rate:.1f}% success)"
                )
            if len(sandboxes_with_failures) > 10:
                print(
                    f"  ... and {len(sandboxes_with_failures) - 10} more sandboxes with failures"
                )
        else:
            print("All sandboxes completed successfully!")

        # Failure distribution analysis
        if sandboxes_with_failures:
            all_failed = all(
                failed == commands_per_sandbox
                for _, _, failed, _ in sandboxes_with_failures
            )
            if all_failed and len(sandboxes_with_failures) == num_sandboxes:
                print("\n  PATTERN: All sandboxes failing - likely a system-wide issue")
            elif len(sandboxes_with_failures) < num_sandboxes * 0.1:
                print(
                    "\n  PATTERN: Only a few sandboxes failing - likely isolated sandbox issues"
                )
            else:
                print(
                    "\n  PATTERN: Mixed failures - may indicate resource constraints or intermittent issues"
                )

        # Cleanup
        print("\nCleaning up...")
        delete_tasks = [client.delete(s.id) for s in sandboxes]
        await asyncio.gather(*delete_tasks)
        print("Done")


if __name__ == "__main__":
    asyncio.run(main())
