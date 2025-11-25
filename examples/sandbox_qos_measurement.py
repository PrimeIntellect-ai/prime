#!/usr/bin/env python3

import argparse
import asyncio
import sys
import time
from statistics import mean, median, stdev

from prime_sandboxes import AsyncSandboxClient, CreateSandboxRequest


async def measure_sandbox_creation(
    client: AsyncSandboxClient, sandbox_index: int
) -> tuple[str, float, bool]:
    """Create a sandbox and measure time until it's ready.

    Returns:
        Tuple of (sandbox_id, creation_time_seconds, success)
    """
    start_time = time.time()

    try:
        # Create sandbox
        sandbox = await client.create(
            CreateSandboxRequest(
                name=f"qos-test-{sandbox_index}",
                docker_image="python:3.11-slim",
                start_command="tail -f /dev/null",
                cpu_cores=1,
                memory_gb=1,
                timeout_minutes=10,
            )
        )

        # Wait for it to be ready using individual wait (faster polling)
        await client.wait_for_creation(sandbox.id, max_attempts=60)

        creation_time = time.time() - start_time
        return sandbox.id, creation_time, True

    except Exception as e:
        creation_time = time.time() - start_time
        print(f"Error creating sandbox {sandbox_index}: {e}")
        return f"failed-{sandbox_index}", creation_time, False


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure QoS for Prime sandbox creation"
    )
    parser.add_argument(
        "num_sandboxes",
        type=int,
        help="Number of concurrent sandboxes to test",
    )

    args = parser.parse_args()
    num_sandboxes = args.num_sandboxes

    if num_sandboxes <= 0:
        print("Error: Number of sandboxes must be positive")
        sys.exit(1)

    print(f"QoS Measurement: Testing {num_sandboxes} concurrent sandboxes")
    print("=" * 60)

    async with AsyncSandboxClient() as client:
        # Start overall timer
        total_start = time.time()

        # Create all sandboxes concurrently
        print(f"\nCreating {num_sandboxes} sandboxes concurrently...")
        tasks = []
        for i in range(num_sandboxes):
            tasks.append(measure_sandbox_creation(client, i))

        results = await asyncio.gather(*tasks)

        total_duration = time.time() - total_start

        # Separate successful and failed results
        successful_results = [r for r in results if r[2]]
        failed_results = [r for r in results if not r[2]]

        sandbox_ids = [r[0] for r in successful_results]
        creation_times = [r[1] for r in successful_results]

        # Print results
        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)

        print(f"\nTotal duration: {total_duration:.2f}s")
        print(f"Successful: {len(successful_results)}/{num_sandboxes}")
        print(f"Failed: {len(failed_results)}")

        if creation_times:
            print("\nSandbox Creation Times:")
            print(f"  Mean:   {mean(creation_times):.2f}s")
            print(f"  Median: {median(creation_times):.2f}s")
            print(f"  Min:    {min(creation_times):.2f}s")
            print(f"  Max:    {max(creation_times):.2f}s")

            if len(creation_times) > 1:
                print(f"  StdDev: {stdev(creation_times):.2f}s")

            # Percentiles
            sorted_times = sorted(creation_times)
            p50_idx = int(len(sorted_times) * 0.50)
            p90_idx = int(len(sorted_times) * 0.90)
            p95_idx = int(len(sorted_times) * 0.95)
            p99_idx = int(len(sorted_times) * 0.99)

            print("\nPercentiles:")
            print(f"  P50: {sorted_times[p50_idx]:.2f}s")
            print(f"  P90: {sorted_times[p90_idx]:.2f}s")
            print(f"  P95: {sorted_times[p95_idx]:.2f}s")
            print(f"  P99: {sorted_times[p99_idx]:.2f}s")

            # Individual results (optional, show for smaller tests)
            if num_sandboxes <= 20:
                print("\nIndividual Sandbox Times:")
                for i, (sandbox_id, create_time, _) in enumerate(successful_results):
                    print(f"  Sandbox {i}: {create_time:.2f}s (ID: {sandbox_id})")

        # Cleanup
        if sandbox_ids:
            print("\nCleaning up sandboxes...")
            delete_tasks = [client.delete(sid) for sid in sandbox_ids]
            await asyncio.gather(*delete_tasks, return_exceptions=True)
            print("Cleanup complete")

        print("\n" + "=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
