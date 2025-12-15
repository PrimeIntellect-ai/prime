#!/usr/bin/env python3
"""
Test long-running command timeouts (up to 15 minutes).

This tests the increased Cloudflare proxy_read_timeout (900s).
For tasks longer than 15 minutes, use background jobs instead.

Usage:
    python sandbox_long_timeout_test.py [--duration SECONDS]

Examples:
    python sandbox_long_timeout_test.py                # default 5 min test
    python sandbox_long_timeout_test.py --duration 600 # 10 min test
    python sandbox_long_timeout_test.py --duration 900 # full 15 min test
"""

import argparse
import time

from prime_sandboxes import APIError, CreateSandboxRequest, SandboxClient


def main() -> None:
    parser = argparse.ArgumentParser(description="Test long-running sandbox commands")
    parser.add_argument(
        "--duration",
        type=int,
        default=300,
        help="How long the command should run in seconds (default: 300, max: 900)",
    )
    args = parser.parse_args()

    duration = min(args.duration, 900)  # cap at 15 minutes
    timeout = duration + 60  # give some buffer

    print(f"Testing long-running command ({duration}s sleep with {timeout}s timeout)")
    print("For tasks > 15 minutes, use background jobs instead.\n")

    try:
        client = SandboxClient()

        # Create sandbox
        request = CreateSandboxRequest(
            name="timeout-test",
            docker_image="python:3.11-slim",
            timeout_minutes=60,
        )

        print("Creating sandbox...")
        sandbox = client.create(request)
        print(f"Created: {sandbox.id}")

        print("Waiting for sandbox to be ready...")
        client.wait_for_creation(sandbox.id, max_attempts=60)
        print("Sandbox ready!\n")

        # Run long command
        cmd = (
            f"echo 'Starting {duration}s sleep...' && sleep {duration} && echo 'Done!'"
        )
        print(f"Running: {cmd}")
        print(f"Timeout: {timeout}s\n")

        start = time.time()
        result = client.execute_command(sandbox.id, cmd, timeout=timeout)
        elapsed = time.time() - start

        print(f"stdout: {result.stdout}")
        print(f"stderr: {result.stderr}")
        print(f"exit_code: {result.exit_code}")
        print(f"elapsed: {elapsed:.1f}s\n")

        if result.exit_code == 0:
            print("SUCCESS - long-running command completed")
        else:
            print("FAILED - command exited with non-zero status")

        # Cleanup
        print("\nDeleting sandbox...")
        client.delete(sandbox.id)
        print("Done!")

    except APIError as e:
        print(f"API Error: {e}")
        print("Make sure you're logged in: run 'prime login' first")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
