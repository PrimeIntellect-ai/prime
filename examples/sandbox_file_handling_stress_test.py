#!/usr/bin/env python3
"""
Sandbox File Upload Error Handling & Performance Test

This example demonstrates error handling and performance measurement for file uploads
with various file sizes. It tests the sandbox's ability to handle different file sizes
and measures upload times to identify potential issues.

The script supports three test modes:
1. Sequential uploads - uploads files one at a time (10, 20, 25, 30 MB)
2. Concurrent uploads - uploads 15 files of 5MB each simultaneously using asyncio.gather()
3. Both - runs both sequential and concurrent tests for comparison

This helps identify:
- Error handling for different file sizes with full API error details
- Upload performance and speed
- Behavior under concurrent load (stress test with 15 simultaneous uploads)
- Disk space issues
- HTTP status codes and response bodies from the API

Usage:
    python examples/sandbox_file_handling_stress_test.py [mode]

    mode can be: sequential, concurrent, or both (default: both)

Examples:
    # Run both sequential and concurrent tests
    python examples/sandbox_file_handling_stress_test.py

    # Run only sequential tests
    python examples/sandbox_file_handling_stress_test.py sequential

    # Run only concurrent tests
    python examples/sandbox_file_handling_stress_test.py concurrent
"""

import asyncio
import os
import sys
import tempfile
import time
from typing import Dict, List, Tuple

import httpx

from prime_sandboxes import (
    APIError,
    AsyncSandboxClient,
    Config,
    CreateSandboxRequest,
    PaymentRequiredError,
    UnauthorizedError,
)


def create_test_file(size_mb: int, file_path: str) -> str:
    """Create a test file of specified size in MB.

    Args:
        size_mb: Size of file to create in megabytes
        file_path: Path where to create the file

    Returns:
        Path to the created file
    """
    # Create file with random data
    chunk_size = 1024 * 1024  # 1 MB chunks
    with open(file_path, "wb") as f:
        for _ in range(size_mb):
            # Write 1 MB of data
            f.write(b"x" * chunk_size)

    return file_path


async def test_file_upload(
    client: AsyncSandboxClient, sandbox_id: str, size_mb: int, test_number: int
) -> Tuple[bool, float, str]:
    """Test file upload with specified size and measure performance.

    Args:
        client: Sandbox client
        sandbox_id: ID of the sandbox
        size_mb: Size of file to upload in MB
        test_number: Test number for identification

    Returns:
        Tuple of (success, duration_seconds, error_message)
    """
    temp_file_path = None
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=f"_test_{size_mb}mb.dat", delete=False
        ) as temp_file:
            temp_file_path = temp_file.name

        print(f"\nCreating test file of {size_mb} MB...")
        create_test_file(size_mb, temp_file_path)

        actual_size = os.path.getsize(temp_file_path)
        print(
            f"   Created: {actual_size:,} bytes ({actual_size / (1024 * 1024):.2f} MB)"
        )

        # Measure upload time
        print(f"Uploading file #{test_number} ({size_mb} MB)...")
        start_time = time.time()

        upload_response = await client.upload_file(
            sandbox_id=sandbox_id,
            file_path=f"/sandbox-workspace/test_file_{size_mb}mb_{test_number}.dat",
            local_file_path=temp_file_path,
        )

        end_time = time.time()
        duration = end_time - start_time

        # Calculate upload speed
        speed_mbps = (actual_size / (1024 * 1024)) / duration if duration > 0 else 0

        print("✅ Upload successful!")
        print(f"   Path: {upload_response.path}")
        print(f"   Size: {upload_response.size:,} bytes")
        print(f"   Duration: {duration:.2f} seconds")
        print(f"   Speed: {speed_mbps:.2f} MB/s")

        return True, duration, ""

    except httpx.HTTPStatusError as e:
        end_time = time.time()
        duration = end_time - start_time if "start_time" in locals() else 0

        # Extract full error details from HTTP response
        status_code = e.response.status_code
        response_text = e.response.text

        # Try to parse JSON response for structured error
        try:
            error_json = e.response.json()
            error_detail = error_json.get("detail", response_text)
        except (ValueError, KeyError):
            error_detail = response_text

        error_msg = f"HTTP {status_code}: {error_detail}"
        print("❌ Upload failed!")
        print("   Error Type: HTTP Status Error")
        print(f"   Status Code: {status_code}")
        print(f"   Response Body: {response_text}")
        if error_detail != response_text:
            print(f"   Error Detail: {error_detail}")
        print(f"   Duration before failure: {duration:.2f} seconds")
        return False, duration, error_msg

    except UnauthorizedError as e:
        end_time = time.time()
        duration = end_time - start_time if "start_time" in locals() else 0
        error_msg = f"Unauthorized: {str(e)}"
        print("❌ Upload failed!")
        print("   Error Type: Unauthorized")
        print(f"   Full Error: {str(e)}")
        print(f"   Duration before failure: {duration:.2f} seconds")
        return False, duration, error_msg

    except PaymentRequiredError as e:
        end_time = time.time()
        duration = end_time - start_time if "start_time" in locals() else 0
        error_msg = f"Payment Required: {str(e)}"
        print("❌ Upload failed!")
        print("   Error Type: Payment Required")
        print(f"   Full Error: {str(e)}")
        print(f"   Duration before failure: {duration:.2f} seconds")
        return False, duration, error_msg

    except APIError as e:
        end_time = time.time()
        duration = end_time - start_time if "start_time" in locals() else 0
        error_msg = f"API Error: {str(e)}"
        print("❌ Upload failed!")
        print("   Error Type: API Error")
        print(f"   Full Error: {str(e)}")
        print(f"   Duration before failure: {duration:.2f} seconds")
        return False, duration, error_msg

    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time if "start_time" in locals() else 0
        error_msg = f"{type(e).__name__}: {str(e)}"
        print("❌ Upload failed!")
        print(f"   Error Type: {type(e).__name__}")
        print(f"   Full Error: {str(e)}")

        # If there's additional exception info, show it
        if hasattr(e, "__dict__") and e.__dict__:
            print(f"   Exception Details: {e.__dict__}")

        print(f"   Duration before failure: {duration:.2f} seconds")
        return False, duration, error_msg

    finally:
        # Clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)


async def verify_sandbox_disk_space(
    client: AsyncSandboxClient, sandbox_id: str
) -> None:
    """Check available disk space in the sandbox.

    Args:
        client: Sandbox client
        sandbox_id: ID of the sandbox
    """
    print("\nChecking sandbox disk space...")
    try:
        result = await client.execute_command(sandbox_id, "df -h /sandbox-workspace")
        print("Disk space information:")
        print(result.stdout)

        # Also check file listing
        ls_result = await client.execute_command(
            sandbox_id, "ls -lh /sandbox-workspace/ | grep test_file"
        )
        if ls_result.stdout.strip():
            print("\nUploaded files:")
            print(ls_result.stdout)

    except Exception as e:
        print(f"Failed to check disk space: {e}")


async def run_concurrent_tests(
    client: AsyncSandboxClient, sandbox_id: str, test_sizes: List[int]
) -> List[Dict]:
    """Run multiple file upload tests concurrently.

    Args:
        client: Sandbox client
        sandbox_id: ID of the sandbox
        test_sizes: List of file sizes in MB to test

    Returns:
        List of test results
    """
    print("\n" + "=" * 60)
    print("Starting Concurrent File Upload Tests")
    print("=" * 60)
    print(f"Uploading {len(test_sizes)} files concurrently: {test_sizes} MB")

    # Start all uploads concurrently
    start_time = time.time()

    tasks = [
        test_file_upload(client, sandbox_id, size_mb, i)
        for i, size_mb in enumerate(test_sizes, 1)
    ]

    results = await asyncio.gather(*tasks, return_exceptions=False)

    end_time = time.time()
    total_duration = end_time - start_time

    print(f"\n{'=' * 60}")
    print(f"Concurrent Upload Completed in {total_duration:.2f} seconds")
    print(f"{'=' * 60}")

    # Build result dictionaries
    result_dicts = []
    for i, (success, duration, error) in enumerate(results, 1):
        result_dicts.append(
            {
                "test_number": i,
                "size_mb": test_sizes[i - 1],
                "success": success,
                "duration": duration,
                "error": error,
            }
        )

    return result_dicts


async def run_sequential_tests(
    client: AsyncSandboxClient, sandbox_id: str, test_sizes: List[int]
) -> Tuple[List[Dict], float]:
    """Run file upload tests sequentially (one at a time).

    Args:
        client: Sandbox client
        sandbox_id: ID of the sandbox
        test_sizes: List of file sizes in MB to test

    Returns:
        Tuple of (results list, total duration)
    """
    print("\n" + "=" * 60)
    print("Starting Sequential File Upload Tests")
    print("=" * 60)

    results: List[Dict] = []
    start_time = time.time()

    for i, size_mb in enumerate(test_sizes, 1):
        print(f"\n{'=' * 60}")
        print(f"Test {i}/{len(test_sizes)}: {size_mb} MB file")
        print(f"{'=' * 60}")

        success, duration, error = await test_file_upload(
            client, sandbox_id, size_mb, i
        )

        results.append(
            {
                "test_number": i,
                "size_mb": size_mb,
                "success": success,
                "duration": duration,
                "error": error,
            }
        )

        # Brief pause between tests
        await asyncio.sleep(1)

    total_duration = time.time() - start_time
    return results, total_duration


def print_test_summary(results: List[Dict], total_time: float, mode: str) -> None:
    """Print summary of test results.

    Args:
        results: List of test results
        total_time: Total time taken for all tests
        mode: Test mode ("Sequential" or "Concurrent")
    """
    print("\n" + "=" * 60)
    print(f"{mode} Test Summary")
    print("=" * 60)

    total_tests = len(results)
    successful_tests = sum(1 for r in results if r["success"])
    failed_tests = total_tests - successful_tests

    print(f"\nTotal Tests: {total_tests}")
    print(f"Successful: {successful_tests} ✅")
    print(f"Failed: {failed_tests} ❌")
    print(f"Total Time: {total_time:.2f} seconds")

    print("\nDetailed Results:")
    print(
        f"{'Test':<8} {'Size (MB)':<12} {'Status':<10} {'Duration (s)':<15} {'Speed (MB/s)':<15} {'Error':<50}"
    )
    print("-" * 120)

    for result in results:
        status = "✅ Success" if result["success"] else "❌ Failed"
        speed = (
            result["size_mb"] / result["duration"]
            if result["duration"] > 0 and result["success"]
            else 0
        )
        error_display = result["error"][:50] if result["error"] else "N/A"

        print(
            f"{result['test_number']:<8} "
            f"{result['size_mb']:<12} "
            f"{status:<10} "
            f"{result['duration']:<15.2f} "
            f"{speed:<15.2f} "
            f"{error_display:<50}"
        )

    # Calculate statistics for successful uploads
    successful_results = [r for r in results if r["success"]]
    if successful_results:
        avg_duration = sum(r["duration"] for r in successful_results) / len(
            successful_results
        )
        avg_speed = sum(r["size_mb"] / r["duration"] for r in successful_results) / len(
            successful_results
        )

        print("\nPerformance Statistics:")
        print(f"  Average Duration: {avg_duration:.2f} seconds")
        print(f"  Average Speed: {avg_speed:.2f} MB/s")

    # Identify potential issues
    if failed_tests > 0:
        print(
            f"\nWarning: {failed_tests} test(s) failed. Check error messages above for details."
        )
        print("Possible causes:")
        print("  - Insufficient disk space")
        print("  - Network timeout")
        print("  - File size limits")
        print("  - API rate limiting")


async def main(mode: str = "both") -> None:
    """Run file upload stress tests.

    Args:
        mode: Test mode - "sequential", "concurrent", or "both"
    """
    # Initialize the client
    client = AsyncSandboxClient()

    # Test configuration: file sizes in MB
    sequential_test_sizes = [10, 20, 25, 30]
    concurrent_test_sizes = [5] * 15  # 15 files of 5MB each

    try:
        print("Creating a sandbox for file upload testing...")

        # Create a sandbox with enough disk space
        request = CreateSandboxRequest(
            name="file-upload-stress-test",
            docker_image="python:3.11-slim",
            start_command="tail -f /dev/null",
            cpu_cores=1,
            memory_gb=2,
            disk_size_gb=20,  # Ensure enough disk space for tests
            timeout_minutes=30,
        )

        sandbox = await client.create(request)
        print(f"Sandbox created: {sandbox.id}")

        # Wait for sandbox to be running
        print("Waiting for sandbox to be running...")
        await client.wait_for_creation(sandbox.id)
        print("✅ Sandbox is running!")

        # Check initial disk space
        await verify_sandbox_disk_space(client, sandbox.id)

        # Run tests based on mode
        if mode in ["sequential", "both"]:
            sequential_results, sequential_time = await run_sequential_tests(
                client, sandbox.id, sequential_test_sizes
            )
            await verify_sandbox_disk_space(client, sandbox.id)
            print_test_summary(sequential_results, sequential_time, "Sequential")

        if mode in ["concurrent", "both"]:
            concurrent_results = await run_concurrent_tests(
                client, sandbox.id, concurrent_test_sizes
            )
            # Calculate total concurrent time (max of individual durations)
            concurrent_time = (
                max(r["duration"] for r in concurrent_results)
                if concurrent_results
                else 0
            )
            await verify_sandbox_disk_space(client, sandbox.id)
            print_test_summary(concurrent_results, concurrent_time, "Concurrent")

        # If both modes were run, print comparison
        if mode == "both":
            print("\n" + "=" * 60)
            print("Sequential vs Concurrent Comparison")
            print("=" * 60)
            print(f"Sequential total time: {sequential_time:.2f} seconds")
            print(f"Concurrent total time: {concurrent_time:.2f} seconds")
            if concurrent_time > 0:
                speedup = sequential_time / concurrent_time
                print(f"Speedup: {speedup:.2f}x faster with concurrent uploads")

        # Clean up the sandbox
        print("\nCleaning up sandbox...")
        await client.delete(sandbox.id)
        print("Sandbox cleaned up!")

    except Exception as e:
        print(f"Error during test execution: {e}")
        import traceback

        traceback.print_exc()

    finally:
        await client.aclose()


if __name__ == "__main__":
    print("Sandbox File Upload Error Handling & Performance Test")
    print("=" * 60)

    # Parse command-line arguments
    mode = "both"  # default
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg in ["sequential", "concurrent", "both"]:
            mode = arg
        else:
            print(f"\nError: Invalid mode '{sys.argv[1]}'")
            print("\nUsage:")
            print("  python sandbox_file_handling_stress_test.py [mode]")
            print("\nModes:")
            print("  sequential  - Run tests one at a time")
            print("  concurrent  - Run all tests simultaneously")
            print("  both        - Run both sequential and concurrent tests (default)")
            print("\nExamples:")
            print("  python sandbox_file_handling_stress_test.py")
            print("  python sandbox_file_handling_stress_test.py sequential")
            print("  python sandbox_file_handling_stress_test.py concurrent")
            exit(1)

    # Check configuration
    config = Config()
    if not config.api_key:
        print("Error: No API key found. Please run 'prime auth login' first.")
        exit(1)

    print(f"Using API endpoint: {config.base_url}")
    if config.team_id:
        print(f"Team: {config.team_id}")
    print(f"Test mode: {mode}")

    # Run the test
    asyncio.run(main(mode))

    print("\nTest completed!")
