#!/usr/bin/env python3
"""
Multi-Sandbox File Upload Stress Test

This comprehensive stress test evaluates file upload performance and reliability across
multiple sandboxes with various file sizes. It combines the patterns from both the
single-sandbox file upload test and the high-volume command execution test.

Features:
- Creates multiple sandboxes concurrently
- Each sandbox gets a DIFFERENT file size pattern for realistic testing
- Tests both concurrent uploads within each sandbox and parallel uploads across sandboxes
- Measures performance metrics (upload speed, success rate, latency)
- Verifies uploaded files exist and have correct sizes
- Provides detailed per-sandbox and aggregate statistics
- Proper error handling and cleanup

Test Patterns (different sandboxes use different patterns):
0. Many small files - 10x 1MB files
1. Small to medium - Mixed 1-5MB files
2. Medium files - 5-10MB files
3. Large files - 10-30MB files
4. Maximum size files - 4x 30MB files (max allowed size)
5. Mixed sizes - Full range 1-30MB
6. Rapid small - 20x 1MB files (stress test)
7. Medium rapid - 10x 5MB files
8. Large batch - Multiple 10-30MB files
9. Extreme contrast - Mix of 1MB and 30MB files

Usage:
    python examples/sandbox_multi_upload_stress_test.py [num_sandboxes] [concurrency]

Examples:
    # Default: 10 sandboxes, 20 concurrent uploads
    python examples/sandbox_multi_upload_stress_test.py

    # Custom: 5 sandboxes, 15 concurrent uploads
    python examples/sandbox_multi_upload_stress_test.py 5 15

    # Heavy load: 20 sandboxes, 30 concurrent uploads
    python examples/sandbox_multi_upload_stress_test.py 20 30
"""

import asyncio
import logging
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import List, Dict, Tuple

from prime_sandboxes import AsyncSandboxClient, CreateSandboxRequest
from prime_core import Config
from tqdm.asyncio import tqdm
import httpx


# Configure logging
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s - %(message)s",
)
for logger_name in ["prime_sandboxes", "prime_core"]:
    logging.getLogger(logger_name).setLevel(getattr(logging, log_level, logging.INFO))


@dataclass
class UploadResult:
    """Result of a single file upload."""

    sandbox_id: str
    file_size_mb: int
    file_number: int
    success: bool
    duration: float
    error: str = ""
    upload_speed_mbps: float = 0.0


@dataclass
class SandboxStats:
    """Statistics for a single sandbox."""

    sandbox_id: str
    total_uploads: int
    successful_uploads: int
    failed_uploads: int
    total_bytes_uploaded: int
    total_duration: float
    avg_speed_mbps: float
    success_rate: float


def create_test_file(size_mb: int) -> str:
    """Create a temporary test file of specified size.

    Args:
        size_mb: Size of file to create in megabytes

    Returns:
        Path to the created temporary file
    """
    temp_file = tempfile.NamedTemporaryFile(
        mode="wb", suffix=f"_{size_mb}mb.dat", delete=False
    )
    temp_file_path = temp_file.name

    # Write file in chunks for efficiency
    chunk_size = 1024 * 1024  # 1 MB
    for _ in range(size_mb):
        temp_file.write(b"x" * chunk_size)

    temp_file.close()
    return temp_file_path


async def upload_file_to_sandbox(
    client: AsyncSandboxClient,
    sandbox_id: str,
    file_size_mb: int,
    file_number: int,
    semaphore: asyncio.Semaphore,
    pbar: tqdm,
) -> UploadResult:
    """Upload a single file to a sandbox.

    Args:
        client: Sandbox client
        sandbox_id: ID of the sandbox
        file_size_mb: Size of file to upload in MB
        file_number: File number for identification
        semaphore: Semaphore for rate limiting
        pbar: Progress bar

    Returns:
        UploadResult with upload statistics
    """
    temp_file_path = None

    async with semaphore:
        try:
            # Create test file
            temp_file_path = create_test_file(file_size_mb)
            actual_size = os.path.getsize(temp_file_path)

            # Upload file
            start_time = time.time()
            remote_path = f"/sandbox-workspace/test_{file_size_mb}mb_{file_number}.dat"

            await client.upload_file(
                sandbox_id=sandbox_id,
                file_path=remote_path,
                local_file_path=temp_file_path,
            )

            duration = time.time() - start_time
            speed_mbps = (actual_size / (1024 * 1024)) / duration if duration > 0 else 0

            pbar.update(1)

            return UploadResult(
                sandbox_id=sandbox_id,
                file_size_mb=file_size_mb,
                file_number=file_number,
                success=True,
                duration=duration,
                upload_speed_mbps=speed_mbps,
            )

        except httpx.HTTPStatusError as e:
            status_code = e.response.status_code
            try:
                error_detail = e.response.json().get("detail", e.response.text)
            except (ValueError, KeyError):
                error_detail = e.response.text

            error_msg = f"HTTP {status_code}: {error_detail}"
            pbar.update(1)

            return UploadResult(
                sandbox_id=sandbox_id,
                file_size_mb=file_size_mb,
                file_number=file_number,
                success=False,
                duration=0,
                error=error_msg,
            )

        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            pbar.update(1)

            return UploadResult(
                sandbox_id=sandbox_id,
                file_size_mb=file_size_mb,
                file_number=file_number,
                success=False,
                duration=0,
                error=error_msg,
            )

        finally:
            # Clean up temporary file
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                except Exception:
                    pass


async def verify_uploads(
    client: AsyncSandboxClient,
    sandbox_id: str,
    expected_files: List[Tuple[int, int]],  # List of (size_mb, file_number)
) -> Dict[str, any]:
    """Verify that uploaded files exist and have correct sizes.

    Args:
        client: Sandbox client
        sandbox_id: ID of the sandbox
        expected_files: List of (size_mb, file_number) tuples

    Returns:
        Dictionary with verification results
    """
    try:
        # List files in sandbox
        result = await client.execute_command(
            sandbox_id, "ls -lh /sandbox-workspace/ | grep test_"
        )

        files_found = (
            len(result.stdout.strip().split("\n")) if result.stdout.strip() else 0
        )
        expected_count = len(expected_files)

        # Check disk usage
        df_result = await client.execute_command(
            sandbox_id, "df -h /sandbox-workspace | tail -1 | awk '{print $3, $4, $5}'"
        )

        disk_info = df_result.stdout.strip().split()

        return {
            "files_found": files_found,
            "files_expected": expected_count,
            "verification_passed": files_found == expected_count,
            "disk_used": disk_info[0] if len(disk_info) > 0 else "N/A",
            "disk_available": disk_info[1] if len(disk_info) > 1 else "N/A",
            "disk_usage_pct": disk_info[2] if len(disk_info) > 2 else "N/A",
        }

    except Exception as e:
        return {
            "files_found": 0,
            "files_expected": len(expected_files),
            "verification_passed": False,
            "error": str(e),
        }


async def upload_files_to_sandbox(
    client: AsyncSandboxClient,
    sandbox_id: str,
    file_sizes: List[int],
    semaphore: asyncio.Semaphore,
    pbar: tqdm,
) -> List[UploadResult]:
    """Upload multiple files to a single sandbox.

    Args:
        client: Sandbox client
        sandbox_id: ID of the sandbox
        file_sizes: List of file sizes in MB to upload
        semaphore: Semaphore for rate limiting
        pbar: Progress bar

    Returns:
        List of UploadResult objects
    """
    tasks = []
    for i, size_mb in enumerate(file_sizes, 1):
        task = upload_file_to_sandbox(client, sandbox_id, size_mb, i, semaphore, pbar)
        tasks.append(task)

    return await asyncio.gather(*tasks, return_exceptions=False)


def calculate_sandbox_stats(results: List[UploadResult]) -> SandboxStats:
    """Calculate statistics for a sandbox from its upload results.

    Args:
        results: List of UploadResult objects

    Returns:
        SandboxStats object
    """
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    total_bytes = sum(r.file_size_mb * 1024 * 1024 for r in successful)
    total_duration = sum(r.duration for r in results)
    avg_speed = (
        sum(r.upload_speed_mbps for r in successful) / len(successful)
        if successful
        else 0
    )

    return SandboxStats(
        sandbox_id=results[0].sandbox_id if results else "unknown",
        total_uploads=len(results),
        successful_uploads=len(successful),
        failed_uploads=len(failed),
        total_bytes_uploaded=total_bytes,
        total_duration=total_duration,
        avg_speed_mbps=avg_speed,
        success_rate=(len(successful) / len(results) * 100) if results else 0,
    )


def print_results(
    sandbox_stats: List[SandboxStats],
    all_results: List[UploadResult],
    total_duration: float,
    setup_duration: float,
    sandbox_patterns: Dict[str, int] = None,
) -> None:
    """Print comprehensive test results.

    Args:
        sandbox_stats: List of SandboxStats objects
        all_results: List of all UploadResult objects
        total_duration: Total test duration
        setup_duration: Setup phase duration
        sandbox_patterns: Mapping of sandbox_id to pattern index
    """
    if sandbox_patterns is None:
        sandbox_patterns = {}
    print("\n" + "=" * 80)
    print("MULTI-SANDBOX FILE UPLOAD STRESS TEST RESULTS")
    print("=" * 80)

    # Overall statistics
    total_uploads = sum(s.total_uploads for s in sandbox_stats)
    total_successful = sum(s.successful_uploads for s in sandbox_stats)
    total_failed = sum(s.failed_uploads for s in sandbox_stats)
    total_bytes = sum(s.total_bytes_uploaded for s in sandbox_stats)
    overall_success_rate = (
        (total_successful / total_uploads * 100) if total_uploads > 0 else 0
    )

    print("\n📊 OVERALL STATISTICS")
    print("-" * 80)
    print(f"Total Sandboxes:      {len(sandbox_stats)}")
    print(f"Total Uploads:        {total_uploads}")
    print(f"Successful:           {total_successful} ✅")
    print(f"Failed:               {total_failed} ❌")
    print(f"Success Rate:         {overall_success_rate:.1f}%")
    print(
        f"Total Data Uploaded:  {total_bytes / (1024 * 1024):.2f} MB ({total_bytes / (1024 * 1024 * 1024):.2f} GB)"
    )
    print(f"Setup Time:           {setup_duration:.2f}s")
    print(f"Upload Time:          {total_duration:.2f}s")
    print(f"Total Test Time:      {setup_duration + total_duration:.2f}s")

    # Calculate throughput
    if total_duration > 0:
        throughput_mbps = (total_bytes / (1024 * 1024)) / total_duration
        upload_rate = total_successful / total_duration
        print(f"Overall Throughput:   {throughput_mbps:.2f} MB/s")
        print(
            f"Upload Rate:          {upload_rate:.2f} uploads/s ({upload_rate * 60:.0f} uploads/min)"
        )

    # Performance by file size
    print("\n📈 PERFORMANCE BY FILE SIZE")
    print("-" * 80)
    size_groups = {}
    for result in all_results:
        if result.success:
            if result.file_size_mb not in size_groups:
                size_groups[result.file_size_mb] = []
            size_groups[result.file_size_mb].append(result)

    print(
        f"{'Size (MB)':<12} {'Count':<10} {'Avg Speed (MB/s)':<20} {'Avg Duration (s)':<20}"
    )
    print("-" * 80)
    for size_mb in sorted(size_groups.keys()):
        results = size_groups[size_mb]
        avg_speed = sum(r.upload_speed_mbps for r in results) / len(results)
        avg_duration = sum(r.duration for r in results) / len(results)
        print(
            f"{size_mb:<12} {len(results):<10} {avg_speed:<20.2f} {avg_duration:<20.2f}"
        )

    # Performance by pattern
    if sandbox_patterns:
        print("\n📊 PERFORMANCE BY PATTERN")
        print("-" * 80)
        pattern_groups = {}
        for result in all_results:
            pattern_idx = sandbox_patterns.get(result.sandbox_id, -1)
            if pattern_idx >= 0:
                if pattern_idx not in pattern_groups:
                    pattern_groups[pattern_idx] = {
                        "results": [],
                        "successful": 0,
                        "failed": 0,
                    }
                pattern_groups[pattern_idx]["results"].append(result)
                if result.success:
                    pattern_groups[pattern_idx]["successful"] += 1
                else:
                    pattern_groups[pattern_idx]["failed"] += 1

        pattern_names = [
            "Many small (10x1MB)",
            "Small-med (1-5MB)",
            "Medium (5-10MB)",
            "Large (10-30MB)",
            "Max size (4x30MB)",
            "Mixed (1-30MB)",
            "Rapid small (20x1MB)",
            "Med rapid (10x5MB)",
            "Large batch (10-30MB)",
            "Extreme (1+30MB)",
        ]

        print(
            f"{'Pattern':<25} {'Files':<10} {'Success':<10} {'Failed':<10} {'Success %':<12} {'Avg Speed':<15}"
        )
        print("-" * 80)
        for pattern_idx in sorted(pattern_groups.keys()):
            group = pattern_groups[pattern_idx]
            total = group["successful"] + group["failed"]
            success_pct = (group["successful"] / total * 100) if total > 0 else 0
            successful_results = [r for r in group["results"] if r.success]
            avg_speed = (
                sum(r.upload_speed_mbps for r in successful_results)
                / len(successful_results)
                if successful_results
                else 0
            )
            pattern_name = (
                pattern_names[pattern_idx]
                if pattern_idx < len(pattern_names)
                else f"Pattern {pattern_idx}"
            )

            print(
                f"{pattern_name:<25} "
                f"{total:<10} "
                f"{group['successful']:<10} "
                f"{group['failed']:<10} "
                f"{success_pct:<12.1f} "
                f"{avg_speed:<15.2f}"
            )

    # Per-sandbox statistics
    print("\n🔍 PER-SANDBOX STATISTICS")
    print("-" * 80)
    print(
        f"{'Sandbox ID':<15} {'Pattern':<10} {'Uploads':<10} {'Success':<10} {'Failed':<10} {'Success %':<12} {'Avg Speed':<15}"
    )
    print("-" * 80)

    # Sort by success rate (worst first)
    sorted_stats = sorted(sandbox_stats, key=lambda s: s.success_rate)

    for stats in sorted_stats[:10]:  # Show first 10
        # Get the pattern index for this sandbox
        pattern_idx = sandbox_patterns.get(stats.sandbox_id, -1)
        pattern_display = f"P{pattern_idx}" if pattern_idx >= 0 else "N/A"

        print(
            f"{stats.sandbox_id[:13]:<15} "
            f"{pattern_display:<10} "
            f"{stats.total_uploads:<10} "
            f"{stats.successful_uploads:<10} "
            f"{stats.failed_uploads:<10} "
            f"{stats.success_rate:<12.1f} "
            f"{stats.avg_speed_mbps:<15.2f}"
        )

    if len(sorted_stats) > 10:
        print(f"... and {len(sorted_stats) - 10} more sandboxes")

    # Failure analysis
    failures = [r for r in all_results if not r.success]
    if failures:
        print("\n⚠️  FAILURE ANALYSIS")
        print("-" * 80)

        # Group failures by error type
        error_types = {}
        for failure in failures:
            error_key = (
                failure.error.split(":")[0] if ":" in failure.error else failure.error
            )
            if error_key not in error_types:
                error_types[error_key] = []
            error_types[error_key].append(failure)

        print(f"Total Failures: {len(failures)}")
        print("\nFailures by Error Type:")
        for error_type, errors in sorted(
            error_types.items(), key=lambda x: len(x[1]), reverse=True
        ):
            print(f"  {error_type}: {len(errors)} occurrences")

        # Show sample failures
        print("\nSample Failures (first 5):")
        for i, failure in enumerate(failures[:5], 1):
            print(
                f"  {i}. Sandbox: {failure.sandbox_id[:13]}, Size: {failure.file_size_mb}MB"
            )
            print(f"     Error: {failure.error[:80]}")

    # Success patterns
    sandboxes_with_failures = [s for s in sandbox_stats if s.failed_uploads > 0]
    if sandboxes_with_failures:
        print("\n🔍 FAILURE PATTERNS")
        print("-" * 80)

        if len(sandboxes_with_failures) == len(sandbox_stats):
            print("⚠️  All sandboxes had failures - likely a system-wide issue")
        elif len(sandboxes_with_failures) < len(sandbox_stats) * 0.1:
            print("✅ Only a few sandboxes had failures - likely isolated issues")
        else:
            print(
                "⚠️  Mixed results - may indicate resource constraints or intermittent issues"
            )
    else:
        print("\n✅ SUCCESS - All uploads completed successfully!")

    print("\n" + "=" * 80)


async def main(num_sandboxes: int = 10, max_concurrent_uploads: int = 20) -> None:
    """Run the multi-sandbox file upload stress test.

    Args:
        num_sandboxes: Number of sandboxes to create
        max_concurrent_uploads: Maximum concurrent uploads
    """
    async with AsyncSandboxClient() as client:
        # Define different file size patterns for different sandboxes
        # This creates a more realistic and varied load pattern
        # Max file size: 30MB per file
        file_size_patterns = [
            # Pattern 0: Many small files
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # 10x 1MB
            # Pattern 1: Small to medium files
            [1, 1, 2, 2, 3, 3, 5, 5],  # Mixed 1-5MB
            # Pattern 2: Medium files
            [5, 5, 5, 5, 10, 10, 10, 10],  # 5-10MB files
            # Pattern 3: Large files
            [10, 15, 20, 25, 30],  # 10-30MB files
            # Pattern 4: Maximum size files
            [30, 30, 30, 30],  # 4x 30MB files (max size)
            # Pattern 5: Mixed sizes (comprehensive)
            [1, 2, 5, 10, 15, 20, 25, 30],  # Full range 1-30MB
            # Pattern 6: Rapid small files (stress test)
            [1] * 20,  # 20x 1MB files
            # Pattern 7: Medium rapid
            [5] * 10,  # 10x 5MB files
            # Pattern 8: Large batch
            [10, 10, 10, 20, 20, 20, 30, 30],  # Larger files
            # Pattern 9: Extreme contrast
            [1, 1, 1, 30, 30, 30],  # Mix of tiny and max size
        ]

        # Assign patterns to sandboxes (cycle through patterns)
        sandbox_file_sizes = {}
        total_files = 0
        total_size_mb = 0

        for i in range(num_sandboxes):
            pattern = file_size_patterns[i % len(file_size_patterns)]
            sandbox_file_sizes[i] = pattern
            total_files += len(pattern)
            total_size_mb += sum(pattern)

        # Calculate pattern distribution
        pattern_usage = {}
        for i in range(num_sandboxes):
            pattern_idx = i % len(file_size_patterns)
            pattern_usage[pattern_idx] = pattern_usage.get(pattern_idx, 0) + 1

        print("\n" + "=" * 80)
        print("MULTI-SANDBOX FILE UPLOAD STRESS TEST")
        print("=" * 80)
        print("\nConfiguration:")
        print(f"  Sandboxes:              {num_sandboxes}")
        print(f"  Total files:            {total_files}")
        print(
            f"  Total data size:        {total_size_mb} MB ({total_size_mb / 1024:.2f} GB)"
        )
        print(f"  Max concurrent uploads: {max_concurrent_uploads}")
        print(f"  File size patterns:     {len(file_size_patterns)} different patterns")

        print("\n  Pattern Distribution:")
        pattern_descriptions = [
            "Many small files (10x 1MB)",
            "Small to medium (1-5MB)",
            "Medium files (5-10MB)",
            "Large files (10-30MB)",
            "Maximum size files (4x 30MB)",
            "Mixed sizes (1-30MB)",
            "Rapid small (20x 1MB)",
            "Medium rapid (10x 5MB)",
            "Large batch (10-30MB)",
            "Extreme contrast (1-30MB)",
        ]
        for pattern_idx, count in sorted(pattern_usage.items()):
            desc = (
                pattern_descriptions[pattern_idx]
                if pattern_idx < len(pattern_descriptions)
                else f"Pattern {pattern_idx}"
            )
            pattern_size = sum(file_size_patterns[pattern_idx])
            print(
                f"    Pattern {pattern_idx}: {count} sandbox(es) - {desc} - {pattern_size}MB total"
            )

        # Phase 1: Create sandboxes
        print("\n📦 Phase 1: Creating Sandboxes")
        print("-" * 80)
        setup_start = time.time()

        create_tasks = []
        for i in range(num_sandboxes):
            create_tasks.append(
                client.create(
                    CreateSandboxRequest(
                        name=f"upload-test-{i}",
                        docker_image="python:3.11-slim",
                        start_command="tail -f /dev/null",
                        cpu_cores=1,
                        memory_gb=2,
                        disk_size_gb=10,
                        timeout_minutes=30,
                    )
                )
            )

        sandboxes = await asyncio.gather(*create_tasks)
        print(f"✅ Created {len(sandboxes)} sandboxes")

        # Phase 2: Wait for sandboxes to be ready
        print("\n⏳ Phase 2: Waiting for Sandboxes to be Ready")
        print("-" * 80)

        sandbox_ids = [s.id for s in sandboxes]

        def status_callback(
            elapsed_time: float, state_counts: dict, attempt: int
        ) -> None:
            status_str = ", ".join(
                [f"{state}: {count}" for state, count in sorted(state_counts.items())]
            )
            print(f"  [{elapsed_time:.1f}s] {status_str}")

        await client.bulk_wait_for_creation(
            sandbox_ids, status_callback=status_callback
        )

        setup_duration = time.time() - setup_start
        print(f"✅ All sandboxes ready in {setup_duration:.2f}s")

        # Phase 3: Upload files
        print("\n📤 Phase 3: Uploading Files")
        print("-" * 80)
        print(f"Uploading {total_files} files across {num_sandboxes} sandboxes...")

        upload_start = time.time()
        semaphore = asyncio.Semaphore(max_concurrent_uploads)

        all_results = []

        with tqdm(total=total_files, desc="Upload Progress", unit="file") as pbar:
            upload_tasks = []
            for i, sandbox in enumerate(sandboxes):
                # Use the specific file size pattern for this sandbox
                file_sizes = sandbox_file_sizes[i]
                task = upload_files_to_sandbox(
                    client, sandbox.id, file_sizes, semaphore, pbar
                )
                upload_tasks.append(task)

            results_per_sandbox = await asyncio.gather(*upload_tasks)

            # Flatten results
            for results in results_per_sandbox:
                all_results.extend(results)

        upload_duration = time.time() - upload_start
        print(f"✅ Upload phase completed in {upload_duration:.2f}s")

        # Phase 4: Verify uploads
        print("\n🔍 Phase 4: Verifying Uploads")
        print("-" * 80)

        verify_tasks = []
        for i, sandbox in enumerate(sandboxes):
            # Use the specific file sizes for this sandbox
            file_sizes = sandbox_file_sizes[i]
            expected_files = [(size, idx) for idx, size in enumerate(file_sizes, 1)]
            verify_tasks.append(verify_uploads(client, sandbox.id, expected_files))

        verification_results = await asyncio.gather(*verify_tasks)

        successful_verifications = sum(
            1 for v in verification_results if v.get("verification_passed", False)
        )
        print(
            f"✅ Verification: {successful_verifications}/{len(sandboxes)} sandboxes passed"
        )

        # Calculate statistics
        sandbox_stats = []
        sandbox_pattern_map = {}
        for i, sandbox in enumerate(sandboxes):
            sandbox_results = [r for r in all_results if r.sandbox_id == sandbox.id]
            if sandbox_results:
                stats = calculate_sandbox_stats(sandbox_results)
                sandbox_stats.append(stats)
            # Map sandbox_id to its pattern index
            pattern_idx = i % len(file_size_patterns)
            sandbox_pattern_map[sandbox.id] = pattern_idx

        # Print results
        print_results(
            sandbox_stats,
            all_results,
            upload_duration,
            setup_duration,
            sandbox_pattern_map,
        )

        # Phase 5: Cleanup
        print("\n🧹 Phase 5: Cleaning Up")
        print("-" * 80)

        delete_tasks = [client.delete(s.id) for s in sandboxes]
        await asyncio.gather(*delete_tasks)
        print(f"✅ Cleaned up {len(sandboxes)} sandboxes")


if __name__ == "__main__":
    print("Multi-Sandbox File Upload Stress Test")
    print("=" * 80)

    # Parse command-line arguments
    num_sandboxes = 10
    max_concurrent = 20

    if len(sys.argv) > 1:
        try:
            num_sandboxes = int(sys.argv[1])
        except ValueError:
            print(f"Error: Invalid number of sandboxes '{sys.argv[1]}'")
            sys.exit(1)

    if len(sys.argv) > 2:
        try:
            max_concurrent = int(sys.argv[2])
        except ValueError:
            print(f"Error: Invalid concurrency '{sys.argv[2]}'")
            sys.exit(1)

    # Validate configuration
    if num_sandboxes < 1 or num_sandboxes > 100:
        print("Error: Number of sandboxes must be between 1 and 100")
        sys.exit(1)

    if max_concurrent < 1 or max_concurrent > 100:
        print("Error: Concurrency must be between 1 and 100")
        sys.exit(1)

    # Check API configuration
    config = Config()
    if not config.api_key:
        print("Error: No API key found. Please run 'prime auth login' first.")
        sys.exit(1)

    print(f"API endpoint: {config.base_url}")
    if config.team_id:
        print(f"Team: {config.team_id}")

    # Run the test
    asyncio.run(main(num_sandboxes, max_concurrent))

    print("\n✅ Test completed!")
