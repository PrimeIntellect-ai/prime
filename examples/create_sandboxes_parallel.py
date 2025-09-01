#!/usr/bin/env python3
"""
IMPORTANT: Run this script from the prime-cli directory with the proper environment:

1. Navigate to prime-cli directory:
2. Activate virtual environment (if using one):
   source .venv/bin/activate.fish  # or source .venv/bin/activate for bash
3. Install dependencies if needed:
   uv pip install ".[dev]"
4. Run the script:
   python create_sandboxes_parallel.py                    # Create 100 NEW sandboxes (default)
   python create_sandboxes_parallel.py --count 200        # Create 200 NEW sandboxes
   python create_sandboxes_parallel.py --count 50 --batch-size 10  # Create 50 NEW sandboxes in batches of 10
"""
"""
Parallel Sandbox Creation Script

This script uses the prime-cli SDK to create a specified number of NEW sandboxes in parallel.

Specifications:
- Image: python
- Timeout: 15 minutes
- Start command: "sleep infinity"
- CPU: 1 core
- Memory: 1GB
- Disk: 1GB
- Creates sandboxes in parallel batches (default: 10 at a time)
- Default: Creates 100 NEW sandboxes
"""

import argparse
import csv
import random
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Set
from queue import Queue

from prime_cli.api.client import APIClient, APIError
from prime_cli.api.sandbox import CreateSandboxRequest, SandboxClient

# Global flag for graceful shutdown
graceful_shutdown = threading.Event()

def signal_handler(signum, frame):
    """Handle SIGINT (Ctrl+C) and SIGTERM gracefully"""
    signal_name = "SIGINT" if signum == signal.SIGINT else "SIGTERM"
    print(f"\n🛑 Received {signal_name}! Initiating graceful shutdown...")
    print("📊 Will generate summary with current data and exit...")
    graceful_shutdown.set()

class RateLimitError(APIError):
    """Raised when API returns 429 rate limit error"""
    pass


class SandboxWatcher:
    """Real-time sandbox status watcher that runs in a separate thread"""

    def __init__(self, sandbox_client: SandboxClient, stats: 'SandboxCreationStats'):
        self.sandbox_client = sandbox_client
        self.stats = stats
        self.watch_queue: Queue[str] = Queue()  # Queue of sandbox IDs to watch
        self.stop_event = threading.Event()
        self.watcher_thread: Optional[threading.Thread] = None
        self.watching_sandboxes: Set[str] = set()  # Currently being watched
        self._lock = threading.Lock()
        self.current_user_id: Optional[str] = None  # Track current user ID for filtering

    def start_watching(self):
        """Start the watcher thread"""
        if self.watcher_thread is None or not self.watcher_thread.is_alive():
            self.stop_event.clear()
            self.watcher_thread = threading.Thread(target=self._watch_loop, daemon=True)
            self.watcher_thread.start()
            print("🔍 Started sandbox watcher thread")

    def stop_watching(self):
        """Stop the watcher thread"""
        self.stop_event.set()
        if self.watcher_thread and self.watcher_thread.is_alive():
            self.watcher_thread.join(timeout=5)
            print("🛑 Stopped sandbox watcher thread")

    def add_sandbox_to_watch(self, sandbox_id: str, user_id: Optional[str] = None):
        """Add a sandbox ID to the watch queue"""
        with self._lock:
            # Set current user ID if provided and not already set
            if user_id and self.current_user_id is None:
                self.current_user_id = user_id
                print(f"🔐 Tracking sandboxes for user {user_id[:8]}...")

            if sandbox_id not in self.watching_sandboxes:
                self.watch_queue.put(sandbox_id)
                self.watching_sandboxes.add(sandbox_id)
                print(f"👀 Added {sandbox_id} to watch queue")

    def _watch_loop(self):
        """Main watcher loop that runs in the background thread"""
        check_interval = 30  # Check every 30 seconds using list API (precision comes from API timestamps)
        pending_sandboxes: Set[str] = set()
        running_sandboxes: Set[str] = set()  # Track sandboxes that transitioned to RUNNING

        while not self.stop_event.is_set() and not graceful_shutdown.is_set():
            try:
                # Add new sandboxes from the queue
                while not self.watch_queue.empty():
                    try:
                        sandbox_id = self.watch_queue.get_nowait()
                        pending_sandboxes.add(sandbox_id)
                    except:
                        break

                # Check status of pending sandboxes using list API
                if pending_sandboxes:
                    completed_sandboxes = set()

                    # Use list API instead of individual get calls to avoid rate limiting
                    sandbox_status_map = self._get_sandbox_statuses_batch(pending_sandboxes)

                    if sandbox_status_map is not None:
                        for sandbox_id in list(pending_sandboxes):
                            if self.stop_event.is_set() or graceful_shutdown.is_set():
                                break

                            if sandbox_id not in sandbox_status_map:
                                # Sandbox not found in list, may be terminated or deleted
                                print(f"⚠️  Sandbox {sandbox_id} not found in list API, removing from watch")
                                completed_sandboxes.add(sandbox_id)
                                continue

                            status, sandbox_user_id = sandbox_status_map[sandbox_id]

                            # Verify sandbox belongs to current user (if we have user ID info)
                            if self.current_user_id and sandbox_user_id and sandbox_user_id != self.current_user_id:
                                print(f"⚠️  Skipping {sandbox_id} - belongs to different user ({sandbox_user_id[:8]}...)")
                                completed_sandboxes.add(sandbox_id)
                                continue

                            if status == "RUNNING":
                                # Mark as running but don't complete watching yet
                                if sandbox_id not in running_sandboxes:
                                    running_sandboxes.add(sandbox_id)
                                    print(f"🔄 Sandbox {sandbox_id} is RUNNING, getting detailed ready time...")

                                    # Get actual startup duration from API timestamps
                                    actual_startup_duration = self._get_sandbox_ready_time(sandbox_id)
                                    if actual_startup_duration is not None:
                                        self.stats.add_ready_sandbox(sandbox_id, actual_startup_duration)
                                        completed_sandboxes.add(sandbox_id)
                                        print(f"✅ Watcher detected {sandbox_id} is ready! ({self.stats.ready_count}/{len(self.stats.created_sandbox_ids)}) - API Duration: {actual_startup_duration:.2f}s")
                                    else:
                                        # Will retry getting details in next iteration
                                        print(f"⚠️  Could not get startup duration for {sandbox_id}, will retry")

                            elif status in ["ERROR", "TERMINATED"]:
                                print(f"❌ Watcher detected {sandbox_id} failed with status: {status}")
                                completed_sandboxes.add(sandbox_id)

                    else:
                        # List API failed, fall back to individual checks but with longer delay
                        print("⚠️  List API failed, using fallback individual checks")
                        for sandbox_id in list(pending_sandboxes)[:3]:  # Limit to 3 to avoid rate limiting
                            if self.stop_event.is_set() or graceful_shutdown.is_set():
                                break

                            result = self._check_sandbox_status_with_retry(sandbox_id)
                            if result is not None:
                                status, sandbox_user_id = result
                                if status == "RUNNING":
                                    self.stats.add_ready_sandbox(sandbox_id)
                                    completed_sandboxes.add(sandbox_id)
                                    print(f"✅ Watcher detected {sandbox_id} is ready! ({self.stats.ready_count}/{len(self.stats.created_sandbox_ids)})")
                                elif status in ["ERROR", "TERMINATED"]:
                                    print(f"❌ Watcher detected {sandbox_id} failed with status: {status}")
                                    completed_sandboxes.add(sandbox_id)

                    # Remove completed sandboxes from pending and running sets
                    for sandbox_id in completed_sandboxes:
                        pending_sandboxes.discard(sandbox_id)
                        running_sandboxes.discard(sandbox_id)
                        with self._lock:
                            self.watching_sandboxes.discard(sandbox_id)

                # Sleep before next check cycle
                if not self.stop_event.is_set():
                    time.sleep(check_interval)

            except Exception as e:
                print(f"⚠️  Error in watcher loop: {e}")
                time.sleep(check_interval)

    def _get_sandbox_statuses_batch(self, sandbox_ids: Set[str], max_retries: int = 3) -> Optional[Dict[str, Tuple[str, Optional[str]]]]:
        """Get status for multiple sandboxes using the list API to avoid rate limiting"""
        base_delay = 1.0
        max_delay = 5.0

        for attempt in range(max_retries + 1):
            try:
                # Use list API with pagination to get all user sandboxes
                sandbox_status_map = {}
                page = 1
                per_page = 100

                while True:
                    response = self.sandbox_client.list(exclude_terminated=True, per_page=per_page, page=page)

                    if not response.sandboxes:
                        break

                    # Build map for sandboxes we're watching
                    for sandbox in response.sandboxes:
                        if sandbox.id in sandbox_ids:
                            sandbox_status_map[sandbox.id] = (sandbox.status, sandbox.user_id)

                    # Check if there are more pages
                    if not response.has_next or len(response.sandboxes) < per_page:
                        break

                    page += 1

                return sandbox_status_map

            except Exception as e:
                error_str = str(e)

                # Check if this is a rate limiting error (429)
                if "HTTP 429" in error_str or "rate limit" in error_str.lower():
                    if attempt < max_retries:
                        # Calculate exponential backoff delay with jitter
                        delay = min(base_delay * (2 ** attempt), max_delay)
                        jitter = random.uniform(0, 0.1 * delay)
                        total_delay = delay + jitter

                        print(f"🚫 List API rate limited, retrying in {total_delay:.1f}s (attempt {attempt + 1}/{max_retries})")
                        time.sleep(total_delay)
                        continue
                    else:
                        print(f"❌ List API rate limit exceeded after {max_retries} retries")
                        return None
                else:
                    # Non-rate-limit error
                    print(f"⚠️  Error in list API: {e}")
                    return None

        return None

    def _get_sandbox_ready_time(self, sandbox_id: str, max_retries: int = 3) -> Optional[float]:
        """Get detailed sandbox information to determine actual startup duration"""
        base_delay = 1.0
        max_delay = 5.0

        for attempt in range(max_retries + 1):
            try:
                sandbox = self.sandbox_client.get(sandbox_id)

                # Calculate actual startup duration using API timestamps
                if sandbox.started_at and sandbox.created_at:
                    # Duration from creation request to actually running
                    duration = sandbox.started_at - sandbox.created_at
                    startup_seconds = duration.total_seconds()
                    print(f"📊 {sandbox_id} actual startup time: {startup_seconds:.2f}s (created: {sandbox.created_at}, started: {sandbox.started_at})")
                    return startup_seconds
                else:
                    print(f"⚠️  {sandbox_id} missing timestamp data - created_at: {sandbox.created_at}, started_at: {sandbox.started_at}")
                    return None

            except Exception as e:
                error_str = str(e)

                # Check if this is a rate limiting error (429)
                if "HTTP 429" in error_str or "rate limit" in error_str.lower():
                    if attempt < max_retries:
                        # Calculate exponential backoff delay with jitter
                        delay = min(base_delay * (2 ** attempt), max_delay)
                        jitter = random.uniform(0, 0.1 * delay)
                        total_delay = delay + jitter

                        print(f"🚫 Get details rate limited for {sandbox_id}, retrying in {total_delay:.1f}s")
                        time.sleep(total_delay)
                        continue
                    else:
                        print(f"❌ Get details rate limit exceeded for {sandbox_id}")
                        return None
                else:
                    # Non-rate-limit error
                    print(f"⚠️  Error getting details for {sandbox_id}: {e}")
                    return None

        return None

    def _check_sandbox_status_with_retry(self, sandbox_id: str, max_retries: int = 3) -> Optional[Tuple[str, Optional[str]]]:
        """Check sandbox status with exponential backoff retry logic"""
        base_delay = 1.0
        max_delay = 8.0

        for attempt in range(max_retries + 1):
            try:
                sandbox = self.sandbox_client.get(sandbox_id)
                return (sandbox.status, sandbox.user_id)

            except Exception as e:
                error_str = str(e)

                # Check if this is a rate limiting error (429)
                if "HTTP 429" in error_str or "rate limit" in error_str.lower():
                    if attempt < max_retries:
                        # Calculate exponential backoff delay with jitter
                        delay = min(base_delay * (2 ** attempt), max_delay)
                        jitter = random.uniform(0, 0.1 * delay)
                        total_delay = delay + jitter

                        print(f"🚫 Watcher rate limited checking {sandbox_id}, retrying in {total_delay:.1f}s")
                        time.sleep(total_delay)
                        continue
                    else:
                        print(f"❌ Watcher rate limit exceeded for {sandbox_id}")
                        return None
                else:
                    # Non-rate-limit error, don't retry
                    print(f"⚠️  Watcher error checking {sandbox_id}: {e}")
                    return None

        return None


class SandboxCreationStats:
    """Track sandbox creation statistics (thread-safe)"""

    def __init__(self, target_count: int = None, batch_size: int = None):
        # Configuration
        self.target_count = target_count
        self.batch_size = batch_size

        # Thread safety
        self._lock = threading.Lock()

        self.total_created = 0
        self.total_failed = 0
        self.total_retries = 0
        self.rate_limit_hits = 0
        self.start_time = time.time()
        self.creation_times: List[float] = []
        self.failed_attempts: List[str] = []

        # Readiness tracking
        self.created_sandbox_ids: List[str] = []
        self.sandbox_creation_times: Dict[str, float] = {}  # sandbox_id -> creation_timestamp
        self.sandbox_startup_times: List[float] = []  # Individual startup times (creation to ready)
        self.first_ready_time: Optional[float] = None
        self.last_ready_time: Optional[float] = None
        self.ready_count = 0

        # Detailed tracking for CSV export
        self.detailed_records: List[Dict] = []  # Individual sandbox records

        # Individual creation durations for statistical analysis
        self.individual_creation_durations: List[float] = []  # Individual creation times

    def add_success(self, creation_time: float):
        """Record a successful sandbox creation (thread-safe)"""
        with self._lock:
            self.total_created += 1
            self.creation_times.append(creation_time)
            self.individual_creation_durations.append(creation_time)

    def add_failure(self, error_msg: str, batch_num: Optional[int] = None, creation_duration: Optional[float] = None):
        """Record a failed sandbox creation (thread-safe)"""
        with self._lock:
            self.total_failed += 1
            self.failed_attempts.append(error_msg)

            # Track failed creation duration for statistics (if available)
            if creation_duration is not None:
                self.individual_creation_durations.append(creation_duration)

            # Record failed attempt for CSV export
            record = {
                'sandbox_id': f'failed-{self.total_failed}',
                'batch_number': batch_num,
                'creation_request_time': datetime.fromtimestamp(time.time()).isoformat(),
                'creation_duration_seconds': creation_duration,
                'creation_timestamp': time.time(),
                'ready_time': None,
                'startup_time_seconds': None,
                'status': 'failed',
                'error_message': error_msg
            }
            self.detailed_records.append(record)

    def add_retry(self):
        """Record a retry attempt (thread-safe)"""
        with self._lock:
            self.total_retries += 1

    def add_rate_limit_hit(self):
        """Record a rate limit hit (thread-safe)"""
        with self._lock:
            self.rate_limit_hits += 1

    def add_created_sandbox(self, sandbox_id: str, creation_timestamp: Optional[float] = None, batch_num: Optional[int] = None, creation_duration: Optional[float] = None):
        """Record a successfully created sandbox ID with its creation timestamp (thread-safe)"""
        with self._lock:
            self.created_sandbox_ids.append(sandbox_id)
            if creation_timestamp is None:
                creation_timestamp = time.time()
            self.sandbox_creation_times[sandbox_id] = creation_timestamp

            # Record detailed information for CSV export
            record = {
                'sandbox_id': sandbox_id,
                'batch_number': batch_num,
                'creation_request_time': datetime.fromtimestamp(creation_timestamp).isoformat(),
                'creation_duration_seconds': creation_duration,
                'creation_timestamp': creation_timestamp,
                'ready_time': None,
                'startup_time_seconds': None,
                'status': 'created'
            }
            self.detailed_records.append(record)

    def add_ready_sandbox(self, sandbox_id: str, actual_startup_duration: Optional[float] = None):
        """Record when a sandbox becomes ready and its startup time (thread-safe)"""
        current_time = time.time()

        with self._lock:
            self.ready_count += 1

            # Use actual API startup duration if provided, otherwise calculate from our timestamps
            startup_time = actual_startup_duration
            if startup_time is None and sandbox_id in self.sandbox_creation_times:
                # Fallback to our approximation if API duration not available
                startup_time = current_time - self.sandbox_creation_times[sandbox_id]
                print(f"📊 Using approximate startup time for {sandbox_id}: {startup_time:.2f}s (API timestamps not available)")

            if startup_time is not None:
                self.sandbox_startup_times.append(startup_time)

            # Update detailed record for CSV export
            for record in self.detailed_records:
                if record['sandbox_id'] == sandbox_id:
                    record['ready_time'] = datetime.fromtimestamp(current_time).isoformat()
                    record['startup_time_seconds'] = startup_time
                    record['status'] = 'ready'
                    if actual_startup_duration is not None:
                        record['duration_source'] = 'api_timestamps'  # Track data source
                    else:
                        record['duration_source'] = 'approximated'
                    break

            if self.first_ready_time is None:
                self.first_ready_time = current_time

            self.last_ready_time = current_time

    def calculate_percentile(self, data: List[float], percentile: float) -> float:
        """Calculate the given percentile of a dataset"""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = (percentile / 100) * (len(sorted_data) - 1)
        if index.is_integer():
            return sorted_data[int(index)]
        else:
            lower = sorted_data[int(index)]
            upper = sorted_data[int(index) + 1]
            return lower + (upper - lower) * (index - int(index))

    def export_to_csv(self, filename: str) -> None:
        """Export detailed measurements to CSV file"""
        if not self.detailed_records:
            print("⚠️  No data to export to CSV")
            return

        # Define CSV headers
        headers = [
            'sandbox_id',
            'batch_number',
            'creation_request_time',
            'creation_duration_seconds',
            'ready_time',
            'startup_time_seconds',
            'duration_source',
            'status',
            'error_message'
        ]

        try:
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=headers)
                writer.writeheader()

                for record in self.detailed_records:
                    # Only write the fields we want in the CSV
                    csv_record = {key: record.get(key) for key in headers}
                    writer.writerow(csv_record)

            print(f"📊 Exported {len(self.detailed_records)} records to {filename}")

        except Exception as e:
            print(f"❌ Error exporting to CSV: {e}")

    def get_summary(self) -> str:
        """Get a summary of the statistics"""
        total_time = time.time() - self.start_time
        avg_creation_time = sum(self.creation_times) / len(self.creation_times) if self.creation_times else 0

        # Calculate readiness timing
        readiness_summary = ""
        if self.first_ready_time and self.last_ready_time:
            first_ready_offset = self.first_ready_time - self.start_time
            last_ready_offset = self.last_ready_time - self.start_time
            provisioning_duration = self.last_ready_time - self.first_ready_time

            readiness_summary = f"""
🚀 Readiness Timing:
   📍 First sandbox ready: {first_ready_offset:.2f}s after start
   🏁 Last sandbox ready: {last_ready_offset:.2f}s after start
   ⏳ Provisioning duration: {provisioning_duration:.2f}s (first to last ready)
   ✅ Ready sandboxes: {self.ready_count}/{len(self.created_sandbox_ids)}"""
        elif len(self.created_sandbox_ids) > 0 and self.ready_count == 0:
            readiness_summary = f"""
🚀 Readiness Timing:
   ⏭️  Readiness monitoring was skipped
   📦 Created sandboxes: {len(self.created_sandbox_ids)}
   💡 Use --wait-until-ready to monitor sandbox readiness"""

        # Calculate creation time percentiles (available in both wait and no-wait modes)
        creation_summary = ""
        if self.individual_creation_durations:
            avg_creation = sum(self.individual_creation_durations) / len(self.individual_creation_durations)
            min_creation = min(self.individual_creation_durations)
            max_creation = max(self.individual_creation_durations)
            p50_creation = self.calculate_percentile(self.individual_creation_durations, 50)
            p90_creation = self.calculate_percentile(self.individual_creation_durations, 90)
            p95_creation = self.calculate_percentile(self.individual_creation_durations, 95)
            p99_creation = self.calculate_percentile(self.individual_creation_durations, 99)

            creation_summary = f"""
📊 Individual Creation Times (API request duration):
   📊 Count: {len(self.individual_creation_durations)} requests
   ⚡ Average: {avg_creation:.2f}s
   🏃 Min: {min_creation:.2f}s
   🐌 Max: {max_creation:.2f}s
   📊 P50 (median): {p50_creation:.2f}s
   📊 P90: {p90_creation:.2f}s
   📊 P95: {p95_creation:.2f}s
   📊 P99: {p99_creation:.2f}s"""

        # Calculate startup time percentiles (only available in wait mode)
        startup_summary = ""
        if self.sandbox_startup_times:
            avg_startup = sum(self.sandbox_startup_times) / len(self.sandbox_startup_times)
            min_startup = min(self.sandbox_startup_times)
            max_startup = max(self.sandbox_startup_times)
            p50_startup = self.calculate_percentile(self.sandbox_startup_times, 50)
            p90_startup = self.calculate_percentile(self.sandbox_startup_times, 90)
            p95_startup = self.calculate_percentile(self.sandbox_startup_times, 95)
            p99_startup = self.calculate_percentile(self.sandbox_startup_times, 99)

            startup_summary = f"""
📈 Individual Startup Times (request to ready):
   📊 Count: {len(self.sandbox_startup_times)} sandboxes
   ⚡ Average: {avg_startup:.2f}s
   🏃 Min: {min_startup:.2f}s
   🐌 Max: {max_startup:.2f}s
   📊 P50 (median): {p50_startup:.2f}s
   📊 P90: {p90_startup:.2f}s
   📊 P95: {p95_startup:.2f}s
   📊 P99: {p99_startup:.2f}s"""
        elif len(self.created_sandbox_ids) > 0 and len(self.sandbox_startup_times) == 0:
            startup_summary = f"""
📈 Individual Startup Times (request to ready):
   ⏭️  Startup time analysis was skipped (readiness monitoring disabled)
   📦 Created sandboxes: {len(self.created_sandbox_ids)}
   💡 Use --wait-until-ready to get startup time metrics"""

        return f"""
Sandbox Creation Summary:
========================
🎯 Target count: {getattr(self, 'target_count', 'N/A')}
📦 Batch size: {getattr(self, 'batch_size', 'N/A')}
✅ Successfully created: {self.total_created}
❌ Failed: {self.total_failed}
🔄 Total retries: {self.total_retries}
🚫 Rate limit hits: {self.rate_limit_hits}
⏱️  Total time: {total_time:.2f}s
📊 Success rate: {(self.total_created / (self.total_created + self.total_failed) * 100):.1f}%
⚡ Average creation time: {avg_creation_time:.2f}s{creation_summary}{readiness_summary}{startup_summary}
"""


def create_single_sandbox_with_retry(
    sandbox_client: SandboxClient,
    sandbox_name: str,
    stats: SandboxCreationStats,
    batch_num: int,
    watcher: Optional[SandboxWatcher] = None,
    max_retries: int = 5
) -> Tuple[bool, str, float, Optional[str], Optional[float]]:
    """
    Create a single sandbox with exponential backoff retry for rate limiting.

    Args:
        sandbox_client: The sandbox client to use
        sandbox_name: Name for the sandbox
        stats: Statistics tracker for recording retries and rate limits
        max_retries: Maximum number of retry attempts

    Returns:
        Tuple of (success, message, creation_time, sandbox_id, creation_timestamp)
    """
    start_time = time.time()
    base_delay = 5.0  # Base delay in seconds
    max_delay = 120.0  # Maximum delay in seconds

    for attempt in range(max_retries + 1):
        try:
            # Create sandbox request with specified parameters
            request = CreateSandboxRequest(
                name=sandbox_name,
                docker_image="python",
                start_command="sleep infinity",
                cpu_cores=1,
                memory_gb=1,
                disk_size_gb=1,
                timeout_minutes=15
            )

            # Create the sandbox
            sandbox = sandbox_client.create(request)
            creation_time = time.time() - start_time
            creation_timestamp = time.time()  # Record when the sandbox was created

            # Add to watcher if available (eliminates race condition)
            if watcher:
                watcher.add_sandbox_to_watch(sandbox.id, sandbox.user_id)

            return True, f"Created sandbox {sandbox.id}", creation_time, sandbox.id, creation_timestamp

        except APIError as e:
            error_str = str(e)

            # Check if this is a rate limiting error (429)
            if "HTTP 429" in error_str or "rate limit" in error_str.lower():
                stats.add_rate_limit_hit()

                if attempt < max_retries:
                    # Calculate exponential backoff delay with jitter
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    jitter = random.uniform(0, 0.1 * delay)  # Add up to 10% jitter
                    total_delay = delay + jitter

                    stats.add_retry()
                    print(f"🚫 Rate limited for {sandbox_name}, retrying in {total_delay:.1f}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(total_delay)
                    continue
                else:
                    creation_time = time.time() - start_time
                    return False, f"Rate limit exceeded after {max_retries} retries: {error_str}", creation_time, None, None
            else:
                # Non-rate-limit API error, don't retry
                creation_time = time.time() - start_time
                return False, f"API Error: {error_str}", creation_time, None, None

        except Exception as e:
            # Unexpected error, don't retry
            creation_time = time.time() - start_time
            return False, f"Unexpected error: {str(e)}", creation_time, None, None

    # Should never reach here, but just in case
    creation_time = time.time() - start_time
    return False, f"Max retries exceeded", creation_time, None, None


def get_current_user_sandbox_count(sandbox_client: SandboxClient) -> int:
    """Get the current number of active sandboxes for the current user only"""
    try:
        # Get all active sandboxes with pagination
        all_user_sandboxes = []
        page = 1
        per_page = 100  # API limit
        current_user_id = None  # Declare outside the loop

        while True:
            response = sandbox_client.list(exclude_terminated=True, per_page=per_page, page=page)

            if not response.sandboxes:
                break

            # The API should already filter by current user, but let's be explicit
            # In case the API returns sandboxes from other users, we'll filter them out
            user_sandboxes = []

            # First, try to determine the current user ID from any sandbox
            for sandbox in response.sandboxes:
                if sandbox.user_id:
                    if current_user_id is None:
                        current_user_id = sandbox.user_id

                    # Only count sandboxes that belong to the same user as the first one
                    # This assumes all returned sandboxes belong to the current user
                    if sandbox.user_id == current_user_id:
                        user_sandboxes.append(sandbox)
                else:
                    # If user_id is not set, assume it belongs to current user
                    # (this handles cases where the API doesn't populate user_id)
                    user_sandboxes.append(sandbox)

            all_user_sandboxes.extend(user_sandboxes)

            # Check if there are more pages
            if not response.has_next or len(response.sandboxes) < per_page:
                break

            page += 1

        total_count = len(all_user_sandboxes)

        if current_user_id:
            print(f"📊 Found {total_count} active sandboxes for user {current_user_id[:8]}...")
        else:
            print(f"📊 Found {total_count} active sandboxes for current user")

        return total_count

    except APIError as e:
        error_str = str(e)
        if "per_page" in error_str and "100" in error_str:
            print(f"⚠️  API pagination limit error (fixed in this version): {e}")
        else:
            print(f"⚠️  API Error getting sandbox count: {e}")
        return 0
    except Exception as e:
        print(f"⚠️  Warning: Could not get current sandbox count: {e}")
        return 0


def create_sandboxes_batch(sandbox_client: SandboxClient, batch_size: int, batch_num: int, stats: SandboxCreationStats, watcher: Optional[SandboxWatcher] = None) -> List[Tuple[bool, str, float, Optional[str], Optional[float]]]:
    """Create a batch of sandboxes in parallel using ThreadPoolExecutor"""
    results = []

    with ThreadPoolExecutor(max_workers=min(batch_size, 20)) as executor:  # Limit concurrent connections
        # Submit all sandbox creation tasks
        future_to_name = {
            executor.submit(
                create_single_sandbox_with_retry,
                sandbox_client,
                f"batch-{batch_num}-sandbox-{i+1}",
                stats,
                batch_num,
                watcher
            ): f"batch-{batch_num}-sandbox-{i+1}"
            for i in range(batch_size)
        }

        # Collect results as they complete
        for future in as_completed(future_to_name):
            sandbox_name = future_to_name[future]
            try:
                success, message, creation_time, sandbox_id, creation_timestamp = future.result()
                results.append((success, message, creation_time, sandbox_id, creation_timestamp))

                # Track successful sandbox IDs with creation timestamp
                if success and sandbox_id and creation_timestamp:
                    stats.add_created_sandbox(sandbox_id, creation_timestamp, batch_num, creation_time)

                # Print immediate feedback
                status_icon = "✅" if success else "❌"
                print(f"{status_icon} {sandbox_name}: {message} ({creation_time:.2f}s)")

            except Exception as e:
                results.append((False, f"Exception in {sandbox_name}: {str(e)}", 0.0, None, None))
                print(f"❌ {sandbox_name}: Exception: {str(e)}")

    return results


def check_sandbox_status_with_retry(sandbox_client: SandboxClient, sandbox_id: str, max_retries: int = 3) -> Optional[str]:
    """
    Check sandbox status with exponential backoff retry for rate limiting.

    Args:
        sandbox_client: The sandbox client to use
        sandbox_id: ID of the sandbox to check
        max_retries: Maximum number of retry attempts

    Returns:
        Sandbox status string or None if failed after all retries
    """
    base_delay = 1.0  # Base delay in seconds
    max_delay = 30.0  # Maximum delay in seconds

    for attempt in range(max_retries + 1):
        try:
            sandbox = sandbox_client.get(sandbox_id)
            return sandbox.status

        except APIError as e:
            error_str = str(e)

            # Check if this is a rate limiting error (429)
            if "HTTP 429" in error_str or "rate limit" in error_str.lower():
                if attempt < max_retries:
                    # Calculate exponential backoff delay with jitter
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    jitter = random.uniform(0, 0.1 * delay)  # Add up to 10% jitter
                    total_delay = delay + jitter

                    print(f"🚫 Rate limited checking {sandbox_id}, retrying in {total_delay:.1f}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(total_delay)
                    continue
                else:
                    print(f"❌ Rate limit exceeded for {sandbox_id} after {max_retries} retries")
                    return None
            else:
                # Non-rate-limit API error, don't retry
                print(f"⚠️  Error checking sandbox {sandbox_id}: {e}")
                return None

        except Exception as e:
            # Unexpected error, don't retry
            print(f"⚠️  Unexpected error checking sandbox {sandbox_id}: {e}")
            return None

    return None


def wait_for_sandboxes_ready(sandbox_client: SandboxClient, stats: SandboxCreationStats, max_wait_time: int = 600) -> None:
    """
    Wait for all created sandboxes to be ready (RUNNING status).

    Args:
        sandbox_client: The sandbox client to use
        stats: Statistics tracker for recording readiness timing
        max_wait_time: Maximum time to wait in seconds (default 10 minutes)
    """
    start_wait_time = time.time()
    check_interval = 5  # Check every 5 seconds

    pending_sandboxes = set(stats.created_sandbox_ids)
    last_progress_update = 0

    print(f"⏳ Monitoring {len(pending_sandboxes)} sandboxes for readiness...")

    while pending_sandboxes and (time.time() - start_wait_time) < max_wait_time:
        # Check status of remaining sandboxes in smaller batches to avoid overwhelming the API
        batch_size = min(10, len(pending_sandboxes))  # Check up to 10 sandboxes at a time
        current_batch = list(pending_sandboxes)[:batch_size]

        for sandbox_id in current_batch:
            status = check_sandbox_status_with_retry(sandbox_client, sandbox_id)

            if status == "RUNNING":
                pending_sandboxes.remove(sandbox_id)
                stats.add_ready_sandbox(sandbox_id)

                # Calculate and display startup time for this sandbox
                if sandbox_id in stats.sandbox_creation_times:
                    startup_time = time.time() - stats.sandbox_creation_times[sandbox_id]
                    print(f"✅ Sandbox {sandbox_id} is ready! ({stats.ready_count}/{len(stats.created_sandbox_ids)}) - Startup: {startup_time:.2f}s")
                else:
                    print(f"✅ Sandbox {sandbox_id} is ready! ({stats.ready_count}/{len(stats.created_sandbox_ids)})")

            elif status in ["ERROR", "TERMINATED"]:
                print(f"❌ Sandbox {sandbox_id} failed with status: {status}")
                pending_sandboxes.remove(sandbox_id)

            elif status is None:
                # Failed to get status after retries, but don't remove from pending
                # It will be retried in the next iteration
                print(f"⚠️  Could not get status for {sandbox_id}, will retry later")

        # Add a small delay between batches to be more API-friendly
        if pending_sandboxes:
            time.sleep(1)  # 500ms delay between batches

        # Progress update every 30 seconds
        elapsed = time.time() - start_wait_time
        if elapsed - last_progress_update >= 30:
            ready_count = len(stats.created_sandbox_ids) - len(pending_sandboxes)
            print(f"📊 Progress: {ready_count}/{len(stats.created_sandbox_ids)} ready after {elapsed:.0f}s")
            last_progress_update = elapsed

        # If there are still pending sandboxes, wait before next full check cycle
        if pending_sandboxes:
            time.sleep(check_interval)

    # Final status
    if not pending_sandboxes:
        total_wait_time = time.time() - start_wait_time
        print(f"🎉 All sandboxes are ready! Total wait time: {total_wait_time:.2f}s")
    else:
        timeout_reached = (time.time() - start_wait_time) >= max_wait_time
        if timeout_reached:
            print(f"⏰ Timeout reached after {max_wait_time}s. {len(pending_sandboxes)} sandboxes still pending:")
            for sandbox_id in list(pending_sandboxes)[:5]:  # Show first 5 pending
                print(f"   - {sandbox_id}")
            if len(pending_sandboxes) > 5:
                print(f"   ... and {len(pending_sandboxes) - 5} more")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Create a specific number of NEW sandboxes in parallel",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python create_sandboxes_parallel.py                    # Create 100 NEW sandboxes (default)
  python create_sandboxes_parallel.py --no-wait          # Create sandboxes but don't wait for readiness
  python create_sandboxes_parallel.py --count 100        # Create exactly 100 NEW sandboxes
  python create_sandboxes_parallel.py --count 50 --batch-size 10  # Create 50 NEW sandboxes in batches of 10
        """
    )

    parser.add_argument(
        "--wait-until-ready",
        action="store_true",
        default=False,
        help="Wait for all sandboxes to be ready (default: False)"
    )

    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Don't wait for sandboxes to be ready (same as default behavior)"
    )

    parser.add_argument(
        "--count",
        type=int,
        default=100,
        help="Number of NEW sandboxes to create (default: 100)"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Number of sandboxes to create in each batch (default: 50)"
    )

    args = parser.parse_args()

    # Handle the --no-wait flag
    if args.no_wait:
        args.wait_until_ready = False

    return args


def main():
    """Main function to create sandboxes in parallel"""
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Termination signal

    args = parse_arguments()

    print("🚀 Starting parallel sandbox creation script")
    print("=" * 60)
    print(f"📊 Configuration:")
    print(f"   🎯 NEW sandboxes to create: {args.count}")
    print(f"   📦 Batch size: {args.batch_size}")
    print(f"   ⏳ Wait until ready: {'Yes' if args.wait_until_ready else 'No (default)'}")
    print("=" * 60)
    print("💡 Press Ctrl+C anytime for graceful shutdown with current data summary")

    # Initialize the API client and sandbox client
    try:
        api_client = APIClient()
        sandbox_client = SandboxClient(api_client)
        print("✅ Successfully initialized Prime CLI API client")
    except Exception as e:
        print(f"❌ Failed to initialize API client: {e}")
        print("💡 Make sure you're logged in with 'prime login' or have PRIME_API_KEY set")
        return

    # Get current sandbox count (for information only)
    print("\n📊 Checking current sandbox count...")
    current_count = get_current_user_sandbox_count(sandbox_client)
    print(f"📈 Current active sandboxes: {current_count}")

    # We will create exactly the requested number of NEW sandboxes
    sandboxes_to_create = args.count
    print(f"🎯 Will create {sandboxes_to_create} NEW sandboxes")
    print(f"📊 After creation, you will have approximately {current_count + sandboxes_to_create} total sandboxes")

    # Initialize statistics
    stats = SandboxCreationStats(target_count=sandboxes_to_create, batch_size=args.batch_size)

    # Determine if we should wait for readiness
    wait_until_ready = args.wait_until_ready

    # Initialize watcher if we're waiting for readiness
    watcher = None
    if wait_until_ready:
        watcher = SandboxWatcher(sandbox_client, stats)
        watcher.start_watching()

    # Create sandboxes in batches
    batch_size = args.batch_size
    batch_num = 1

    while stats.total_created < sandboxes_to_create and not graceful_shutdown.is_set():
        # Calculate how many to create in this batch
        sandboxes_left = sandboxes_to_create - stats.total_created
        current_batch_size = min(batch_size, sandboxes_left)

        print(f"\n🔄 Starting batch {batch_num} - Creating {current_batch_size} sandboxes...")
        batch_start_time = time.time()

        # Create batch of sandboxes
        batch_results = create_sandboxes_batch(sandbox_client, current_batch_size, batch_num, stats, watcher)

        # Process batch results
        batch_successes = 0
        batch_failures = 0

        for success, message, creation_time, sandbox_id, creation_timestamp in batch_results:
            if success:
                stats.add_success(creation_time)
                batch_successes += 1
            else:
                stats.add_failure(message, batch_num, creation_time)
                batch_failures += 1

        batch_time = time.time() - batch_start_time

        # Print batch summary
        print(f"\n📊 Batch {batch_num} completed in {batch_time:.2f}s:")
        print(f"   ✅ Successful: {batch_successes}")
        print(f"   ❌ Failed: {batch_failures}")
        print(f"   📈 Total created so far: {stats.total_created}/{sandboxes_to_create}")

        # Check if we've reached our target
        if stats.total_created >= sandboxes_to_create:
            print(f"\n🎉 Target reached! Created {stats.total_created} NEW sandboxes.")
            break

        batch_num += 1

        # Small delay between batches to avoid overwhelming the API
        if stats.total_created < sandboxes_to_create and not graceful_shutdown.is_set():
            print("⏸️  Waiting 2 seconds before next batch...")
            time.sleep(2)

    # Check if we were interrupted during creation
    if graceful_shutdown.is_set():
        print(f"\n🛑 Graceful shutdown during creation. Created {stats.total_created} out of {sandboxes_to_create} requested NEW sandboxes.")

    # Wait for all sandboxes to be ready (if enabled)
    if wait_until_ready and stats.created_sandbox_ids:
        print(f"\n🕒 Waiting for watcher to detect all {len(stats.created_sandbox_ids)} sandboxes as ready...")

        # Wait for watcher to complete (with timeout)
        max_wait_time = 600  # 10 minutes
        start_wait = time.time()
        last_progress = 0

        while stats.ready_count < len(stats.created_sandbox_ids) and (time.time() - start_wait) < max_wait_time and not graceful_shutdown.is_set():
            elapsed = time.time() - start_wait

            # Progress update every 30 seconds
            if elapsed - last_progress >= 30:
                print(f"📊 Watcher progress: {stats.ready_count}/{len(stats.created_sandbox_ids)} ready after {elapsed:.0f}s")
                last_progress = elapsed

            time.sleep(5)  # Check every 5 seconds

        # Final status
        if graceful_shutdown.is_set():
            total_wait_time = time.time() - start_wait
            print(f"🛑 Graceful shutdown requested. {stats.ready_count}/{len(stats.created_sandbox_ids)} sandboxes ready after {total_wait_time:.2f}s")
        elif stats.ready_count >= len(stats.created_sandbox_ids):
            total_wait_time = time.time() - start_wait
            print(f"🎉 Watcher detected all sandboxes are ready! Total wait time: {total_wait_time:.2f}s")
        else:
            print(f"⏰ Timeout reached. {stats.ready_count}/{len(stats.created_sandbox_ids)} sandboxes ready")

        # Stop the watcher
        if watcher:
            watcher.stop_watching()

    elif not wait_until_ready and stats.created_sandbox_ids:
        print(f"\n⏭️  Skipping readiness wait for {len(stats.created_sandbox_ids)} sandboxes (fast mode)")
        print("💡 Sandboxes will continue provisioning in the background")

        # Stop the watcher if it was started (shouldn't happen, but just in case)
        if watcher:
            watcher.stop_watching()

    # Export detailed data to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"sandbox_creation_detailed_{timestamp}.csv"
    print(f"\n📊 Exporting detailed measurements to CSV...")
    stats.export_to_csv(csv_filename)

    # Print final statistics
    print("\n" + "=" * 60)
    print(stats.get_summary())

    # Print failed attempts if any
    if stats.failed_attempts:
        print("\n❌ Failed attempts:")
        for i, error in enumerate(stats.failed_attempts[:10], 1):  # Show first 10 errors
            print(f"   {i}. {error}")
        if len(stats.failed_attempts) > 10:
            print(f"   ... and {len(stats.failed_attempts) - 10} more")

        # Final verification
    print("\n🔍 Verifying final sandbox count...")
    final_count = get_current_user_sandbox_count(sandbox_client)
    print(f"📈 Final active sandboxes: {final_count}")

    expected_final = current_count + stats.total_created
    print(f"📊 Expected total: {expected_final} (started with {current_count}, created {stats.total_created} new)")

    if stats.total_created >= sandboxes_to_create:
        print(f"🎉 SUCCESS! Created {stats.total_created} NEW sandboxes as requested!")
    else:
        print(f"⚠️  Warning: Only created {stats.total_created} new sandboxes (target: {sandboxes_to_create})")

    print(f"\n📄 Detailed measurements saved to: {csv_filename}")
    print(f"💡 Use this CSV file for further analysis and visualization")


if __name__ == "__main__":
    main()
