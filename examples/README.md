# Prime CLI Examples

This directory contains example scripts and demos for using the Prime CLI.

## Evals Example

Prime CLI supports two modes for pushing evaluation results:

2. **Verifiers Format**: Directory with `metadata.json` and `results.jsonl`
1. **JSON Format**: Single JSON file with all evaluation data

### Pushing Evals

1. **Push from current directory** (if it contains metadata.json/results.jsonl):
   ```bash
   cd outputs/evals/gsm8k--gpt-4/abc123
   prime eval push
   ```

2. **Auto-discover and push all** (from root with outputs/evals/):
   ```bash
   prime eval push
   ```

3. **Push specific directory**:
   ```bash
   prime eval push examples/verifiers_example
   ```

4. **Push from JSON file with environment**:
   ```bash
   prime eval push examples/eval_example.json --env gsm8k
   ```

5. **Push with run ID** (link to existing training run):
   ```bash
   prime eval push examples/eval_example.json --run-id abc123
   ```

6. **List all evals**:
   ```bash
   prime eval list
   ```

7. **Get specific eval**:
   ```bash
   prime eval get <eval_id>
   ```

8. **View eval samples**:
   ```bash
   prime eval samples <eval_id>
   ```


## Sandbox Demo

The `sandbox_demo.py` script demonstrates both programmatic and CLI usage of the sandbox functionality.

### Running the Demo

From the repository root:

```bash
# Run the basic demo
uv run python examples/sandbox_demo.py

# Run async demo
uv run python examples/sandbox_async_demo.py

# Run high-volume async demo
uv run python examples/sandbox_async_high_volume_demo.py

# Run with debug logging to see detailed progress
LOG_LEVEL=DEBUG uv run python examples/sandbox_async_high_volume_demo.py

# Run file operations demo
uv run python examples/sandbox_file_operations.py

# Run file upload stress test with error handling (both modes)
uv run python examples/sandbox_file_handling_stress_test.py

# Run only sequential tests
uv run python examples/sandbox_file_handling_stress_test.py sequential

# Run only concurrent tests
uv run python examples/sandbox_file_handling_stress_test.py concurrent

# Run multi-sandbox file upload stress test (default: 10 sandboxes, 20 concurrent)
uv run python examples/sandbox_multi_upload_stress_test.py

# Custom configuration: 5 sandboxes, 15 concurrent uploads
uv run python examples/sandbox_multi_upload_stress_test.py 5 15

# Heavy load test: 100 sandboxes, 50 concurrent uploads
uv run python examples/sandbox_multi_upload_stress_test.py 100 50

# Extreme stress test: 500 sandboxes, 100 concurrent uploads
uv run python examples/sandbox_multi_upload_stress_test.py 500 100
```

### Prerequisites

- Repository cloned and set up: `uv sync`
- Valid API key (run `uv run prime login` first)

### What the Demo Shows

**Programmatic Usage:**

- Creating sandboxes with custom configurations
- Listing and filtering sandboxes
- Getting detailed sandbox information
- Updating sandbox settings
- Retrieving logs
- Deleting sandboxes
- Error handling

**CLI Usage Examples:**

- All available sandbox commands
- Common parameter combinations
- Environment variable handling

### File Handling & Error Testing

The `sandbox_file_handling_stress_test.py` demonstrates:

- **File Upload Testing**:
  - Sequential mode: Tests 10, 20, 25, 30 MB files one at a time
  - Concurrent mode: Stress tests with 15 simultaneous 5MB file uploads
- **Multiple Test Modes**: Choose between sequential, concurrent, or both modes via command-line argument
- **Performance Measurement**: Measures upload time and calculates transfer speeds
- **Concurrency Comparison**: Compares sequential vs concurrent upload performance
- **Full Error Details**: Shows complete API error information including HTTP status codes and response bodies
- **Error Type Detection**: Catches and displays specific error types (HTTPStatusError, UnauthorizedError, PaymentRequiredError, APIError)
- **Disk Space Monitoring**: Checks available disk space before and after uploads
- **Test Summary**: Provides detailed statistics including success rates and performance metrics

This example is useful for:
- Testing sandbox file upload limits
- Identifying performance bottlenecks
- Validating error handling for large file transfers
- Debugging API errors with full response details
- Benchmarking upload speeds (sequential vs concurrent)
- Stress testing concurrent upload behavior with 15 simultaneous uploads

### Multi-Sandbox File Upload Stress Testing

The `sandbox_multi_upload_stress_test.py` demonstrates comprehensive stress testing across multiple sandboxes:

- **Multiple Sandboxes**: Creates and tests many sandboxes simultaneously (configurable: default 10)
- **Diverse File Patterns**: Each sandbox receives a different file size pattern for realistic load testing:
  - Pattern 0: Many small files (10x 1MB)
  - Pattern 1: Small to medium (1-5MB mixed)
  - Pattern 2: Medium files (5-10MB)
  - Pattern 3: Large files (10-30MB)
  - Pattern 4: Maximum size files (4x 30MB - max allowed)
  - Pattern 5: Mixed sizes (1-30MB comprehensive)
  - Pattern 6: Rapid small (20x 1MB stress test)
  - Pattern 7: Medium rapid (10x 5MB)
  - Pattern 8: Large batch (10-30MB)
  - Pattern 9: Extreme contrast (1-30MB mixed)
- **Concurrent Upload Control**: Configurable semaphore limiting (default: 20 concurrent uploads)
- **Comprehensive Metrics**:
  - Overall statistics (success rate, throughput, data volume)
  - Performance by file size
  - Performance by pattern (identifies which patterns are problematic)
  - Per-sandbox statistics with pattern tracking
  - Failure analysis with error grouping and pattern detection
- **File Verification**: Validates all uploaded files exist with correct sizes
- **Disk Space Monitoring**: Tracks disk usage per sandbox
- **Automatic Cleanup**: Deletes all sandboxes after testing

This example is useful for:
- Large-scale stress testing of file upload infrastructure
- Identifying performance differences across file sizes
- Testing system behavior under varied concurrent load
- Validating resource management at scale
- Detecting pattern-specific issues (e.g., small files vs large files)
- Benchmarking multi-sandbox upload throughput
- Testing failure isolation (system-wide vs per-sandbox issues)

## Sandbox API Reference

### Creating Sandboxes Programmatically

```python
from prime_core import APIClient
from prime_sandboxes import SandboxClient, CreateSandboxRequest

# Initialize client
client = APIClient()
sandbox_client = SandboxClient(client)

# Create sandbox
request = CreateSandboxRequest(
    name="my-sandbox",
    docker_image="python:3.11-slim",
    start_command="python app.py",
    cpu_cores=2,
    memory_gb=4,
    disk_size_gb=20,
    gpu_count=0,
    timeout_minutes=60,
    environment_vars={"ENV": "production"},
    team_id=None  # Use None for personal account
)

sandbox = sandbox_client.create(request)
print(f"Created sandbox: {sandbox.id}")
```

### CLI Command Reference

```bash
# List sandboxes
prime sandbox list [--team_id TEAM] [--status STATUS] [--page N] [--per_page N]

# Create sandbox
prime sandbox create NAME --docker_image IMAGE [OPTIONS]

# Get sandbox details
prime sandbox get SANDBOX_ID

# Update sandbox
prime sandbox update SANDBOX_ID [OPTIONS]

# Delete sandbox
prime sandbox delete SANDBOX_ID

# Get logs
prime sandbox logs SANDBOX_ID
```

## Error Handling

Both programmatic and CLI usage include proper error handling for:

- API authentication errors
- Network connectivity issues
- Invalid parameters
- Resource not found errors
- Rate limiting

See the demo script for examples of proper error handling patterns.
