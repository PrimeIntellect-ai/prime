# Prime CLI Examples

This directory contains example scripts and demos for using the Prime CLI.

## Evals

Push evaluation results to the Environments Hub. Expects the [verifiers](https://github.com/PrimeIntellect-ai/verifiers) output format, a directory containing `metadata.json` and `results.jsonl`.

### Pushing Evals

1. **Push from current directory** (if it contains metadata.json and results.jsonl):
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
   prime eval push outputs/evals/gsm8k--gpt-4/abc123
   ```

4. **Push with environment override**:
   ```bash
   prime eval push outputs/evals/gsm8k--gpt-4/abc123 --env primeintellect/gsm8k
   ```

5. **Push to existing evaluation**:
   ```bash
   prime eval push outputs/evals/gsm8k--gpt-4/abc123 --eval <eval_id>
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

# Run file operations demo
uv run python examples/sandbox_file_operations.py

# Run file upload stress test with error handling (both modes)
uv run python examples/sandbox_file_handling_stress_test.py

# Run only sequential tests
uv run python examples/sandbox_file_handling_stress_test.py sequential

# Run only concurrent tests
uv run python examples/sandbox_file_handling_stress_test.py concurrent
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
    gpu_type=None,
    timeout_minutes=60,
    environment_vars={"ENV": "production"},
    secrets={"API_KEY": "your-secret-key"},
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
prime sandbox create IMAGE [OPTIONS]

# Create GPU sandbox
prime sandbox create --gpu-count 1 --gpu-type H100_80GB

# Note: DOCKER_IMAGE is not supported for GPU sandboxes

# With environment variables and secrets:
prime sandbox create python:3.11-slim --env KEY=VALUE --secret API_KEY=secret123

# Run command in sandbox
prime sandbox run SANDBOX_ID -- python script.py

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
