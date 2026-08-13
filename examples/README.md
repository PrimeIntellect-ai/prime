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
- Executing commands in a sandbox
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
from prime_sandboxes import APIClient, SandboxClient, CreateSandboxRequest, StartCommand

# Initialize client
client = APIClient()
sandbox_client = SandboxClient(client)

# Create sandbox
request = CreateSandboxRequest(
    name="my-sandbox",
    docker_image="python:3.11-slim",
    cpu_cores=2,
    memory_gb=4,
    disk_size_gb=20,
    timeout_minutes=60,
    environment_vars={"ENV": "production"},
    secrets={"API_KEY": "your-secret-key"},
    team_id=None,  # Use None for personal account
)

sandbox = sandbox_client.create(request)
print(f"Created sandbox: {sandbox.id}")
```

VM sandboxes are opt-in with `vm=True`. They take a structured argv start command
(`StartCommand`) instead of a command string, and support GPUs and network rules:

```python
vm_request = CreateSandboxRequest(
    name="my-vm-sandbox",
    docker_image="user-1/vm-image:latest",
    vm=True,
    start_command=StartCommand(executable="python", args=["serve.py", "--port", "8000"]),
    gpu_count=1,
    gpu_type="RTX_PRO_6000",  # required when gpu_count > 0; GPUs require vm=True
    network_allowlist=["api.openai.com"],  # mutually exclusive with network_denylist
)

vm = sandbox_client.create(vm_request)
```

### CLI Command Reference

```bash
# List sandboxes
prime sandbox list [--team-id TEAM] [--status STATUS] [--label LABEL] [--page N] [--num N] [--all]

# Create sandbox
prime sandbox create IMAGE [OPTIONS]

# Create VM sandbox with GPUs (GPUs require --vm, and --gpu-type when --gpu-count > 0)
prime sandbox create user-1/vm-image:latest --vm --gpu-count 1 --gpu-type RTX_PRO_6000

# Create CPU-only VM sandbox
prime sandbox create user-1/vm-image:latest --vm

# VM start command: each argv token is separate after --, no shell is involved
prime sandbox create user-1/vm-image:latest --vm -- python serve.py --port 8000

# Restrict VM egress (--network-allow/--network-deny are VM only, repeatable, mutually exclusive)
prime sandbox create user-1/vm-image:latest --vm --network-allow api.openai.com
prime sandbox create user-1/vm-image:latest --vm --network-deny 0.0.0.0/0

# With environment variables and secrets:
prime sandbox create python:3.11-slim --env KEY=VALUE --secret API_KEY=secret123

# Other create options: --name, --cpu-cores, --memory-gb, --disk-size-gb,
# --timeout-minutes, --idle-timeout-minutes, --team-id, --region, --label, --yes

# Run command in sandbox
prime sandbox run SANDBOX_ID -- python script.py

# Get sandbox details
prime sandbox get SANDBOX_ID [--output json]

# Show or replace VM network rules (VM only)
prime sandbox network SANDBOX_ID
prime sandbox network SANDBOX_ID --allow api.openai.com,10.0.0.0/8

# Delete sandboxes (by ID, by label, or all)
prime sandbox delete SANDBOX_ID
prime sandbox delete --label experiment-1
prime sandbox delete --all --yes

# Get logs
prime sandbox logs SANDBOX_ID

# Upload/download files
prime sandbox upload SANDBOX_ID local_file.py /remote/path/file.py
prime sandbox download SANDBOX_ID /remote/file.txt ./local/file.txt

# Expose ports and SSH (container sandboxes)
prime sandbox expose SANDBOX_ID 8000 [--protocol HTTP|TCP]
prime sandbox list-ports [SANDBOX_ID]
prime sandbox unexpose SANDBOX_ID EXPOSURE_ID
prime sandbox ssh SANDBOX_ID
```

### Image Command Reference

```bash
# Build and push an image from a Dockerfile (linux/amd64 by default)
prime images push myapp:v1.0.0 --context ./app --dockerfile ./app/Dockerfile

# Copy an existing public image into Prime instead of building
prime images push myubuntu:22.04 --source-image ubuntu:22.04

# Pre-build the VM artifact for an existing image (otherwise the first --vm
# sandbox using that image triggers a one-time conversion)
prime images build-vm myapp:v1.0.0

# List images
prime images list [--search TERM] [--page N] [--num N] [--output json]

# Change visibility
prime images publish myapp:v1.0.0
prime images unpublish myapp:v1.0.0

# Rename, retag, or move an image
prime images update myapp:v1 --name myapp-final --tag v2

# Delete an image
prime images delete myapp:v1.0.0 --yes
```

## Error Handling

Both programmatic and CLI usage include proper error handling for:

- API authentication errors
- Network connectivity issues
- Invalid parameters
- Resource not found errors
- Rate limiting

See the demo script for examples of proper error handling patterns.
