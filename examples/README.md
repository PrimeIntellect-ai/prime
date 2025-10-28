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
