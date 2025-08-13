# Prime CLI Examples

This directory contains example scripts and demos for using the Prime CLI.

## Sandbox Demo

The `sandbox_demo.py` script demonstrates both programmatic and CLI usage of the sandbox functionality.

### Running the Demo

1. **Interactive Demo** (recommended for first-time users):
   ```bash
   python examples/sandbox_demo.py
   ```

2. **Programmatic Demo Only**:
   ```bash
   python examples/sandbox_demo.py --programmatic
   ```

3. **CLI Examples Only**:
   ```bash
   python examples/sandbox_demo.py --cli
   ```

### Prerequisites

- Prime CLI installed and configured
- Valid API key (run `prime login` first)
- Python environment with prime-cli package available

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
from prime_cli.api.client import APIClient
from prime_cli.api.sandbox import SandboxClient, CreateSandboxRequest

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

# Update status from Kubernetes
prime sandbox status SANDBOX_ID
```

## Error Handling

Both programmatic and CLI usage include proper error handling for:
- API authentication errors
- Network connectivity issues
- Invalid parameters
- Resource not found errors
- Rate limiting

See the demo script for examples of proper error handling patterns.