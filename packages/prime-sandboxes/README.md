# Prime Sandboxes SDK

Lightweight Python SDK for managing Prime Intellect sandboxes - secure remote code execution environments.

## Features

- **Synchronous and async clients** - Use with sync or async/await code
- **Full sandbox lifecycle** - Create, list, execute commands, upload/download files, delete
- **Type-safe** - Full type hints and Pydantic models
- **Authentication caching** - Automatic token management
- **Bulk operations** - Create and manage multiple sandboxes efficiently
- **No CLI dependencies** - Pure SDK, ~50KB installed

## Installation

```bash
uv pip install prime-sandboxes
```

Or with pip:
```bash
pip install prime-sandboxes
```

## Quick Start

```python
from prime_sandboxes import APIClient, SandboxClient, CreateSandboxRequest

# Initialize
client = APIClient(api_key="your-api-key")
sandbox_client = SandboxClient(client)

# Create a sandbox
request = CreateSandboxRequest(
    name="my-sandbox",
    docker_image="python:3.11-slim",
    cpu_cores=2,
    memory_gb=4,
)

sandbox = sandbox_client.create(request)
print(f"Created: {sandbox.id}")

# Wait for it to be ready
sandbox_client.wait_for_creation(sandbox.id)

# Execute commands
result = sandbox_client.execute_command(sandbox.id, "python --version")
print(result.stdout)

# Clean up
sandbox_client.delete(sandbox.id)
```

### GPU Sandbox Example

```python
gpu_request = CreateSandboxRequest(
    name="gpu-sandbox",
    docker_image="nvidia/cuda:12.4.1-base-ubuntu22.04",
    gpu_count=1,
    gpu_type="H100_80GB",  # Required when gpu_count > 0
)

gpu_sandbox = sandbox_client.create(gpu_request)
```

## Async Usage

```python
import asyncio
from prime_sandboxes import AsyncSandboxClient, CreateSandboxRequest

async def main():
    async with AsyncSandboxClient(api_key="your-api-key") as client:
        # Create sandbox
        sandbox = await client.create(CreateSandboxRequest(
            name="async-sandbox",
            docker_image="python:3.11-slim",
        ))

        # Wait and execute
        await client.wait_for_creation(sandbox.id)
        result = await client.execute_command(sandbox.id, "echo 'Hello from async!'")
        print(result.stdout)

        # Clean up
        await client.delete(sandbox.id)

asyncio.run(main())
```

## Authentication

The SDK looks for credentials in this order:

1. **Direct parameter**: `APIClient(api_key="sk-...")`
2. **Environment variable**: `export PRIME_API_KEY="sk-..."`
3. **Config file**: `~/.prime/config.json` (created by `prime login` CLI command)

## Advanced Features

### Environment Variables and Secrets

```python
# Create sandbox with environment variables and secrets
request = CreateSandboxRequest(
    name="my-sandbox",
    docker_image="python:3.11-slim",
    environment_vars={
        "DEBUG": "true",
        "LOG_LEVEL": "info"
    },
    secrets={
        "API_KEY": "sk-secret-key-here",
        "DATABASE_PASSWORD": "super-secret-password"
    }
)

sandbox = sandbox_client.create(request)
```

**Note:** Secrets are never displayed in logs or outputs. When retrieving sandbox details, only the secret keys are shown with values masked as `***`.

### File Operations

```python
# Upload a file
sandbox_client.upload_file(
    sandbox_id=sandbox.id,
    file_path="/app/script.py",
    local_file_path="./local_script.py"
)

# Download a file
sandbox_client.download_file(
    sandbox_id=sandbox.id,
    file_path="/app/output.txt",
    local_file_path="./output.txt"
)
```

### Bulk Operations

```python
# Create multiple sandboxes
sandbox_ids = []
for i in range(5):
    sandbox = sandbox_client.create(CreateSandboxRequest(
        name=f"sandbox-{i}",
        docker_image="python:3.11-slim",
    ))
    sandbox_ids.append(sandbox.id)

# Wait for all to be ready
statuses = sandbox_client.bulk_wait_for_creation(sandbox_ids)

# Delete by IDs or labels
sandbox_client.bulk_delete(sandbox_ids=sandbox_ids)
# OR by labels
sandbox_client.bulk_delete(labels=["experiment-1"])
```

### Labels & Filtering

```python
# Create with labels
sandbox = sandbox_client.create(CreateSandboxRequest(
    name="labeled-sandbox",
    docker_image="python:3.11-slim",
    labels=["experiment", "ml-training"],
))

# List with filters
sandboxes = sandbox_client.list(
    status="RUNNING",
    labels=["experiment"],
    page=1,
    per_page=50,
)

for s in sandboxes.sandboxes:
    print(f"{s.name}: {s.status}")
```

### Long-Running Tasks

Use `start_background_job` to run long-running tasks that continue after the API call returns. Poll for completion with `get_background_job`.

```python
from prime_sandboxes import SandboxClient, CreateSandboxRequest

sandbox_client = SandboxClient()

# Create sandbox with extended timeout
sandbox = sandbox_client.create(CreateSandboxRequest(
    name="training-job",
    docker_image="python:3.11-slim",
    timeout_minutes=1440,  # 24 hours
    cpu_cores=4,
    memory_gb=16,
))
sandbox_client.wait_for_creation(sandbox.id)

# Start a long-running job in the background
job = sandbox_client.start_background_job(
    sandbox.id,
    "python train.py --epochs 100"
)
print(f"Job started: {job.job_id}")

# Poll for completion
import time
while True:
    status = sandbox_client.get_background_job(sandbox.id, job)
    if status.completed:
        print(f"Job finished with exit code: {status.exit_code}")
        print(status.stdout)
        break
    print("Still running...")
    time.sleep(30)

# Download results
sandbox_client.download_file(sandbox.id, "/app/model.pt", "./model.pt")
```

#### Async version

```python
import asyncio
from prime_sandboxes import AsyncSandboxClient, CreateSandboxRequest

async def run_training():
    async with AsyncSandboxClient() as client:
        sandbox = await client.create(CreateSandboxRequest(
            name="async-training",
            docker_image="python:3.11-slim",
            timeout_minutes=720,
        ))
        await client.wait_for_creation(sandbox.id)

        # Start background job
        job = await client.start_background_job(
            sandbox.id,
            "python train.py"
        )

        # Poll until done
        while True:
            status = await client.get_background_job(sandbox.id, job)
            if status.completed:
                print(status.stdout)
                break
            await asyncio.sleep(30)

        await client.delete(sandbox.id)

asyncio.run(run_training())
```

## Documentation

Full API reference: https://github.com/PrimeIntellect-ai/prime-cli/tree/main/packages/prime-sandboxes

## Related Packages

- **`prime`** - Full CLI + SDK with pods, inference, and more (includes this package)

## License

MIT License - see LICENSE file for details
