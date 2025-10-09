<p align="center">
  <picture>
    <source media="(prefers-color-scheme: light)" srcset="https://github.com/user-attachments/assets/40c36e38-c5bd-4c5a-9cb3-f7b902cd155d">
    <source media="(prefers-color-scheme: dark)" srcset="https://github.com/user-attachments/assets/6414bc9b-126b-41ca-9307-9e982430cde8">
    <img alt="Prime Intellect" src="https://github.com/user-attachments/assets/40c36e38-c5bd-4c5a-9cb3-f7b902cd155d" width="312" style="max-width: 100%;">
  </picture>
</p>

---

<h3 align="center">
Prime Intellect CLI & SDKs
</h3>

---

<div align="center">

[![PyPI version](https://img.shields.io/pypi/v/prime?cacheSeconds=60)](https://pypi.org/project/prime/)
[![Python versions](https://img.shields.io/pypi/pyversions/prime?cacheSeconds=60)](https://pypi.org/project/prime/)
[![Downloads](https://img.shields.io/pypi/dm/prime)](https://pypi.org/project/prime/)

Command line interface and SDKs for managing Prime Intellect GPU resources, sandboxes, and environments.
</div>

## Overview

Prime is the official CLI and Python SDK for [Prime Intellect](https://primeintellect.ai), providing seamless access to GPU compute infrastructure, remote code execution environments (sandboxes), and AI inference capabilities.

**What can you do with Prime?**

- Deploy GPU pods with H100, A100, and other high-performance GPUs
- Create and manage isolated sandbox environments for running code
- Access hundreds of pre-configured development environments
- SSH directly into your compute instances
- Manage team resources and permissions
- Run OpenAI-compatible inference requests

## Installation

### Using uv (recommended)

First, install uv if you haven't already:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then install prime:

```bash
uv tool install prime
```

### Using pip

```bash
pip install prime
```

## Quick Start

### Authentication

```bash
# Interactive login (recommended)
prime login

# Or set API key directly
prime config set-api-key

# Or use environment variable
export PRIME_API_KEY="your-api-key-here"
```

Get your API key from the [Prime Intellect Dashboard](https://app.primeintellect.ai).

### Basic Usage

```bash
# Browse verified environments
prime env list

# List available GPUs
prime availability list

# Create a GPU pod
prime pods create --gpu A100 --count 1

# SSH into a pod
prime pods ssh <pod-id>

# Create a sandbox
prime sandbox create --image python:3.11
```

## Features

### Environments Hub

Access hundreds of verified environments on our community hub with deep integrations with sandboxes, training, and evaluation stack.

```bash
# Browse available environments
prime env list

# View environment details
prime env info <environment-name>

# Install an environment locally
prime env install <environment-name>

# Create and push your own environment
prime env init my-environment
prime env push my-environment
```

Environments provide pre-configured setups for machine learning, data science, and development workflows, tested and verified by the Prime Intellect community.

### GPU Pod Management

Deploy and manage GPU compute instances:

```bash
# Browse available configurations
prime availability list --gpu-type H100_80GB

# Create a pod with specific configuration
prime pods create --id <config-id> --name my-training-pod

# Monitor pod status
prime pods status <pod-id>

# SSH access
prime pods ssh <pod-id>

# Terminate when done
prime pods terminate <pod-id>
```

### Sandboxes

Isolated environments for running code remotely:

```bash
# Create a sandbox
prime sandbox create --image python:3.11

# List sandboxes
prime sandbox list

# Execute commands
prime sandbox exec <sandbox-id> "python script.py"

# Upload/download files
prime sandbox upload <sandbox-id> local_file.py /remote/path/
prime sandbox download <sandbox-id> /remote/file.txt ./local/

# Clean up
prime sandbox delete <sandbox-id>
```

### Team Management

Manage resources across team contexts:

```bash
# List your teams
prime teams list

# Set team context
prime config set-team-id <team-id>

# All subsequent commands use team context
prime pods list  # Shows team's pods
```

## Configuration

### API Key

Multiple ways to configure your API key:

```bash
# Option 1: Interactive (hides input)
prime config set-api-key

# Option 2: Direct
prime config set-api-key YOUR_API_KEY

# Option 3: Environment variable
export PRIME_API_KEY="your-api-key"
```

Configuration priority: CLI config > Environment variable

### SSH Key

Configure SSH key for pod access:

```bash
prime config set-ssh-key-path ~/.ssh/id_rsa.pub
```

### View Configuration

```bash
prime config view
```

## Python SDK

Prime also provides a Python SDK for programmatic access:

```python
from prime_sandboxes import APIClient, SandboxClient, CreateSandboxRequest

# Initialize client
client = APIClient(api_key="your-api-key")
sandbox_client = SandboxClient(client)

# Create a sandbox
sandbox = sandbox_client.create(CreateSandboxRequest(
    name="my-sandbox",
    docker_image="python:3.11-slim",
    cpu_cores=2,
    memory_gb=4,
))

# Wait for creation
sandbox_client.wait_for_creation(sandbox.id)

# Execute commands
result = sandbox_client.execute_command(sandbox.id, "python --version")
print(result.stdout)

# Clean up
sandbox_client.delete(sandbox.id)
```

### Async SDK

```python
import asyncio
from prime_sandboxes import AsyncSandboxClient, CreateSandboxRequest

async def main():
    async with AsyncSandboxClient(api_key="your-api-key") as client:
        sandbox = await client.create(CreateSandboxRequest(
            name="async-sandbox",
            docker_image="python:3.11-slim",
        ))

        await client.wait_for_creation(sandbox.id)
        result = await client.execute_command(sandbox.id, "echo 'Hello'")
        print(result.stdout)

        await client.delete(sandbox.id)

asyncio.run(main())
```

## Use Cases

### Machine Learning Training

```bash
# Deploy a pod with 8x H100 GPUs
prime pods create --gpu H100 --count 8 --name ml-training

# SSH and start training
prime pods ssh <pod-id>
```
## Support & Resources

- **Documentation**: [github.com/PrimeIntellect-ai/prime-cli](https://github.com/PrimeIntellect-ai/prime-cli)
- **Dashboard**: [app.primeintellect.ai](https://app.primeintellect.ai)
- **API Docs**: [api.primeintellect.ai/docs](https://api.primeintellect.ai/docs)
- **Discord**: [discord.gg/primeintellect](https://discord.gg/primeintellect)
- **Website**: [primeintellect.ai](https://primeintellect.ai)

## Related Packages

- **prime-sandboxes** - Lightweight SDK for sandboxes only (if you don't need the full CLI)

## License

MIT License - see [LICENSE](https://github.com/PrimeIntellect-ai/prime-cli/blob/main/LICENSE) file for details.
