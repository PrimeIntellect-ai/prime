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

Command line interface and SDKs for Hosted Training, hosted evaluations, GPU resources, sandboxes, and environments.

</div>

## Overview

Prime is the official CLI and Python SDK for [Prime Intellect](https://primeintellect.ai), providing seamless access to Hosted Training, hosted evaluations, GPU compute infrastructure, remote code execution environments (sandboxes), and AI inference capabilities.

**What can you do with Prime?**

- Deploy GPU pods with H100, A100, and other high-performance GPUs
- Set up Lab workspaces for verifiers environments, evals, GEPA, and training
- Discover and launch Hosted Training runs against verifiers environments
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

# Cap what the key minted by this login can do (see `prime login --help`)
prime login --max-concurrent-sandboxes 10 --max-concurrent-tunnels 2

# Or set API key directly
prime config set-api-key

# Or use environment variable
export PRIME_API_KEY="your-api-key-here"
```

Get your API key from the [Prime Intellect Dashboard](https://app.primeintellect.ai).

### Basic Usage

```bash
# Browse environments on the hub
prime env list

# Set up a Lab workspace for environments, evals, GEPA, and Hosted Training
prime lab setup

# See available Hosted Training models, capacity, and pricing
prime train models

# Generate and launch a Hosted Training config
prime train init
prime train rl.toml

# List available GPUs
prime availability list

# Create a GPU pod
prime pods create --gpu A100 --count 1

# SSH into a pod
prime pods ssh <pod-id>

# Create a sandbox
prime sandbox create python:3.11
```

## Features

### Lab and Hosted Training

Start with `prime lab setup` to create a local workspace with starter configs, coding-agent skills and a verifiers install; build and evaluate environments there with verifiers' `vf-init` and `vf-eval`. Then use `prime train models` to choose a Hosted Training model with current capacity and pricing.

```bash
# Set up a Lab workspace
prime lab setup

# List trainable models, capacity, and token pricing
prime train models

# Generate a Hosted Training config
prime train init

# Launch the run from the generated config
prime train rl.toml

# Inspect and manage Hosted Training runs
prime train list
prime train logs <run-id> -f
prime train metrics <run-id>
prime train checkpoints <run-id>
```

### Environments Hub

Access hundreds of RL environments on our community hub with deep integrations with sandboxes, training, and evaluation stack.

```bash
# Browse available environments
prime env list

# View environment details
prime env info <environment-name>

# Inspect environment source without downloading the archive
prime env inspect <environment-name>

# Install an environment locally
prime env install <environment-name>

# Push your own environment (scaffold one with verifiers' `vf-init`)
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
# Create a sandbox (VM-backed)
prime sandbox create python:3.11

# Create a sandbox with GPUs
prime sandbox create user-1/vm-image:latest --gpu-count 1 --gpu-type RTX_PRO_6000

# Create a one-shot workload (arguments after -- are preserved exactly)
prime sandbox create user-1/vm-image:latest -- /worker --platform linux/amd64

# List sandboxes
prime sandbox list

# Execute commands
prime sandbox run <sandbox-id> -- python script.py

# Run as an existing guest account
prime sandbox run <sandbox-id> --user ubuntu -- id

# Request a filesystem checkpoint; check again until it is DURABLE
prime sandbox checkpoint create <sandbox-id>
prime sandbox checkpoint list <sandbox-id> [--checkpoint-id <checkpoint-id>]
prime sandbox checkpoint restore <checkpoint-id> --name restored-sandbox
# Checkpoint + restore in one step, inheriting the source's resources
prime sandbox fork <sandbox-id> [--name forked-sandbox]

# Upload/download files
prime sandbox upload <sandbox-id> local_file.py /remote/path/
prime sandbox download <sandbox-id> /remote/file.txt ./local/

# Clean up
prime sandbox delete <sandbox-id>
```

### Team Management

Manage resources across personal and team contexts:

```bash
# List your teams
prime teams list

# Switch context directly
prime switch
prime switch personal
prime switch <team-slug>
prime switch <team-id>  # fallback for teams without a slug

# All subsequent commands use the selected context
prime pods list
```

## Portable training volumes (pilot)

Where JuiceFS is enabled for your team (or you), `prime volumes create` makes a
portable JuiceFS volume by default. It can be attached to other JuiceFS clusters
without copying its data. Elsewhere, and with `--backend cluster`, the volume uses
the cluster's own storage and stays on that cluster.

```bash
prime volumes create models --size 1Ti
prime volumes attach models --cluster e2e-spk
prime volumes list
```

`--cluster` is optional. For a JuiceFS volume it only picks the cluster where the
volume is first mounted and where `prime volumes ssh` sessions run; other clusters
use it through `prime volumes attach`. A `--cluster` without JuiceFS gets a cluster
volume there.

Creation and attachment are asynchronous. Wait for the volume and attachment to
show `RUNNING` before using them. An attachment makes the cluster eligible for
training; it does not select that cluster or move a running job.

```bash
prime train config.toml --volume models
prime volumes detach models --cluster e2e-spk
```

Detaching removes an unused cluster binding, not the data. The backend rejects a
detach while consumers still use it, and volume deletion requires the remote
attachments to be removed first. `prime volumes list -o json` includes the backend
and each attachment's cluster and status.

This pilot requires the matching platform backend and storage configuration.
Portable-volume resizing, SSH on a remote attachment, and serving an exported
model through managed inference are separate backend rollout steps; these CLI
commands do not enable them.

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

# Request a filesystem checkpoint and wait until it is durable
checkpoint = sandbox_client.checkpoint(sandbox.id)
checkpoint = sandbox_client.wait_for_checkpoint(checkpoint.id, timeout_seconds=300)
restored = sandbox_client.create(CreateSandboxRequest(
    name="restored-sandbox", checkpoint_id=checkpoint.id
))

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

- **Documentation**: [github.com/PrimeIntellect-ai/prime](https://github.com/PrimeIntellect-ai/prime)
- **Dashboard**: [app.primeintellect.ai](https://app.primeintellect.ai)
- **API Docs**: [api.primeintellect.ai/docs](https://api.primeintellect.ai/docs)
- **Discord**: [discord.gg/primeintellect](https://discord.gg/primeintellect)
- **Website**: [primeintellect.ai](https://primeintellect.ai)

## Related Packages

- **prime-sandboxes** - Lightweight SDK for sandboxes only (if you don't need the full CLI)

## License

MIT License - see [LICENSE](https://github.com/PrimeIntellect-ai/prime/blob/main/LICENSE) file for details.
