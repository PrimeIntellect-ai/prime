# CLI Reference

The Prime CLI provides commands for managing GPU pods, sandboxes, environments, and more.

## Global Options

```bash
prime --help          # Show all commands
prime --version       # Show version
```

## Authentication

```bash
# Interactive login (opens browser)
prime login

# Check current user
prime whoami
```

## GPU Availability

Query available GPU resources across providers:

```bash
# List all available GPUs
prime availability list

# Filter by GPU type
prime availability list --gpu-type H100_80GB
prime availability list --gpu-type A100_80GB

# Show available GPU types
prime availability gpu-types

# Filter by region
prime availability list --region us-east

# Show detailed pricing
prime availability list --show-pricing
```

## Pod Management

Deploy and manage GPU compute instances:

```bash
# List your pods
prime pods list

# Create a pod (interactive)
prime pods create

# Create with specific configuration
prime pods create --id <config-id>
prime pods create --name my-training-pod
prime pods create --gpu H100 --count 8

# Monitor pod status
prime pods status <pod-id>

# SSH into a pod
prime pods ssh <pod-id>

# Terminate a pod
prime pods terminate <pod-id>
```

## Sandboxes

Manage isolated code execution environments:

```bash
# Create a sandbox
prime sandbox create python:3.11
prime sandbox create ubuntu:22.04 --name my-sandbox

# List sandboxes
prime sandbox list
prime sandbox list --status RUNNING

# Execute commands
prime sandbox exec <sandbox-id> "python script.py"
prime sandbox exec <sandbox-id> "pip install numpy && python -c 'import numpy; print(numpy.__version__)'"

# File operations
prime sandbox upload <sandbox-id> local_file.py /remote/path/
prime sandbox download <sandbox-id> /remote/file.txt ./local/

# Delete sandbox
prime sandbox delete <sandbox-id>
```

## Environments Hub

Access hundreds of verified environments:

```bash
# Browse available environments
prime env list

# Search environments
prime env list --search "math"

# View environment details
prime env info <environment-name>

# Install an environment locally
prime env install <environment-name>

# Create your own environment
prime env init my-environment

# Push environment to hub
prime env push my-environment
```

## Evaluations

Push and manage evaluation results:

```bash
# Auto-discover and push evaluations from current directory
prime eval push

# Push specific directory
prime eval push examples/verifiers_example

# List all evaluations
prime eval list

# Get evaluation details
prime eval get <eval-id>

# View evaluation samples
prime eval samples <eval-id>

# Delete an evaluation
prime eval delete <eval-id>
```

## Teams

Manage team resources:

```bash
# List teams
prime teams list

# Set team context
prime config set-team-id
prime config set-team-id <team-id>
```

## Configuration

```bash
# View current configuration
prime config view

# Set API key
prime config set-api-key

# Set SSH key path
prime config set-ssh-key-path

# Set team ID
prime config set-team-id
```

## Inference

Run inference requests:

```bash
# Chat completion
prime inference chat "What is 2+2?"

# With specific model
prime inference chat "Explain quantum computing" --model gpt-4

# Streaming response
prime inference chat "Write a poem" --stream
```

## Images

Manage container images:

```bash
# List available images
prime images list

# Get image details
prime images info <image-id>
```

## Registry

Manage private registries:

```bash
# List registry credentials
prime registry list

# Add registry credentials
prime registry add --name my-registry --server ghcr.io
```

## Disks

Manage persistent storage:

```bash
# List disks
prime disks list

# Create disk
prime disks create --name my-data --size 100

# Delete disk
prime disks delete <disk-id>
```

