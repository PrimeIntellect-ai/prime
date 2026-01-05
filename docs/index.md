# Docs

This directory maintains the documentation for Prime CLI & SDKs. It is organized into the following sections:

- [**Installation**](installation.md) - Installation instructions for CLI and SDKs
- [**Configuration**](configuration.md) - API key setup, SSH keys, and team configuration
- [**CLI**](cli.md) - Command-line interface reference for pods, sandboxes, and more
- [**Environments**](environments.md) - Access and manage verified environments on the community hub
- [**Sandboxes**](sandboxes.md) - Python SDK for remote code execution environments
- [**Evals**](evals.md) - SDK for pushing and managing evaluation results
- [**MCP Server**](mcp.md) - Model Context Protocol server for AI assistants

## Overview

Prime is the official CLI and Python SDK for [Prime Intellect](https://primeintellect.ai), providing seamless access to GPU compute infrastructure, remote code execution environments (sandboxes), and AI inference capabilities.

**What can you do with Prime?**

- Deploy GPU pods with H100, A100, and other high-performance GPUs
- Create and manage isolated sandbox environments for running code
- Access hundreds of pre-configured development environments
- SSH directly into your compute instances
- Manage team resources and permissions
- Push and track model evaluation results

## Packages

| Package | Description | Install |
|---------|-------------|---------|
| `prime` | Full CLI + SDK with pods, sandboxes, inference, and more | `uv tool install prime` |
| `prime-sandboxes` | Lightweight SDK for sandboxes only (~50KB) | `uv pip install prime-sandboxes` |
| `prime-evals` | SDK for managing evaluation results | `uv pip install prime-evals` |
| `prime-mcp-server` | MCP server for AI assistants | `uv pip install prime-mcp-server` |

## Quick Start

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install prime
uv tool install prime

# Authenticate
prime login

# Browse verified environments
prime env list

# List available GPU resources
prime availability list
```

