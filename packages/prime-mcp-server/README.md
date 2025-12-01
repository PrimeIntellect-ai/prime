# Prime MCP Server

Model Context Protocol (MCP) server for Prime Intellect - manage GPU pods, check availability, and control compute resources through MCP.

## Features

- **GPU Availability Checking** - Search for available GPU instances across providers
- **Pod Management** - Create, list, monitor, and delete GPU pods
- **Cluster Support** - Check multi-node cluster availability
- **SSH Key Management** - Manage SSH keys for pod access

## Demo

![Prime MCP Demo](mcp-demo.gif)

## Installation

```bash
pip install prime-mcp-server
```

Or with uv:
```bash
uv pip install prime-mcp-server
```

## Configuration

Prime MCP uses `prime-core` for configuration, which supports multiple authentication methods:

### Option 1: Environment Variable
```bash
export PRIME_API_KEY="your-api-key"
```

### Option 2: Prime CLI Login (Recommended)
```bash
prime login
```
This stores your API key in `~/.prime/config.json` and is shared across all Prime tools.

### Option 3: Config File
Manually edit `~/.prime/config.json`:
```json
{
  "api_key": "your-api-key",
  "base_url": "https://api.primeintellect.ai"
}
```

## Quick Start

### Running the MCP Server

```bash
python -m prime_mcp.mcp
```

The server runs over stdio and can be integrated with MCP clients (like Claude Desktop, IDEs, or other MCP-compatible tools).

## Available Tools

For complete documentation of all available tools and their parameters, see [docs.primeintellect.ai](https://docs.primeintellect.ai/api-reference/introduction).

## Development

```bash
# Clone the repo
git clone https://github.com/PrimeIntellect-ai/prime.git
cd prime/packages/prime-mcp-server

# Install dependencies
uv sync

# Run tests
uv run pytest

# Run linter
uv run ruff check .
```

## Related Packages

- **`prime`** - Full CLI + SDK with pods, sandboxes, inference, and more
- **`prime-sandboxes`** - SDK for managing remote code execution environments
- **`prime-evals`** - SDK for managing model evaluations
