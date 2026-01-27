# MCP Server

Model Context Protocol (MCP) server for Prime Intellect - manage GPU pods, check availability, and control compute resources through MCP-compatible AI assistants.

## Features

- **GPU Availability Checking** - Search for available GPU instances across providers
- **Pod Management** - Create, list, monitor, and delete GPU pods
- **Cluster Support** - Check multi-node cluster availability
- **SSH Key Management** - Manage SSH keys for pod access
- **Sandbox Operations** - Create and manage remote code execution environments

## Installation

```bash
uv pip install prime-mcp-server
```

Or with pip:

```bash
pip install prime-mcp-server
```

## Configuration

Prime MCP uses `prime-core` for configuration, which supports multiple authentication methods.

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

## Running the Server

```bash
python -m prime_mcp.mcp
```

The server runs over stdio and can be integrated with MCP clients (like Claude Desktop, Cursor, or other MCP-compatible tools).

## Integration with Claude Desktop

Add to your Claude Desktop configuration (`~/Library/Application Support/Claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "prime": {
      "command": "python",
      "args": ["-m", "prime_mcp.mcp"],
      "env": {
        "PRIME_API_KEY": "your-api-key"
      }
    }
  }
}
```

## Integration with Cursor

Add to your Cursor MCP configuration:

```json
{
  "mcpServers": {
    "primeintellect": {
      "command": "uvx",
      "args": ["prime-mcp-server"]
    }
  }
}
```

## Available Tools

### GPU Availability

| Tool | Description |
|------|-------------|
| `check_gpu_availability` | Search for available GPU instances |
| `check_cluster_availability` | Check multi-node cluster options |

### Pod Management

| Tool | Description |
|------|-------------|
| `list_pods` | List all pods in your account |
| `get_pod_details` | Get detailed pod information |
| `get_pods_status` | Get status of specific pods |
| `get_pods_history` | Get historical pod data |
| `create_pod` | Create a new GPU pod |
| `delete_pod` | Terminate a pod |

### SSH Keys

| Tool | Description |
|------|-------------|
| `manage_ssh_keys` | List, add, delete, or set primary SSH keys |

### Sandboxes

| Tool | Description |
|------|-------------|
| `create_sandbox` | Create a new sandbox environment |
| `list_sandboxes` | List all sandboxes |
| `get_sandbox` | Get sandbox details |
| `delete_sandbox` | Delete a sandbox |
| `bulk_delete_sandboxes` | Delete multiple sandboxes |
| `execute_sandbox_command` | Run commands in a sandbox |
| `get_sandbox_logs` | Get sandbox container logs |
| `expose_sandbox_port` | Expose a port to the internet |
| `unexpose_sandbox_port` | Remove port exposure |
| `list_sandbox_exposed_ports` | List exposed ports |

### Docker Registry

| Tool | Description |
|------|-------------|
| `list_registry_credentials` | List private registry credentials |
| `check_docker_image` | Validate Docker image accessibility |

## Example Usage

Once configured with an MCP client, you can interact naturally:

**User:** "Show me available H100 GPUs"

**Assistant:** Uses `check_gpu_availability` with `gpu_type="H100_80GB"`

**User:** "Create a pod with 4 A100 GPUs for training"

**Assistant:** 
1. Checks SSH keys with `manage_ssh_keys`
2. Finds availability with `check_gpu_availability`
3. Creates pod with `create_pod`

**User:** "Create a sandbox and run my Python script"

**Assistant:**
1. Creates sandbox with `create_sandbox`
2. Waits for ready state with `get_sandbox`
3. Executes code with `execute_sandbox_command`

## Development

```bash
# Clone the repo
git clone https://github.com/PrimeIntellect-ai/prime-cli
cd prime-cli/packages/prime-mcp-server

# Install dependencies
uv sync

# Run tests
uv run pytest

# Run linter
uv run ruff check .
```

## Troubleshooting

### Authentication Errors

Ensure your API key is correctly configured:

```bash
# Check if key is set
echo $PRIME_API_KEY

# Or verify config file
cat ~/.prime/config.json
```

### Server Not Starting

Check Python path and package installation:

```bash
# Verify installation
python -c "import prime_mcp; print(prime_mcp.__file__)"

# Run with verbose output
python -m prime_mcp.mcp --debug
```

### MCP Client Not Connecting

Verify the MCP configuration path and command:

```bash
# Test the server directly
python -m prime_mcp.mcp

# Should output MCP protocol messages
```

