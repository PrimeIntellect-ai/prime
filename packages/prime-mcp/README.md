# Prime MCP Server

Model Context Protocol (MCP) server for Prime Intellect - manage GPU pods, check availability, and control compute resources through MCP.

## Features

- **GPU Availability Checking** - Search for available GPU instances across providers
- **Pod Management** - Create, list, monitor, and delete GPU pods
- **Cluster Support** - Check multi-node cluster availability
- **SSH Key Management** - Manage SSH keys for pod access
- **Type-safe** - Full type hints and proper error handling
- **FastMCP Integration** - Built on FastMCP for easy MCP server development

## Installation

```bash
pip install prime-mcp
```

Or with uv:
```bash
uv pip install prime-mcp
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

### Using as a Library

```python
from prime_mcp import availability, pods, ssh

# Check GPU availability
result = await availability.check_gpu_availability(
    gpu_type="A100_80GB",
    regions=["united_states", "eu_west"],
    security="secure_cloud"
)

# Create a pod
pod = await pods.create_pod(
    cloud_id="cloud-123",
    gpu_type="A100_80GB",
    provider_type="runpod",
    name="my-training-pod",
    gpu_count=1,
    image="cuda_12_1_pytorch_2_4"
)

# List pods
pods_list = await pods.list_pods(limit=10)

# Manage SSH keys
keys = await ssh.manage_ssh_keys(action="list")
```

## Available Tools

### Availability Tools

#### `check_gpu_availability`
Check GPU availability across different providers.

**Parameters:**
- `gpu_type` (optional): GPU model (e.g., "A100_80GB", "H100_80GB", "RTX4090_24GB")
- `regions` (optional): List of regions to filter. Valid options: `africa`, `asia_south`, `asia_northeast`, `australia`, `canada`, `eu_east`, `eu_north`, `eu_west`, `middle_east`, `south_america`, `united_states`
- `socket` (optional): Socket type ("PCIe", "SXM2", "SXM3", "SXM4", "SXM5", "SXM6")
- `security` (optional): Security type ("secure_cloud" or "community_cloud")
- `gpu_count` (optional): Number of GPUs to filter by

#### `check_cluster_availability`
Check cluster availability for multi-node deployments.

**Parameters:**
- `regions` (optional): List of regions to filter. Valid options: `africa`, `asia_south`, `asia_northeast`, `australia`, `canada`, `eu_east`, `eu_north`, `eu_west`, `middle_east`, `south_america`, `united_states`
- `gpu_count` (optional): Desired number of GPUs
- `gpu_type` (optional): GPU model (e.g., "H100_80GB", "A100_80GB", "RTX4090_24GB")
- `socket` (optional): Socket type ("PCIe", "SXM2", "SXM3", "SXM4", "SXM5", "SXM6")
- `security` (optional): Security type ("secure_cloud", "community_cloud")

### Pod Management Tools

#### `create_pod`
Create a new GPU pod (compute instance).

**Required Parameters:**
- `cloud_id`: Cloud provider ID from availability check
- `gpu_type`: GPU model name
- `provider_type`: Provider type (e.g., "runpod", "fluidstack", "lambdalabs")

**Optional Parameters:**
- `name`: Pod name
- `gpu_count`: Number of GPUs (default: 1)
- `socket`: GPU socket type (default: "PCIe")
- `disk_size`: Disk size in GB
- `vcpus`: Number of virtual CPUs
- `memory`: Memory in GB
- `max_price`: Maximum price per hour
- `image`: Environment image (default: "ubuntu_22_cuda_12")
- `custom_template_id`: Custom template ID
- `data_center_id`: Specific data center ID
- `country`: Country code
- `security`: Security level
- `auto_restart`: Auto-restart on failure
- `jupyter_password`: Jupyter password
- `env_vars`: Environment variables (dict)
- `team_id`: Team ID

**Available Images:**
- `ubuntu_22_cuda_12`
- `cuda_12_1_pytorch_2_2`, `cuda_11_8_pytorch_2_1`, `cuda_12_1_pytorch_2_3`, `cuda_12_1_pytorch_2_4`
- `cuda_12_4_pytorch_2_4`, `cuda_12_4_pytorch_2_5`, `cuda_12_6_pytorch_2_7`
- `stable_diffusion`, `axolotl`, `bittensor`, `hivemind`, `petals_llama`
- `vllm_llama_8b`, `vllm_llama_70b`, `vllm_llama_405b`
- `flux`, `custom_template`

#### `list_pods`
List all pods in your account.

**Parameters:**
- `offset`: Number of pods to skip (default: 0)
- `limit`: Maximum pods to return (default: 100)

#### `get_pod_details`
Get detailed information about a specific pod.

**Parameters:**
- `pod_id`: Unique identifier of the pod

#### `get_pods_status`
Get status information for pods.

**Parameters:**
- `pod_ids` (optional): List of specific pod IDs

#### `get_pods_history`
Get pods history with sorting and pagination.

**Parameters:**
- `limit`: Maximum entries (default: 100)
- `offset`: Entries to skip (default: 0)
- `sort_by`: Field to sort by (default: "terminatedAt", options: "terminatedAt", "createdAt")
- `sort_order`: Sort order (default: "desc", options: "asc", "desc")

#### `delete_pod`
Delete/terminate a pod.

**Parameters:**
- `pod_id`: Unique identifier of the pod to delete

### SSH Key Management

#### `manage_ssh_keys`
Manage SSH keys for pod access.

**Parameters:**
- `action`: Action to perform ("list", "add", "delete", "set_primary")
- `key_name`: Name for the SSH key (required for "add")
- `public_key`: SSH public key content (required for "add")
- `key_id`: Key ID (required for "delete" and "set_primary")
- `offset`: Items to skip for "list" (default: 0)
- `limit`: Maximum items for "list" (default: 100)

## Examples

### Check GPU Availability and Create Pod

```python
import asyncio
from prime_mcp import availability, pods

async def main():
    # Check what's available
    available = await availability.check_gpu_availability(
        gpu_type="A100_80GB",
        security="secure_cloud"
    )
    
    if available and not available.get("error"):
        # Get the first available cloud ID
        first_option = available.get("data", [])[0]
        cloud_id = first_option.get("cloudId")
        
        # Create a pod
        pod = await pods.create_pod(
            cloud_id=cloud_id,
            gpu_type="A100_80GB",
            provider_type="runpod",
            name="training-pod",
            gpu_count=1,
            disk_size=50,
            image="cuda_12_4_pytorch_2_5"
        )
        
        print(f"Created pod: {pod.get('id')}")
        print(f"SSH: {pod.get('sshConnection')}")

asyncio.run(main())
```

### Monitor Pods

```python
import asyncio
from prime_mcp import pods

async def main():
    # List all active pods
    active_pods = await pods.list_pods()
    
    for pod in active_pods.get("data", []):
        print(f"Pod {pod.get('name')}: {pod.get('status')}")
    
    # Get status for specific pods
    pod_ids = [pod.get("id") for pod in active_pods.get("data", [])]
    status = await pods.get_pods_status(pod_ids=pod_ids)
    
    for pod_status in status.get("data", []):
        print(f"Pod {pod_status.get('podId')}: {pod_status.get('installationStatus')}")

asyncio.run(main())
```

### Manage SSH Keys

```python
import asyncio
from prime_mcp import ssh

async def main():
    # List existing keys
    keys = await ssh.manage_ssh_keys(action="list")
    print(f"Found {len(keys.get('data', []))} SSH keys")
    
    # Add a new key
    with open("~/.ssh/id_rsa.pub") as f:
        public_key = f.read()
    
    result = await ssh.manage_ssh_keys(
        action="add",
        key_name="my-laptop",
        public_key=public_key
    )
    
    print(f"Added key: {result.get('id')}")

asyncio.run(main())
```

## MCP Client Configuration

### Claude Desktop

Add to your Claude Desktop config (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "primeintellect": {
      "command": "python",
      "args": ["-m", "prime_mcp.mcp"],
      "env": {
        "PRIME_API_KEY": "your-api-key"
      }
    }
  }
}
```

## Architecture

The package is organized into:

- `client.py` - Thin wrapper around `prime-core`'s `AsyncAPIClient`
- `mcp.py` - MCP server implementation with FastMCP
- `tools/` - Tool implementations
  - `availability.py` - GPU availability checking
  - `pods.py` - Pod management
  - `ssh.py` - SSH key management

Prime MCP builds on `prime-core` for:
- Authentication and configuration management
- HTTP client with proper error handling
- Shared config across Prime tools (`~/.prime/config.json`)

## Error Handling

All tools return dictionaries with either the result or an error key:

```python
result = await availability.check_gpu_availability(
    gpu_type="InvalidGPU",
    regions=["united_states"]
)

if "error" in result:
    print(f"Error: {result['error']}")
else:
    print(f"Success: {result}")
```

## Development

```bash
# Clone the repo
git clone https://github.com/PrimeIntellect-ai/prime-cli.git
cd prime-cli/packages/prime-mcp

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
