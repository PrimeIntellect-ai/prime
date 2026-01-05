# Configuration

## API Key

Get your API key from the [Prime Intellect Dashboard](https://app.primeintellect.ai).

### Interactive Login (Recommended)

```bash
prime login
```

This opens a browser for authentication and stores your credentials securely.

### Set API Key Directly

```bash
# Interactive mode (hides input)
prime config set-api-key

# Non-interactive mode (for automation)
prime config set-api-key YOUR_API_KEY
```

### Environment Variable

```bash
export PRIME_API_KEY="your-api-key-here"
```

**Security Note**: When using non-interactive mode, the API key may be visible in your shell history. For enhanced security, use interactive mode or environment variables.

### Configuration Priority

The SDK looks for credentials in this order:

1. **Direct parameter**: `APIClient(api_key="sk-...")`
2. **Environment variable**: `PRIME_API_KEY`
3. **Config file**: `~/.prime/config.json` (created by `prime login`)

## SSH Key

Configure SSH key for pod access:

```bash
# Interactive selection
prime config set-ssh-key-path

# Or specify directly
prime config set-ssh-key-path ~/.ssh/id_rsa.pub
```

## Team Context

Set team context for managing shared resources:

```bash
# List your teams
prime teams list

# Set team context (interactive)
prime config set-team-id

# Set team context directly
prime config set-team-id <team-id>
```

All subsequent commands will use the team context:

```bash
prime pods list  # Shows team's pods
prime sandbox list  # Shows team's sandboxes
```

## View Configuration

```bash
prime config view
```

## Configuration File

Configuration is stored in `~/.prime/config.json`:

```json
{
  "api_key": "your-api-key",
  "base_url": "https://api.primeintellect.ai",
  "ssh_key_path": "/Users/you/.ssh/id_rsa.pub",
  "team_id": "team-123"
}
```

## SDK Configuration

### Python SDK

```python
from prime_sandboxes import APIClient, SandboxClient

# Option 1: Direct API key
client = APIClient(api_key="your-api-key")

# Option 2: From environment variable
import os
client = APIClient(api_key=os.environ["PRIME_API_KEY"])

# Option 3: Auto-detect (uses env var or config file)
client = APIClient()

sandbox_client = SandboxClient(client)
```

### Async Client

```python
from prime_sandboxes import AsyncSandboxClient

# Context manager handles cleanup
async with AsyncSandboxClient(api_key="your-api-key") as client:
    # Use client
    pass
```

