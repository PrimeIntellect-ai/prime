# Prime Tunnel SDK

Expose local services via secure tunnels on Prime infrastructure.

## Installation

```bash
uv pip install prime-tunnel
```

Or with pip:

```bash
pip install prime-tunnel
```

## Quick Start

```python
from prime_tunnel import Tunnel

# Create and start a tunnel
async with Tunnel(local_port=8765) as tunnel:
    print(f"Tunnel URL: {tunnel.url}")
    # Your local service on port 8765 is now accessible at tunnel.url
```

## CLI Usage

```bash
# Start a tunnel
prime tunnel start --port 8765

# List active tunnels
prime tunnel list

# Get tunnel status
prime tunnel status <tunnel_id>
```

## Documentation

Full API reference: https://github.com/PrimeIntellect-ai/prime-cli/tree/main/packages/prime-tunnel

## Related Packages

- **`prime`** - Full CLI + SDK with pods, inference, and more (includes this package)

## License

MIT License - see LICENSE file for details
