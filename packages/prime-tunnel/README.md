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

If a tunnel stops serving while it should stay at the same URL, restart its
client. The registration is kept, so `tunnel.url` does not change:

```python
await tunnel.restart()
```

A tunnel that was deleted, expired, or stayed disconnected for too long cannot
be restarted: `restart()` raises `TunnelGoneError`, and `tunnel.is_gone` turns
true as soon as the tunnel service rejects the client. Stop it and start a new
tunnel, which gets a new URL. `await tunnel.status()` reports the client
process and the service's record of the tunnel:

```python
status = await tunnel.status()
if status.gone:
    await tunnel.stop()
```

### Keeping a tunnel up

`keep_serving()` checks that requests reach your service through the tunnel's
URL, and restarts the client if they stop. Give it a path on your service that
returns 2xx, and run it as a task:

```python
watch = asyncio.create_task(tunnel.keep_serving("/health"))
```

If your service itself is down, nothing is restarted. If the tunnel was deleted
or expired, the task raises `TunnelGoneError`, and you need a new tunnel with a
new URL. Cancel the task to stop watching.

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

Full API reference: https://github.com/PrimeIntellect-ai/prime/tree/main/packages/prime-tunnel

## Related Packages

- **`prime`** - Full CLI + SDK with pods, inference, and more (includes this package)

## License

MIT License - see LICENSE file for details
