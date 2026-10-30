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

A running client and a connected registration do not prove that requests get
through. `keep_serving()` checks that they do, and restarts the client under
the same URL when they stop. Run it as a task, and give it a path on your
service that answers 2xx quickly:

```python
watch = asyncio.create_task(tunnel.keep_serving("/health"))
```

It sends a GET through the URL every 15 seconds. It blames the tunnel only if
the same request sent straight to your service succeeds, so a service that is
down does not cause restarts. Cancel the task to stop watching. It ends with
`TunnelGoneError` when the registration is gone; only a new tunnel, with a new
URL, helps then:

```python
try:
    await watch
except TunnelGoneError:
    await tunnel.stop()
    # start a new Tunnel; it gets a new URL
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

Full API reference: https://github.com/PrimeIntellect-ai/prime/tree/main/packages/prime-tunnel

## Related Packages

- **`prime`** - Full CLI + SDK with pods, inference, and more (includes this package)

## License

MIT License - see LICENSE file for details
