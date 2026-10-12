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

Pass `supervise=True` to have the tunnel watch and repair itself after
`start()`. It restarts its client under the same URL when the client exits or
when requests through the URL stop reaching your service, and it marks the
tunnel dead when the registration is gone:

```python
tunnel = Tunnel(local_port=8765, supervise=True, on_state_change=print)
await tunnel.start()

if await tunnel.wait_dead():  # False if the tunnel was stopped instead
    await tunnel.stop()
    # start a new Tunnel; it gets a new URL
```

`tunnel.state` is `connected`, `recovering`, `dead` or `stopped`, and
`on_state_change` is called with each new state.

The supervisor sends a GET to `probe_path` (default `/`) through the URL
every 15 seconds. It blames the tunnel only if the same request sent straight
to your service succeeds, so a service that is down or returns errors itself
does not cause restarts. If your service sets a header on every response,
pass its name as `probe_header`: a response without it is then known not to
come from your service, whatever its status code.

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
