# prime-core

Shared HTTP client and configuration for Prime Intellect packages.

This is a core package used by:
- [prime](https://pypi.org/project/prime/) - Prime Intellect CLI + SDK
- [prime-sandboxes](https://pypi.org/project/prime-sandboxes/) - Sandboxes SDK
- [prime-evals](https://pypi.org/project/prime-evals/) - Evals SDK

## Installation

```bash
pip install prime-core
```

## Usage

```python
from prime_core import APIClient, Config

# Uses PRIME_API_KEY env var or ~/.prime/config.json
client = APIClient()
response = client.get("/some/endpoint")
```

## Configuration

Set your API key:
```bash
export PRIME_API_KEY=your-api-key
```

Or use the CLI:
```bash
prime login
```
