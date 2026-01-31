# Installation

## CLI Installation

### Using uv (recommended)

First, install uv if you haven't already:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then install prime as a global tool:

```bash
uv tool install prime
```

This installs the `prime` command globally, making it available from any directory.

### Using pip

```bash
pip install prime
```

### Verify Installation

```bash
prime --version
prime --help
```

## SDK Installation

### Sandboxes SDK

If you only need the sandboxes SDK (lightweight, ~50KB):

```bash
uv pip install prime-sandboxes
```

Or with pip:

```bash
pip install prime-sandboxes
```

### Evals SDK

For managing evaluation results:

```bash
uv pip install prime-evals
```

Or with pip:

```bash
pip install prime-evals
```

### MCP Server

For AI assistant integration:

```bash
uv pip install prime-mcp-server
```

Or with pip:

```bash
pip install prime-mcp-server
```

## Development Installation

For contributing to Prime CLI:

```bash
# Clone the repository
git clone https://github.com/PrimeIntellect-ai/prime-cli
cd prime-cli

# Set up workspace (installs all packages in editable mode)
uv sync

# Install CLI globally in editable mode
uv tool install -e packages/prime

# Now you can use the CLI directly
prime --help

# Run tests
uv run pytest packages/prime/tests
uv run pytest packages/prime-sandboxes/tests
```

All packages (prime, prime-sandboxes, prime-evals, prime-mcp-server) are installed in editable mode. Changes to code are immediately reflected.

## Updating

### Update CLI

```bash
uv tool upgrade prime
```

Or with pip:

```bash
pip install --upgrade prime
```

### Update SDKs

```bash
uv pip install --upgrade prime-sandboxes prime-evals
```

