<p align="center">
  <picture>
    <source media="(prefers-color-scheme: light)" srcset="https://github.com/user-attachments/assets/40c36e38-c5bd-4c5a-9cb3-f7b902cd155d">
    <source media="(prefers-color-scheme: dark)" srcset="https://github.com/user-attachments/assets/6414bc9b-126b-41ca-9307-9e982430cde8">
    <img alt="Prime Intellect" src="https://github.com/user-attachments/assets/6414bc9b-126b-41ca-9307-9e982430cde8" width="312" style="max-width: 100%;">
  </picture>
</p>

---

<h3 align="center">
Prime Intellect CLI & SDKs
</h3>

---

<div align="center">

[![PyPI version](https://img.shields.io/pypi/v/prime?cacheSeconds=60)](https://pypi.org/project/prime/)
[![Python versions](https://img.shields.io/pypi/pyversions/prime?cacheSeconds=60)](https://pypi.org/project/prime/)
[![Downloads](https://img.shields.io/pypi/dm/prime)](https://pypi.org/project/prime/)

Command line interface and SDKs for the Environments Hub, evals, Hosted Training, GPU resources, and sandboxes.
</div>

## Quick Start

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install prime
uv tool install prime

# Authenticate
prime login

# Evaluate a model on an environment (runs prime-rl in a sandbox)
prime eval run gsm8k -n 32 -r 4

# See available Hosted Training models, capacity, and pricing
prime train models

# Generate and launch a Hosted Training config
prime train init
prime train rl.toml

# Browse verified environments
prime env list

# List available GPU resources
prime availability list
```

## Features

- **Environments** - Access hundreds of verified environments on our community hub
- **Evaluations** - Run prime-rl evals in a sandbox and manage results on the platform
- **Hosted Training** - Train models against environments and inspect runs, logs, metrics, and checkpoints
- **GPU Resource Management** - Query and filter available GPU resources
- **Pod Management** - Create, monitor, and terminate compute pods
- **Sandboxes** - Easily run AI-generated code in the cloud
- **SSH Access** - Direct SSH access to running pods
- **Team Support** - Manage resources across team environments

## Installation

### Using uv (recommended)

First, install uv if you haven't already:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then install prime:

```bash
uv tool install prime
```

### Using pip

```bash
pip install prime
```

### Sandboxes SDK Only

If you only need the sandboxes SDK (lightweight, ~50KB):

```bash
uv pip install prime-sandboxes
```

See [prime-sandboxes documentation](./packages/prime-sandboxes/) for SDK usage.

## Usage

### Configuration

#### API Key Setup

```bash
# Interactive mode (recommended - hides input)
prime config set-api-key

# Non-interactive mode (for automation)
prime config set-api-key YOUR_API_KEY

# Environment variable (most secure for scripts)
export PRIME_API_KEY="your-api-key-here"
```

#### Other Configuration

```bash
# Configure SSH key for pod access
prime config set-ssh-key-path

# View current configuration
prime config view
```

**Security Note**: When using non-interactive mode, the API key may be visible in your shell history. For enhanced security, use interactive mode or environment variables.

### Environments Hub

Access hundreds of verified environments on our community hub with deep integrations with sandboxes, training, and evaluation stack.

```bash
# Browse available environments
prime env list

# View environment details
prime env info <environment-name>

# Install an environment locally
prime env install <environment-name>

# Push your own environment (scaffold one with `vf-init` from verifiers)
prime env push my-environment
```

### Hosted Training

Use `prime train models` to choose a Hosted Training model with current capacity and pricing.

```bash
# List trainable models, capacity, and token pricing
prime train models

# Generate a Hosted Training config
prime train init

# Launch the run from the generated config
prime train rl.toml

# Inspect and manage Hosted Training runs
prime train list
prime train logs <run-id> -f
prime train metrics <run-id>
prime train checkpoints <run-id>
```

### GPU Resources

```bash
# List all available GPUs
prime availability list

# Filter by GPU type
prime availability list --gpu-type H100_80GB

# Show available GPU types
prime availability gpu-types
```

### Pod Management

```bash
# List your pods
prime pods list

# Create a pod
prime pods create
prime pods create --id <ID>     # With specific GPU config
prime pods create --name my-pod # With custom name

# Monitor and manage pods
prime pods status <pod-id>
prime pods terminate <pod-id>
prime pods ssh <pod-id>
```

### Evaluations

`prime eval run` creates a sandbox, installs [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl), and runs `uv run eval` with every other argument passed through. Results stream to the platform via the prime monitor.

```bash
# Evaluate a model (flags after the environment go to `uv run eval`)
prime eval run gsm8k -n 32 -r 4 -m openai/gpt-4.1-mini

# Multi-source eval from a local TOML, pinned to a prime-rl ref
prime eval run @ eval.toml --ref v0.3.0

# Auto-discover and push evaluations from current directory
prime eval push

# Push specific eval directory (verifiers format)
prime eval push outputs/evals/gsm8k--gpt-4/abc123

# Push a public evaluation (default is private)
prime eval push --public

# List all evaluations
prime eval list

# Get evaluation details
prime eval get <eval-id>

# View evaluation samples
prime eval samples <eval-id>
```

### Team Management

```bash
# List teams
prime teams list

# Switch context
prime switch
prime switch personal
prime switch <team-slug>
```

## Development

```bash
# Clone the repository
git clone https://github.com/PrimeIntellect-ai/prime
cd prime

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

All workspace packages (`prime`, `prime-sandboxes`, `prime-evals`, `prime-tunnel`) are installed in editable mode. Changes to code are immediately reflected.

### Releasing

This monorepo contains independently versioned packages: `prime` (CLI + full SDK), `prime-sandboxes`, `prime-evals`, and `prime-tunnel` (lightweight SDKs).

Versions are single-sourced from each package's `__init__.py` file:
- **prime**: `packages/prime/src/prime_cli/__init__.py`
- **prime-sandboxes**: `packages/prime-sandboxes/src/prime_sandboxes/__init__.py`
- **prime-evals**: `packages/prime-evals/src/prime_evals/__init__.py`
- **prime-tunnel**: `packages/prime-tunnel/src/prime_tunnel/__init__.py`

#### To release a new version:

1. Update the `__version__` string in the appropriate `__init__.py` file
2. Commit and push the change

Tagging and publishing to PyPI is handled automatically by CI.

#### Version sync considerations:

When releasing `prime`, consider whether `prime-sandboxes` or `prime-tunnel` should also be bumped, as `prime` depends on both. The packages can be released independently or together depending on what changed.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Links

- [Website](https://primeintellect.ai)
- [Dashboard](https://app.primeintellect.ai)
- [API Docs](https://api.primeintellect.ai/docs)
- [Discord Community](https://discord.gg/primeintellect)
