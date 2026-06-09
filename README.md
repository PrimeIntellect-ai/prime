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

Command line interface and SDKs for Prime Lab, Hosted Training, GPU resources, sandboxes, and environments.
</div>

## Quick Start

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install prime
uv tool install prime

# Authenticate
prime login

# Set up a Lab workspace for environments, evals, GEPA, and Hosted Training
prime lab setup

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

- **Lab Workspaces** - Set up local verifiers workspaces for environments, evals, GEPA, and training
- **Hosted Training** - Train models against verifiers environments and inspect runs, logs, metrics, and checkpoints
- **Environments** - Access hundreds of verified environments on our community hub
- **Evaluations** - Push and manage evaluation results
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
# Interactive login (recommended)
prime login

# Store a key in the active Prime config
prime config set-api-key

# Process-scoped override for one command
PRIME_API_KEY="your-api-key-here" python script.py
```

Avoid exporting `PRIME_API_KEY` globally unless you want it to override your active Prime config in every project. For project-specific tools that require env vars, prefer a project-scoped `.env`/direnv setup.

#### Other Configuration

```bash
# Configure SSH key for pod access
prime config set-ssh-key-path

# View current configuration
prime config view
```

**Security Note**: Passing API keys as command arguments can leave them in shell history. Prefer `prime login` or the interactive `prime config set-api-key` prompt; use environment variables only as process- or project-scoped overrides.

### Environments Hub

Access hundreds of verified environments on our community hub with deep integrations with sandboxes, training, and evaluation stack.

```bash
# Browse available environments
prime env list

# View environment details
prime env info <environment-name>

# Inspect environment source without downloading the archive
prime env inspect <environment-name>

# Install an environment locally
prime env install <environment-name>

# Create and push your own environment
prime env init my-environment
prime env push my-environment
```

### Lab and Hosted Training

Prime Lab connects verifiers environments to evaluations, GEPA prompt optimization, and Hosted Training. Start with `prime lab setup` to create a local workspace with starter configs, then use `prime train models` to choose a Hosted Training model with current capacity and pricing.

```bash
# Set up a Lab workspace
prime lab setup

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

### Prime Inference from Python

Prime Inference is OpenAI-compatible. Configure the standard OpenAI client from the active Prime context instead of reading `PRIME_API_KEY` directly:

```python
from openai import OpenAI
from prime_cli import Config

prime = Config()
client_kwargs = {
    "base_url": prime.inference_url,
    "api_key": prime.api_key,
}
if prime.team_id:
    client_kwargs["default_headers"] = {"X-Prime-Team-ID": prime.team_id}

client = OpenAI(**client_kwargs)
response = client.chat.completions.create(
    model="qwen/qwen3-30b-a3b-instruct-2507",
    messages=[{"role": "user", "content": "Say hi."}],
)
print(response.choices[0].message.content)
```

### Evaluations

Push and manage evaluation results to the Environments Hub.

```bash
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

All packages (prime-core, prime-sandboxes, prime) are installed in editable mode. Changes to code are immediately reflected.

### Releasing

This monorepo contains two independently versioned packages: `prime` (CLI + full SDK) and `prime-sandboxes` (lightweight SDK).

Versions are single-sourced from each package's `__init__.py` file:
- **prime**: `packages/prime/src/prime_cli/__init__.py`
- **prime-sandboxes**: `packages/prime-sandboxes/src/prime_sandboxes/__init__.py`

#### To release a new version:

1. Update the `__version__` string in the appropriate `__init__.py` file
2. Commit and push the change

Tagging and publishing to PyPI is handled automatically by CI.

#### Version sync considerations:

When releasing `prime`, consider whether `prime-sandboxes` should also be bumped, as `prime` depends on `prime-sandboxes`. The packages can be released independently or together depending on what changed.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Links

- [Website](https://primeintellect.ai)
- [Dashboard](https://app.primeintellect.ai)
- [API Docs](https://api.primeintellect.ai/docs)
- [Discord Community](https://discord.gg/primeintellect)
