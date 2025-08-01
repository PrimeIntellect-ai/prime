# Prime Intellect CLI

<div align="center">

[![PyPI version](https://img.shields.io/pypi/v/prime-cli?cacheSeconds=60)](https://pypi.org/project/prime-cli/)
[![Python versions](https://img.shields.io/pypi/pyversions/prime-cli?cacheSeconds=60)](https://pypi.org/project/prime-cli/)
[![Downloads](https://img.shields.io/pypi/dm/prime-cli)](https://pypi.org/project/prime-cli/)

Command line interface for managing Prime Intellect resources and environments.
</div>

## 🚀 Quick Start

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install prime-cli with uv
uv tool install prime-cli

# Authenticate 
prime login 

# List available GPU resources
prime availability list
```

## 🔧 Features

- **GPU Resource Management**: Query and filter available GPU resources
- **Pod Management**: Create, monitor, and terminate compute pods
- **SSH Access**: Direct SSH access to running pods
- **Team Support**: Manage resources across team environments

## 📦 Installation

### Recommended: Using uv
```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install prime-cli
uv tool install prime-cli
```

### Alternative: Using pip
```bash
# If you prefer traditional pip
pip install prime-cli
```

### For Development
```bash
# Clone the repository
git clone https://github.com/PrimeIntellect-ai/prime-cli
cd prime-cli

# Create virtual environment with uv
uv venv

# Install in development mode with all dependencies
uv pip install -e ".[dev]"

# Set up pre-commit hooks
pre-commit install
```

## 🛠️ Usage

### Configuration
```bash
# Set up API key
prime config set-api-key

# Configure SSH key for pod access
prime config set-ssh-key-path

# View current configuration
prime config view
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

### Team Management
```bash
# Set team context
prime config set-team-id
```

### Working Without uv

While we recommend using uv for the best experience, all commands work with standard pip:

```bash
# Install without uv
pip install prime-cli

# Development setup without uv
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Note: Without uv, operations will be slower and you'll need to manage virtual environments manually.

## 💻 Development

### Prerequisites
- Python 3.11+
- uv (recommended): `curl -LsSf https://astral.sh/uv/install.sh | sh`

### Setup
```bash
# Clone and setup with uv
git clone https://github.com/PrimeIntellect-ai/prime-cli
cd prime-cli
uv venv
uv pip install -e ".[dev]"

# Run code quality checks
uv run ruff format src/prime_cli
uv run ruff check src/prime_cli
uv run mypy src/prime_cli
uv run pytest
```

### Code Quality
```bash
# Format code
uv run ruff format src/prime_cli

# Run linter
uv run ruff check src/prime_cli

# Run tests
uv run pytest

# Or if you have the venv activated:
ruff format src/prime_cli
ruff check src/prime_cli
pytest
```

### Release Process
We use semantic versioning. Releases are automatically created when changes are merged to main.

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Set up development environment with uv:
   ```bash
   uv venv
   uv pip install -e ".[dev]"
   pre-commit install
   ```
4. Run code quality checks:
   ```bash
   uv run ruff format .
   uv run ruff check .
   uv run pytest
   ```
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Links

- [Documentation](https://docs.primeintellect.ai)
- [PyPI Package](https://pypi.org/project/prime-cli/)
- [Issue Tracker](https://github.com/PrimeIntellect-ai/prime-cli/issues)