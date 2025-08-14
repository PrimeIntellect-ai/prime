<p align="center">
  <picture>
    <source media="(prefers-color-scheme: light)" srcset="https://github.com/user-attachments/assets/40c36e38-c5bd-4c5a-9cb3-f7b902cd155d">
    <source media="(prefers-color-scheme: dark)" srcset="https://github.com/user-attachments/assets/6414bc9b-126b-41ca-9307-9e982430cde8">
    <img alt="Prime Intellect" src="https://github.com/user-attachments/assets/6414bc9b-126b-41ca-9307-9e982430cde8" width="312" style="max-width: 100%;">
  </picture>
</p>

---

<h3 align="center">
Prime Intellect CLI
</h3>

---

<div align="center">

[![PyPI version](https://img.shields.io/pypi/v/prime?cacheSeconds=60)](https://pypi.org/project/prime/)
[![Python versions](https://img.shields.io/pypi/pyversions/prime?cacheSeconds=60)](https://pypi.org/project/prime/)
[![Downloads](https://img.shields.io/pypi/dm/prime)](https://pypi.org/project/prime/)

Command line interface for managing Prime Intellect resources and environments.
</div>

## 🚀 Quick Start

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install prime with uv
uv tool install prime

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

# Install prime
uv tool install prime
```

### Alternative: Using pip
```bash
# If you prefer traditional pip
pip install prime
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

#### API Key Setup (Multiple Options)

```bash
# Option 1: Interactive mode (recommended - hides input)
prime config set-api-key

# Option 2: Non-interactive mode (for automation)
prime config set-api-key YOUR_API_KEY

# Option 3: Environment variable (most secure for scripts)
export PRIME_API_KEY="your-api-key-here"
```

#### Other Configuration
```bash
# Configure SSH key for pod access
prime config set-ssh-key-path

# Set base URL (interactive or non-interactive)
prime config set-base-url
prime config set-base-url https://api.primeintellect.ai

# Set frontend URL (interactive or non-interactive)  
prime config set-frontend-url
prime config set-frontend-url https://app.primeintellect.ai

# View current configuration
prime config view
```

**Security Note**: When using the non-interactive mode, be aware that the API key may be visible in your shell history. For enhanced security:
- Use the interactive mode (no arguments) which hides your input
- Use environment variables (`PRIME_API_KEY`) as fallback if no CLI config is set
- Clear your shell history after setting sensitive values

**Configuration Priority**: CLI config takes precedence over environment variables. If you set an API key via `prime auth login`, it will override any `PRIME_API_KEY` environment variable.

### Environment Management
```bash
# Save current config as an environment (includes API key)
prime config save develop

# Switch between environments
prime config use production
prime config use develop

# List available environments
prime config envs
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
pip install prime

# Development setup without uv
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Note: Without uv, operations will be slower and you'll need to manage virtual environments manually.

## 🔐 Security Best Practices

### API Key Management
1. **Never commit API keys to version control**
2. **Use environment variables for automation**:
   ```bash
   export PRIME_API_KEY="your-api-key-here"
   prime pods list  # Will use the environment variable
   ```
3. **Use interactive mode for one-time setup**:
   ```bash
   prime config set-api-key  # Prompts securely without showing input
   ```
4. **For CI/CD pipelines**, use secrets management:
   - GitHub Actions: Use repository secrets
   - GitLab CI: Use CI/CD variables
   - Jenkins: Use credentials plugin

### Secure Storage
- API keys are stored in `~/.prime/config.json` with user-only permissions
- Environment variables take precedence over stored configuration
- Consider using a password manager for long-term API key storage

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
4. Set up your development environment:
   ```bash
   # Save your local development config
   prime config set-base-url http://localhost:8000
   prime config set-frontend-url http://localhost:3000
   prime config save local
   
   # Switch between environments as needed
   prime config use local       # For development
   prime config use production  # For testing against prod
   ```
5. Run code quality checks:
   ```bash
   uv run ruff format .
   uv run ruff check .
   uv run pytest
   ```
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Links

- [Documentation](https://docs.primeintellect.ai)
- [PyPI Package](https://pypi.org/project/prime/)
- [Issue Tracker](https://github.com/PrimeIntellect-ai/prime-cli/issues)
