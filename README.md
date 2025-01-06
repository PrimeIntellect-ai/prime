# Prime Intellect CLI

<div align="center">

[![PyPI version](https://img.shields.io/pypi/v/prime-cli)](https://pypi.org/project/prime-cli/)
[![Python versions](https://img.shields.io/pypi/pyversions/prime-cli)](https://pypi.org/project/prime-cli/)
[![Downloads](https://img.shields.io/pypi/dm/prime-cli)](https://pypi.org/project/prime-cli/)

Command line interface for managing Prime Intellect GPU resources, enabling seamless deployment and management of compute pods.
</div>

## 🚀 Quick Start

```bash
# Install from PyPI
pip install prime-cli

# Set up your API key
prime config set-api-key

# List available GPU resources
prime availability list
```

## 🔧 Features

- **GPU Resource Management**: Query and filter available GPU resources
- **Pod Management**: Create, monitor, and terminate compute pods
- **SSH Access**: Direct SSH access to running pods
- **Team Support**: Manage resources across team environments

## 📦 Installation

### From PyPI
```bash
pip install prime-cli
```

### For Development
```bash
# Clone the repository
git clone https://github.com/PrimeIntellect-ai/prime-cli
cd prime-cli

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

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

## 💻 Development

### Code Quality
```bash
# Format code
ruff format src/prime_cli

# Run linter
ruff check src/prime_cli

# Run tests
pytest
```

### Release Process
We use semantic versioning. Releases are automatically created when changes are merged to main.

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Run code quality checks
   ```bash
   ruff format .
   ruff check .
   pytest
   ```
4. Commit your changes (`git commit -m 'Add amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Links

- [Documentation](https://docs.primeintellect.ai)
- [PyPI Package](https://pypi.org/project/prime-cli/)
- [Issue Tracker](https://github.com/PrimeIntellect-ai/prime-cli/issues)