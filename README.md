# Prime Intellect CLI

Command line interface for managing Prime Intellect AI resources.

## Development Setup

1. Create and activate a virtual environment:```bash
# Create a new virtual environment
python -m venv venv

# Activate it:
source venv/bin/activate
```

2. Clone and install the package in development mode:
```bash
git clone https://github.com/PrimeIntellect-ai/prime-cli
cd prime-cli
pip install -e ".[dev]"  # This installs the package with development dependencies
```

3. Set up your API key:
```bash
# You can either set it in your environment:
export PRIME_API_KEY=your_api_key

# Or configure it via the CLI:
prime config set-api-key
```

## Basic Usage

```bash
# Check GPU availability
prime gpu availability
```

## Development Commands

```bash
# Run linter and formatter (includes import sorting)
ruff format src/prime_cli
ruff check src/prime_cli
```

## Project Structure

```
prime-cli/
├── src/
│   └── prime_cli/         # Main package code
│       ├── api/           # API client code
│       ├── commands/      # CLI commands
│       └── config.py      # Configuration
├── tests/                 # Test files
└── pyproject.toml        # Project configuration
```

## Contributing

1. Create a new branch for your feature
2. Make your changes
3. Run the formatting tools (black, isort)
4. Run the linter (ruff)
5. Run the tests (pytest)
6. Submit a pull request

## License

MIT License - see LICENSE file for details
