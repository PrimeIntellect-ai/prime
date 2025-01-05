# Prime Intellect CLI

Command line interface for managing Prime Intellect resources.

## Setup

1. Create and activate a virtual environment:
```bash
# Create a new virtual environment
python -m venv venv

# Activate it:
source venv/bin/activate
```

2. Clone and install the package in development mode:
```bash
git clone https://github.com/PrimeIntellect-ai/prime-cli
cd prime-cli
pip install -e ".[dev]" 
```

3. Set up your API key:
```bash
prime config set-api-key
# Check details
prime config view
```

## Basic Usage

```bash
# Check GPU availability 
prime availability list                          # List all available GPU resources
prime availability list --gpu-type H100_80GB     # Filter by GPU type
prime availability gpu-types                     # List available GPU types

# Manage compute pods
prime pods list                                  # List your running pods
prime pods status <pod-id>                       # Get detailed status of a pod
prime pods create                                # Create a pod interactively
prime pods create --id <ID>                      # Create a pod with a specific GPU config
prime pods create --name my-pod                  # Create a pod with a custom name
prime pods terminate <pod-id>                    # Terminate a pod

# SSH into pods
prime pods ssh <pod-id>                          # SSH into a running pod

# Before using SSH:
# 1. Download your SSH private key from Prime Intellect dashboard
# 2. Set the SSH key path in your config:
prime config set-ssh-key-path

## Team Usage
prime config set-team-id 
```

## Development Commands

```bash
# Run linter and formatter (includes import sorting)
ruff format src/prime_cli
ruff check src/prime_cli
```

## Contributing

1. Create a new branch for your feature
2. Make your changes
3. Run the linter (ruff)
4. Submit a pull request

## License
MIT License - see LICENSE file for details
