# Environments Hub

Access hundreds of RL environments on our community hub with deep integrations with sandboxes, training, and evaluation stack.

## Overview

Environments provide pre-configured setups for machine learning, data science, and development workflows, tested and verified by the Prime Intellect community. They integrate seamlessly with:

- **Sandboxes** - Run environments in isolated execution environments
- **Training** - Use environments for RL training with PRIME-RL
- **Evaluation** - Evaluate models against environment benchmarks

## Browsing Environments

```bash
# List all available environments
prime env list

# Search for specific environments
prime env list --search "math"
prime env list --search "code"

# Filter by category
prime env list --category reasoning
```

## Environment Details

```bash
# View environment information
prime env info gsm8k
prime env info hendrycks-math

# Shows:
# - Description
# - Author
# - Version
# - Dependencies
# - Usage examples
```

## Installing Environments

```bash
# Install an environment locally
prime env install gsm8k

# Install specific version
prime env install gsm8k@1.0.0

# Install to custom path
prime env install gsm8k --path ./my-envs/
```

After installation, environments are available for use with verifiers:

```python
from verifiers import load_environment

env = load_environment("gsm8k")
```

## Creating Environments

### Initialize a New Environment

```bash
prime env init my-environment
```

This creates a new environment scaffold:

```
my-environment/
├── pyproject.toml
├── README.md
├── metadata.json
└── src/
    └── my_environment/
        └── __init__.py
```

### Environment Structure

**pyproject.toml** - Package configuration:

```toml
[project]
name = "my-environment"
version = "0.1.0"
description = "My custom environment"

[project.optional-dependencies]
all = []

[tool.verifiers]
environment = "my_environment"
```

**metadata.json** - Hub metadata:

```json
{
  "name": "my-environment",
  "description": "A custom RL environment",
  "category": "reasoning",
  "tags": ["math", "custom"],
  "author": "your-username"
}
```

**src/my_environment/__init__.py** - Environment implementation:

```python
from verifiers import SingleTurnEnv

def load_environment(**kwargs):
    """Entry point for loading the environment."""
    return MyEnvironment(**kwargs)

class MyEnvironment(SingleTurnEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize environment
    
    def get_dataset(self):
        # Return evaluation dataset
        pass
    
    def get_reward_functions(self):
        # Return reward functions
        pass
```

### Publishing Environments

```bash
# Validate environment
prime env validate my-environment

# Push to hub
prime env push my-environment

# Push with version tag
prime env push my-environment --version 1.0.0
```

## Using with Training

Environments integrate with PRIME-RL for training:

```bash
# Install environment for training
prime env install gsm8k

# Use in training config
uv run rl @ config.toml --env gsm8k
```

## Using with Evaluation

Push evaluation results for environments:

```bash
# Run evaluation and push results
prime eval push ./outputs/evals/gsm8k--gpt-4o/
```

Results appear on the environment's leaderboard on the hub.

## Environment Categories

| Category | Description | Examples |
|----------|-------------|----------|
| `reasoning` | Math and logical reasoning | gsm8k, hendrycks-math |
| `code` | Code generation and completion | humaneval, mbpp |
| `qa` | Question answering | triviaqa, naturalquestions |
| `games` | Interactive games | wordle, chess |
| `agents` | Multi-turn agent tasks | webshop, alfworld |

