# RL Training Configuration Guide

Run RL training jobs using TOML config files with the `prime rl run` command.

## Quick Start

```bash
# Create a config file
prime rl init > my-config.toml

# Start a training run
prime rl run @my-config.toml

# List runs
prime rl list

# View logs
prime rl logs <run-id>

# Stop a run
prime rl stop <run-id>
```

## Config File Examples

### Basic Config

```toml
model = "PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT"
max_steps = 100
batch_size = 8
rollouts_per_example = 2

[sampling]
max_tokens = 128
temperature = 0.8

[[env]]
id = "your-username/your-env"
name = "my-training-run"
```

### Config with Environment Args

```toml
model = "PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT"
max_steps = 200
batch_size = 16
rollouts_per_example = 4

[sampling]
max_tokens = 256
temperature = 0.9

[[env]]
id = "your-username/your-env"
name = "custom-args-run"
args = { dataset_split = "train", system_prompt = "You are a helpful assistant." }
```

### Config with Evaluation

```toml
model = "PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT"
max_steps = 500
batch_size = 8
rollouts_per_example = 2

[sampling]
max_tokens = 128
temperature = 0.8

[[env]]
id = "your-username/your-env"
name = "training-with-eval"

[eval]
interval = 50  # Run eval every 50 steps

[[eval.env]]
id = "your-username/your-env"
num_examples = 10
rollouts_per_example = 2
```

### Config with W&B Logging

```toml
model = "PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT"
max_steps = 100
batch_size = 8
rollouts_per_example = 2

[sampling]
max_tokens = 128
temperature = 0.8

[[env]]
id = "your-username/your-env"
name = "wandb-run"

[wandb]
project = "my-rl-project"
entity = "my-team"
name = "experiment-1"
```

Then run with your W&B API key:
```bash
prime rl run @config.toml --wandb-api-key YOUR_KEY
# or
WANDB_API_KEY=YOUR_KEY prime rl run @config.toml
```

## Configuration Reference

### Top-Level Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `model` | string | Yes | HuggingFace model name |
| `max_steps` | int | Yes | Number of training steps |
| `batch_size` | int | Yes | Training batch size |
| `rollouts_per_example` | int | Yes | Rollouts per training example |

### `[sampling]` Section

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_tokens` | int | Required | Max tokens to generate |
| `temperature` | float | 1.0 | Sampling temperature (>= 0) |

### `[[env]]` Section (Training Environment)

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | string | Yes | Environment ID (username/env-name) |
| `name` | string | No | Run name for display |
| `args` | table | No | Extra args passed to environment |

### `[eval]` Section (Optional)

| Field | Type | Description |
|-------|------|-------------|
| `interval` | int | Run eval every N steps |

### `[[eval.env]]` Section (Eval Environments)

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | string | Yes | Environment ID |
| `num_examples` | int | No | Number of examples to evaluate |
| `rollouts_per_example` | int | No | Rollouts per eval example |

### `[wandb]` Section (Optional)

| Field | Type | Description |
|-------|------|-------------|
| `project` | string | W&B project name |
| `entity` | string | W&B team/entity |
| `name` | string | Run name in W&B |

## Notes

- **seq_len** is NOT user-configurable - it's set at the cluster level
- **max_tokens** controls generation length and should be less than cluster's seq_len
- Temperature must be >= 0 (negative values are rejected)
- Use `prime rl models` to see available models

## Commands Reference

```bash
# Initialize a template config
prime rl init > config.toml

# Start a run
prime rl run @config.toml

# List available models
prime rl models

# List your runs
prime rl list

# View run logs
prime rl logs <run-id>

# Stop a run
prime rl stop <run-id>

# Delete a run
prime rl delete <run-id>
```
