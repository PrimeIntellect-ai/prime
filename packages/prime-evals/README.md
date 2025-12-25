# Prime Evals SDK

Lightweight Python SDK for managing Prime Intellect evaluations - push, track, and analyze your model evaluation results.

## Features

- **Simple evaluation management** - Create, push samples, and finalize evaluations
- **Type-safe** - Full type hints and Pydantic models
- **Authentication caching** - Automatic token management
- **Environment checking** - Validate environments before pushing
- **No CLI dependencies** - Pure SDK, lightweight installation
- **Context manager support** - Automatic resource cleanup

## Installation

```bash
uv pip install prime-evals
```

Or with pip:
```bash
pip install prime-evals
```

## Quick Start

```python
from prime_evals import APIClient, EvalsClient

# Initialize client
api_client = APIClient(api_key="your-api-key")
client = EvalsClient(api_client)

# Create an evaluation
eval_response = client.create_evaluation(
    name="gsm8k-gpt4o-baseline",
    model_name="gpt-4o-mini",
    dataset="gsm8k",
    framework="verifiers",
    metadata={
        "version": "1.0",
        "num_examples": 10,
        "temperature": 0.7,
    }
)

eval_id = eval_response["evaluation_id"]
print(f"Created evaluation: {eval_id}")

# Push samples
samples = [
    {
        "example_id": 0,
        "reward": 1.0,
        "correct": True,
        "answer": "18",
        "prompt": [{"role": "user", "content": "What is 9+9?"}],
        "completion": [{"role": "assistant", "content": "The answer is 18."}],
    }
]

client.push_samples(eval_id, samples)

# Finalize with metrics
metrics = {
    "avg_reward": 0.87,
    "avg_correctness": 0.82,
    "success_rate": 0.87,
}

client.finalize_evaluation(eval_id, metrics=metrics)
print("Evaluation finalized!")
```

## Async Usage

```python
import asyncio
from prime_evals import AsyncEvalsClient

async def main():
    async with AsyncEvalsClient(api_key="your-api-key") as client:
        # Create evaluation
        eval_response = client.create_evaluation(
            name="my-evaluation",
            model_name="gpt-4o-mini",
            dataset="gsm8k",
        )
        
        eval_id = eval_response["evaluation_id"]
    
        # Push samples
        await client.push_samples(eval_id, samples)
        
        # Finalize
        await client.finalize_evaluation(eval_id)
        
# Client automatically closed

asyncio.run(main())
```

## Authentication

The SDK looks for credentials in this order:

1. **Direct parameter**: `APIClient(api_key="sk-...")`
2. **Environment variable**: `export PRIME_API_KEY="sk-..."`
3. **Config file**: `~/.prime/config.json` (created by `prime login` CLI command)

## Complete Example

```python
from prime_evals import APIClient, EvalsClient

# Initialize
api_client = APIClient(api_key="your-api-key")
client = EvalsClient(api_client)

# Create evaluation with full metadata
eval_response = client.create_evaluation(
    name="gsm8k-experiment-1",
    model_name="gpt-4o-mini",
    dataset="gsm8k",
    framework="verifiers",
    task_type="math",
    description="Baseline evaluation on GSM8K dataset",
    tags=["baseline", "math", "gsm8k"],
    metadata={
        "version": "1.0",
        "timestamp": "2025-10-09T12:00:00Z",
        "num_examples": 100,
        "temperature": 0.7,
        "max_tokens": 2048,
    }
)

eval_id = eval_response["evaluation_id"]

# Push samples in batches
samples_batch = [
    {
        "example_id": i,
        "task": "gsm8k",
        "reward": 1.0 if i % 2 == 0 else 0.5,
        "correct": i % 2 == 0,
        "format_reward": 1.0,
        "correctness": 1.0 if i % 2 == 0 else 0.0,
        "answer": str(i * 2),
        "prompt": [
            {"role": "system", "content": "Solve the math problem."},
            {"role": "user", "content": f"What is {i} + {i}?"}
        ],
        "completion": [
            {"role": "assistant", "content": f"The answer is {i * 2}."}
        ],
        "info": {"batch": 1}
    }
    for i in range(10)
]

client.push_samples(eval_id, samples_batch)

# Finalize with computed metrics
final_metrics = {
    "avg_reward": 0.75,
    "avg_format_reward": 1.0,
    "avg_correctness": 0.50,
    "success_rate": 0.75,
    "total_samples": len(samples_batch),
}

client.finalize_evaluation(eval_id, metrics=final_metrics)

# Retrieve evaluation details
eval_details = client.get_evaluation(eval_id)
print(f"Evaluation Status: {eval_details.get('status')}")

# List all evaluations
evaluations = client.list_evaluations(limit=10)
for eval in evaluations.get("evaluations", []):
    print(f"{eval['name']}: {eval.get('total_samples', 0)} samples")

# Get samples
samples_response = client.get_samples(eval_id, page=1, limit=100)
print(f"Retrieved {len(samples_response.get('samples', []))} samples")
```

## Push from JSON File

You can also push evaluations from a JSON file:

```python
import json
from prime_evals import APIClient, EvalsClient

with open("eval_results.json") as f:
    eval_data = json.load(f)

api_client = APIClient()
client = EvalsClient(api_client)
# Create
eval_response = client.create_evaluation(
    name=eval_data["eval_name"],
    model_name=eval_data["model_name"],
    dataset=eval_data["dataset"],
    metadata=eval_data.get("metadata"),
    metrics=eval_data.get("metrics"),
)

eval_id = eval_response["evaluation_id"]

# Push samples
if "results" in eval_data:
    client.push_samples(eval_id, eval_data["results"])

# Finalize
client.finalize_evaluation(eval_id, metrics=eval_data.get("metrics"))

print(f"Successfully pushed evaluation: {eval_id}")
```

## API Reference

### EvalsClient

Main client for interacting with the Prime Evals API.

**Methods:**

- `create_evaluation()` - Create a new evaluation
- `push_samples()` - Push evaluation samples
- `finalize_evaluation()` - Finalize an evaluation with final metrics
- `get_evaluation()` - Get evaluation details by ID
- `list_evaluations()` - List evaluations with optional filters
- `get_samples()` - Get samples for an evaluation

### AsyncEvalsClient

Async version of EvalsClient with the same methods (all async).

### Models

**Evaluation**
- Full evaluation object with metadata

**Sample**
- Individual evaluation sample with prompt/completion/scores

**CreateEvaluationRequest**
- Request model for creating evaluations

**EvaluationStatus**
- Enum: PENDING, RUNNING, COMPLETED, FAILED, CANCELLED

## Error Handling

```python
from prime_evals import APIClient, EvalsClient, EvalsAPIError, EvaluationNotFoundError

try:
    api_client = APIClient()
    client = EvalsClient(api_client)
    client.get_evaluation("non-existent-id")
except EvaluationNotFoundError:
    print("Evaluation not found")
except EvalsAPIError as e:
    print(f"API error: {e}")
```

## Related Packages

- **`prime`** - Full CLI + SDK with pods, sandboxes, inference, and more (includes this package)
- **`prime-sandboxes`** - SDK for managing remote code execution environments

## License

MIT License - see LICENSE file for details

