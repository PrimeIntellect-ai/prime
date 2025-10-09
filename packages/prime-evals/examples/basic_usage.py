"""
Basic usage example for Prime Evals SDK.

This example demonstrates:
- Creating an evaluation
- Pushing samples
- Finalizing with metrics
- Retrieving evaluation data
"""

from prime_evals import APIClient, EvalsClient


def main():
    """Run the basic evaluation example."""
    # Initialize client (uses PRIME_API_KEY env var or ~/.prime/config.json)
    api_client = APIClient()
    client = EvalsClient(api_client)

    print("Creating evaluation...")

    eval_response = client.create_evaluation(
        name="gsm8k-example-evaluation",
        model_name="gpt-4o-mini",
        dataset="gsm8k",
        framework="verifiers",
        task_type="math",
        description="Example evaluation using Prime Evals SDK",
        tags=["example", "gsm8k", "math"],
        metadata={
            "version": "1.0",
            "num_examples": 3,
            "temperature": 0.7,
            "max_tokens": 2048,
        },
    )

    eval_id = eval_response.get("evaluation_id") or eval_response.get("id")
    print(f"✓ Created evaluation: {eval_id}\n")

    print("Pushing samples...")
    samples = [
        {
            "example_id": 0,
            "task": "gsm8k",
            "reward": 1.0,
            "correct": True,
            "format_reward": 1.0,
            "correctness": 1.0,
            "answer": "18",
            "prompt": [
                {
                    "role": "system",
                    "content": "Solve the math problem and put your final answer in \\boxed{}.",
                },
                {
                    "role": "user",
                    "content": "Janet's ducks lay 16 eggs per day. She eats three for breakfast "
                    "every morning and bakes muffins for her friends every day with four. "
                    "She sells the remainder at the farmers' market daily for $2 per fresh "
                    "duck egg. How much in dollars does she make every day at the farmers' market?",
                },
            ],
            "completion": [
                {
                    "role": "assistant",
                    "content": "Let me solve this step by step.\n\n"
                    "Janet's ducks lay 16 eggs per day.\n"
                    "She eats 3 eggs for breakfast.\n"
                    "She uses 4 eggs for muffins.\n"
                    "So she uses 3 + 4 = 7 eggs total.\n\n"
                    "The remainder is 16 - 7 = 9 eggs.\n"
                    "She sells each egg for $2.\n"
                    "So she makes 9 × $2 = $18.\n\n"
                    "The answer is \\boxed{18}.",
                }
            ],
            "metadata": {"difficulty": "easy"},
        },
        {
            "example_id": 1,
            "task": "gsm8k",
            "reward": 1.0,
            "correct": True,
            "format_reward": 1.0,
            "correctness": 1.0,
            "answer": "3",
            "prompt": [
                {
                    "role": "system",
                    "content": "Solve the math problem and put your final answer in \\boxed{}.",
                },
                {
                    "role": "user",
                    "content": "A robe takes 2 bolts of blue fiber and half that much white "
                    "fiber. How many bolts in total does it take?",
                },
            ],
            "completion": [
                {
                    "role": "assistant",
                    "content": "The robe takes 2 bolts of blue fiber.\n"
                    "It takes half that much white fiber, so 2/2 = 1 bolt of white fiber.\n"
                    "Total: 2 + 1 = 3 bolts.\n\n"
                    "\\boxed{3}",
                }
            ],
            "metadata": {"difficulty": "easy"},
        },
        {
            "example_id": 2,
            "task": "gsm8k",
            "reward": 0.0,
            "correct": False,
            "format_reward": 1.0,
            "correctness": 0.0,
            "answer": "12",
            "prompt": [
                {
                    "role": "system",
                    "content": "Solve the math problem and put your final answer in \\boxed{}.",
                },
                {
                    "role": "user",
                    "content": "Josh decides to try flipping a house. He buys a house for $80,000 "
                    "and then puts in $50,000 in repairs. This increased the value of the house "
                    "by 150%. How much profit did he make?",
                },
            ],
            "completion": [
                {
                    "role": "assistant",
                    "content": "Josh bought the house for $80,000.\n"
                    "He put in $50,000 in repairs.\n"
                    "The value increased by 150%, so the new value is $80,000 * 1.5 = $120,000.\n"
                    "His profit is $120,000 - $80,000 - $50,000 = -$10,000.\n\n"
                    "\\boxed{-10000}",
                }
            ],
            "metadata": {"difficulty": "medium"},
        },
    ]

    # Push samples
    client.push_samples(eval_id, samples)
    print(f"✓ Pushed {len(samples)} samples\n")

    # Calculate final metrics
    print("Computing metrics...")
    total_reward = sum(s["reward"] for s in samples)
    total_correct = sum(1 for s in samples if s["correct"])
    avg_reward = total_reward / len(samples)
    avg_correctness = total_correct / len(samples)

    metrics = {
        "avg_reward": avg_reward,
        "avg_format_reward": 1.0,
        "avg_correctness": avg_correctness,
        "success_rate": avg_correctness,
        "total_samples": len(samples),
    }

    # Finalize evaluation
    print("Finalizing evaluation...")
    client.finalize_evaluation(eval_id, metrics=metrics)
    print("✓ Evaluation finalized\n")

    # Retrieve and display results
    print("Retrieving evaluation details...")
    eval_details = client.get_evaluation(eval_id)
    print(f"   Name: {eval_details.get('name')}")
    print(f"   Model: {eval_details.get('model_name')}")
    print(f"   Dataset: {eval_details.get('dataset')}")
    print(f"   Status: {eval_details.get('status')}")
    print(f"   Total Samples: {eval_details.get('total_samples')}")
    print(f"   Metrics: {eval_details.get('metrics')}\n")

    # Retrieve samples
    print("Retrieving samples...")
    samples_response = client.get_samples(eval_id, page=1, limit=10)
    retrieved_samples = samples_response.get("samples", [])
    print(f"   Retrieved {len(retrieved_samples)} samples")

    print("\n✓ Example completed successfully!")
    print(f"   Evaluation ID: {eval_id}")
    print(f"   Avg Reward: {metrics['avg_reward']:.2f}")
    print(f"   Avg Correctness: {metrics['avg_correctness']:.2f}")


if __name__ == "__main__":
    main()
