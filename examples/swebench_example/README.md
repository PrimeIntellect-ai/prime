# Prime SWE-bench Example

This directory contains an example script to run SWE-bench with Prime sandboxes.

### Prerequisites

- Prime CLI installed and configured with optional `swebench` package
- Valid API key (run `prime login` first)
- Python environment with prime-cli package available

### Installation

```bash
uv pip install -e ".[swebench]"
```

### Usage

The script accepts a subset of arguments from the SWE-bench run_evaluation script.

```bash
python examples/swebench_example/swebench_example.py \
    --dataset_name rasdani/SWE-bench_Verified_oracle-parsed_commits_32k_2 \
    --predictions_path Qwen__Qwen3-14B_preds.jsonl \
    --run_id Qwen__Qwen3-14B
```
