# Prime SWE-bench Example

This directory contains an example script to run SWE-bench with Prime sandboxes.
Each sandbox runs execution based verification of predicted patches. I.e. it runs the original test suite of the patched repository with a Docker image specific to every example.

### Prerequisites

- Prime CLI installed and configured with optional `swebench` package
- Valid API key (run `prime login` first)
- Python environment with prime-cli package available

### Installation

```bash
uv pip install -e ".[swebench]"
```

### Predictions

You can pull 100 predicted patches from Qwen3-14B with the following script:

```python
import datasets

ds = datasets.load_dataset("rasdani/Qwen__Qwen3-14B_preds_100", split="train")
instance_ids = [
    "sympy__sympy-17655",   # successful patch
    "matplotlib__matplotlib-24637", # unsuccessful patch
]
ds = ds.filter(lambda x: x["instance_id"] in instance_ids)

# ds.to_json("Qwen__Qwen3-14B_preds_100.jsonl", orient="records", lines=True)
ds.to_json("Qwen__Qwen3-14B_preds_2.jsonl", orient="records", lines=True)
```

### Usage

The script accepts a subset of arguments from the SWE-bench run_evaluation script.

```bash
# DATASET_NAME=rasdani/SWE-bench_Verified_oracle-parsed_commits_32k_100
DATASET_NAME=rasdani/SWE-bench_Verified_oracle-parsed_commits_32k_2

# PREDS_PATH=Qwen__Qwen3-14B_preds_100.jsonl
PREDS_PATH=Qwen__Qwen3-14B_preds_2.jsonl

# RUN_ID=Qwen__Qwen3-14B_100
RUN_ID=Qwen__Qwen3-14B_2

python examples/swebench_example/swebench_example.py \
    --dataset_name $DATASET_NAME \
    --predictions_path $PREDS_PATH \
    --run_id $RUN_ID
```
