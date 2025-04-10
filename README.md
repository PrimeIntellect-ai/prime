# prime - decentralized training at scale
prime (previously called ZeroBand) is a framework for efficient, globally distributed training of AI models over the internet.

https://github.com/user-attachments/assets/c034d2a2-400c-4bf8-acd0-c84b6c897d69

## Key Features

- **`ElasticDeviceMesh` for Fault Tolerant Training:**
    - In Prime, we’ve added a new distributed abstraction called `ElasticDeviceMesh` which encapsulates dynamic global process groups for fault-tolerant communication across the internet and local process groups for communication within a node or datacenter.
    - The `ElasticDeviceMesh` manages the resizing of the global process groups when nodes join or leave, unlike the standard `DeviceMesh` in torch distributed, which will crash and require a cold restart to resize the process group.
    - In order to know when to resize the process groups, we use a heartbeat mechanism to discover dead nodes and remove them from the process group. Crashing nodes will attempt a best effort deathrattle to fail their own heartbeat quickly, saving its comrades the timeout.
- **Asynchronous distributed checkpointing**
    - Due to the size of the model, checkpointing can be an expensive operation, taking up to 20 minutes on the nodes we tested. This would reduce our compute utilisation if it blocked the main training process.
    - In order to minimize the blocking time, we first checkpoint into `/dev/shm` which is a RAM backed filesystem. This operation is much faster and we can unblock the main training process once the checkpoint has been created in `/dev/shm`.
    - We then use two subprocesses to asynchronously copy the checkpoint out of `/dev/shm` into the checkpoint directory on disk as well as upload it to the remote.
- **Live checkpoint recovery**
    - Nodes that wish to join the run mid-training need to be able to get the most recent state of the model and optimiser before being able to contribute to the training. They must complete this operation in the time window between two outer steps, otherwise, the checkpoint they receive would be stale.
    - In order to do this quickly, we have the joining nodes request the checkpoints from its peers which all host a sidecar HTTP server serving the latest checkpoint out of `/dev/shm`.
    - Once the joining node has downloaded and initialized the model, it skips the inner steps and joins the outer step with zero pseudo-gradients. This is to prevent the joining node from stalling the existing nodes. If the joining node also performed the inner steps, it would be late to the outer step by the time it took to download and load the checkpoint, reducing the clusters compute utilisation.
- **Custom Int8 All-Reduce Kernel**
    - In our experiments, we found that we are able to perform int8 quantization on the pseudo gradients without any impact on the loss curves. This means that we can reduce the payload size of each outer step all-reduce by 4x if we communicate the pseudo-gradients in int8 instead of fp32.
    - However, we need to accumulate the reduce in fp32, dequantizing and re-quantizing intermediate results during the all-reduce. This is not supported by any collective communication libraries.
    - We thus implemented our own fully pipelined ring-reduce kernel in C++ which is JIT compiled as a custom operator using the torch library.
    - However, with the amount of quantization work we needed to perform, using the torch ops (`quantize_per_tensor`, `scatter_add`, `index`, etc) was too slow, resulting in underutilisation of our target network bandwidth of 4 Gbps.
    - We thus implemented our own multithreaded uint8 ops in C++  to perform the quantization and dequantization operations, improving the quantization speed by more than 60x.
- **Maximising bandwidth utilization:**
    - By sharding our DiLoCo pseudo-gradients in a node, we can maximise network bandwidth utilization by opening multiple connections at the same time when performing the all-reduce. This yielded a transfer speed improvement of 8x on some nodes.
    - Relying on the public IP forward resulted in poor or unstable p2p bandwidth on some compute providers. To mitigate this, we employ VPN technology to optimize peer-to-peer connections between nodes, allowing us to better utilize the available internet bandwidth between nodes by modifying the routing of packets through the internet.
    - We’ve improved bandwidth utilization between nodes in similar data center settings by up to 40x compared to our OpenDiLoCo release, achieving up to 4Gb/s connections between data centers across the whole United States.
- **PyTorch FSDP2 / DTensor ZeRO-3 implementation**
    - In order to fit the 10B model training within our given memory resources, we had to do shard the model weights, gradients and optimizer states between intra-node GPUs.
    - We achieved this using the `fully_shard` API from PyTorch FSDP2 which wraps the model parameters as `DTensor`s and registers hooks to schedule all-gather and reduce-scatter on the tensors when they are used. FSDP2 also optimizes the collectives by bucketing the parameters into `FSDPParamGroup`s. This allows us to execute the collectives on larger tensors, improving protocol-to-payload ratio and improving the overlap from pipelining. We employ the same trick for our pseudo-gradients, bucketing them by layer.
- **CPU Off-Loading**
    - Our Diloco optimizer does not add any GPU overhead. All the tensors required by the Diloco optimizer are offloaded to CPU memory.
    - Since we only perform a global sync every hundreds of steps, the reduced speed of copying and calculating the pseudo-gradient on cpu is negligible relative to the time to execute the inner steps and all-reduce.

A research paper about the framework and our INTELLECT-1 10B experiment can be found [here](https://arxiv.org/abs/2412.01152).

## Getting Started

For an easy install that download the data

```
curl -sSL https://raw.githubusercontent.com/PrimeIntellect-ai/prime/main/scripts/install/install.sh | bash
```

step by step :


1. Clone: 

```bash
git clone git@github.com:PrimeIntellect-ai/prime.git
```

2. Install `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

3. Set up the environment:
```bash
sudo apt install iperf -y
uv venv
source .venv/bin/activate
uv sync --extra all
git submodule update --init --recursive
```


4. Log into Hugging Face:
```bash
huggingface-cli login
```

5. Download the data 
```
mkdir -p datasets
uv run python scripts/subset_data.py --dataset_name PrimeIntellect/fineweb-edu --data_world_size 1 --data_rank 0 --max_shards 32
mv fineweb-edu/ datasets/fineweb-edu/
```


### Quick Check

Verify your setup:

```bash
GLOO_SOCKET_IFNAME=lo GLOBAL_ADDR=localhost GLOBAL_RANK=0 GLOBAL_UNIQUE_ID=0 GLOBAL_WORLD_SIZE=1 GLOBAL_PORT=8989  uv run torchrun --nproc_per_node=2 src/zeroband/train.py  @configs/debug/diloco.toml
```

## Usage

### Running DiLoCo

To test DiLoCo locally you can use the helper script `scripts/simulate_multi_node_diloco.sh`

```bash
# Using 4 GPUs (2 diloco workers, each across 2 GPUs)
ZERO_BAND_LOG_LEVEL=DEBUG ./scripts/simulate_multi_node_diloco.sh 2 2 src/zeroband/train.py @configs/debug/diloco.toml

# Using 2 GPUs (2 diloco workers, each on a single GPU)
ZERO_BAND_LOG_LEVEL=DEBUG ./scripts/simulate_multi_node_diloco.sh 2 1 src/zeroband/train.py @configs/debug/diloco.toml
```

### Running Tests

Ensure you have at least two GPU to run the full test suite:
```bash
uv run pytest
```


### Eval

To eval you need first to convert the checkpoint to a huggingface compatible model.

```bash
uv run python scripts/export_dcp.py @configs/10B/H100.toml --ckpt.path CONVERTED_MODEL_PATH --ckpt.resume CHECKPOINT_PATH --torch_dtype bfloat16  --ckpt.interval 1
```


```
uv run accelerate launch -m lm_eval --model hf --model_args pretrained=CONVERTED_MODEL_PATH,add_bos_token=True  --tasks hellaswag --num_fewshot 10
```


## Environment variables
### Global Store Initialization
| Environment Variable  | Description                                      | Default Value |
|-----------------------|--------------------------------------------------|---------------|
| `GLOBAL_UNIQUE_ID`    | Unique identifier worker in global store.        | `None`  |
| `GLOBAL_ADDR`         | IP Address of the global store                   | `None`  |
| `GLOBAL_PORT`         | Port number of the global store.                 | `None` |
| `GLOBAL_WORLD_SIZE`   | The size of the global process group.            | `1` |
| `GLOBAL_RANK`         | Rank of the process in the global process group. | `0` |

### Elastic Device Mesh Configuration
| Environment Variable  | Description                                      | Default Value |
|-----------------------|--------------------------------------------------|---------------|
| `ZERO_BAND_LOG_LEVEL` | Enable debug log lines | `False` |
| `ZERO_BAND_GLOBAL_STORE_TIMEOUT_SECONDS` | Number of seconds before the global store operations timeout | `300` |
| `ZERO_BAND_GLOBAL_PG_TIMEOUT_SECONDS` | Number of seconds before the global process group operations timeout | `600` |
| `ZERO_BAND_GLOBAL_STORE_POLLING_INTERVAL_SECONDS` | Number of seconds between polls to the store when waiting for values | `0.1` |
| `ZERO_BAND_EDM_HEARTBEAT_INTERVAL_SECONDS` | Interval in seconds between heartbeats | `2` |
| `ZERO_BAND_EDM_HEARTBEAT_TIMEOUT_SECONDS` | Time in seconds after which a node is considered dead if no heartbeat is received | `10` |
| `ZERO_BAND_LIVE_RECO_PORT` | Port number for the live recovery server | random |
| `ZERO_BAND_LIVE_RECO_ADDR` | IP Address for the live recovery server | `localhost` |

## Troubleshooting

If you encounter any dataset loading errors at the beginning of training, try setting:

```bash
export HF_HUB_ETAG_TIMEOUT=500
```

## Pre-downloading datasets
Streaming datasets from huggingface hub can sometimes result in http 443 errors which will crash the training process.
To avoid them, you can pre-download the dataset.

Here is an example that downloads all the files in `PrimeIntellect/fineweb-edu` which are used by `data_rank` 5 in a training with `data_world_size` of 12.
```bash
python3 scripts/subset_data.py --dataset_name PrimeIntellect/fineweb-edu --data_world_size 12 --data_rank 5
```

For info about the arguments to the script, do:
```bash
python3 scripts/subset_data.py --help
```

# Exporting checkpoints to huggingface compatible model
You can convert the checkpoints saved by the training script to a model that can be run with any huggingface-compatible inference engine (e.g. transformers, vLLM) using our export script.
The export script takes the training config as a positional argument and 2 keyword arguments, `ckpt.resume` which is the path to the checkpoint, `ckpt.path` which is the path you wish to save the converted model.
You may also pass the `torch_dtype` argument to either `float32` or `bfloat16` to specify the precision of the exported model weights. The default `torch_dtype` is `float32`.

Example export command:
```bash
python scripts/export_dcp.py @configs/10B/H100.toml --ckpt.path /path/to/save/converted_model --ckpt.resume /path/to/ckpt/step_84000 --torch_dtype bfloat16
```

You can then upload the model to huggingface using huggingface-cli:
```bash
# Usage:  huggingface-cli upload [repo_id] [local_path] [path_in_repo]
huggingface-cli upload username/mymodel /path/to/save/converted_model . --private
```
The repo will be created if `repo_id` does not exist. The `--private` will create the repo as a private repo and can be ommited to create a publicly accessible repo.
