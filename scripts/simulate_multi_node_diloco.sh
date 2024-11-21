#!/bin/bash

#
# Simulate multi-node on a single GPU or multiple GPUs.
# Start N torchrun instances on X GPUs locally.
# Example usage:
# ./scripts/simulate_multi_node.sh 2 1 src/zeroband/train.py @configs/debug/normal.toml

# Function to get the total number of available GPUs
get_total_gpus() {
    nvidia-smi --query-gpu=name --format=csv,noheader | wc -l
}

# Function to get CUDA devices based on the number of GPUs and index
get_cuda_devices() {
    local num_gpu=$1
    local index=$2
    local start_gpu=$((num_gpu * index))
    local end_gpu=$((start_gpu + num_gpu - 1))

    if [ "$TOTAL_GPU" -eq 1 ]; then
        # If only one GPU is available, use GPU 0 for all ranks
        echo "0"
    elif [ "$num_gpu" -eq 1 ]; then
        # Assign a single GPU based on the rank index
        echo "$start_gpu"
    else
        # Assign a range of GPUs based on the rank index
        echo "$(seq -s ',' $start_gpu $end_gpu)"
    fi
}

# Array to store PIDs of child processes
child_pids=()

# Function to kill all child processes
cleanup() {
    echo "Cleaning up child processes..."
    local killed=0
    for pid in "${child_pids[@]}"; do
        if kill -TERM "$pid" 2>/dev/null; then
            ((killed++))
        fi
    done
    wait
    echo "All child processes terminated. Killed $killed processes."
    exit
}

# Register the cleanup function to be called on SIGINT (Ctrl+C) and SIGTERM
trap cleanup SIGINT SIGTERM

# Check if at least three arguments were passed
if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <N> <num_gpu_per_node> <python_script> [additional_python_args...]"
    echo "Example: $0 2 1 src/zeroband/train.py @configs/debug/normal.toml"
    exit 1
fi

N=$1               # Number of ranks/nodes
NUM_GPU=$2         # Number of GPUs per node
shift 2            # Shift the first two arguments so that $@ contains only additional Python arguments

# Determine the total number of GPUs available on the system
TOTAL_GPU=$(get_total_gpus)

# Ensure that NUM_GPU does not exceed TOTAL_GPU
if [ "$NUM_GPU" -gt "$TOTAL_GPU" ]; then
    echo "Requested NUM_GPU ($NUM_GPU) exceeds the total available GPUs ($TOTAL_GPU)."
    echo "Setting NUM_GPU to $TOTAL_GPU."
    NUM_GPU=$TOTAL_GPU
fi

mkdir -p logs

export GLOBAL_ADDR=localhost
export GLOBAL_PORT=${GLOBAL_PORT:-5565}
export GLOBAL_WORLD_SIZE=$N
export BASE_PORT=${BASE_PORT:-10001}

for i in $(seq 0 $((N - 1)))
do
    LOG_FILE="logs/log$i.log"
    > "$LOG_FILE"

    CUDA_DEVICES=$(get_cuda_devices "$NUM_GPU" "$i")

    echo "Starting rank $i with CUDA_VISIBLE_DEVICES=$CUDA_DEVICES"

    WANDB_MODE=$([ "$i" -eq 0 ] && echo "online" || echo "online") \
    GLOBAL_UNIQUE_ID=$i \
    GLOBAL_RANK=$i \
    CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" \
    torchrun --nproc_per_node="$NUM_GPU" \
             --node_rank=0 \
             --rdzv_endpoint=localhost:$((BASE_PORT + i)) \
             --rdzv_id=simulate_multi_node \
             --rdzv_backend=c10d \
             --nnodes=1 \
             "$@" \
             --data.data_rank "$i" \
             --data.data_world_size "$N" \
             > "$LOG_FILE" 2>&1 &

    child_pids+=($!)
done

# Optionally, tail the first log file to monitor progress
if [ "$TOTAL_GPU" -ge 1 ]; then
    tail -f "logs/log0.log" &
    child_pids+=($!)
fi

# Wait for all child processes to finish
wait
