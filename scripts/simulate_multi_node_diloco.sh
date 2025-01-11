#!/bin/bash

#
# simulate multi nodes on one gpu. start N torchrun on X gpu locally.
# example how to run ./scripts/simulate_multi_node.sh 2 1  src/zeroband/train.py @configs/debug/normal.toml

# Function to get CUDA devices based on the number of GPUs and index
function get_cuda_devices() {
    local num_gpu=$1
    local index=$2
    local start_gpu=$((num_gpu * index))
    local end_gpu=$((start_gpu + num_gpu - 1))

    if [ "$num_gpu" -eq 1 ]; then
        echo $start_gpu
    else
        echo $(seq -s ',' $start_gpu $end_gpu)
    fi
}

# Array to store PIDs of child processes
child_pids=()

# Modified cleanup function to handle tail separately
cleanup() {
    echo "Cleaning up child processes..."
    local killed=0

    # First kill the main processes
    for pid in "${child_pids[@]}"; do
        if kill -TERM "$pid" 2>/dev/null; then
            ((killed++))
        fi
    done

    # Kill the tail process if it exists
    if [ -n "$tail_pid" ]; then
        kill -TERM "$tail_pid" 2>/dev/null
        ((killed++))
    fi

    wait
    echo "All child processes terminated. Killed $killed processes."
    exit
}

# Check if at least three arguments were passed
if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <N> <initial_peer> <num_gpu> [additional_python_args]"
    exit 1
fi


N=$1         # The number of processes
NUM_GPU=$2   # The number of GPUs used by each process
# Remove the first three arguments so $@ contains only additional Python arguments
shift 2

# Register the cleanup function to be called on SIGINT (Ctrl+C)
trap cleanup SIGINT


mkdir -p logs

export GLOBAL_ADDR=localhost
export GLOBAL_PORT=${GLOBAL_PORT:-5565}
export GLOBAL_WORLD_SIZE=$N
export BASE_PORT=${BASE_PORT:-10001}
export GLOO_SOCKET_IFNAME=lo

for i in $(seq 0 $(($N - 1 )))
do
    > logs/log$i.log
    WANDB_MODE=$([ $i -eq 0 ] && echo "online" || echo "offline") GLOBAL_UNIQUE_ID=$i GLOBAL_RANK=$i CUDA_VISIBLE_DEVICES=$(get_cuda_devices $NUM_GPU $i) uv run torchrun --nproc_per_node=$NUM_GPU --node-rank 0 --rdzv-endpoint localhost:$((BASE_PORT + $i)) --nnodes=1  $@ --data.data_rank $i --data.data_world_size $N > logs/log$i.log 2>&1 &
    child_pids+=($!)
done

# Start tail in background and store its PID separately
tail -f logs/log0.log &
tail_pid=$!

# Wait for the main processes only
for pid in "${child_pids[@]}"; do
    wait $pid
done

# Once main processes are done, kill the tail process
if [ -n "$tail_pid" ]; then
    kill -TERM "$tail_pid"
fi
