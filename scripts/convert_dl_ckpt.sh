#!/bin/bash
set -e

# Wrapper script to run the Python command on 8 checkpoints in parallel
# Usage: ./convert_all.sh /data/10b/step_50800/diloco_0/data

# Input path prefix
INPUT_PATH=$1

# Run the commands for each checkpoint in parallel
for i in {0..7}; do
    CHECKPOINT_PATH="${INPUT_PATH}/_${i}.pt"
    BACKUP_PATH="${INPUT_PATH}/_${i}_old.pt"
    TMP_PATH="${INPUT_PATH}/_${i}_tmp.pt"

    if [ -f "$BACKUP_PATH" ]; then
        echo "Checkpoint ${CHECKPOINT_PATH} has already been processed, skipping." &
    else
        (
            uv run python scripts/convert_dl_state.py @configs/10B/H100.toml \
                --input_path "$CHECKPOINT_PATH" \
                --output_path "$TMP_PATH" \
                --rank "$i" \
                --world_size 8 && \
            mv "$CHECKPOINT_PATH" "$BACKUP_PATH" && \
            mv "$TMP_PATH" "$CHECKPOINT_PATH" && \
            echo "Processed ${CHECKPOINT_PATH} and moved to ${BACKUP_PATH}"
        ) &
    fi
done

# Wait for all background jobs to complete
wait

echo "All checkpoints processed"
