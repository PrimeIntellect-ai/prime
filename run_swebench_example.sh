PREDS_PATH="Qwen__Qwen3-14B_preds_2.jsonl"
RUN_ID="Qwen__Qwen3-14B_preds_2"

if [ -d "logs/run_evaluation/$RUN_ID" ]; then
    rm -rf "logs/run_evaluation/$RUN_ID"
fi


python examples/swebench_example.py \
    --dataset_name rasdani/SWE-bench_Verified_oracle-parsed_commits_32k_2 \
    --predictions_path $PREDS_PATH \
    --run_id $RUN_ID \