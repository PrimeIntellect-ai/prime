for topk in 5 50 500 1000 2000 3000; do
#   ./scripts/simulate_multi_node_diloco.sh 1 8 src/zeroband/train.py @ configs/150M_short/H100.toml \
#     --optim.optim.lr 3e-3 \
#     --optim.optim.precondition_frequency 100 \
#     --optim.optim.topk_compression $topk
    ./scripts/simulate_multi_node_diloco.sh 2 2 src/zeroband/train.py @ configs/debug/normal.toml
    echo "done with $topk"
done



