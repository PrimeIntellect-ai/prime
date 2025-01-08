export TORCHFT_LIGHTHOUSE=http://localhost:29510
export TORCHFT_MANAGER_PORT=29512
export CUDA_VISIBLE_DEVICES=2,3

uv run torchrun --nproc_per_node=2 \
	--rdzv-endpoint localhost:10002 \
	meow.py
