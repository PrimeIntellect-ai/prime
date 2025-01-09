export TORCHFT_LIGHTHOUSE=http://localhost:29510
export TORCHFT_MANAGER_PORT=29513
export CUDA_VISIBLE_DEVICES=4,5

uv run torchrun --nproc_per_node=2 \
	--rdzv-endpoint localhost:10003 \
	meow.py
