export WANDB_MODE=disabled
export GLOBAL_ADDR=localhost
export GLOBAL_PORT=1234
export GLOBAL_WORLD_SIZE=2

export CUDA_VISIBLE_DEVICES=0,1
export GLOBAL_UNIQUE_ID=0 
export GLOBAL_RANK=0 

export ZERO_BAND_LOG_LEVEL=DEBUG
export ZERO_BAND_LOG_ALL_RANK=true

uv run torchrun --nproc_per_node=2 \
	--rdzv-endpoint localhost:1000$GLOBAL_RANK \
	src/zeroband/train.py \
	@configs/150M/3090.toml
