export WANDB_MODE=disabled
#export GLOBAL_ADDR=localhost
#export GLOBAL_PORT=1234
#export GLOBAL_WORLD_SIZE=3
export TORCHFT_LIGHTHOUSE=http://localhost:29510
#export NUM_REPLICA_GROUPS=3

export TORCHFT_MANAGER_PORT=29513
export CUDA_VISIBLE_DEVICES=4,5
#export GLOBAL_RANK=0 
#export GLOBAL_UNIQUE_ID=A0 
#export REPLICA_GROUP_ID=A0

#export GLOO_SOCKET_IFNAME=tailscale0
export ZERO_BAND_LOG_LEVEL=DEBUG
export ZERO_BAND_LOG_ALL_RANK=true

uv run torchrun --nproc_per_node=2 \
	--rdzv-endpoint localhost:10003 \
	src/zeroband/train.py \
	@configs/150M/3090.toml \
	--no-wandb-resume
	#--ckpt.live_recovery_rank_src 0
