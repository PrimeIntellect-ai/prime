export PCCL_LOG_LEVEL=DEBUG
export WANDB_MODE=disabled
export PCCL_MASTER_ADDR=127.0.0.1:48148

export CUDA_VISIBLE_DEVICES=0,1
#export GLOBAL_RANK=0 
#export GLOBAL_UNIQUE_ID=A0 
#export REPLICA_GROUP_ID=A0

#export GLOO_SOCKET_IFNAME=tailscale0
export ZERO_BAND_LOG_LEVEL=DEBUG
export ZERO_BAND_LOG_ALL_RANK=true

uv run torchrun --nproc_per_node=2 \
	--rdzv-endpoint localhost:10001 \
	src/zeroband/train.py \
	@configs/150M/3090.toml \
	--no-wandb-resume
	#--ckpt.live_recovery_rank_src 0
