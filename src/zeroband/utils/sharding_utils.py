import torch.nn
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy, CPUOffloadPolicy

from zeroband.config import HardwareConfig


def apply_sharding(hardware_config: HardwareConfig, model: torch.nn.Module):
    """
    Applies the sharding strategy to the model according to the configuration.
    Will use FSDP with optional re-sharding for backward depending on configuration
    """
    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16, reduce_dtype=torch.float32 if hardware_config.reduce_fp32 else None
    )

    offload_policy = CPUOffloadPolicy(pin_memory=True) if hardware_config.fsdp_cpu_offload else None

    for layer_id, transformer_block in model.layers.items():
        if hardware_config.reshard_after_forward:
            # As an optimization, do not re-shard after forward for the last
            # transformer block since FSDP would prefetch it immediately
            reshard_after_forward = int(layer_id) < len(model.layers) - 1
        else:
            reshard_after_forward = False

        fully_shard(
            transformer_block,
            mp_policy=mp_policy,
            reshard_after_forward=reshard_after_forward,
            offload_policy=offload_policy,
        )
    fully_shard(
        model,
        mp_policy=mp_policy,
        reshard_after_forward=hardware_config.reshard_after_forward,
        offload_policy=offload_policy,
    )
