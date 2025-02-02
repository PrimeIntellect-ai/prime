from zeroband.models.llama.model import Transformer

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

from zeroband.utils.logger import get_logger


def apply_ac_ckpt(model: Transformer, num: int):
    """Apply activation checkpointing to the model.
    Apply to layers multiple of `num`.

    Example if `num=2` only half of the layers are checkpointed.
    """
    logger = get_logger()

    layers_ckpt = 0

    for layer_id, transformer_block in model.layers.named_children():
        if layers_ckpt % num == 0:
            transformer_block = checkpoint_wrapper(transformer_block, preserve_rng_state=False)
            model.layers.register_module(layer_id, transformer_block)
            layers_ckpt += 1

    logger.debug(f"Applied activation checkpointing to {layers_ckpt} layers")
