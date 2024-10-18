import torch
from typing import Literal
import torch.distributed.checkpoint as dcp
from zeroband.models.llama import get_model
from zeroband.checkpoint import ModelWrapper
from zeroband.utils import get_module_signature
from zeroband.train import Config
from zeroband.utils.logging import get_logger
from pydantic_config import parse_argv
from transformers import AutoTokenizer
import math
from pathlib import Path
from safetensors.torch import save_file


class ExportConfig(Config):
    save_format: Literal["pt", "safetensors"]


def main(config: ExportConfig):
    save_path = Path(config.ckpt.path)
    save_path.mkdir(parents=True, exist_ok=True)

    logger.info("Getting tokenizer (for vocab size)")
    if config.type_model == "llama2":
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", use_fast=True)
    elif config.type_model == "llama3":
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B", use_fast=True)
    else:
        raise ValueError(f"Model type {config.type_model} not supported")

    logger.info("Getting model")
    model, model_config = get_model(
        config.name_model,
        config.type_model,
        vocab_size=len(tokenizer),
        seq_length=config.data.seq_length,
        attn_fn=config.train.attn_fn,
    )

    logger.info("Before load: %s", get_module_signature(model))

    states = {
        "model": ModelWrapper(model),
    }
    logger.info("Loading from %s", config.ckpt.resume)
    dcp.load(
        state_dict=states,
        checkpoint_id=config.ckpt.resume,
    )

    logger.info("After load: %s", get_module_signature(model))

    num_shards = int(sum(p.numel() for p in model.parameters()) / 1e9)
    state_dict = model.state_dict()
    if "freqs_cis" in state_dict:  # This should not be persisted
        del state_dict["freqs_cis"]
    state_keys = list(state_dict.keys())
    shard_size = int(math.ceil(len(state_keys) / num_shards))

    logger.info("Saving model to %d shards", num_shards)
    for i in range(num_shards):
        start = i * shard_size
        end = (i + 1) * shard_size
        if i == 9:
            end = len(state_keys)
        shard = {k: state_dict[k] for k in state_keys[start:end]}
        if config.save_format == "pt":
            torch.save(shard, str(save_path / f"model-{i:04}-of-{num_shards:04}.pt"))
        else:
            save_file(shard, str(save_path / f"model-{i:04}-of-{num_shards:04}.safetensors"))


if __name__ == "__main__":
    logger = get_logger()
    config = ExportConfig(**parse_argv())
    logger.debug(f"config: {config.model_dump()}")

    main(config)
