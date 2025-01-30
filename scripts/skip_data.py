"""
This script is simulating a training to exaust the datasets and recover the dataloader ckpt.

It has the same api as the training one. The only difference is that you probably want to change the total_steps and put a data_path.

It can load config from the config file to have the same setup as the real run.

example.
```
uv run torchrun --nproc_per_node=4 scripts/skip_data.py @configs/150M/3090.toml --optim.total_steps 100 --ckpt.data_path out_data
```

"""

import os
import torch
from pydantic_config import parse_argv


from transformers import AutoTokenizer
from zeroband.checkpoint import CkptManager
from zeroband.config import resolve_env_vars
from zeroband.train import Config

from zeroband.data import get_dataloader

from zeroband.utils.world_info import get_world_info
from zeroband.utils.logger import get_logger


def skip_data(config: Config):
    # batch_size is the total batch size for all GPUs
    assert config.optim.batch_size % world_info.local_world_size == 0
    batch_size = config.optim.batch_size // world_info.local_world_size

    assert batch_size % config.train.micro_bs == 0
    gradient_accumulation_steps = batch_size // config.train.micro_bs

    if config.type_model == "llama2":
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", use_fast=True)
    elif config.type_model == "llama3":
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B", use_fast=True)
    else:
        raise ValueError(f"Model type {config.type_model} not supported")

    logger.debug("tokenizer loaded")

    train_dataloader = get_dataloader(
        tokenizer=tokenizer,
        world_size=world_info.world_size,
        rank=world_info.rank,
        batch_size=config.train.micro_bs,
        data_config=config.data,
    )

    train_dataloader_iterator = iter(train_dataloader)

    logger.info("starting skipping data up to step: %d", config.optim.total_steps)

    total_steps = 0

    while True:
        num_inner_steps = config.diloco.inner_steps if config.diloco is not None else 1

        for _inner_step in range(num_inner_steps):
            for _ in range(gradient_accumulation_steps):
                next(train_dataloader_iterator)

        total_steps += num_inner_steps
        logger.info("total steps: %d", total_steps)
        if total_steps >= config.optim.total_steps:
            break

    CkptManager.save_data(os.path.join(config.ckpt.data_path, "data"), train_dataloader, world_info.local_rank)

    logger.info("skipped data up to step: %d", config.optim.total_steps)


if __name__ == "__main__":
    torch.manual_seed(42)

    world_info = get_world_info()
    logger = get_logger()

    config = Config(**parse_argv())
    resolve_env_vars(config)

    skip_data(config)
