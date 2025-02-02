#!/usr/bin/env python
# coding: utf-8
# Example Usage:
# python scripts/convert_dl_state.py @configs/10B/H100.toml --input_path /workspace/step_49200/diloco_0/data/_3.pt --output_path ./meow.pt --rank 3 --world_size 8

import torch
from zeroband.config import resolve_env_vars
from zeroband.data import get_dataloader
from transformers import AutoTokenizer
from zeroband.train import Config
from zeroband.utils.logger import get_logger
from pydantic_config import parse_argv

COMMON_KEYS = [
    "_snapshot._main_snapshot._sampler_iter_yielded",
    "_snapshot._snapshot_step",
    "_snapshot._main_snapshot._index_sampler_state.samples_yielded",
    "_snapshot._main_snapshot._num_workers",
    "_snapshot._main_snapshot._sampler_iter_state",
    "_snapshot._main_snapshot._shared_seed",
    "_snapshot._last_yielded_worker_id",
    "_snapshot._main_snapshot._base_seed",
]


def traverse_dict(d: dict, key: str):
    _k = key.split(".")
    for k in _k:
        d = d[k]
    return d


def transfer_states(old_state_dict: dict, new_state_dict: dict):
    for k in COMMON_KEYS:
        parent, _, child = k.rpartition(".")
        if parent:
            traverse_dict(new_state_dict, parent)[child] = traverse_dict(old_state_dict, parent)[child]
    for worker_id in range(4):
        ex_iterables = [
            ds_state["ex_iterable"]
            for ds_state in traverse_dict(
                old_state_dict, f"_snapshot._worker_snapshots.worker_{worker_id}.dataset_state.ex_iterable.ex_iterables"
            )
        ]
        num_ds = len(ex_iterables)
        new_ds_state = traverse_dict(
            new_state_dict, f"_snapshot._worker_snapshots.worker_{worker_id}.dataset_state.dataset"
        )
        # HACK: dataset_4 is openwebmath which is not always present
        if "dataset_4" not in new_ds_state.keys():
            num_ds -= 1
        new_ds_state = [
            traverse_dict(
                new_state_dict, f"_snapshot._worker_snapshots.worker_{worker_id}.dataset_state.dataset.dataset_{i}"
            )
            for i in range(num_ds)
        ]

        for new_state, old_state in zip(new_ds_state, ex_iterables):
            # HACK: We might index error because of skipping into a different sized shard for dclm
            new_state["file_index"] = (old_state["shard_idx"] + 1) % len(new_state["files"])
            new_state["row_index"] = 0  # old_state["shard_example_idx"]


class ExportConfig(Config):
    input_path: str
    output_path: str
    rank: int
    world_size: int


def main(config: ExportConfig):
    old_state_dict = torch.load(config.input_path)["data_loader"]

    if config.type_model == "llama2":
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", use_fast=True)
    elif config.type_model == "llama3":
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B", use_fast=True)
    else:
        raise ValueError(f"Model type {config.type_model} not supported")

    dl = get_dataloader(
        tokenizer=tokenizer,
        world_size=config.world_size,
        rank=config.rank,
        batch_size=config.train.micro_bs,
        data_config=config.data,
    )

    iter_dl = iter(dl)

    # Needed to init the states because they are lazy
    while True:
        try:
            _ = next(iter_dl)
            new_state_dict = dl.state_dict()
            transfer_states(old_state_dict, new_state_dict)
            break
        except KeyError:
            print("Not inited, sampling again")
            pass

    print(f"Saving to {config.output_path}")
    torch.save({"data_loader": new_state_dict}, config.output_path)

    del dl


def test_dl(config: ExportConfig):
    if config.type_model == "llama2":
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", use_fast=True)
    elif config.type_model == "llama3":
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B", use_fast=True)
    else:
        raise ValueError(f"Model type {config.type_model} not supported")

    dl = get_dataloader(
        tokenizer=tokenizer,
        world_size=config.world_size,
        rank=config.rank,
        batch_size=config.train.micro_bs,
        data_config=config.data,
    )
    dl.load_state_dict(torch.load(config.output_path, weights_only=True)["data_loader"])

    iter_dl = iter(dl)

    # Needed to init the states because they are lazy
    for i in range(10):
        batch = next(iter_dl)
        print(batch.keys(), batch["input_ids"].shape)


if __name__ == "__main__":
    logger = get_logger()
    config = ExportConfig(**parse_argv())
    resolve_env_vars(config)
    logger.debug(f"config: {config.model_dump()}")

    main(config)
    test_dl(config)
