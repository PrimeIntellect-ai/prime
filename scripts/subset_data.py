#!/usr/bin/env python
# coding: utf-8
# Usage:
# python scripts/subset_data.py --dataset_name PrimeIntellect/fineweb-edu --data_world_size 12 --data_rank 1

import argparse
import subprocess
from typing import Dict, List, Optional
import functools
from datasets import load_dataset_builder, BuilderConfig
import logging
from huggingface_hub import get_token
import os
import multiprocessing as mp
from tqdm import tqdm

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)


@functools.lru_cache(maxsize=None)
def _get_ds_config_dict(path: str, name: Optional[str] = None) -> Dict[str, BuilderConfig]:
    ds_builder = load_dataset_builder(path=path, name=name)
    return ds_builder.builder_configs


def _get_datafiles(path: str, name: Optional[str] = None, split: str = "train") -> List[str]:
    builder_config = _get_ds_config_dict(path=path, name=name)
    if name is None:
        if "default" not in builder_config:
            name = next(iter(builder_config.keys()))
        else:
            name = "default"
    return builder_config[name].data_files[split]


def _download_file(data_file: str, save_path: str) -> None:
    """Download a file from huggingface.co

    Args:
        data_file (str): The file to download. e.g. 'hf://datasets/PrimeIntellect/fineweb-edu@14efaa24d7dff8a745bf4918e415878546542346/data1/train-00450.parquet'
        save_path (str): The path to save the file. e.g. 'data1/train-00450.parquet'
    """
    assert data_file.startswith("hf://")
    data_file = data_file.replace("hf://", "").replace("@", "/resolve/")

    if "/" in save_path:
        parent = "/".join(save_path.split("/")[:-1])
        if not os.path.exists(parent):
            logger.debug(f"Creating directory: {parent}")
            os.makedirs(parent, exist_ok=True)

    cmd = [
        "wget",
        f'--header="Authorization: Bearer {get_token()}"',
        f"https://huggingface.co/{data_file}?download=true",
        f"-O {save_path}",
    ]
    result = subprocess.run(" ".join(cmd), shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if result.returncode != 0:
        logger.error(f"Error downloading file: {data_file}")
        logger.error(result.stderr.decode("utf-8"))


def _download_file_wrapper(args):
    return _download_file(*args)


def _get_save_path(data_file: str) -> str:
    ret_list = data_file.split("@")[-1].split("/")[1:]
    return args.dataset_name.split("/")[-1] + "/" + "/".join(ret_list)


def main(args):
    g_data_files = _get_datafiles(args.dataset_name)
    logger.debug(f"Length of data_files: {len(g_data_files)}")
    if len(args.filter) > 0:
        args.filter = args.filter.split(",")
        data_files = []
        for _filter in args.filter:
            data_files.extend([f for f in g_data_files if _filter in f])
    else:
        data_files = g_data_files

    logger.debug(f"Length of data_files: {len(data_files)}")
    data_files = data_files[args.data_rank :: args.data_world_size][: args.max_shards]
    logger.debug(f"Data files: {data_files}")
    logger.debug(f"Length of data_files processing: {len(data_files)}")

    if args.dry_run:
        return

    with mp.Pool(args.num_workers) as pool:
        save_paths = list(pool.imap(_get_save_path, tqdm(data_files, desc="Getting save paths")))
        _ = list(
            tqdm(
                pool.imap(_download_file_wrapper, zip(data_files, save_paths)),
                desc="Downloading files",
                total=len(data_files),
                bar_format="{l_bar}{bar:10}{r_bar}",
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and process data from a HF dataset")
    parser.add_argument("--dataset_name", type=str, default="PrimeIntellect/fineweb-edu", help="dataset name")
    parser.add_argument("--dry_run", action="store_true", help="do not download data")
    parser.add_argument("--filter", type=str, default="", help="search shards by the filter")
    parser.add_argument("--data_rank", type=int, default=0, help="start index")
    parser.add_argument("--data_world_size", type=int, default=4, help="world size")
    parser.add_argument("--max_shards", type=int, default=1000)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()
    main(args)
