import random
from typing import Any, Generator, Optional, List, Dict, TypedDict, Union
from pydantic_config import BaseConfig
from zeroband.utils.logging import get_logger

import torch
from torch.utils.data import DataLoader
from torch.utils.data import IterableDataset, Dataset
from torchdata.stateful_dataloader import StatefulDataLoader
from torch.distributed.checkpoint.stateful import Stateful

from datasets import load_dataset, interleave_datasets, load_dataset_builder, BuilderConfig
from datasets.distributed import split_dataset_by_node
import functools

TEST_VOCAB_SIZE = 1024

# TODO sami: make sure the init of the model is the same on all rank

logger = get_logger(__name__)


class DataConfig(BaseConfig):
    dataset_name_or_paths: str = "allenai/c4:en"
    val_dataset_name_or_paths: Optional[str] = None
    seq_length: int = 1024
    fake: bool = False
    num_workers: int = 4
    streaming: bool = True
    max_train_samples: Optional[int] = None
    max_eval_samples: Optional[int] = None
    dataset_ratio: Optional[str] = None
    data_rank: Optional[int] = None
    data_world_size: Optional[int] = None
    reverse_data_files: bool = False


class FakeTokenizedDataset(IterableDataset):
    """This is a dummy dataset that generates random sequences of length seq_len and vocab_size"""

    def __init__(self, seq_len: int, vocab_size: int):
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        assert vocab_size > 3, "Vocab size must be greater than 3"

    def __iter__(self) -> Generator[dict[str, Any], Any, None]:
        while True:
            len_ = random.randint(1, self.seq_len)
            input_ids = torch.randint(3, self.vocab_size, (len_,)).tolist()
            yield {"input_ids": input_ids}

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass


class BatchOutput(TypedDict):
    input_ids: torch.IntTensor
    labels: torch.IntTensor
    seqlens: list[int]


class SequencePackingDataSet(IterableDataset, Stateful):
    """
    This class wrap a dataset and wrap it into an iterable that return sequence of max_seq_length
    packed
    """

    def __init__(self, dataset: Dataset, max_seq_length: int, eos_token: int):
        self.dataset = dataset
        self.max_seq_length = max_seq_length
        self.eos_token = eos_token

    def __iter__(self) -> Generator[BatchOutput, Any, None]:
        inputs_ids = []
        labels = []
        seqlens = []

        for og_sample in self.dataset:
            og_sample: list[int] = og_sample["input_ids"]

            og_sample = og_sample + [self.eos_token]
            sample_inputs_ids = og_sample[:-1]
            sample_labels = og_sample[1:]

            token_remaining = self.max_seq_length - len(inputs_ids)

            if len(sample_inputs_ids) < token_remaining:
                inputs_ids.extend(sample_inputs_ids)
                labels.extend(sample_labels)
                seqlens.append(len(sample_inputs_ids))

            else:
                inputs_ids.extend(sample_inputs_ids[:token_remaining])
                labels.extend(sample_labels[:token_remaining])
                seqlens.append(token_remaining)

                yield {
                    "input_ids": torch.Tensor(inputs_ids).to(dtype=torch.long),
                    "labels": torch.Tensor(labels).to(dtype=torch.long),
                    "seqlens": seqlens,
                }
                inputs_ids = []
                labels = []
                seqlens = []

    def state_dict(self):
        return self.dataset.state_dict()

    def load_state_dict(self, state_dict):
        self.dataset.load_state_dict(state_dict)


def collate_fn(samples: list[dict[str, torch.LongTensor]]) -> dict[str, torch.LongTensor]:
    assert samples[0].keys() == {"input_ids", "labels", "seqlens"}

    inputs_ids = []
    labels = []
    seqlens = []

    for sample in samples:
        inputs_ids.append(sample["input_ids"])
        labels.append(sample["labels"])

        seqlens.extend(sample["seqlens"])

    return {
        "input_ids": torch.stack(inputs_ids, dim=0),
        "labels": torch.stack(labels, dim=0),
        "seqlens": torch.Tensor(seqlens).long(),
    }


def get_dataloader(
    tokenizer,
    world_size: int,
    rank: int,
    batch_size: int,
    data_config: DataConfig,
) -> DataLoader:
    if data_config.fake:
        train_dataset = FakeTokenizedDataset(data_config.seq_length, TEST_VOCAB_SIZE)
    else:
        ds = load_all_datasets(data_config=data_config, split="train")

        def tokenize_function(data):
            outputs = tokenizer(data["text"], truncation=True, max_length=data_config.seq_length)
            return outputs

        tokenized_datasets = ds.map(tokenize_function, batched=True, remove_columns=["text", "attention_mask"])
        train_dataset = split_dataset_by_node(tokenized_datasets, world_size=world_size, rank=rank)

    dataset = SequencePackingDataSet(train_dataset, data_config.seq_length, eos_token=tokenizer.eos_token_id)

    return StatefulDataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=data_config.num_workers,
    )


@functools.lru_cache(maxsize=None)
def _get_ds_config_dict(path: str, name: Optional[str] = None) -> Dict[str, BuilderConfig]:
    ds_builder = load_dataset_builder(path=path, name=name)
    return ds_builder.builder_configs


def _get_datafiles(path: str, name: Optional[str] = None, split: str = "train") -> List[str]:
    builder_config = _get_ds_config_dict(path=path, name=name)
    if name is None or len(name) == 0:
        if "default" not in builder_config:
            logger.warning(f"Default config not found for {path}. Using first config.")
            name = next(iter(builder_config.keys()))
        else:
            name = "default"
    return builder_config[name].data_files[split]


def _nice_print(kwargs: Dict[str, Union[str, List[str]]]) -> str:
    def _foo(a):
        if isinstance(a, list):
            return str(a[:5]) + "..." + str(a[-5:]) if len(a) > 10 else str(a)
        return str(a)

    return str({k: _foo(v) for k, v in kwargs.items()})


def _load_datasets(
    dataset_names: str,
    split: str,
    data_rank: Optional[int] = None,
    data_world_size: Optional[int] = None,
    streaming: bool = True,
    probabilities: Optional[List[float]] = None,
    reverse_data_files: bool = False,
) -> Dataset:
    logger.debug(dataset_names)
    ds_args = []
    for _ds in dataset_names.split(","):
        _ds_name, _, _ds_config = _ds.partition(":")
        _ds_args = {"path": _ds_name}
        if _ds_config:
            _ds_args["name"] = _ds_config
        _data_files = _get_datafiles(_ds_name, _ds_config, split)
        if reverse_data_files:
            _data_files = _data_files[::-1]
            _ds_args["data_files"] = _data_files
        if data_rank is not None and data_world_size is not None:
            _ds_args["data_files"] = _data_files[data_rank::data_world_size]
        ds_args.append(_ds_args)

    # logger.debug(f"Datasets ({split}):\n" + "\n".join(map(_nice_print, ds_args)))
    # logger.debug(f"Probabilities: {probabilities}")
    logger.debug(f"Loading datasets{' in streaming mode' if streaming else ''}")
    datasets = []
    for ds_arg in ds_args:
        # logger.debug(f"Loading dataset: {ds_arg}")
        _ds = load_dataset(**ds_arg, split=split, streaming=streaming)
        _ds = _ds.remove_columns([i for i in _ds.column_names if i not in ["text"]])
        datasets.append(_ds)
        # logger.debug(f"Loaded dataset: {ds_arg}")

    ds = interleave_datasets(datasets=datasets, probabilities=probabilities, stopping_strategy="all_exhausted")
    logger.info(f"Loaded datasets ({split})")
    return ds


def _get_probabilities(data_config: DataConfig) -> Optional[List[float]]:
    if data_config.dataset_ratio is None:
        return None
    if len(data_config.dataset_name_or_paths.split(",")) != len(data_config.dataset_ratio.split(":")):
        raise ValueError("Number of datasets and dataset ratios must be the same")
    nums = [float(i) for i in data_config.dataset_ratio.split(":")]
    denom = sum(nums)
    return [i / denom for i in nums]


def load_all_datasets(data_config: DataConfig, split: str, max_samples: Optional[int] = None) -> IterableDataset:
    """Load all datasets and interleave them"""
    if max_samples is not None and not data_config.streaming:
        split = f"{split}[:{max_samples}]"
    ds = _load_datasets(
        dataset_names=data_config.dataset_name_or_paths,
        split=split,
        data_rank=data_config.data_rank,
        data_world_size=data_config.data_world_size,
        streaming=data_config.streaming,
        probabilities=_get_probabilities(data_config),
        reverse_data_files=data_config.reverse_data_files,
    )
    if max_samples is not None and data_config.streaming:
        if data_config.max_train_samples is not None:
            ds = ds.take(data_config.max_train_samples)
    logger.info(f"Train dataset:\n{ds}")

    return ds
