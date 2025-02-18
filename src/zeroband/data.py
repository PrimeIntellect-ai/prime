from dataclasses import dataclass, asdict
import random
from typing import Any, Generator, Optional, List, Dict, TypedDict, Union
import functools
import threading

from zeroband.models.llama.model import create_block_mask_from_seqlens
from zeroband.utils.logger import get_logger
from zeroband.utils.world_info import get_world_info
from zeroband.config import DataConfig

import torch
from torch.utils.data import IterableDataset, Dataset
from torchdata.stateful_dataloader import StatefulDataLoader
from torch.distributed.checkpoint.stateful import Stateful

from datasets import load_dataset_builder, BuilderConfig
from pyarrow import parquet as pq
from transformers import PreTrainedTokenizer


TEST_VOCAB_SIZE = 1024


class FakeTokenizedDataset(IterableDataset):
    """This is a dummy dataset that generates random sequences of length seq_len and vocab_size"""

    def __init__(self, seq_len: int, vocab_size: int):
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        assert vocab_size > 3, "Vocab size must be greater than 3"
        self.step = 0

    def __iter__(self) -> Generator[dict[str, Any], Any, None]:
        while True:
            len_ = random.randint(1, self.seq_len)
            input_ids = torch.randint(3, self.vocab_size, (len_,)).tolist()
            self.step += 1
            yield {"input_ids": input_ids}

    def state_dict(self):
        return {"step": self.step}

    def load_state_dict(self, state_dict):
        self.step = state_dict["step"]
        itera = iter(self)
        for _ in range(self.step):
            next(itera)


class BatchOutput(TypedDict):
    input_ids: torch.IntTensor
    labels: torch.IntTensor
    seqlens: list[int]


@dataclass
class SequencePackingDataSetState:
    inputs_ids: list[int]
    labels: list[int]
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

        self.state = SequencePackingDataSetState(inputs_ids=[], labels=[], seqlens=[])

    def __iter__(self) -> Generator[BatchOutput, Any, None]:
        for og_sample in self.dataset:
            og_sample: list[int] = og_sample["input_ids"]

            og_sample = og_sample + [self.eos_token]
            sample_inputs_ids = og_sample[:-1]
            sample_labels = og_sample[1:]

            token_remaining = self.max_seq_length - len(self.state.inputs_ids)

            if len(sample_inputs_ids) < token_remaining:
                self.state.inputs_ids.extend(sample_inputs_ids)
                self.state.labels.extend(sample_labels)
                self.state.seqlens.append(len(sample_inputs_ids))

            else:
                self.state.inputs_ids.extend(sample_inputs_ids[:token_remaining])
                self.state.labels.extend(sample_labels[:token_remaining])
                self.state.seqlens.append(token_remaining)

                data = {
                    "input_ids": torch.Tensor(self.state.inputs_ids).to(dtype=torch.long),
                    "labels": torch.Tensor(self.state.labels).to(dtype=torch.long),
                    "seqlens": self.state.seqlens,
                }
                self.state.inputs_ids = []
                self.state.labels = []
                self.state.seqlens = []

                yield data

    def state_dict(self):
        return {"dataset": self.dataset.state_dict(), "state": asdict(self.state)}

    def load_state_dict(self, state_dict):
        self.dataset.load_state_dict(state_dict["dataset"])
        self.state = SequencePackingDataSetState(**state_dict["state"])


def collate_fn(samples: list[dict[str, torch.LongTensor]]) -> dict[str, torch.LongTensor | list[torch.LongTensor]]:
    assert samples[0].keys() == {"input_ids", "labels", "seqlens"}

    inputs_ids = []
    labels = []
    seqlens = []

    for sample in samples:
        inputs_ids.append(sample["input_ids"])
        labels.append(sample["labels"])

        seqlens.append(torch.Tensor(sample["seqlens"]).long())

    return {
        "input_ids": torch.stack(inputs_ids, dim=0),
        "labels": torch.stack(labels, dim=0),
        "seqlens": seqlens,
    }


@dataclass
class PQDatasetState:
    files: List[str]
    file_index: int
    row_index: int
    increment: int
    init_row_index: int


class ParquetDataset(IterableDataset, Stateful):
    """
    this class is a wrapper around a parquet dataset compatible with datasets and statefull compatible. The dataset is infinite and will restart from the last state if the iterator is exhausted.
    TODO:
    * [ ] handle mutli proc dataloader pytorch
    """

    def __init__(self, files: List[str], tokenizer: PreTrainedTokenizer):
        self.arg_files = files
        self.tokenizer = tokenizer

        self.state = None

    def _lazy_init(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            if worker_info.num_workers > len(self.arg_files):
                get_logger().warning(
                    f"dataloader rank {worker_info.id} Number of workers {worker_info.num_workers} is greater than the number of files {len(self.arg_files)}"
                )
                self.state = PQDatasetState(
                    files=self.arg_files,
                    file_index=0,
                    row_index=worker_info.id,
                    increment=worker_info.num_workers,
                    init_row_index=worker_info.id,
                )
                return

            files = self.arg_files[worker_info.id :: worker_info.num_workers]
        else:
            files = self.arg_files

        self.state = PQDatasetState(files=files, file_index=0, row_index=0, increment=1, init_row_index=0)

    def __iter__(self):
        # we lazy init the parquet dataset to get the worker info from dataloader multi process
        if self.state is None:
            self._lazy_init()

        while True:
            file = self.state.files[self.state.file_index]

            parquet_file = pq.ParquetFile(file)
            table = parquet_file.read()["text"]

            while True:
                row = table[self.state.row_index]

                self.state.row_index += self.state.increment
                if self.state.row_index >= len(table):
                    self.state.row_index = self.state.init_row_index
                    self.state.file_index += 1
                    if self.state.file_index >= len(self.state.files):  # infinite datasets
                        self.state.file_index = 0

                yield {"input_ids": self.tokenizer.encode(str(row))}

    @property
    def is_empty(self):
        return len(self.arg_files) == 0

    def state_dict(self) -> dict[str, Any]:
        return asdict(self.state) if self.state is not None else {}

    def load_state_dict(self, state_dict):
        self.state = PQDatasetState(**state_dict)


@dataclass
class InterleaveDatasetState:
    current_index: int
    seed: int


class InterleaveDataset(IterableDataset, Stateful):
    """This class take a list of datasets and interleave them. It is stateful and can be used with pytorch dataloader.

    It draw a sample from each dataset with a probability given by the probabilities list.

    The state can be saved and restored. Under the hood we just fast forward the random generator to the current position.
    """

    def __init__(self, datasets: List[ParquetDataset], probabilities: List[float], seed: int = 42):
        assert len(datasets) > 0, "At least one dataset is required"
        assert len(datasets) == len(probabilities), "The number of datasets and probabilities must be the same"

        self.probabilities = []
        self.datasets = []

        for dataset, prob in zip(datasets, probabilities):
            if not dataset.is_empty:
                self.datasets.append(dataset)
                self.probabilities.append(prob)
            else:
                get_logger().warning(f"Dataset {dataset} is empty. Skipping.")

        self.state = InterleaveDatasetState(current_index=0, seed=seed)
        self._init_random_state()

    def _init_random_state(self):
        """Initialize random generator and advance to current position"""
        self.random_generator = random.Random(self.state.seed)
        # Advance the RNG to the current position
        for _ in range(self.state.current_index):
            self._get_dataset_to_yield_from()

    def _get_dataset_to_yield_from(self) -> int:
        return self.random_generator.choices(range(len(self.datasets)), weights=self.probabilities, k=1)[0]

    def __iter__(self):
        data_iters = [iter(dataset) for dataset in self.datasets]
        while True:
            dataset_to_yield_from = self._get_dataset_to_yield_from()

            sample = next(data_iters[dataset_to_yield_from])
            self.state.current_index += 1

            yield sample

    def state_dict(self):
        state = {"interleave_state": asdict(self.state)}

        for i, dataset in enumerate(self.datasets):
            state[f"dataset_{i}"] = dataset.state_dict()
        return state

    def load_state_dict(self, state_dict):
        self.state = InterleaveDatasetState(**state_dict["interleave_state"])
        for i, dataset in enumerate(self.datasets):
            dataset.load_state_dict(state_dict[f"dataset_{i}"])
        self._init_random_state()


class PrefetchDataLoader(StatefulDataLoader):
    """
    This class is a wrapper around a dataloader that prefetches the next batch asynchronously on another thread.
    This is useful to hide the latency of transferring the batch to GPU.
    We can't integrate this into the StatefulDataloader's collate_fn() because it runs in another process.
    We're also using it to hide the latency of torch compiling FlexAttention block masks.
    """

    def __init__(self, original_dataloader: StatefulDataLoader, data_config: DataConfig):
        self.config = data_config
        self.original_dataloader = original_dataloader
        self._prefetch_iterator = None

    def __iter__(self):
        if self._prefetch_iterator is not None:
            return self._prefetch_iterator

        self._prefetch_iterator = self._PrefetchIterator(
            self.original_dataloader,
            self.config,
        )
        return self._prefetch_iterator

    def state_dict(self):
        if self._prefetch_iterator is not None:
            self._prefetch_iterator._await_prefetch()

        # Only keep around one or the other
        return {
            'dataloader_state': None if self._prefetch_iterator else self.original_dataloader.state_dict(),
            '_prefetch_iterator': None if self._prefetch_iterator is None else self._prefetch_iterator.state_dict(),
        }

    def load_state_dict(self, state_dict):
        if state_dict['dataloader_state'] is not None:
            self.original_dataloader.load_state_dict(state_dict['dataloader_state'])
        if state_dict['_prefetch_iterator'] is not None:
            self._prefetch_iterator = self._PrefetchIterator(self.original_dataloader, self.config)

    class _PrefetchIterator(Stateful):
        def __init__(self, original_dataloader: StatefulDataLoader, config: DataConfig):
            self.dataloader_iter = iter(original_dataloader)
            self.config = config
            self.ready_batch = None
            self.thread = None

            # Immediately transfer first batch async.
            self._prefetch_next()

        def state_dict(self) -> Dict[str, Any]:
            self._await_prefetch()
            return {
                'dataloader_iter': self.dataloader_iter.state_dict(),
                'ready_batch': self.ready_batch
            }

        def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
            self.dataloader_iter = state_dict['dataloader_iter']
            self.ready_batch = state_dict['ready_batch']

        def _prefetch_next(self):
            def _task() -> None:
                # NOTE: Each thread gets its own threadlocal CUDA context and has to reset the device.
                local_rank = get_world_info().local_rank
                torch.cuda.set_device(local_rank)

                # Grab batch or return sentinel
                try:
                    batch = next(self.dataloader_iter)
                except StopIteration:
                    self.ready_batch = StopIteration
                    return None

                # Transfer to CUDA asynchronously and create block mask in another cuda stream
                newstream = torch.cuda.Stream(local_rank)
                with torch.cuda.stream(newstream): #type: ignore (cuda stream is a cuda stream :) )
                    input_ids = batch["input_ids"].to("cuda", non_blocking=True)
                    labels = batch["labels"].to("cuda", non_blocking=True)

                    # Create block mask if needed
                    block_mask = None
                    if self.config.sequence_packing:
                        seqlens = batch.get("seqlens")
                        if seqlens is not None:
                            seqlens = [s.to("cuda", non_blocking=True) for s in seqlens]
                            block_mask = create_block_mask_from_seqlens(seqlens)

                # Construct and return processed batch.
                self.ready_batch = {
                    "input_ids": input_ids,
                    "labels": labels,
                    "block_mask": block_mask
                }
                return None

            self.thread = threading.Thread(target=_task)
            self.thread.start()

        def __next__(self):
            self._await_prefetch()
            if self.ready_batch is StopIteration:
                raise StopIteration

            batch = self.ready_batch
            self.ready_batch = None

            self._prefetch_next()
            return batch

        def _await_prefetch(self):
            if self.thread is not None and self.thread.is_alive():
                self.thread.join()
                self.thread = None

        def __del__(self):
            self._await_prefetch()

def get_dataloader(
    tokenizer,
    world_size: int,
    rank: int,
    batch_size: int,
    data_config: DataConfig,
) -> StatefulDataLoader:
    if data_config.fake:
        train_dataset = FakeTokenizedDataset(data_config.seq_length, TEST_VOCAB_SIZE)
    else:
        train_dataset = load_all_datasets(
            data_config=data_config, split="train", tokenizer=tokenizer, rank=rank, world_size=world_size
        )

    dataset = SequencePackingDataSet(train_dataset, data_config.seq_length, eos_token=tokenizer.eos_token_id)
    mp_batch_dataloader = StatefulDataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=data_config.num_workers,
    )
    return PrefetchDataLoader(mp_batch_dataloader, data_config)


@functools.lru_cache(maxsize=None)
def _get_ds_config_dict(path: str, name: Optional[str] = None) -> Dict[str, BuilderConfig]:
    ds_builder = load_dataset_builder(path=path, name=name)
    return ds_builder.builder_configs


def _get_datafiles(path: str, name: Optional[str] = None, split: str = "train") -> List[str]:
    builder_config = _get_ds_config_dict(path=path, name=name)
    if name is None or len(name) == 0:
        if "default" not in builder_config:
            get_logger().warning(f"Default config not found for {path}. Using first config.")
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
    tokenizer: PreTrainedTokenizer,
    data_rank: Optional[int] = None,
    data_world_size: Optional[int] = None,
    streaming: bool = True,
    probabilities: Optional[List[float]] = None,
    reverse_data_files: bool = False,
) -> InterleaveDataset:
    get_logger().debug(dataset_names)
    ds_args = []
    for _ds in dataset_names.split(","):
        _ds_name, _, _ds_config = _ds.partition(":")
        _ds_args: dict[str, Any] = {"path": _ds_name}
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
    get_logger().debug(f"Loading datasets{' in streaming mode' if streaming else ''}")
    datasets = []
    for ds_arg in ds_args:
        # logger.debug(f"Loading dataset: {ds_arg['data_files']}")
        _ds = ParquetDataset(files=ds_arg["data_files"], tokenizer=tokenizer)
        datasets.append(_ds)

    if len(datasets) > 1:
        ds = InterleaveDataset(datasets=datasets, probabilities=probabilities)
    else:
        ds = datasets[0]

    get_logger().info(f"Loaded datasets ({split})")
    return ds


def _get_probabilities(data_config: DataConfig) -> Optional[List[float]]:
    if data_config.dataset_ratio is None:
        return None
    if len(data_config.dataset_name_or_paths.split(",")) != len(data_config.dataset_ratio.split(":")):
        raise ValueError("Number of datasets and dataset ratios must be the same")
    nums = [float(i) for i in data_config.dataset_ratio.split(":")]
    denom = sum(nums)
    return [i / denom for i in nums]


def load_all_datasets(
    data_config: DataConfig,
    split: str,
    tokenizer: PreTrainedTokenizer,
    rank: int,
    world_size: int,
) -> InterleaveDataset:
    """Load all datasets and interleave them"""

    if data_config.split_by_data_rank and (
        data_config.data_rank is not None and data_config.data_world_size is not None
    ):
        split_rank = data_config.data_rank * world_size + rank
        split_world_size = data_config.data_world_size * world_size
    else:
        split_rank = rank
        split_world_size = world_size


    get_logger().info("Loading Train dataset(s)")

    ds = _load_datasets(
        dataset_names=data_config.dataset_name_or_paths,
        split=split,
        data_rank=split_rank,
        data_world_size=split_world_size,
        probabilities=_get_probabilities(data_config),
        reverse_data_files=data_config.reverse_data_files,
        tokenizer=tokenizer,
    )

    get_logger().info(f"Train dataset: {ds}")

    return ds
