import os.path
import random
from abc import ABC
from dataclasses import dataclass, asdict
from typing import Any, Generator, List, Dict

import torch
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizer, AutoTokenizer

from zeroband.config import DataConfig
from zeroband.utils import rand_utils, nibble_utils, math_utils
from zeroband.utils.tokenizer_utils import TokenizerInfo

from pyarrow import parquet as pq, Table

DEBUG_VOCAB_SIZE = 1024


class StatefulDataset(IterableDataset, Stateful, ABC):
    ...


class FakeTokenizedDataset(StatefulDataset):
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


class NibbleDataset(StatefulDataset):

    def __init__(self, nibble_file: str, seq_len: int, tokenizer_info: TokenizerInfo, n_bits: int, seed: int):
        if not os.path.exists(nibble_file):
            raise ValueError("Supplied Nibble-file does not exist!")

        self.nibble_file = nibble_file
        self.file_size = os.path.getsize(nibble_file)
        self.seq_len = seq_len
        self.tokenizer_info = tokenizer_info
        self.vocab_size = tokenizer_info.vocab_size

        if self.vocab_size > 2 ** n_bits:
            raise ValueError(
                f"Cannot represent vocab_size with supplied number of bits! vocab_size: {tokenizer_info.vocab_size}, n_bits: {n_bits}, 2**n_bits: {2 ** n_bits}")

        self.n_bits = n_bits

        self.lsr_seed = seed

    def __iter__(self) -> Generator[dict[str, Any], Any, None]:
        with open(self.nibble_file, "rb") as f:
            while True:
                seek_pos, self.lsr_seed = rand_utils.lsfr_rand_u64(self.lsr_seed, hi=self.file_size)

                # floor to nearest nibble start
                num_nibbles = (seek_pos * 8) // self.n_bits
                nibble_start_bit = num_nibbles * self.n_bits

                # find the next bit where a nibble starts that aligns with a byte boundary
                no_carry_start_bit = math_utils.next_common_multiple(self.n_bits, 8, nibble_start_bit)

                chunk_start_pos = no_carry_start_bit // 8

                # compute num bytes to read to be (seq_len + 1) * n_bits ceil-ed to next byte
                chunk_size = ((self.seq_len + 1) * self.n_bits + 7) // 8

                f.seek(chunk_start_pos)
                data = f.read(chunk_size)

                tokens, _, _ = nibble_utils.read_nibbles(data, self.n_bits)
                tokens = torch.tensor(tokens, dtype=torch.int64, device='cpu')

                input_ids = tokens[:-1]
                labels = tokens[1:]

                # create document lengths from where eot tokens are placed inside chunk
                document_lengths = []
                cur_len = 0
                for t in input_ids:
                    cur_len += 1
                    if t == self.tokenizer_info.eot_token:
                        document_lengths.append(cur_len)
                        cur_len = 0
                document_lengths.append(cur_len)

                document_lengths = torch.tensor(document_lengths, dtype=torch.int64, device='cpu')
                yield {'input_ids': input_ids, 'labels': labels, 'seqlens': document_lengths}

    def state_dict(self):
        return {"lsfr_seed": self.lsr_seed}

    def load_state_dict(self, state_dict):
        self.lsr_seed = state_dict["lsfr_seed"]


@dataclass
class InterleaveDatasetState:
    current_index: int
    seed: int


class InterleaveDataset(StatefulDataset):
    """This class take a list of datasets and interleave them. It is stateful and can be used with pytorch dataloader.

    It draw a sample from each dataset with a probability given by the probabilities list.

    The state can be saved and restored. Under the hood we just fast forward the random generator to the current position.
    """

    def __init__(self, datasets: List[StatefulDataset], probabilities: List[float], seed: int = 42):
        assert len(datasets) > 0, "At least one dataset is required"
        assert len(datasets) == len(probabilities), "The number of datasets and probabilities must be the same"

        self.probabilities = []
        self.datasets = []

        for dataset, prob in zip(datasets, probabilities):
            self.datasets.append(dataset)
            self.probabilities.append(prob)

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


@dataclass
class PQDatasetState:
    files: List[str]
    lsfr_seed: int


class ParquetDataset(StatefulDataset):

    def __init__(self, files: List[str], seq_len: int, seed: int, tokenizer: PreTrainedTokenizer):
        self.arg_files = files
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        self.state = PQDatasetState(
            files=files,
            lsfr_seed=seed
        )
        self.parquet_tables: Dict[str, Table] = {}

    def __iter__(self):
        while True:
            # choose random file
            file_idx, self.state.lsfr_seed = rand_utils.lsfr_rand_u64(self.state.lsfr_seed, len(self.state.files))
            parquet_table = self._lazy_get_table(file_idx)

            tokens = []
            document_lengths = []
            while True:
                # choose random row
                row_idx, self.state.lsfr_seed = rand_utils.lsfr_rand_u64(self.state.lsfr_seed, len(parquet_table))
                row = parquet_table[row_idx]

                new_tokens = self.tokenizer.encode(str(row), add_special_tokens=True)

                target_num_tokens = self.seq_len + 1

                new_length = len(tokens) + len(new_tokens)
                if new_length > target_num_tokens:
                    over_shoot = new_length - target_num_tokens
                    new_tokens = new_tokens[:-over_shoot]

                # Track document length
                # Note: target_num_tokens is seq_len + 1 because we obtain one block that we extract
                # both input and labels from. however, seqlens declares document lengths within only the inputs.
                # hence, we need to account for the discrepancy in the last document, which becomes one token shorter.
                if new_length >= target_num_tokens:
                    document_lengths.append(len(new_tokens) - 1)
                else:
                    document_lengths.append(len(new_tokens))

                tokens.extend(new_tokens)

                if len(tokens) >= target_num_tokens:
                    break

            tokens = torch.tensor(tokens, dtype=torch.int64, device='cpu')

            input_ids = tokens[:-1]
            labels = tokens[1:]

            document_lengths = torch.tensor(document_lengths, dtype=torch.int64, device='cpu')
            yield {'input_ids': input_ids, 'labels': labels, 'seqlens': document_lengths}

    @property
    def is_empty(self):
        return len(self.arg_files) == 0

    def state_dict(self) -> dict[str, Any]:
        return asdict(self.state) if self.state is not None else {}

    def load_state_dict(self, state_dict):
        self.state = PQDatasetState(**state_dict)

    def _lazy_get_table(self, file_idx: int) -> Table:
        file_path = self.state.files[file_idx]
        parquet_table = self.parquet_tables.get(file_path, None)
        if parquet_table is None:
            parquet_table = pq.ParquetFile(file_path)
            parquet_table = parquet_table.read()['text']
            self.parquet_tables[file_path] = parquet_table
        return parquet_table


def collate_fn(samples: list[dict[str, torch.LongTensor]]) -> dict[str, torch.LongTensor | list[torch.LongTensor]]:
    assert samples[0].keys() == {"input_ids", "labels", "seqlens"}

    inputs_ids = []
    labels = []
    sequence_lengths = []

    doc_id = 0
    for sample in samples:
        inputs_ids.append(sample["input_ids"])
        labels.append(sample["labels"])
        sequence_lengths.append(sample["seqlens"])
        doc_id += 1

    return {
        "input_ids": torch.stack(inputs_ids, dim=0),
        "labels": torch.stack(labels, dim=0),
        "seqlens": sequence_lengths,
    }


def make_mixed_nibble_dataset(data_config: DataConfig, tokenizer_info: TokenizerInfo) -> StatefulDataset:
    dataset_paths = data_config.dataset_name_or_paths.split(',')
    probabilities = [int(ratio) / 100 for ratio in data_config.dataset_ratio.split(':')]

    for dataset_path in dataset_paths:
        if not os.path.exists(dataset_path):
            raise ValueError(f"Dataset path {dataset_path} does not exist")
        if not dataset_path.endswith('.bin'):
            raise ValueError(f"Cannot mix nibble- with non-nibble dataset files!")

    rand = random.Random()

    # iterator seed *must* be random to avoid different peers training on same data, killing the point of DDP
    # There is no "rank" in PCCL, and even if there was it would still not be a safe seed.
    # There are internal UUIDs, but they are not exposed for now.
    # Random is fine for now.
    iterator_seed = rand.randint(0, 2 ** 31 - 1)

    return InterleaveDataset(
        [NibbleDataset(dataset_path, data_config.seq_length, tokenizer_info, data_config.token_bit_size,
                       iterator_seed) for dataset_path in dataset_paths],
        probabilities
    )

def get_parquet_files(dataset_path: str) -> List[str]:
    files = os.listdir(dataset_path)
    parquet_files = []
    for file in files:
        if file.endswith('.parquet'):
            parquet_files.append(os.path.join(dataset_path, file))
    return parquet_files

def get_hf_tokenizer(tokenizer_info: TokenizerInfo) -> PreTrainedTokenizer:
    return AutoTokenizer.from_pretrained(tokenizer_info.hf_name, use_fast=True)

def make_mixed_parquet_dataset(data_config: DataConfig, tokenizer_info: TokenizerInfo) -> StatefulDataset:
    dataset_paths = data_config.dataset_name_or_paths.split(',')
    probabilities = [int(ratio) / 100 for ratio in data_config.dataset_ratio.split(':')]

    for dataset_path in dataset_paths:
        if not os.path.exists(dataset_path):
            raise ValueError(f"Dataset path {dataset_path} does not exist")
        if not os.path.isdir(dataset_path):
            raise ValueError(f"Dataset path {dataset_path} must be a directory containing .parquet files")
        if dataset_path.endswith('.bin'):
            raise ValueError(f"Cannot mix nibble- with non-nibble dataset files!")

    rand = random.Random()

    # iterator seed *must* be random to avoid different peers training on same data, killing the point of DDP
    # There is no "rank" in PCCL, and even if there was it would still not be a safe seed.
    # There are internal UUIDs, but they are not exposed for now.
    # Random is fine for now.
    iterator_seed = rand.randint(0, 2 ** 31 - 1)

    return InterleaveDataset(
        [ParquetDataset(get_parquet_files(dataset_path), data_config.seq_length, iterator_seed, get_hf_tokenizer(tokenizer_info)) for dataset_path in dataset_paths],
        probabilities
    )


def make_dataloader(
        tokenizer_info: TokenizerInfo,
        mpi_world_size: int,
        mpi_rank: int,
        batch_size: int,
        data_config: DataConfig,
) -> StatefulDataLoader:
    if data_config.fake:
        train_dataset = FakeTokenizedDataset(data_config.seq_length, DEBUG_VOCAB_SIZE)
    else:
        is_nibble_file = any([path.endswith('.bin') for path in data_config.dataset_name_or_paths.split(',')])
        if is_nibble_file:
            train_dataset = make_mixed_nibble_dataset(data_config, tokenizer_info)
        else:
            train_dataset = make_mixed_parquet_dataset(data_config, tokenizer_info)

    return StatefulDataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=data_config.num_workers,
    )
