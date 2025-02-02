import copy
import torch
from zeroband.data import InterleaveDataset, ParquetDataset, SequencePackingDataSet, collate_fn
from torch.utils.data import DataLoader
from zeroband.data import load_all_datasets, DataConfig
from zeroband.utils.logger import get_logger
from collections import Counter
from itertools import chain
import pytest
import logging
import pyarrow as pa
import pyarrow.parquet as pq
from faker import Faker
from typing import List
import string
from torchdata.stateful_dataloader import StatefulDataLoader


@pytest.mark.skip(reason="not using hf for now")
@pytest.mark.parametrize(
    "ratio, lower, upper",
    [
        ("3:2", 1.2821, 1.7549),
        ("0.5:1", 0.4247, 0.5886),
    ],
)
def test_load_all_datasets_vanilla(ratio: str, lower: float, upper: float):
    config = DataConfig(
        dataset_name_or_paths="Jackmin108/abc-testing:A,Jackmin108/abc-testing:C",
        dataset_ratio=ratio,
        streaming=True,
        fake=False,
    )

    ds = load_all_datasets(config, "train")
    print(ds)

    dl = DataLoader(ds, batch_size=256)
    batches = [i["text"] for i, _ in zip(dl, range(10))]
    assert len(batches) == 10

    # Check that the ratio is correct
    letter_count = Counter(i[0] for i in chain(*batches))
    print(letter_count, letter_count["A"] / letter_count["C"])
    assert letter_count["A"] / letter_count["C"] < upper
    assert letter_count["A"] / letter_count["C"] > lower


@pytest.mark.skip(reason="not using hf for now")
@pytest.mark.parametrize(
    "ratio, lower, upper, data_rank, data_world_size",
    [
        ("3:2", 1.2821, 1.7549, 1, 4),
        ("0.5:1", 0.4247, 0.5886, 0, 3),
    ],
)
def test_load_all_datasets_data_rank(ratio: str, lower: float, upper: float, data_rank: int, data_world_size: int):
    get_logger().setLevel(logging.DEBUG)
    config = DataConfig(
        dataset_name_or_paths="Jackmin108/abc-testing:A,Jackmin108/abc-testing:C",
        dataset_ratio=ratio,
        streaming=True,
        fake=False,
        data_world_size=data_world_size,
        data_rank=data_rank,
    )

    ds = load_all_datasets(config, "train")
    print(ds)

    dl = DataLoader(ds, batch_size=256)
    batches = [i["text"] for i, _ in zip(dl, range(10))]
    assert len(batches) == 10

    # Check that the ratio is correct
    letter_count = Counter(i[0] for i in chain(*batches))
    print(letter_count, letter_count["A"] / letter_count["C"])
    assert letter_count["A"] / letter_count["C"] < upper
    assert letter_count["A"] / letter_count["C"] > lower

    c_num_set = {int(i[1:]) for i in chain(*batches) if i[0] == "C"}
    a_num_set = {int(i[1:]) for i in chain(*batches) if i[0] == "A"}

    # Check that the data is correctly sharded
    first_a_shard = set(range(data_rank * (2**12), (data_rank + 1) * (2**12)))
    first_10_c_shard = set()
    for i in range(data_rank, data_world_size * 10, data_world_size):
        first_10_c_shard = first_10_c_shard.union(set(range(i * (2**8), (i + 1) * (2**8))))
    assert all(i in first_a_shard for i in a_num_set)
    assert all(i in first_10_c_shard for i in c_num_set)


def test_squence_packing():
    class FakeDataset(torch.utils.data.Dataset):
        def __init__(self):
            self.data = [[6, 1, 2, 3, 4], [6, 3, 3, 4, 1, 7], [3, 2], [1, 2], [1, 4, 5, 3, 4, 1, 7, 8]]

        def __len__(self):
            return len(self.data)

        def __getitem__(self, index):
            return {"input_ids": self.data[index]}

    MAX_SEQ_LEN = 8
    dataset = SequencePackingDataSet(FakeDataset(), max_seq_length=MAX_SEQ_LEN, eos_token=0)

    input_ids = []
    labels = []
    for data in dataset:
        assert data["input_ids"].shape[0] == MAX_SEQ_LEN
        assert data["labels"].shape[0] == MAX_SEQ_LEN
        assert sum(data["seqlens"]) == MAX_SEQ_LEN

        input_ids.append(data["input_ids"].tolist())
        labels.append(data["labels"].tolist())

    assert input_ids == [[6, 1, 2, 3, 4, 6, 3, 3], [3, 2, 1, 2, 1, 4, 5, 3]]
    assert labels == [[1, 2, 3, 4, 0, 3, 3, 4], [2, 0, 2, 0, 4, 5, 3, 4]]


class SimpleTokenizer:
    def __init__(self):
        # Create vocabulary: a-z (0-25) and unknown token (26)
        self.char_to_id = {char: idx for idx, char in enumerate(string.ascii_lowercase)}
        self.unknown_token = 26

    def encode(self, text: str) -> List[int]:
        """Convert text to list of token ids"""
        return [self.char_to_id.get(char.lower(), self.unknown_token) for char in text]


@pytest.fixture
def fake_sentences():
    """Generate 500 fake sentences (100 per file * 5 files)"""
    fake = Faker()
    return [fake.sentence() for _ in range(10_000)]


@pytest.fixture
def parquet_files(tmp_path, fake_sentences):
    """Create 10 parquet files with 100 sentences each"""
    files = []
    for i in range(10):
        # Create data for this file
        start_idx = i * 100
        sentences = fake_sentences[start_idx : start_idx + 100]

        # Create arrow table
        table = pa.Table.from_arrays([pa.array(sentences)], names=["text"])

        # Write to parquet file
        file_path = tmp_path / f"data_{i}.parquet"
        pq.write_table(table, file_path)
        files.append(str(file_path))

    return files


@pytest.fixture
def tokenizer():
    """Get a simple character-based tokenizer"""
    return SimpleTokenizer()


def test_parquet_dataset_ckpt(parquet_files, tokenizer):
    # Create first dataset and iterate halfway
    dataset1 = ParquetDataset(parquet_files, tokenizer)
    halfway_point = 100

    for _, data in zip(range(halfway_point), dataset1):
        pass
    # Save state
    state_dict = dataset1.state_dict()

    # Create new dataset and load state
    dataset2 = ParquetDataset(parquet_files, tokenizer)
    dataset2.load_state_dict(state_dict)

    max_to_yield = 200
    # Continue first dataset

    for _, data1, data2 in zip(range(max_to_yield), dataset1, dataset2):
        assert data1["input_ids"] == data2["input_ids"]


def test_sequence_packing_dataset_ckpt(parquet_files, tokenizer):
    dataset1 = SequencePackingDataSet(ParquetDataset(parquet_files, tokenizer), max_seq_length=16, eos_token=0)

    halfway_point = 100

    for _, data in zip(range(halfway_point), dataset1):
        pass
    # Save state
    state_dict = dataset1.state_dict()

    # Create new dataset and load state
    dataset2 = SequencePackingDataSet(ParquetDataset(parquet_files, tokenizer), max_seq_length=16, eos_token=0)
    dataset2.load_state_dict(state_dict)

    assert dataset1.state_dict() == dataset2.state_dict()

    max_to_yield = 199
    # Continue first dataset

    for _, data1, data2 in zip(range(max_to_yield), dataset1, dataset2):
        assert (data1["input_ids"] == data2["input_ids"]).all()
        assert (data1["labels"] == data2["labels"]).all()
        assert data1["seqlens"] == data2["seqlens"]


def test_interleave_dataset_ckpt(parquet_files, tokenizer):
    # Split parquet files into two groups to create two datasets
    files1 = parquet_files[:2]  # First two files
    files2 = parquet_files[2:4]  # Next two files

    # Create first dataset and iterate halfway
    dataset1 = InterleaveDataset(
        [ParquetDataset(files1, tokenizer), ParquetDataset(files2, tokenizer)], probabilities=[0.5, 0.5]
    )

    halfway_point = 100

    for _, data in zip(range(halfway_point), dataset1):
        pass
    # Save state
    state_dict = dataset1.state_dict()

    # Create new dataset and load state
    dataset2 = InterleaveDataset(
        [ParquetDataset(files1, tokenizer), ParquetDataset(files2, tokenizer)], probabilities=[0.5, 0.5]
    )
    dataset2.load_state_dict(state_dict=copy.deepcopy(state_dict))

    assert dataset1.state_dict() == dataset2.state_dict()

    max_to_yield = 250

    for _, data1, data2 in zip(range(max_to_yield), dataset1, dataset2):
        assert data1["input_ids"] == data2["input_ids"]


@pytest.mark.skip(reason="not working for now")
@pytest.mark.parametrize("num_workers", [0, 2, 16])
def test_dataloader_parquet_dataset(parquet_files, tokenizer, num_workers):
    dataset = SequencePackingDataSet(ParquetDataset(parquet_files, tokenizer), max_seq_length=8, eos_token=0)

    loader = StatefulDataLoader(dataset, batch_size=8, num_workers=num_workers, collate_fn=collate_fn)

    total_samples = 100

    for _, _batch in zip(range(total_samples), loader):
        ...

    # Save state
    state_dict = loader.state_dict()

    # Create new loader and load state
    dataset2 = SequencePackingDataSet(ParquetDataset(parquet_files, tokenizer), max_seq_length=8, eos_token=0)

    loader2 = StatefulDataLoader(dataset2, batch_size=8, num_workers=num_workers, collate_fn=collate_fn)

    print(state_dict)

    loader2.load_state_dict(state_dict)

    warmup = 10

    for i, batch1, batch2 in zip(range(total_samples), loader, loader2):
        if i > warmup:
            assert (batch1["input_ids"] == batch2["input_ids"]).all()
            assert (batch1["labels"] == batch2["labels"]).all()
            assert (batch1["seqlens"] == batch2["seqlens"]).all()
