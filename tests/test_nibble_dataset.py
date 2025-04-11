from transformers import AutoTokenizer

from zeroband.data import NibbleDataset
from zeroband.utils.tokenizer_utils import TokenizerInfo


def test_nibble_dataset():
    seq_len = 1024
    ds = NibbleDataset("test_data/val_1.bin", seq_len, TokenizerInfo(vocab_size=128255, eot_token=128001, bot_token=0), 17, 42)

    ds_it = iter(ds)

    for i in range(4):
        data = next(ds_it)
        input_ids = data['input_ids']
        label = data['labels']

        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B", use_fast=True)

        print(input_ids)
        text = tokenizer.decode(input_ids, skip_special_tokens=False)
        print(text)
        assert len(input_ids) == seq_len
        assert len(label) == seq_len