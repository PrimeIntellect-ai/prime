from transformers import AutoTokenizer

from zeroband.config import Config
from zeroband.data import DEBUG_VOCAB_SIZE


class FakeTokenizer(object):
    def __init__(self):
        self.vocab_size = DEBUG_VOCAB_SIZE
        self.bos_token_id = 0
        self.eos_token_id = 1
        self.pad_token_id = 2

    def __len__(self):
        return self.vocab_size


def make_tokenizer(config: Config):
    if config.data.fake and config.model_name == "debugmodel":
        tokenizer = FakeTokenizer()
    elif config.model_type == "llama2":
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", use_fast=True)
    elif config.model_type == "llama3":
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B", use_fast=True)
    else:
        raise ValueError(f"Model type {config.model_type} not supported")
    return tokenizer
