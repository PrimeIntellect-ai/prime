from dataclasses import dataclass
from typing import TYPE_CHECKING
from zeroband.config import Config

if TYPE_CHECKING:
    from zeroband.data import DEBUG_VOCAB_SIZE

@dataclass
class TokenizerInfo:
    vocab_size: int
    bot_token: int
    eot_token: int

def get_tokenizer_info(config: Config) -> TokenizerInfo:
    if config.data.fake and config.model_name == "debugmodel":
        return DEBUG_VOCAB_SIZE
    elif config.model_type == "llama2":
        return TokenizerInfo(
            # print(len(AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", use_fast=True)))
            vocab_size=32000,
            # print(AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", use_fast=True).bos_token_id)
            bot_token=1,
            # print(AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", use_fast=True).eos_token_id)
            eot_token=2,
        )
    elif config.model_type == "llama3":
        return TokenizerInfo(
            # print(len(AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B", use_fast=True)))
            vocab_size=128256,
            # print(AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", use_fast=True).bos_token_id)
            bot_token=128000,
            # print(AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B", use_fast=True).eos_token_id)
            eot_token=128001,
        )
    else:
        raise ValueError(f"Model type {config.model_type} not supported")
