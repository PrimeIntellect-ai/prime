#!/usr/bin/env python
# coding: utf-8
# Example Usage:
# python scripts/export_dcp.py @configs/10B/H100.toml --ckpt.path /data/intellect-1-step17000 --ckpt.resume /data/10b/step_17000/diloco_0

import torch
from typing import Literal
import torch.distributed.checkpoint as dcp
from zeroband.models.llama import get_model
from zeroband.config import resolve_env_vars
from zeroband.checkpoint import ModelWrapper
from zeroband.utils import get_module_signature
from zeroband.train import Config
from zeroband.utils.logger import get_logger
from pydantic_config import parse_argv
from transformers import AutoTokenizer
import math
from pathlib import Path
from safetensors.torch import save_file
import json
from zeroband.models.llama import ModelArgs
from transformers import LlamaConfig
from transformers.generation import GenerationConfig


class ExportConfig(Config):
    save_format: Literal["pt", "safetensors"] = "safetensors"
    torch_dtype: Literal["float32", "bfloat16"] = "float32"
    with_debug_automap: bool = False


def remap_keys_llama(k: str) -> str:
    """Maps ZeroBand keys to HuggingFace keys"""
    return ("model." if "output.weight" not in k else "") + k.replace("tok_embeddings", "embed_tokens").replace(
        "attention.wq", "self_attn.q_proj"
    ).replace("attention.wk", "self_attn.k_proj").replace("attention.wv", "self_attn.v_proj").replace(
        "attention.wo", "self_attn.o_proj"
    ).replace("attention_norm", "input_layernorm").replace("feed_forward.w3", "mlp.up_proj").replace(
        "feed_forward.w2", "mlp.down_proj"
    ).replace("feed_forward.w1", "mlp.gate_proj").replace("ffn_norm", "post_attention_layernorm").replace(
        "output.weight", "lm_head.weight"
    )


def _get_ffn_dim(hidden_dim: int, ffn_dim_multiplier: float, multiple_of: int) -> int:
    """Get the FFN dimension from ZeroBand args"""
    hidden_dim = int(8 * hidden_dim / 3)
    # custom dim factor multiplier
    if ffn_dim_multiplier is not None:
        hidden_dim = int(ffn_dim_multiplier * hidden_dim)
    hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
    return hidden_dim


def convert_config_zb_to_hf(
    zb_config: ModelArgs, with_debug_automap: bool = False, type_model: str = "llama3"
) -> LlamaConfig:
    """Convert ZeroBand config to HuggingFace config"""
    config = LlamaConfig()
    config.hidden_size = zb_config.dim
    config.num_hidden_layers = zb_config.n_layers
    config.num_attention_heads = zb_config.n_heads
    config.num_key_value_heads = zb_config.n_kv_heads
    config.vocab_size = zb_config.vocab_size
    config.intermediate_size = _get_ffn_dim(zb_config.dim, zb_config.ffn_dim_multiplier, zb_config.multiple_of)
    config.rms_norm_eps = zb_config.norm_eps
    config.rope_theta = float(zb_config.rope_theta)
    config.max_position_embeddings = zb_config.max_seq_len

    if type_model == "llama2":
        config.bos_token_id = [1]
        config.eos_token_id = [2]
    else:
        config.bos_token_id = [128000]
        config.eos_token_id = [128001, 128008, 128009]

    config.architectures = ["LlamaForCausalLM"]

    # Rope scaling
    config.rope_scaling = {
        "original_max_position_embeddings": 8192,
        "rope_type": "default",
    }

    if with_debug_automap:
        config.auto_map = {
            "AutoConfig": "PrimeIntellect/prime-llama-debug--configuration_llama.LlamaConfig",
            "AutoModelForCausalLM": "PrimeIntellect/prime-llama-debug--modeling_llama.LlamaForCausalLM",
        }

    return config


@torch.no_grad
def convert_qk_from_complex_to_rotate_half(linear_weight: torch.FloatTensor, head_dim: int) -> torch.FloatTensor:
    """Converts the Q/K weight from complex to rotate half form.
    This is required because the rotary implementation in ZeroBand uses complex numbers which encodes even elements as real and odd number as complex.
    [0, 1, 2, 3] -> [0 + 1j, 2 + 3j]
    However, the HuggingFace implementation uses rotate_half which encodes top half as real and bottom half as complex.
    [0, 1, 2, 3] -> [0, 1] + [2, 3]j

    We thus need to permute the QK outputs to match the HuggingFace implementation.
    """
    new_weight = torch.zeros_like(linear_weight)

    num_heads = linear_weight.size(0) // head_dim
    hhd = head_dim // 2

    # This applies the riffle shuffle permutation to the outputs of the linear for each attn head
    # Even numbers go to the top half, odd numbers go to the bottom half
    for i in range(num_heads):
        new_weight[i * head_dim : (i * head_dim + hhd), :].copy_(
            linear_weight[i * head_dim + 0 : (i + 1) * head_dim : 2, :]
        )
        new_weight[i * head_dim + hhd : (i + 1) * head_dim, :].copy_(
            linear_weight[i * head_dim + 1 : (i + 1) * head_dim : 2, :]
        )

    return new_weight


def main(config: ExportConfig):
    # Create save path
    save_path = Path(config.ckpt.path)
    save_path.mkdir(parents=True, exist_ok=True)

    # Load model
    logger.info("Getting tokenizer (for vocab size)")
    if config.type_model == "llama2":
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", use_fast=True)
    elif config.type_model == "llama3":
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B", use_fast=True)
    else:
        raise ValueError(f"Model type {config.type_model} not supported")

    logger.info("Getting model")
    model, model_config = get_model(
        config.name_model,
        config.type_model,
        vocab_size=len(tokenizer),
        seq_length=config.data.seq_length,
        attn_fn=config.train.attn_fn,
    )

    # Convert ZeroBand config to HuggingFace config
    hf_config = convert_config_zb_to_hf(
        model_config, with_debug_automap=config.with_debug_automap, type_model=config.type_model
    )
    hf_config.to_json_file(save_path / "config.json")

    # Load checkpoint
    logger.info("Before load: %s", get_module_signature(model))
    states = {
        "model": ModelWrapper(model),
    }
    logger.info("Loading from %s", config.ckpt.resume)
    dcp.load(
        state_dict=states,
        checkpoint_id=config.ckpt.resume,
    )

    logger.info("After load: %s", get_module_signature(model))

    # Convert model to HuggingFace format
    num_shards = int(sum(p.numel() for p in model.parameters()) / 1e9)
    state_dict = model.state_dict()

    index_json = {}
    total_size = 0
    state_dict = {remap_keys_llama(k): v for k, v in state_dict.items()}
    if not config.with_debug_automap:  # The debug uses complex rotary impl
        with torch.no_grad():
            for i in range(hf_config.num_hidden_layers):
                old_q = state_dict[f"model.layers.{i}.self_attn.q_proj.weight"]
                old_k = state_dict[f"model.layers.{i}.self_attn.k_proj.weight"]
                new_q = convert_qk_from_complex_to_rotate_half(old_q, 128)
                new_k = convert_qk_from_complex_to_rotate_half(old_k, 128)
                state_dict[f"model.layers.{i}.self_attn.q_proj.weight"].copy_(new_q)
                state_dict[f"model.layers.{i}.self_attn.k_proj.weight"].copy_(new_k)
    if "model.freqs_cis" in state_dict:  # This should not be persisted
        del state_dict["model.freqs_cis"]
    if config.torch_dtype == "bfloat16":
        state_dict = {k: v.to(torch.bfloat16) for k, v in state_dict.items()}

    # Save model
    state_keys = list(state_dict.keys())
    shard_size = int(math.ceil(len(state_keys) / num_shards))
    logger.info("Saving model to %d shards", num_shards)

    for i in range(num_shards):
        _file = save_path / f"model-{i:04}-of-{num_shards:04}.{config.save_format}"
        start = i * shard_size
        end = min((i + 1) * shard_size, len(state_keys))
        shard = {k: state_dict[k] for k in state_keys[start:end]}

        index_json.update({k: _file.name for k in shard.keys()})
        total_size += sum(p.numel() for p in shard.values())
        if config.save_format == "pt":
            torch.save(shard, _file)
        else:
            save_file(shard, _file, metadata=dict(format="pt"))

    json.dump(
        {
            "weight_map": index_json,
            "metadata": {
                "total_size": total_size * (2 if config.torch_dtype == "bfloat16" else 4),
            },
        },
        (save_path / "model.safetensors.index.json").open("w"),
        indent=2,
    )

    # Save Tokenizer
    tokenizer.save_pretrained(save_path)

    # Save Generation Config
    gconfig = GenerationConfig(max_length=100, use_cache=False, temperature=0.7, top_k=None, do_sample=True)
    gconfig.save_pretrained(save_path)


if __name__ == "__main__":
    logger = get_logger()
    config = ExportConfig(**parse_argv())
    resolve_env_vars(config)
    logger.debug(f"config: {config.model_dump()}")

    main(config)
