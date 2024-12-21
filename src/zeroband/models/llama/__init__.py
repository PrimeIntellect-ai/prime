# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Llama 2 is licensed under the LLAMA 2 Community License,
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.

from zeroband.models.llama.model import ModelArgs, Transformer

__all__ = ["Transformer"]

llama2_configs = {
    "debugmodel": ModelArgs(dim=256, n_layers=2, n_heads=8),
    "150M": ModelArgs(dim=1024, n_layers=12, n_heads=16),  # todo(sami): double check this
    "271M": ModelArgs(dim=1024, n_layers=16, n_heads=8),
    "1B": ModelArgs(dim=2048, n_layers=18, n_heads=16),
    "7B": ModelArgs(dim=4096, n_layers=32, n_heads=32),
    "10B": ModelArgs(dim=5120, n_layers=32, n_heads=40),
    "13B": ModelArgs(dim=5120, n_layers=40, n_heads=40),
    "26B": ModelArgs(dim=5120, n_layers=80, n_heads=40),
    "70B": ModelArgs(
        dim=8192,
        n_layers=80,
        n_heads=64,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        multiple_of=4096,
    ),
}

llama3_configs = {
    "debugmodel": ModelArgs(dim=256, n_layers=8, n_heads=16, rope_theta=500000),
    "1B": ModelArgs(
        dim=2048,
        n_layers=18,
        n_heads=16,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        multiple_of=512,
        rope_theta=500000,
    ),
    "8B": ModelArgs(
        dim=4096,
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        multiple_of=1024,
        rope_theta=500000,
    ),
    "10B": ModelArgs(
        dim=4096,
        n_layers=42,
        n_heads=32,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        multiple_of=1024,
        rope_theta=500000,
    ),
    "70B": ModelArgs(
        dim=8192,
        n_layers=80,
        n_heads=64,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        multiple_of=4096,
        rope_theta=500000,
    ),
    "405B": ModelArgs(
        dim=16384,
        n_layers=126,
        n_heads=128,
        n_kv_heads=8,
        ffn_dim_multiplier=1.2,
        multiple_of=4096,
        rope_theta=500000,
    ),
}


def get_model(
    name_model: str,
    type_model: str,
    vocab_size: int,
    seq_length: int,
    math_attn: bool,
) -> tuple[Transformer, ModelArgs]:
    """get the transformer model"""

    if type_model == "llama2":
        config = llama2_configs[name_model]
    elif type_model == "llama3":
        config = llama3_configs[name_model]
    else:
        raise ValueError(f"Model type {type_model} not supported")

    config.vocab_size = vocab_size
    config.max_seq_len = seq_length
    config.math_attn = math_attn

    return Transformer(config), config
