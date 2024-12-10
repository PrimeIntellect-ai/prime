import random
import pytest
import torch
from zeroband.models.llama import Transformer, llama2_configs
from zeroband.models.llama.model import Attention, ModelArgs, create_block_mask_from_seqlens


VOCAB_SIZE = 1024

ERROR_ATOL = {
    torch.float: 3e-4,
    torch.half: 4e-3,
    torch.bfloat16: 2e-2,
}
ERROR_RTOL = {
    torch.float: 2e-5,
    torch.half: 4e-4,
    torch.bfloat16: 5e-3,
}


@pytest.fixture
def llama_config() -> ModelArgs:
    config = llama2_configs["debugmodel"]
    config.vocab_size = VOCAB_SIZE
    return config


def test_llama(llama_config: ModelArgs):
    seq_len = 512
    bs = 8
    model = Transformer(llama_config).to("cuda")
    input_ = torch.randint(0, llama_config.vocab_size, (bs, seq_len)).to("cuda")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(input_)

    assert output.shape == (bs, seq_len, llama_config.vocab_size)


def get_freqs_cis(llama_config: ModelArgs):
    model = Transformer(llama_config).to("cuda")
    return model.freqs_cis


def test_attn(llama_config: ModelArgs):
    seq_len = 512
    bs = 8

    freqs_cis = get_freqs_cis(llama_config)
    input_ = torch.rand(bs, seq_len, llama_config.dim).to("cuda")
    seqlens = [torch.Tensor([seq_len]).int().to("cuda") for _ in range(bs)]
    block_mask = create_block_mask_from_seqlens(seqlens)

    attn = Attention(llama_config).to("cuda")

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output_sdpa = attn(input_, freqs_cis)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output_flex = attn(input_, freqs_cis, block_mask=block_mask)

    rtol = ERROR_RTOL[torch.bfloat16]
    atol = ERROR_ATOL[torch.bfloat16]
    assert output_sdpa.shape == output_flex.shape
    torch.testing.assert_close(output_sdpa, output_flex, rtol=rtol, atol=atol)


def test_packing_simple(llama_config: ModelArgs):
    seq_len = 512
    bs = 8

    freqs_cis = get_freqs_cis(llama_config)
    input_ = torch.rand(bs, seq_len, llama_config.dim).to("cuda")
    seqlens = [torch.Tensor([seq_len // 4] * 4).int().to("cuda") for _ in range(bs)]
    block_mask = create_block_mask_from_seqlens(seqlens)

    attn = Attention(llama_config).to("cuda")

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = attn(input_, freqs_cis, block_mask=block_mask)

    assert output.shape == (bs, seq_len, llama_config.dim)


def test_sequence_packing_two_time_same_sequence(llama_config: ModelArgs):
    """
    In this test we take a sequence and pack it with itself along the seqlen dimension.
    We then pass the packed sequence to the attention layer and check that the output for each sequence is the same.
    """

    model = Attention(llama_config).to("cuda")

    emb = torch.nn.Embedding(10, llama_config.dim).to("cuda")

    seq = [2, 1, 4, 8]
    input_stuff_raw = torch.Tensor([seq + seq]).long().to("cuda")
    seqlens = [torch.Tensor([len(seq), len(seq)]).int().to("cuda")]
    block_mask = create_block_mask_from_seqlens(seqlens)

    input_stuff = emb(input_stuff_raw)

    freqs_cis = get_freqs_cis(llama_config)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(input_stuff, freqs_cis, block_mask=block_mask)

    output_left = output[:, :4, :]
    output_right = output[:, 4:, :]

    ### TESTING
    assert output_left.shape == output_right.shape

    rtol = ERROR_RTOL[torch.bfloat16]
    atol = ERROR_ATOL[torch.bfloat16]
    torch.testing.assert_close(output_left, output_right, atol=atol, rtol=rtol)


def test_sequence_packing_vs_normal(llama_config: ModelArgs):
    """
    take two sequences and compare the outout of attention on individual sequences vs the output of attention on the packed sequence
    """

    model = Attention(llama_config).to("cuda")
    emb = torch.nn.Embedding(10, llama_config.dim).to("cuda")

    freqs_cis = get_freqs_cis(llama_config)

    seq_1 = [2, 1, 4, 8]
    seq_2 = [3, 7, 5, 6]

    input_packed_raw = torch.Tensor([seq_1 + seq_2]).long().to("cuda")
    seqlens = [torch.Tensor([len(seq_1), len(seq_2)]).int().to("cuda")]
    block_mask = create_block_mask_from_seqlens(seqlens)

    input_packed = emb(input_packed_raw)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(input_packed, freqs_cis, block_mask=block_mask)

    output_packed_1 = output[:, :4, :]
    output_packed_2 = output[:, 4:, :]

    input_raw_1 = torch.Tensor([seq_1]).long().to("cuda")
    input_raw_2 = torch.Tensor([seq_2]).long().to("cuda")

    emb_1 = emb(input_raw_1)
    emb_2 = emb(input_raw_2)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output_1 = model(emb_1, freqs_cis)
        output_2 = model(emb_2, freqs_cis)

    rtol = ERROR_RTOL[torch.bfloat16]
    atol = ERROR_ATOL[torch.bfloat16]

    ### TESTING
    assert output_1.shape == output_packed_1.shape
    assert output_2.shape == output_packed_2.shape

    torch.testing.assert_close(output_1, output_packed_1, atol=atol, rtol=rtol)
    torch.testing.assert_close(output_2, output_packed_2, atol=atol, rtol=rtol)


def test_sequence_packing_vs_normal_random(llama_config: ModelArgs):
    """
    take two sequences and compare the outout of attention on individual sequences vs the output of attention on the packed sequence
    """

    model = Attention(llama_config).to("cuda")

    freqs_cis = get_freqs_cis(llama_config)

    MAX_SEQ_LEN = 256

    for _ in range(10):
        seq_len_cutoff = random.randint(1, MAX_SEQ_LEN)

        seq1 = seq_len_cutoff
        seq2 = MAX_SEQ_LEN - seq_len_cutoff
        input_1 = torch.rand(1, seq1, llama_config.dim).to("cuda")
        input_2 = torch.rand(1, seq2, llama_config.dim).to("cuda")

        seqlens = [torch.Tensor([seq1, seq2]).int().to("cuda")]
        block_mask = create_block_mask_from_seqlens(seqlens)

        packed_input = torch.cat([input_1, input_2], dim=1)

        # packed output
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(packed_input, freqs_cis, block_mask=block_mask)

        output_packed_1 = output[:, :seq_len_cutoff, :]
        output_packed_2 = output[:, seq_len_cutoff:, :]

        # normal output
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output_1 = model(input_1, freqs_cis)
            output_2 = model(input_2, freqs_cis)

        rtol = ERROR_RTOL[torch.bfloat16]
        atol = ERROR_ATOL[torch.bfloat16]

        ### TESTING
        assert output_1.shape == output_packed_1.shape
        assert output_2.shape == output_packed_2.shape

        torch.testing.assert_close(output_1, output_packed_1, atol=atol, rtol=rtol)
        torch.testing.assert_close(output_2, output_packed_2, atol=atol, rtol=rtol)


def test_end_to_end_packing(llama_config: ModelArgs):
    model = Transformer(llama_config).to("cuda")

    BS = 8
    SEQ_LEN = 128

    input_ = torch.randint(1, llama_config.vocab_size, (BS, SEQ_LEN)).to("cuda")

    seqlens = [torch.Tensor([SEQ_LEN // 4, SEQ_LEN // 4, SEQ_LEN // 2]).int().to("cuda") for _ in range(BS)]
    block_mask = create_block_mask_from_seqlens(seqlens)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(input_, block_mask=block_mask)

    assert output.shape == (BS, SEQ_LEN, llama_config.vocab_size)

    loss = output.mean()
    loss.backward()  # test that the backward for fa2
