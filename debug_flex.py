import torch
from torch.nn.attention.flex_attention import create_block_mask
# flex_attention = torch.compile(flex_attention, dynamic=False)
# create_block_mask = torch.compile(create_block_mask, dynamic=False)


def seqlens_to_docs_tensor(seqlens: list[torch.Tensor]) -> torch.Tensor:
    """Converts list of sequence lengths to document indices tensor.
    Example:
        seqlens = [tensor([2,2,1]), tensor([2,2,1])]  # List of 2 tensors
        docs = [[0,0,1,1,2], [0,0,1,1,2]] # Each doc_id repeated per its length
    """
    return torch.stack([torch.repeat_interleave(torch.arange(len(seq), device=seq.device), seq) for seq in seqlens])


SEQ_LEN = 16
BS = 8


seqlens = [torch.Tensor([16 // 4] * 4).int().to("cuda") for _ in range(BS)]
docs = seqlens_to_docs_tensor(seqlens)


def document_masking(b, h, q_idx, kv_idx):
    return docs[b, q_idx] == docs[b, kv_idx]


# block_mask = create_block_mask(document_masking, BS, None, SEQ_LEN, SEQ_LEN, device="cuda", _compile=True)
block_mask = create_block_mask(document_masking, BS, None, SEQ_LEN, SEQ_LEN, device="cuda", _compile=False)
