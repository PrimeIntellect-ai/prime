from torch import Tensor
import torch
import torch.nn.functional as F

def compute_cross_entropy_loss(
        logits: Tensor,
        labels: Tensor,
        z_weight: float | None = None,
        num_chunks: int | None = None,
        ignore_index: int = -100,
        fused_linear_weight: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
    """
    Compute cross entropy loss in fp32, optionally chunked, and optionally with max z loss.

    Do not torch compile this function if you set num_chunks >= 1. It will unroll the chunking loop, thus removing the benefit of chunking.

    Max z loss is from the baichuan2 paper: https://arxiv.org/abs/2309.10305

    .. math::
        z_{loss} = weight z^{2}
    where z is the max logit
    """

    if fused_linear_weight is None:
        num_elements = (labels != ignore_index).sum().float()

        if num_chunks is not None and not num_chunks <= 1:
            l_labels: list[Tensor] = [target_chunk.reshape(-1) for target_chunk in labels.chunk(num_chunks, dim=0)]
            l_logits: list[Tensor] = [logit_chunk.reshape(-1, logit_chunk.size(-1)) for logit_chunk in logits.reshape(-1, logits.size(-1)).chunk(num_chunks, dim=0)]
        else:
            l_labels: list[Tensor] = [labels.reshape(-1)]
            l_logits: list[Tensor] = [logits.reshape(-1, logits.size(-1))]

        loss = 0.0
        ce_loss = None if z_weight is None else 0.0
        for logits_chunk, labels_chunk in zip(l_logits, l_labels):
            if z_weight is None:
                loss += _upcast_cross_entropy(logits_chunk, labels_chunk, ignore_index=ignore_index)
            else:
                ce, z = _upcast_cross_entropy_max_z(logits_chunk, labels_chunk, z_weight, ignore_index=ignore_index)
                loss += ce
                ce_loss += z

        return (loss / num_elements), (None if ce_loss is None else ce_loss / num_elements)

    else:
        # Ignore number of chunks, since it is not confugrable in liger.
        from liger_kernel.ops.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyFunction
        ret = LigerFusedLinearCrossEntropyFunction.apply(
            logits,                                      # _input
            fused_linear_weight,                         # weight
            labels,                                      # target
            None,                                        # ce_weight
            None,                                        # bias
            ignore_index,                                # ce_weight=None
            z_weight if z_weight is not None else 0.0,   # lse_square_scale
            0.0,                                         # label_smoothing
            "mean",                                      # reduction
            None,                                        # softcap
            fused_linear_weight is not None,             # return_z_loss
        )
        if not isinstance(ret, tuple):
            assert isinstance(ret, Tensor)
            ret = (ret, None)
        return ret


# Compile the upcast into the CE calculation
@torch.compile
def _upcast_cross_entropy(logit_chunk, label_chunk, ignore_index) -> Tensor:
    return F.cross_entropy(logit_chunk.float(), label_chunk, ignore_index=ignore_index, reduction="sum")


@torch.compile
def _upcast_cross_entropy_max_z(
    logits: Tensor,
    targets: Tensor,
    z_loss_weight: float,
    ignore_index: int = -100,
) -> tuple[Tensor, Tensor]:
    # max is not differentiable. But here we just pick the indices of the max value, so it's fine for backpropagation.
    loss = F.cross_entropy(logits.float(), targets, ignore_index=ignore_index, reduction="sum")
    max_logits = logits.max(dim=-1)[0]
    max_logits = max_logits.where(targets != ignore_index, 0)
    z_loss = z_loss_weight * max_logits.pow(2).mean()
    return loss, z_loss
