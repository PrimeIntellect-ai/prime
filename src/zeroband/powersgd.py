import torch
import torch.distributed as dist


@torch.compile
def _orthogonalize_gram_schmidt(matrices, epsilon=0):
    """
    Apply Gram-Schmidt procedure to orthogonalize a batch of matrices.

    If epsilon is 0, this is equivalent to `torch.qr(matrices, out=(matrices, _))`,
    """
    num_cols = matrices.shape[2]
    for i in range(num_cols):
        # Normalize the i'th column.
        col = matrices[:, :, i : i + 1]
        # If no epsilon is added here, division by zero may be caused by vanishing gradients.
        # This epsilon is not needed if the input batch of matrices covers the gradients of at least one entire layer
        # in the neural network.
        if epsilon == 0:
            # Note that col ** 2 can underflow/overflow if we use FP16.
            # May need to consider multiplying a scaling factor and dividing it later, or using bfloat16 instead.
            try:
                col /= torch.norm(col, dim=1, keepdim=True)
            except ZeroDivisionError:
                # logger.error(
                #     "The matrices to be orthogonalized has at least a column of all 0s. Please set a small value such as 1e-8 "
                #     "as `orthogonalization_epsilon` in PowerSGD state."
                # )
                # Recover the values from NaNs to 0s.
                col.fill_(0.0)
        else:
            col /= torch.norm(col, dim=1, keepdim=True) + epsilon
        # Project it on the rest and remove it.
        if i + 1 < num_cols:
            rest = matrices[:, :, i + 1 :]
            rest -= torch.sum(col * rest, dim=1, keepdim=True) * col


class PowerSGD:
    def __init__(self, params: list[torch.nn.Parameter], rank: int, warmup_steps: int):
        self.params = list(params)
        self.rank = rank
        self.warmup_steps = warmup_steps

        self.no_compress_param = [param for param in self.params if len(param.shape) != 2]
        self.low_rank_param = [param for param in self.params if len(param.shape) == 2]

        self.q = [torch.randn(param.shape[1], self.rank).to(param.device) for param in self.low_rank_param]
        self.error = [torch.zeros_like(param).to(param.device) for param in self.low_rank_param]

    def all_reduce(self, step: int):
        for param in self.no_compress_param:
            dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)

        if step < self.warmup_steps:
            for param in self.low_rank_param:
                param.grad = None
        else:
            for param, q, error in zip(self.low_rank_param, self.q, self.error):
                delta = param.grad + error

                P = delta @ q  # n×r matrix
                dist.all_reduce(P, op=dist.ReduceOp.AVG)  # Average P across workers
                error.copy_(delta - P @ q.T)

                P = P.unsqueeze(0)
                _orthogonalize_gram_schmidt(P)
                P = P.squeeze(0)

                Q = param.grad.T @ P  # m×r matrix
                dist.all_reduce(Q, op=dist.ReduceOp.AVG)  # Average Q across workers

                q.copy_(Q)
                param.grad = P @ Q.T
