import torch
import torch.distributed as dist
from torch import Tensor

@torch.compile
def low_rank_momentum_newton_schulz(G: Tensor, rank: int, steps: int = 5) -> Tensor:
    """
    Compute low-rank momentum approximation using Newton-Schulz iteration.
    This combines PowerSGD's low-rank idea with Muon's Newton-Schulz orthogonalization.
    
    Args:
        G: Input tensor of shape (m, n)
        rank: Target rank for approximation
        steps: Number of Newton-Schulz iterations
    """
    assert G.ndim == 2
    assert rank > 0 and rank <= min(G.size(0), G.size(1))

    # Constants for quintic iteration (same as Muon)
    a, b, c = (3.4445, -4.7750, 2.0315)

    # Convert to bfloat16 for efficiency
    G = G.bfloat16()

    # Initialize random projection matrix Q
    n = G.size(1)
    Q = torch.randn(n, rank, device=G.device).bfloat16()
    Q = Q / (Q.norm() + 1e-7)

    # Project momentum buffer onto low-rank space
    P = G @ Q  # [m x n] @ [n x r] = [m x r]
    
    # Handle non-square matrices like in original Muon
    if P.size(0) > P.size(1):
        P = P.t()
        transposed = True
    else:
        transposed = False

    # Normalize P
    P = P / (P.norm() + 1e-7)

    # Newton-Schulz iterations for orthogonalization
    for _ in range(steps):
        A = P @ P.t()
        B = b * A + c * A @ A
        P = a * P + B @ P

    if transposed:
        P = P.t()

    # Compute final factors and expand
    V = G.t() @ P  # [n x m] @ [m x r] = [n x r]
    return P @ V.t()  # [m x r] @ [r x n] = [m x n]

class Muon(torch.optim.Optimizer):
    """
    Fuses Muon and PowerSGD approaches:
    - Uses momentum accumulation from Muon
    - Applies low-rank projection like PowerSGD
    - Uses Newton-Schulz for orthogonalization
    """
    def __init__(
        self,
        params,
        lr=0.02,
        momentum=0.95,
        nesterov=True,
        rank=4,  # New parameter for low-rank approximation
        ns_steps=5,
        world_size=1,
    ):
        self.rank = rank
        self.world_size = world_size
        self._step_count = 0

        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        params = [*params]
        assert all(isinstance(p, Tensor) for p in params)
        sizes = {p.numel() for p in params}

        def create_update_buffer(size: int):
            b = torch.empty(self.world_size, size, dtype=torch.bfloat16, device="cuda")
            return dict(update_buffer=b, update_buffer_views=[b[i] for i in range(self.world_size)])

        param_groups = [
            dict(params=[p for p in params if p.numel() == size], **create_update_buffer(size))
            for size in sizes
        ]
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self):
        self._step_count += 1

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            update_buffer = group["update_buffer"]
            update_buffer_views = group["update_buffer_views"]
            params = group["params"]
            handle = None
            params_world = None

            def update_prev():
                if params_world is None:
                    return
                assert handle is not None
                handle.wait()
                for p_world, g_world in zip(params_world, update_buffer_views):
                    p_world.add_(
                        g_world.view_as(p_world),
                        alpha=-lr * max(1, p_world.size(-2) / p_world.size(-1)) ** 0.5,
                    )

            for base_i in range(len(params))[:: self.world_size]:
                if base_i + self.rank < len(params):
                    p = params[base_i + self.rank]
                    g = p.grad
                    assert g is not None
                    
                    # Get momentum buffer
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    
                    # Update momentum (same as original Muon)
                    buf.lerp_(g, 1 - momentum)
                    g = g.lerp_(buf, momentum) if nesterov else buf

                    # Apply low-rank momentum approximation
                    g = low_rank_momentum_newton_schulz(g, self.rank, steps=ns_steps).flatten()
                else:
                    g = update_buffer_views[self.rank]
                    
                update_prev()
                handle = dist.all_gather_into_tensor(update_buffer, g, async_op=True)
                params_world = params[base_i : base_i + self.world_size]
            update_prev()
