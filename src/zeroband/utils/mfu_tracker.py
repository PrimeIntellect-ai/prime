import math
from dataclasses import dataclass
from enum import Enum

import torch
import torch._dynamo
from torch.nn.attention.flex_attention import BlockMask


@dataclass
class FlagshipPerformance:
    tflops_tf32: float
    tflops_bf16_32: float
    tflops_fp16_32: float
    tflops_fp16_16: float
    tflops_fp8_32: float
    tflops_fp8_16: float
    num_tensor_cores: int
    clock_mhz: float


@dataclass
class DeviceEntry:
    generation: str
    num_tensor_cores: int
    clock_mhz: float


generation_db = {
    'VOLTA': FlagshipPerformance(125., -1., 125., -1., -1., -1., 640, 1530.),
    'AMPERE_DATACENTER': FlagshipPerformance(156., 312., 312., 312., -1., -1., 432, 1410.),
    'AMPERE_CONSUMER': FlagshipPerformance(40., 80., 80., 160., -1., -1., 336, 1860.),
    'HOPPER': FlagshipPerformance(500., 1000., 1000., 1000., 2000., 2000., 528, 1830.),
    'ADA_CONSUMER': FlagshipPerformance(82.6, 165.2, 165.2, 330.3, 330.3, 660.6, 512, 2520.),
    'BLACKWELL_CONSUMER': FlagshipPerformance(104.8, 209.5, 209.5, 419, 419, 838, 680, 2407.)
}

gpu_db = {
    "Tesla V100-SXM2-16GB": DeviceEntry(generation='VOLTA', num_tensor_cores=640, clock_mhz=1530),
    "Tesla V100-PCIE-32GB": DeviceEntry(generation='VOLTA', num_tensor_cores=640, clock_mhz=1530),
    "NVIDIA A100-PCIE-40GB": DeviceEntry(generation='AMPERE_DATACENTER', num_tensor_cores=432, clock_mhz=1410),
    "NVIDIA A100-PCIE-80GB": DeviceEntry(generation='AMPERE_DATACENTER', num_tensor_cores=432, clock_mhz=1410),
    "NVIDIA A100-SXM4-40GB": DeviceEntry(generation='AMPERE_DATACENTER', num_tensor_cores=432, clock_mhz=1410),
    "NVIDIA A100-SXM4-80GB": DeviceEntry(generation='AMPERE_DATACENTER', num_tensor_cores=432, clock_mhz=1410),
    "NVIDIA RTX A2000": DeviceEntry(generation='AMPERE_CONSUMER', num_tensor_cores=104, clock_mhz=1200),
    "NVIDIA RTX A4000": DeviceEntry(generation='AMPERE_CONSUMER', num_tensor_cores=192, clock_mhz=1560),
    "NVIDIA RTX A4500": DeviceEntry(generation='AMPERE_CONSUMER', num_tensor_cores=224, clock_mhz=1650),
    "NVIDIA RTX A5000": DeviceEntry(generation='AMPERE_CONSUMER', num_tensor_cores=256, clock_mhz=1695),
    "NVIDIA RTX A5500": DeviceEntry(generation='AMPERE_CONSUMER', num_tensor_cores=320, clock_mhz=1770),
    "NVIDIA RTX A6000": DeviceEntry(generation='AMPERE_CONSUMER', num_tensor_cores=336, clock_mhz=1800),
    'NVIDIA RTX A40': DeviceEntry(generation='AMPERE_CONSUMER', num_tensor_cores=336, clock_mhz=1740),
    "NVIDIA GeForce RTX 3090 Ti": DeviceEntry(generation='AMPERE_CONSUMER', num_tensor_cores=336, clock_mhz=1860),
    "NVIDIA GeForce RTX 3090": DeviceEntry(generation='AMPERE_CONSUMER', num_tensor_cores=328, clock_mhz=1695),
    "NVIDIA GeForce RTX 3080 Ti": DeviceEntry(generation='AMPERE_CONSUMER', num_tensor_cores=320, clock_mhz=1665),
    "NVIDIA GeForce RTX 3080": DeviceEntry(generation='AMPERE_CONSUMER', num_tensor_cores=272, clock_mhz=1710),
    "NVIDIA GeForce RTX 3070 Ti": DeviceEntry(generation='AMPERE_CONSUMER', num_tensor_cores=192, clock_mhz=1770),
    "NVIDIA GeForce RTX 3070": DeviceEntry(generation='AMPERE_CONSUMER', num_tensor_cores=184, clock_mhz=1725),
    "NVIDIA GeForce RTX 3060 Ti": DeviceEntry(generation='AMPERE_CONSUMER', num_tensor_cores=152, clock_mhz=1665),
    "NVIDIA GeForce RTX 3060": DeviceEntry(generation='AMPERE_CONSUMER', num_tensor_cores=112, clock_mhz=1777),
    "NVIDIA RTX A2000 ADA": DeviceEntry(generation='ADA_CONSUMER', num_tensor_cores=88, clock_mhz=2130),
    "NVIDIA RTX A4000 ADA": DeviceEntry(generation='ADA_CONSUMER', num_tensor_cores=192, clock_mhz=2175),
    "NVIDIA RTX A4500 ADA": DeviceEntry(generation='ADA_CONSUMER', num_tensor_cores=224, clock_mhz=2580),
    "NVIDIA RTX A5000 ADA": DeviceEntry(generation='ADA_CONSUMER', num_tensor_cores=400, clock_mhz=2550),
    "NVIDIA RTX A5880 ADA": DeviceEntry(generation='ADA_CONSUMER', num_tensor_cores=440, clock_mhz=2460),
    "NVIDIA RTX A6000 ADA": DeviceEntry(generation='ADA_CONSUMER', num_tensor_cores=568, clock_mhz=2505),
    "NVIDIA GeForce RTX 4090": DeviceEntry(generation='ADA_CONSUMER', num_tensor_cores=512, clock_mhz=2520),
    "NVIDIA GeForce RTX 4080 SUPER": DeviceEntry(generation='ADA_CONSUMER', num_tensor_cores=320, clock_mhz=2550),
    "NVIDIA GeForce RTX 4080": DeviceEntry(generation='ADA_CONSUMER', num_tensor_cores=304, clock_mhz=2505),
    "NVIDIA GeForce RTX 4070 Ti SUPER": DeviceEntry(generation='ADA_CONSUMER', num_tensor_cores=264, clock_mhz=2610),
    "NVIDIA GeForce RTX 4070 Ti": DeviceEntry(generation='ADA_CONSUMER', num_tensor_cores=240, clock_mhz=2610),
    "NVIDIA GeForce RTX 4070 SUPER": DeviceEntry(generation='ADA_CONSUMER', num_tensor_cores=224, clock_mhz=2475),
    "NVIDIA GeForce RTX 4070": DeviceEntry(generation='ADA_CONSUMER', num_tensor_cores=184, clock_mhz=2475),
    "NVIDIA GeForce RTX 4060 Ti": DeviceEntry(generation='ADA_CONSUMER', num_tensor_cores=136, clock_mhz=2535),
    "NVIDIA GeForce RTX 4060": DeviceEntry(generation='ADA_CONSUMER', num_tensor_cores=96, clock_mhz=2460),
    "NVIDIA H100 PCIe": DeviceEntry(generation='HOPPER', num_tensor_cores=456, clock_mhz=1695),
    "NVIDIA H100 80GB HBM3": DeviceEntry(generation='HOPPER', num_tensor_cores=528, clock_mhz=1830),
    "NVIDIA GeForce RTX 5090": DeviceEntry(generation='BLACKWELL_CONSUMER', num_tensor_cores=680, clock_mhz=2407),
    "NVIDIA GeForce RTX 5080": DeviceEntry(generation='BLACKWELL_CONSUMER', num_tensor_cores=336, clock_mhz=2617),
    "NVIDIA GeForce RTX 5070 Ti": DeviceEntry(generation='BLACKWELL_CONSUMER', num_tensor_cores=280, clock_mhz=2452),
    "NVIDIA GeForce RTX 5070": DeviceEntry(generation='BLACKWELL_CONSUMER', num_tensor_cores=192, clock_mhz=2512),
}


class PrecisionMode(Enum):
    PRECISION_TF32 = 1
    PRECISION_FP16 = 2
    PRECISION_BF16 = 3


def _get_peak_flops(performance: FlagshipPerformance, precision_mode: PrecisionMode):
    if precision_mode == PrecisionMode.PRECISION_TF32:
        return performance.tflops_tf32
    elif precision_mode == PrecisionMode.PRECISION_BF16:
        return performance.tflops_bf16_32
    elif precision_mode == PrecisionMode.PRECISION_FP16:
        return performance.tflops_fp16_32
    else:
        raise ValueError(f'Unknown precision mode {precision_mode}')


def _interpolate_performance(flagship_performance: FlagshipPerformance,
                             device_entry: DeviceEntry,
                             precision_mode: PrecisionMode) -> float:
    flagship_tflops = _get_peak_flops(flagship_performance, precision_mode)
    adjusted_tflops = flagship_tflops * (device_entry.num_tensor_cores / flagship_performance.num_tensor_cores) * (
            device_entry.clock_mhz / flagship_performance.clock_mhz)
    return adjusted_tflops


def get_flops_promised_pt(device: torch.device, precision_mode: PrecisionMode):
    assert device.type == 'cuda', 'get_flops_promised_torch cannot be invoked for non-cuda torch device!'
    device_name = torch.cuda.get_device_name(device)
    return get_flops_promised(device_name, precision_mode)


def get_flops_promised(device_name: str, precision_mode: PrecisionMode):
    db_entry: DeviceEntry = gpu_db.get(device_name, None)
    assert db_entry is not None, f"Cannot obtain promised flops for unknown GPU {device_name}"

    flagship_performance = generation_db.get(db_entry.generation, None)
    assert flagship_performance is not None, f"Unknown gpu generation {db_entry.generation}"

    return _interpolate_performance(flagship_performance, db_entry, precision_mode)

class FlopCounter:
    """
    Flop counter object used to track flops performed by performed tensor operations.
    The flop counter will infer forward and backward flops given the tracked operation type and supplied operand shapes.
    Backward flops inference can be optionally disabled.
    """

    def __init__(self, no_infer_bwd_flops: bool = False):
        self._num_forward_flops: int = 0
        self._num_backward_flops: int = 0
        self.no_infer_bwd_flops = no_infer_bwd_flops

    def track_forward_flops(self, num_flops: int):
        self._num_forward_flops += num_flops

    def track_backward_flops(self, num_flops: int, force_track_bwd: bool = False):
        if not self.no_infer_bwd_flops or force_track_bwd:
            self._num_backward_flops += num_flops

    def get_performed_flops(self) -> int:
        return self._num_forward_flops + self._num_backward_flops

    def track_linear(self, linear: torch.nn.Linear, x: torch.Tensor):
        """
        Tracks the number of flops for both the forward and backward passes of a linear layer.

        Forward pass:
          - For a matrix multiplication (x @ weight.T):
                (m, k) @ (k, n) = (m, n)
                Flops: m * n * (2 * k - 1)
          - If bias is present, one addition per output element:
                Flops: m * n

        Backward pass (assuming grad_output has the same shape as the forward output):
          - grad_input = grad_output @ weight:
                Shape: (m, n) @ (n, k) = (m, k)
                Flops: m * k * (2 * n - 1)
          - grad_weight = grad_output^T @ x:
                Shape: (n, m) @ (m, k) = (n, k)
                Flops: n * k * (2 * m - 1)
          - grad_bias = sum(grad_output) (if bias exists):
                Flops: m * n
        """
        # Determine dimensions for the forward pass
        m = x.numel() // x.size(-1)  # batch size
        k = x.size(-1)  # in_features
        n = linear.weight.size(0)  # out_features

        # Forward flop computation
        matmul_flops = m * n * (2 * k - 1)
        bias_flops = m * n if linear.bias is not None else 0
        total_forward_flops = matmul_flops + bias_flops
        self.track_forward_flops(total_forward_flops)

        # Backward flop computation (assuming grad_output of shape (m, n))
        grad_input_flops = m * k * (2 * n - 1)  # dL/dX ; becomes dL/dZ where Z = f(x) in autograd chain
        grad_weight_flops = n * k * (2 * m - 1)  # dL/dW
        grad_bias_flops = m * n if linear.bias is not None else 0
        total_backward_flops = grad_input_flops + grad_weight_flops + grad_bias_flops
        self.track_backward_flops(total_backward_flops)

    def track_binary(self, a: torch.Tensor, b: torch.Tensor):
        """
        Tracks the amount of flops that are performed when performing a binary elementwise operator of
        the same shape as the supplied inputs
        """
        result_shape = torch.broadcast_shapes(a.shape, b.shape)
        num_flops = result_shape.numel()
        self.track_forward_flops(num_flops)
        self.track_backward_flops(num_flops)

    def track_unary(self, x: torch.Tensor):
        """
        Tracks the amount of flops that are performed when performing a unary elementwise operator on the same shape
        as the supplied operand
        """
        self.track_forward_flops(x.numel())
        self.track_backward_flops(x.numel())

    def track_mha_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal=False):
        # Refer to shape legend:
        # https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
        #
        # q: (N, ..., H_q, L, E)
        # k: (N, ..., H, S, E)
        # v: (N, ..., H, S, E_v)
        *batch_dims_q, H_q, L, E_q = q.shape
        *batch_dims_k, H_k, S_k, E_k = k.shape
        *batch_dims_v, H_v, S_v, E_v = v.shape

        N_q = math.prod(batch_dims_q)
        N_k = math.prod(batch_dims_k)
        N_v = math.prod(batch_dims_v)

        assert (N_q == N_k) and (N_k == N_v), "batch size must batch across q, k & v"
        assert H_k == H_v, "head size must match for keys and values"  # H_q may differ from H_k & H_v
        assert E_q == E_k, "embedding dim must match for q & k"  # E_v may differ from E_k & E_q
        assert S_k == S_v, "source sequence length must match for k & v"  # L may differ from S_k & S_v

        N = N_q
        S = S_k
        E = E_q

        mha_flops = 4.0 * N * (S * L) * E
        if is_causal:
            mha_flops /= 2

        self.track_forward_flops(mha_flops)
        self.track_backward_flops(math.floor(mha_flops * 2.5))

    @torch.compiler.disable
    def track_flex_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask_sparsity: BlockMask):
        # Refer to shape legend:
        # https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
        #
        # q: (N, ..., H_q, L, E)
        # k: (N, ..., H, S, E)
        # v: (N, ..., H, S, E_v)
        *batch_dims_q, H_q, L, E_q = q.shape
        *batch_dims_k, H_k, S_k, E_k = k.shape
        *batch_dims_v, H_v, S_v, E_v = v.shape

        N_q = math.prod(batch_dims_q)
        N_k = math.prod(batch_dims_k)
        N_v = math.prod(batch_dims_v)

        assert (N_q == N_k) and (N_k == N_v), "batch size must batch across q, k & v"
        assert H_k == H_v, "head size must match for keys and values"  # H_q may differ from H_k & H_v
        assert E_q == E_k, "embedding dim must match for q & k"  # E_v may differ from E_k & E_q
        assert S_k == S_v, "source sequence length must match for k & v"  # L may differ from S_k & S_v

        N = N_q
        S = S_k
        E = E_q
        mha_flops = 4.0 * N * (S * L) * E

        mask_occupancy = 1.0 - (mask_sparsity.sparsity() / 100.0)
        forward_flops = math.floor(mha_flops * mask_occupancy)
        self.track_forward_flops(forward_flops)
        self.track_backward_flops(math.floor(forward_flops * 2.5))


    def track_norm(self, norm: torch.nn.Module, x: torch.Tensor):
        d = x.size(-1)
        if isinstance(norm, torch.nn.LayerNorm):
            if norm.elementwise_affine:
                if norm.bias is not None:
                    # With both gamma and beta.
                    flops = 7 * d + 2
                else:
                    # With only gamma scaling.
                    flops = 6 * d + 2
            else:
                flops = 5 * d + 2
        elif "rmsnorm" in type(norm).__name__.lower():
            if hasattr(norm, "weight") and norm.weight is not None:
                flops = 4 * d + 2
            else:
                flops = 3 * d + 2
        else:
            raise NotImplementedError(f"Normalization type {type(norm)} not supported for flop tracking.")

        self.track_forward_flops(flops)

    def track_optimizer_step(self, optimizer: torch.optim.Optimizer, num_param_scalars: int):
        if isinstance(optimizer, torch.optim.Adam):
            flops_per_param = 14
        elif isinstance(optimizer, torch.optim.AdamW):
            flops_per_param = 16
        else:
            raise NotImplementedError(f"Optimizer type {type(optimizer)} not supported for flop tracking.")
        self.track_backward_flops(flops_per_param * num_param_scalars, force_track_bwd=True)

    def track_cross_entropy(self, logits: torch.Tensor):
        """
           Tracks the FLOPs performed for the cross entropy loss computation.

           Assumes logits is of shape (N, C) where:
             - N is the number of samples (or tokens)
             - C is the number of classes (e.g. vocabulary size)

           The estimated FLOP breakdown per sample is:
             - Exponential: 4 FLOPs per element -> 4 * C
             - Summation: Approximately C FLOPs
             - Logarithm: ~4 FLOPs
             - Subtraction: 1 FLOP

           Total per sample: (4C + C + 4 + 1) = 5C + 5 FLOPs.

           The function tracks both forward and an estimated backward pass cost.
           """
        N, C = logits.shape
        forward_flops = N * (5 * C + 5)
        self.track_forward_flops(forward_flops)
        self.track_backward_flops(int(forward_flops * 2.5), force_track_bwd=True)


def get_num_flop_per_token(num_params: int, model_config, seq_len) -> int:
    l, h, q, t = (  # noqa: E741
        model_config.n_layers,
        model_config.n_heads,
        model_config.dim // model_config.n_heads,
        seq_len,
    )
    # Reasoning behind the factor of 12 for the self-attention part of the formula:
    # 1. each self-attention has 2 matmul in the forward and 4 in the backward (6)
    # 2. the flash attention does 1 more matmul recomputation in the backward
    #    but recomputation should not be counted in calculating MFU           (+0)
    # 3. each matmul performs 1 multiplication and 1 addition                 (*2)
    # 4. we follow the convention and do not account for sparsity in causal attention
    flop_per_token = 6 * num_params + 12 * l * h * q * t

    return flop_per_token