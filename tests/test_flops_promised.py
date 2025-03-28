# NOTE: TFLOP Numbers sourced from Nvidia Ampere & ADA whitepaper:
# https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf
# https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.pdf
# https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf
# Asserts interpolated performance from peak matches Nvidia provided numbers

import pytest

from zeroband.utils import mfu_tracker
from zeroband.utils.mfu_tracker import PrecisionMode


def test_get_flops_promised_4090():
    assert (mfu_tracker.get_flops_promised('NVIDIA GeForce RTX 4090', PrecisionMode.PRECISION_BF16)
            == pytest.approx(165.2))
    assert (mfu_tracker.get_flops_promised('NVIDIA GeForce RTX 4090', PrecisionMode.PRECISION_FP16)
            == pytest.approx(165.2))
    assert (mfu_tracker.get_flops_promised('NVIDIA GeForce RTX 4090', PrecisionMode.PRECISION_TF32)
            == pytest.approx(82.6))


def test_get_flops_promised_4080__non_flagship():
    assert (mfu_tracker.get_flops_promised('NVIDIA GeForce RTX 4080', PrecisionMode.PRECISION_BF16)
            == pytest.approx(97.5, abs=1e-1))
    assert (mfu_tracker.get_flops_promised('NVIDIA GeForce RTX 4080', PrecisionMode.PRECISION_FP16)
            == pytest.approx(97.5, abs=1e-1))
    assert (mfu_tracker.get_flops_promised('NVIDIA GeForce RTX 4080', PrecisionMode.PRECISION_TF32)
            == pytest.approx(48.7, abs=1e-1))


def test_get_flops_promised_4070_ti__non_flagship():
    assert (mfu_tracker.get_flops_promised('NVIDIA GeForce RTX 4070 Ti', PrecisionMode.PRECISION_BF16)
            == pytest.approx(80.2, abs=1e-1))
    assert (mfu_tracker.get_flops_promised('NVIDIA GeForce RTX 4070 Ti', PrecisionMode.PRECISION_FP16)
            == pytest.approx(80.2, abs=1e-1))
    assert (mfu_tracker.get_flops_promised('NVIDIA GeForce RTX 4070 Ti', PrecisionMode.PRECISION_TF32)
            == pytest.approx(40.1, abs=1e-1))


def test_get_flops_promised_4070__non_flagship():
    assert (mfu_tracker.get_flops_promised('NVIDIA GeForce RTX 4070', PrecisionMode.PRECISION_BF16)
            == pytest.approx(58.3, abs=1e-1))
    assert (mfu_tracker.get_flops_promised('NVIDIA GeForce RTX 4070', PrecisionMode.PRECISION_FP16)
            == pytest.approx(58.3, abs=1e-1))
    assert (mfu_tracker.get_flops_promised('NVIDIA GeForce RTX 4070', PrecisionMode.PRECISION_TF32)
            == pytest.approx(29.1, abs=1e-1))


def test_get_flops_promised_3090_ti():
    assert (mfu_tracker.get_flops_promised('NVIDIA GeForce RTX 3090 Ti', PrecisionMode.PRECISION_BF16)
            == pytest.approx(80))
    assert (mfu_tracker.get_flops_promised('NVIDIA GeForce RTX 3090 Ti', PrecisionMode.PRECISION_FP16)
            == pytest.approx(80))
    assert (mfu_tracker.get_flops_promised('NVIDIA GeForce RTX 3090 Ti', PrecisionMode.PRECISION_TF32)
            == pytest.approx(40))


def test_get_flops_promised_3090__non_flagship():
    assert (mfu_tracker.get_flops_promised('NVIDIA GeForce RTX 3090', PrecisionMode.PRECISION_BF16)
            == pytest.approx(71, abs=2e-1))
    assert (mfu_tracker.get_flops_promised('NVIDIA GeForce RTX 3090', PrecisionMode.PRECISION_FP16)
            == pytest.approx(71, abs=2e-1))
    assert (mfu_tracker.get_flops_promised('NVIDIA GeForce RTX 3090', PrecisionMode.PRECISION_TF32)
            == pytest.approx(35.6, abs=1e-1))


def test_get_flops_promised_3080_ti__non_flagship():
    assert (mfu_tracker.get_flops_promised('NVIDIA GeForce RTX 3080 Ti', PrecisionMode.PRECISION_BF16)
            == pytest.approx(68.2, abs=1e-1))
    assert (mfu_tracker.get_flops_promised('NVIDIA GeForce RTX 3080 Ti', PrecisionMode.PRECISION_FP16)
            == pytest.approx(68.2, abs=1e-1))
    assert (mfu_tracker.get_flops_promised('NVIDIA GeForce RTX 3080 Ti', PrecisionMode.PRECISION_TF32)
            == pytest.approx(34.1, abs=1e-1))


def test_get_flops_promised_3080__non_flagship():
    assert (mfu_tracker.get_flops_promised('NVIDIA GeForce RTX 3080', PrecisionMode.PRECISION_BF16)
            == pytest.approx(59.5, abs=1e-1))
    assert (mfu_tracker.get_flops_promised('NVIDIA GeForce RTX 3080', PrecisionMode.PRECISION_FP16)
            == pytest.approx(59.5, abs=1e-1))
    assert (mfu_tracker.get_flops_promised('NVIDIA GeForce RTX 3080', PrecisionMode.PRECISION_TF32)
            == pytest.approx(29.8, abs=1e-1))


def test_get_flops_promised_3070_ti__non_flagship():
    assert (mfu_tracker.get_flops_promised('NVIDIA GeForce RTX 3070 Ti', PrecisionMode.PRECISION_BF16)
            == pytest.approx(43.5, abs=1e-1))
    assert (mfu_tracker.get_flops_promised('NVIDIA GeForce RTX 3070 Ti', PrecisionMode.PRECISION_FP16)
            == pytest.approx(43.5, abs=1e-1))
    assert (mfu_tracker.get_flops_promised('NVIDIA GeForce RTX 3070 Ti', PrecisionMode.PRECISION_TF32)
            == pytest.approx(21.7, abs=1e-1))


def test_get_flops_promised_3070__non_flagship():
    assert (mfu_tracker.get_flops_promised('NVIDIA GeForce RTX 3070', PrecisionMode.PRECISION_BF16)
            == pytest.approx(40.6, abs=1e-1))
    assert (mfu_tracker.get_flops_promised('NVIDIA GeForce RTX 3070', PrecisionMode.PRECISION_FP16)
            == pytest.approx(40.6, abs=1e-1))
    assert (mfu_tracker.get_flops_promised('NVIDIA GeForce RTX 3070', PrecisionMode.PRECISION_TF32)
            == pytest.approx(20.3, abs=1e-1))


def test_get_flops_promised_rtx_a6000():
    assert (mfu_tracker.get_flops_promised('NVIDIA RTX A6000', PrecisionMode.PRECISION_BF16)
            == pytest.approx(77.4, abs=1e-1))
    assert (mfu_tracker.get_flops_promised('NVIDIA RTX A6000', PrecisionMode.PRECISION_FP16)
            == pytest.approx(77.4, abs=1e-1))
    assert (mfu_tracker.get_flops_promised('NVIDIA RTX A6000', PrecisionMode.PRECISION_TF32)
            == pytest.approx(38.7, abs=1e-1))


def test_get_flops_promised_rtx_a40():
    assert (mfu_tracker.get_flops_promised('NVIDIA RTX A40', PrecisionMode.PRECISION_BF16)
            == pytest.approx(74.8, abs=1e-1))
    assert (mfu_tracker.get_flops_promised('NVIDIA RTX A40', PrecisionMode.PRECISION_FP16)
            == pytest.approx(74.8, abs=1e-1))
    assert (mfu_tracker.get_flops_promised('NVIDIA RTX A40', PrecisionMode.PRECISION_TF32)
            == pytest.approx(37.4, abs=1e-1))


def test_get_flops_promised_a100():
    assert mfu_tracker.get_flops_promised('NVIDIA A100-PCIE-80GB', PrecisionMode.PRECISION_BF16) \
           == pytest.approx(312, abs=1e-1)
    assert mfu_tracker.get_flops_promised('NVIDIA A100-PCIE-80GB', PrecisionMode.PRECISION_FP16) \
           == pytest.approx(312, abs=1e-1)
    assert mfu_tracker.get_flops_promised('NVIDIA A100-PCIE-80GB', PrecisionMode.PRECISION_TF32) \
           == pytest.approx(156, abs=1e-1)


def test_get_flops_promised_h100_sxm():
    assert mfu_tracker.get_flops_promised('NVIDIA H100 80GB HBM3', PrecisionMode.PRECISION_BF16) \
           == pytest.approx(1000, abs=1e-1)
    assert mfu_tracker.get_flops_promised('NVIDIA H100 80GB HBM3', PrecisionMode.PRECISION_FP16) \
           == pytest.approx(1000, abs=1e-1)
    assert mfu_tracker.get_flops_promised('NVIDIA H100 80GB HBM3', PrecisionMode.PRECISION_TF32) \
           == pytest.approx(500, abs=1e-1)


def test_get_flops_promised_h100_pcie__non_flagship():
    assert mfu_tracker.get_flops_promised('NVIDIA H100 PCIe', PrecisionMode.PRECISION_BF16) \
           == pytest.approx(800, abs=1e-1)
    assert mfu_tracker.get_flops_promised('NVIDIA H100 PCIe', PrecisionMode.PRECISION_FP16) \
           == pytest.approx(800, abs=1e-1)
    assert mfu_tracker.get_flops_promised('NVIDIA H100 PCIe', PrecisionMode.PRECISION_TF32) \
           == pytest.approx(400, abs=1e-1)


def test_get_flops_promised_rtx_5090():
    assert mfu_tracker.get_flops_promised('NVIDIA GeForce RTX 5090', PrecisionMode.PRECISION_BF16) \
           == pytest.approx(209.5, abs=1e-1)
    assert mfu_tracker.get_flops_promised('NVIDIA GeForce RTX 5090', PrecisionMode.PRECISION_FP16) \
           == pytest.approx(209.5, abs=1e-1)
    assert mfu_tracker.get_flops_promised('NVIDIA GeForce RTX 5090', PrecisionMode.PRECISION_TF32) \
           == pytest.approx(104.8, abs=1e-1)


def test_get_flops_promised_rtx_5080__non_flagship():
    assert mfu_tracker.get_flops_promised('NVIDIA GeForce RTX 5080', PrecisionMode.PRECISION_BF16) \
           == pytest.approx(112.6, abs=1e-1)
    assert mfu_tracker.get_flops_promised('NVIDIA GeForce RTX 5080', PrecisionMode.PRECISION_FP16) \
           == pytest.approx(112.6, abs=1e-1)
    assert mfu_tracker.get_flops_promised('NVIDIA GeForce RTX 5080', PrecisionMode.PRECISION_TF32) \
           == pytest.approx(56.3, abs=1e-1)


def test_get_flops_promised_rtx_5070_ti__non_flagship():
    assert mfu_tracker.get_flops_promised('NVIDIA GeForce RTX 5070 Ti', PrecisionMode.PRECISION_BF16) \
           == pytest.approx(87.9, abs=1e-1)
    assert mfu_tracker.get_flops_promised('NVIDIA GeForce RTX 5070 Ti', PrecisionMode.PRECISION_FP16) \
           == pytest.approx(87.9, abs=1e-1)
    assert mfu_tracker.get_flops_promised('NVIDIA GeForce RTX 5070 Ti', PrecisionMode.PRECISION_TF32) \
           == pytest.approx(43.9, abs=1e-1)


def test_get_flops_promised_rtx_5070__non_flagship():
    assert mfu_tracker.get_flops_promised('NVIDIA GeForce RTX 5070', PrecisionMode.PRECISION_BF16) \
           == pytest.approx(61.7, abs=1e-1)
    assert mfu_tracker.get_flops_promised('NVIDIA GeForce RTX 5070', PrecisionMode.PRECISION_FP16) \
           == pytest.approx(61.7, abs=1e-1)
    assert mfu_tracker.get_flops_promised('NVIDIA GeForce RTX 5070', PrecisionMode.PRECISION_TF32) \
           == pytest.approx(30.9, abs=1e-1)
