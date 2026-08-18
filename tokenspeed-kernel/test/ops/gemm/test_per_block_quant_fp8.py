from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel import mm
from tokenspeed_kernel.ops.gemm.fp8_utils import per_block_quant_fp8


def _dequantize(
    q: torch.Tensor, scales: torch.Tensor, block_size: tuple[int, int]
) -> torch.Tensor:
    block_n, block_k = block_size
    n, k = q.shape
    out = q.float()
    for i in range(scales.shape[0]):
        for j in range(scales.shape[1]):
            rows = slice(i * block_n, min((i + 1) * block_n, n))
            cols = slice(j * block_k, min((j + 1) * block_k, k))
            out[rows, cols] *= scales[i, j]
    return out


@pytest.mark.parametrize(
    "n, k, block_size",
    [
        (256, 256, (128, 128)),
        (6144, 2048, (128, 128)),
        # Dimensions that are not a multiple of the block shape.
        (130, 300, (128, 128)),
        # Block shape that is not a power of two.
        (192, 256, (96, 128)),
    ],
)
def test_per_block_quant_fp8_roundtrip(
    device: str, n: int, k: int, block_size: tuple[int, int]
) -> None:
    torch.manual_seed(0)
    x = torch.randn(n, k, device=device, dtype=torch.bfloat16) * 0.05

    q, scales = per_block_quant_fp8(x, block_size)

    block_n, block_k = block_size
    assert q.shape == x.shape
    assert q.dtype == torch.float8_e4m3fn
    assert scales.dtype == torch.float32
    assert scales.shape == (
        (n + block_n - 1) // block_n,
        (k + block_k - 1) // block_k,
    )

    dequantized = _dequantize(q, scales, block_size)
    cosine = torch.nn.functional.cosine_similarity(
        dequantized.flatten(), x.float().flatten(), dim=0
    )
    assert cosine > 0.99


def test_per_block_quant_fp8_scale_is_per_block(device: str) -> None:
    """Each block must be scaled independently, so a huge outlier in one block
    must not degrade the resolution of its neighbours."""
    torch.manual_seed(0)
    block_size = (128, 128)
    x = torch.full((128, 256), 0.01, device=device, dtype=torch.bfloat16)
    x[0, 0] = 1000.0

    q, scales = per_block_quant_fp8(x, block_size)

    assert scales[0, 0] > scales[0, 1] * 100
    dequantized = _dequantize(q, scales, block_size)
    torch.testing.assert_close(
        dequantized[:, 128:], x[:, 128:].float(), atol=1e-4, rtol=1e-2
    )


def test_per_block_quant_fp8_rejects_non_2d(device: str) -> None:
    x = torch.randn(4, 8, 16, device=device, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="2D tensor"):
        per_block_quant_fp8(x)


def test_per_block_quant_fp8_feeds_block_scaled_gemm(device: str) -> None:
    """The produced weights must be consumable by the block-scaled FP8 GEMM."""
    torch.manual_seed(0)
    m, n, k = 16, 256, 256
    block_size = (128, 128)
    a = torch.randn(m, k, device=device, dtype=torch.bfloat16) * 0.1
    weight = torch.randn(n, k, device=device, dtype=torch.bfloat16) * 0.1

    q_weight, weight_scales = per_block_quant_fp8(weight, block_size)
    out = mm(
        a,
        q_weight,
        B_scales=weight_scales,
        out_dtype=torch.bfloat16,
        quant="mxfp8",
        block_size=list(block_size),
        override="triton_mm_fp8_blockscale",
    )

    reference = a.float() @ _dequantize(q_weight, weight_scales, block_size).t()
    torch.testing.assert_close(out.float(), reference, atol=0.05, rtol=0.05)
