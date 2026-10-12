from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.ops.gemm import _online_quantize_mxfp8, mm
from tokenspeed_kernel.ops.quantization import quantize_fp8


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
        (130, 300, (96, 128)),
        (65, 160, (32, 64)),
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_fp8_block_quantization_roundtrip(
    device: str, n: int, k: int, block_size: tuple[int, int], dtype: torch.dtype
) -> None:
    torch.manual_seed(0)
    x = torch.randn(n, k, device=device, dtype=dtype) * 0.05

    q, scales = quantize_fp8(x, granularity="block", block_size=block_size)

    block_n, block_k = block_size
    assert q.shape == x.shape
    assert q.dtype == torch.float8_e4m3fn
    assert scales.dtype == torch.float32
    assert scales.shape == (
        (n + block_n - 1) // block_n,
        (k + block_k - 1) // block_k,
    )

    dequantized = _dequantize(q, scales, block_size)
    # Round-to-nearest E4M3 (3 mantissa bits) is off by at most half an ulp:
    # 2^-4 relative for normal values, 2^-10 * scale for subnormal ones.
    torch.testing.assert_close(
        dequantized,
        x.float(),
        atol=scales.max().item() * 2**-10,
        rtol=2**-4,
    )


def test_fp8_block_quantization_scale_is_per_block(device: str) -> None:
    """Each block must be scaled independently, so a huge outlier in one block
    must not degrade the resolution of its neighbours."""
    torch.manual_seed(0)
    block_size = (128, 128)
    x = torch.full((128, 256), 0.01, device=device, dtype=torch.bfloat16)
    x[0, 0] = 1000.0

    q, scales = quantize_fp8(x, granularity="block", block_size=block_size)

    assert scales[0, 0] > scales[0, 1] * 100
    dequantized = _dequantize(q, scales, block_size)
    torch.testing.assert_close(
        dequantized[:, 128:], x[:, 128:].float(), atol=1e-4, rtol=1e-2
    )


def test_fp8_block_quantization_rejects_non_2d(device: str) -> None:
    x = torch.randn(4, 8, 16, device=device, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="2-D tensor"):
        quantize_fp8(x, granularity="block", block_size=(128, 128))


def test_fp8_block_quantization_feeds_block_scaled_gemm(device: str) -> None:
    """The produced weights must be consumable by the block-scaled FP8 GEMM."""
    torch.manual_seed(0)
    m, n, k = 16, 256, 256
    block_size = (128, 128)
    a = torch.randn(m, k, device=device, dtype=torch.bfloat16) * 0.1
    weight = torch.randn(n, k, device=device, dtype=torch.bfloat16) * 0.1

    q_weight, weight_scales = quantize_fp8(
        weight, granularity="block", block_size=block_size
    )
    out = mm(
        a,
        q_weight,
        B_scales=weight_scales,
        out_dtype=torch.bfloat16,
        quant="mxfp8",
        block_size=list(block_size),
        override="triton_mm_fp8_blockscale",
    )

    # Compare against the activation as mm() quantized it online.
    q_a, a_scales = _online_quantize_mxfp8(
        a, list(block_size), "float32", enable_pdl=False
    )
    activation = q_a.float() * a_scales.repeat_interleave(block_size[1], dim=1)[:, :k]
    reference = activation @ _dequantize(q_weight, weight_scales, block_size).t()
    torch.testing.assert_close(out.float(), reference, atol=1e-3, rtol=5e-3)


def test_fp8_block_quantization_reuses_compilation(device: str) -> None:
    from tokenspeed_kernel.ops.quantization.triton import _fp8_block_quantize_kernel
    from utils import assert_no_triton_compile

    def run(rows: int) -> None:
        quantize_fp8(
            torch.zeros((rows, 256), device=device, dtype=torch.bfloat16),
            granularity="block",
            block_size=(128, 128),
        )

    for rows in (1, 16, 17):
        run(rows)
    with assert_no_triton_compile(_fp8_block_quantize_kernel):
        for rows in (2, 3, 31, 32, 33, 64, 65, 127, 128, 129):
            run(rows)
