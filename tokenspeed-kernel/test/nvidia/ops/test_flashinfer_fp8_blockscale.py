"""FlashInfer FP8 block-scale prepared-layout and fused-padding tests."""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel import fp8_linear, mm, prepare_fp8_linear
from tokenspeed_kernel.ops.gemm.flashinfer import (
    gemm_fp8_nt_groupwise,
    has_flashinfer_fp8_blockscale,
    prepare_flashinfer_fp8_blockscale_weight_scales,
    use_flashinfer_fp8_blockscale_prepacked,
)
from tokenspeed_kernel.ops.gemm.fp8_utils import (
    flashinfer_fp8_blockscale_quantize_prepacked,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not has_flashinfer_fp8_blockscale(),
    reason="requires SM100 CUDA and FlashInfer FP8 block-scale GEMM",
)


@pytest.mark.parametrize("m", [1, 2, 3, 4, 5, 8])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_quantize_prepacked_writes_native_scales_and_padding(
    device: str, m: int, dtype: torch.dtype
) -> None:
    torch.manual_seed(0)
    k = 512
    x = torch.randn(m, k, device=device, dtype=dtype)

    q, scales = flashinfer_fp8_blockscale_quantize_prepacked(x)

    padded_m = (m + 3) // 4 * 4
    assert q.shape == (padded_m, k)
    assert scales.shape == (k // 128, padded_m)
    assert q.is_contiguous()
    assert scales.is_contiguous()

    expected_scales = (
        x.float().view(m, k // 128, 128).abs().amax(dim=-1).transpose(0, 1)
        / torch.finfo(torch.float8_e4m3fn).max
    )
    torch.testing.assert_close(scales[:, :m], expected_scales, atol=1e-6, rtol=1e-5)
    if padded_m != m:
        assert torch.count_nonzero(q[m:]).item() == 0
        assert torch.equal(scales[:, m:], torch.ones_like(scales[:, m:]))


@pytest.mark.parametrize("m", [1, 2, 3, 4, 5, 8])
def test_canonical_and_prepacked_gemm_match(device: str, m: int) -> None:
    torch.manual_seed(1)
    n, k = 256, 512
    x = torch.randn(m, k, device=device, dtype=torch.bfloat16)
    weight = (torch.randn(n, k, device=device) * 0.02).to(torch.float8_e4m3fn)
    weight_scales = (
        torch.rand(n // 128, k // 128, device=device, dtype=torch.float32) * 0.02
        + 0.001
    )

    quantized_x, activation_scales = flashinfer_fp8_blockscale_quantize_prepacked(x)
    prepared_weight_scales = prepare_flashinfer_fp8_blockscale_weight_scales(
        weight_scales
    )
    expected = gemm_fp8_nt_groupwise(
        quantized_x,
        weight,
        activation_scales,
        prepared_weight_scales,
        scale_major_mode="MN",
        out_dtype=torch.bfloat16,
    )[:m]
    canonical = mm(
        x,
        weight,
        B_scales=weight_scales,
        out_dtype=torch.bfloat16,
        quant="mxfp8",
        block_size=[128, 128],
        override="flashinfer_mm_fp8_blockscale",
    )
    prepacked = mm(
        x,
        weight,
        B_scales=prepared_weight_scales,
        out_dtype=torch.bfloat16,
        quant="mxfp8",
        block_size=[128, 128],
        override="flashinfer_mm_fp8_blockscale",
        prepacked_scales=True,
    )

    torch.testing.assert_close(prepacked, expected, atol=0, rtol=0)
    torch.testing.assert_close(canonical, prepacked, atol=5e-4, rtol=2e-3)


def test_prepacked_gemm_rejects_canonical_weight_scales(device: str) -> None:
    torch.manual_seed(2)
    m, n, k = 4, 256, 512
    x = torch.randn(m, k, device=device, dtype=torch.bfloat16)
    weight = (torch.randn(n, k, device=device) * 0.02).to(torch.float8_e4m3fn)
    canonical_scales = torch.rand(
        n // 128, k // 128, device=device, dtype=torch.float32
    )

    with pytest.raises(ValueError, match="prepacked weight scales"):
        mm(
            x,
            weight,
            B_scales=canonical_scales,
            out_dtype=torch.bfloat16,
            quant="mxfp8",
            block_size=[128, 128],
            override="flashinfer_mm_fp8_blockscale",
            prepacked_scales=True,
        )


def test_prepacked_gemm_is_cuda_graph_safe(device: str) -> None:
    torch.manual_seed(2)
    m, n, k = 1, 256, 512
    static_x = torch.randn(m, k, device=device, dtype=torch.bfloat16)
    weight = (torch.randn(n, k, device=device) * 0.02).to(torch.float8_e4m3fn)
    weight_scales = prepare_flashinfer_fp8_blockscale_weight_scales(
        torch.rand(n // 128, k // 128, device=device, dtype=torch.float32)
    )

    def run() -> torch.Tensor:
        return mm(
            static_x,
            weight,
            B_scales=weight_scales,
            out_dtype=torch.bfloat16,
            quant="mxfp8",
            block_size=[128, 128],
            override="flashinfer_mm_fp8_blockscale",
            prepacked_scales=True,
        )

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run()

    static_x.copy_(torch.randn_like(static_x))
    graph.replay()
    expected = run()
    torch.testing.assert_close(captured, expected, atol=0, rtol=0)


@pytest.mark.parametrize(
    "num_tokens, expected",
    [
        (1, True),
        (3, True),
        (4, True),
        (64, True),
        (256, True),
        (257, False),
        (260, True),
        (4097, False),
    ],
)
def test_prepacked_selection_threshold(num_tokens: int, expected: bool) -> None:
    assert use_flashinfer_fp8_blockscale_prepacked(num_tokens) is expected


@pytest.mark.parametrize("m", [1, 4, 8, 64])
def test_prepared_plan_takes_the_prepacked_path(device: str, m: int) -> None:
    torch.manual_seed(3)
    n, k = 256, 512
    x = torch.randn(m, k, device=device, dtype=torch.bfloat16)
    weight = (torch.randn(n, k, device=device) * 0.02).to(torch.float8_e4m3fn)
    weight_scales = (
        torch.rand(n // 128, k // 128, device=device, dtype=torch.float32) * 0.02
        + 0.001
    )

    plan = prepare_fp8_linear(weight, weight_scales, [128, 128])
    planned = fp8_linear(plan, x, weight, weight_scales, out_dtype=torch.bfloat16)
    prepacked = mm(
        x,
        weight,
        B_scales=prepare_flashinfer_fp8_blockscale_weight_scales(weight_scales),
        out_dtype=torch.bfloat16,
        quant="mxfp8",
        block_size=[128, 128],
        override="flashinfer_mm_fp8_blockscale",
        prepacked_scales=True,
    )
    torch.testing.assert_close(planned, prepacked, atol=0, rtol=0)


def test_prepared_plan_falls_back_above_the_padding_threshold(device: str) -> None:
    torch.manual_seed(4)
    m, n, k = 257, 256, 512
    x = torch.randn(m, k, device=device, dtype=torch.bfloat16)
    weight = (torch.randn(n, k, device=device) * 0.02).to(torch.float8_e4m3fn)
    weight_scales = (
        torch.rand(n // 128, k // 128, device=device, dtype=torch.float32) * 0.02
        + 0.001
    )

    plan = prepare_fp8_linear(weight, weight_scales, [128, 128])
    planned = fp8_linear(plan, x, weight, weight_scales, out_dtype=torch.bfloat16)
    canonical = mm(
        x,
        weight,
        B_scales=weight_scales,
        out_dtype=torch.bfloat16,
        quant="mxfp8",
        block_size=[128, 128],
        override="flashinfer_mm_fp8_blockscale",
    )
    torch.testing.assert_close(planned, canonical, atol=0, rtol=0)


@pytest.mark.parametrize("m", [16, 17, 24, 31, 32, 33])
def test_prepared_plan_is_exact_for_partial_row_tiles(device: str, m: int) -> None:
    """FlashInfer's K-major scale mode mis-reads activation scales for
    17 <= M <= 32 on SM10x; a prepared layer must not route through it."""
    torch.manual_seed(5)
    n, k = 2048, 512
    x = torch.randn(m, k, device=device, dtype=torch.bfloat16)
    weight = (torch.randn(n, k, device=device) / 8).to(torch.float8_e4m3fn)
    weight_scales = (
        torch.rand(n // 128, k // 128, device=device, dtype=torch.float32) + 0.5
    )

    plan = prepare_fp8_linear(weight, weight_scales, [128, 128])
    got = fp8_linear(plan, x, weight, weight_scales, out_dtype=torch.bfloat16)

    # Compare against the exact product of the quantized operands.
    quantized_x, activation_scales = flashinfer_fp8_blockscale_quantize_prepacked(x)
    activation = quantized_x[:m].float() * activation_scales[:, :m].transpose(
        0, 1
    ).repeat_interleave(128, dim=1)
    dequantized = weight.float() * weight_scales.repeat_interleave(
        128, dim=0
    ).repeat_interleave(128, dim=1)
    reference = activation @ dequantized.t()

    error = (got.float() - reference).abs().max()
    assert error < 0.01 * reference.abs().max()
