# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Weight-only per-block FP8 (W8A16) triton GEMM correctness.

The kernel must match a bf16 GEMM over the bf16-dequantized weight: with
weight-only semantics the two are mathematically identical, so any deviation
beyond fp32-accumulator tiling order (the same rounding a plain bf16 GEMM
exhibits) is a bug — in particular there must be no activation-quantization
staircase.
"""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.platform import current_platform

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not current_platform().is_nvidia,
    reason="FP8 weight-only GEMM targets NVIDIA GPUs.",
)

_BLOCK = [128, 128]
_FP8_MAX = torch.finfo(torch.float8_e4m3fn).max


def _quantize_per_block(
    w: torch.Tensor, block_n: int, block_k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-[block_n, block_k] amax quantization to FP8 with f32 dequant scales."""
    n, k = w.shape
    nb = (n + block_n - 1) // block_n
    kb = (k + block_k - 1) // block_k
    padded = torch.zeros(
        nb * block_n, kb * block_k, dtype=torch.float32, device=w.device
    )
    padded[:n, :k] = w.float()
    blocks = padded.view(nb, block_n, kb, block_k)
    amax = blocks.abs().amax(dim=(1, 3)).clamp(min=1e-12)
    scales = (amax / _FP8_MAX).contiguous()
    q = (blocks / scales[:, None, :, None]).clamp(-_FP8_MAX, _FP8_MAX)
    q = q.view(nb * block_n, kb * block_k)[:n, :k].to(torch.float8_e4m3fn)
    return q.contiguous(), scales


def _dequant_bf16(
    q: torch.Tensor, scales: torch.Tensor, block_n: int, block_k: int
) -> torch.Tensor:
    n, k = q.shape
    s_full = scales.repeat_interleave(block_n, dim=0)[:n].repeat_interleave(
        block_k, dim=1
    )[:, :k]
    return (q.float() * s_full).to(torch.bfloat16)


@pytest.mark.parametrize(
    "m,n,k",
    [
        (1, 512, 256),  # decode-shaped
        (7, 96, 384),  # ragged N (< one block)
        (33, 384, 200),  # ragged K
        (128, 256, 640),
    ],
)
def test_matches_bf16_dequant_reference(m: int, n: int, k: int) -> None:
    from tokenspeed_kernel.ops.gemm.triton import wo_block_fp8_matmul_triton

    torch.manual_seed(0)
    device = "cuda"
    x = torch.randn(m, k, device=device, dtype=torch.bfloat16)
    # Vary magnitudes across blocks so per-block scales actually differ.
    w = torch.randn(n, k, device=device, dtype=torch.float32)
    w *= torch.logspace(-2, 1, steps=n, device=device)[:, None]
    q, scales = _quantize_per_block(w, *_BLOCK)

    out = wo_block_fp8_matmul_triton(x, q, scales, block_size=_BLOCK)
    torch.cuda.synchronize()
    assert out.shape == (m, n) and out.dtype == torch.bfloat16

    w_dq = _dequant_bf16(q, scales, *_BLOCK)
    ref32 = x.float() @ w_dq.float().t()
    ref_bf16 = x @ w_dq.t()

    # Both the kernel and cublas bf16 GEMM consume identical bf16 operands and
    # accumulate in fp32; only tiling order may differ. Gate the kernel error
    # (vs the exact fp32 product) by the bf16 GEMM's own rounding.
    scale_ref = ref32.abs().amax().clamp(min=1.0)
    err_kernel = (out.float() - ref32).abs().amax() / scale_ref
    err_torch = (ref_bf16.float() - ref32).abs().amax() / scale_ref
    assert err_kernel <= 4 * err_torch + 1e-5, (
        f"kernel deviates beyond bf16 GEMM rounding: "
        f"{err_kernel=:.3e} {err_torch=:.3e}"
    )


def test_no_activation_quantization_staircase() -> None:
    """Mixed-magnitude activations survive: w8a8 group quant would crush them."""
    from tokenspeed_kernel.ops.gemm.triton import wo_block_fp8_matmul_triton

    torch.manual_seed(1)
    device = "cuda"
    m, n, k = 4, 128, 256
    x = torch.randn(m, k, device=device, dtype=torch.bfloat16)
    # One huge column per 128-group: per-token-group fp8 quantization of x
    # would collapse the remaining ~1e-4 entries to zero.
    x *= 1e-4
    x[:, ::128] = 500.0
    w = torch.randn(n, k, device=device, dtype=torch.float32)
    q, scales = _quantize_per_block(w, *_BLOCK)

    out = wo_block_fp8_matmul_triton(x, q, scales, block_size=_BLOCK)
    torch.cuda.synchronize()
    w_dq = _dequant_bf16(q, scales, *_BLOCK)
    ref32 = x.float() @ w_dq.float().t()
    ref_bf16 = x @ w_dq.t()
    scale_ref = ref32.abs().amax().clamp(min=1.0)
    err_kernel = (out.float() - ref32).abs().amax() / scale_ref
    err_torch = (ref_bf16.float() - ref32).abs().amax() / scale_ref
    assert err_kernel <= 4 * err_torch + 1e-5


def test_registered_wrapper_and_batched_input() -> None:
    from tokenspeed_kernel.ops.gemm.triton import (
        triton_mm_fp8_weight_only_blockscale,
        wo_block_fp8_matmul_triton,
    )

    torch.manual_seed(2)
    device = "cuda"
    b, s, n, k = 2, 3, 256, 384
    x = torch.randn(b, s, k, device=device, dtype=torch.bfloat16)
    w = torch.randn(n, k, device=device, dtype=torch.float32)
    q, scales = _quantize_per_block(w, *_BLOCK)

    out3d = wo_block_fp8_matmul_triton(x, q, scales, block_size=_BLOCK)
    assert out3d.shape == (b, s, n)
    out_wrapper = triton_mm_fp8_weight_only_blockscale(
        x.view(-1, k), q, None, scales, torch.bfloat16, block_size=_BLOCK
    )
    torch.cuda.synchronize()
    assert torch.equal(out3d.view(-1, n), out_wrapper)

    with pytest.raises(AssertionError, match="unquantized activations"):
        triton_mm_fp8_weight_only_blockscale(
            x.view(-1, k), q, scales, scales, torch.bfloat16, block_size=_BLOCK
        )
