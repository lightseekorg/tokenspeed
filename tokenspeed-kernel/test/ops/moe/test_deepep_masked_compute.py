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

"""The DeepEP low-latency expert compute: two masked grouped GEMMs around the
masked UE8M0 activation quantizer. Runs the padded ``[experts, capacity, ...]``
layouts the low-latency apply path builds, minus the all-to-all legs, against a
dequantized torch reference.

Scales are powers of two throughout, because on sm100+ DeepGEMM's only FP8
kernel reads scales as UE8M0 and silently misreads anything else.
"""

from __future__ import annotations

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

deep_gemm = pytest.importorskip(
    "tokenspeed_kernel.thirdparty.deep_gemm",
    reason="DeepGEMM is an optional dependency",
)

from tokenspeed_kernel.ops.activation.triton import (  # noqa: E402
    fused_swiglu_fp8_ue8m0_masked,
)
from tokenspeed_kernel.ops.moe.deep_gemm.ue8m0 import is_ue8m0  # noqa: E402

_BLOCK = 128


def _quantize_masked(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row 1x128 block FP8 quantize with UE8M0 scales, [E, M, K] layout."""
    experts, rows, cols = x.shape
    blocks = x.view(experts, rows, cols // _BLOCK, _BLOCK).float()
    amax = blocks.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)
    scales = deep_gemm.ceil_to_ue8m0(amax / 448.0)
    quantized = (blocks / scales).to(torch.float8_e4m3fn).view(experts, rows, cols)
    return quantized, scales.squeeze(-1).contiguous()


def _quantize_weight(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-expert 128x128 block FP8 quantize with UE8M0 scales."""
    experts, n, k = w.shape
    blocks = w.view(experts, n // _BLOCK, _BLOCK, k // _BLOCK, _BLOCK).float()
    amax = blocks.abs().amax(dim=(2, 4), keepdim=True).clamp(min=1e-6)
    scales = deep_gemm.ceil_to_ue8m0(amax / 448.0)
    quantized = (blocks / scales).to(torch.float8_e4m3fn).view(experts, n, k)
    return quantized, scales.view(experts, n // _BLOCK, k // _BLOCK).contiguous()


def _dequantize(q: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    return q.float() * scales.repeat_interleave(_BLOCK, dim=-1)


def _dequantize_weight(q: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    expanded = scales.repeat_interleave(_BLOCK, dim=1).repeat_interleave(_BLOCK, dim=2)
    return q.float() * expanded


def test_masked_swiglu_ue8m0_quantizer_matches_reference_and_skips_padding():
    """The activation between the two masked GEMMs.

    The apply path hands the kernel an ``[E, blocks, M]`` scale buffer viewed
    as ``[E, M, blocks]`` (mn-major), and the padded rows beyond ``masked_m``
    must stay untouched so the zero-initialized scales mark them dead.
    """
    torch.manual_seed(0)
    experts, capacity, ispp = 4, 64, 256
    gateup = torch.randn(
        experts, capacity, 2 * ispp, device="cuda", dtype=torch.bfloat16
    )
    masked_m = torch.tensor([64, 17, 0, 33], dtype=torch.int32, device="cuda")

    out = torch.empty(
        (experts, capacity, ispp), dtype=torch.float8_e4m3fn, device="cuda"
    )
    out_probe = out.view(torch.uint8)
    out_probe.fill_(0)
    scales = torch.zeros(
        (experts, ispp // _BLOCK, capacity), dtype=torch.float32, device="cuda"
    ).permute(0, 2, 1)

    fused_swiglu_fp8_ue8m0_masked(gateup, masked_m, out, scales)

    assert scales.shape == (experts, capacity, ispp // _BLOCK)
    assert is_ue8m0(scales)

    reference = (
        torch.nn.functional.silu(gateup[..., :ispp].float())
        * gateup[..., ispp:].float()
    )
    for expert in range(experts):
        valid = int(masked_m[expert])
        if valid:
            torch.testing.assert_close(
                _dequantize(out[expert, :valid], scales[expert, :valid]),
                reference[expert, :valid],
                rtol=6e-2,
                atol=6e-2 * reference.abs().max(),
            )
        # Padding rows were never written: FP8 payload still zero.
        assert (out_probe[expert, valid:] == 0).all()
        assert (scales[expert, valid:] == 0).all()


def test_low_latency_expert_compute_matches_dequantized_reference():
    torch.manual_seed(0)
    experts, capacity, hidden, ispp = 4, 128, 512, 256
    device = "cuda"

    recv_x, recv_scales = _quantize_masked(
        torch.randn(experts, capacity, hidden, device=device)
    )
    masked_m = torch.tensor([128, 96, 1, 0], dtype=torch.int32, device=device)
    expected_m = 64

    w13, w13_scales = _quantize_weight(
        torch.randn(experts, 2 * ispp, hidden, device=device) * 0.1
    )
    w2, w2_scales = _quantize_weight(
        torch.randn(experts, hidden, ispp, device=device) * 0.1
    )

    gateup = torch.empty(
        (experts, capacity, 2 * ispp), dtype=torch.bfloat16, device=device
    )
    deep_gemm.m_grouped_fp8_gemm_nt_masked(
        (recv_x, deep_gemm.get_mn_major_tma_aligned_tensor(recv_scales)),
        (w13, w13_scales),
        gateup,
        masked_m,
        expected_m,
    )

    down_in = torch.empty(
        (experts, capacity, ispp), dtype=torch.float8_e4m3fn, device=device
    )
    down_scales = torch.zeros(
        (experts, ispp // _BLOCK, capacity), dtype=torch.float32, device=device
    ).permute(0, 2, 1)
    fused_swiglu_fp8_ue8m0_masked(gateup, masked_m, down_in, down_scales)

    out = torch.empty((experts, capacity, hidden), dtype=torch.bfloat16, device=device)
    deep_gemm.m_grouped_fp8_gemm_nt_masked(
        (down_in, deep_gemm.get_mn_major_tma_aligned_tensor(down_scales)),
        (w2, w2_scales),
        out,
        masked_m,
        expected_m,
    )

    # Reference on the kernels' own operands so only the masked layout wiring
    # is under test, not quantization noise.
    x_ref = _dequantize(recv_x, recv_scales)
    w13_ref = _dequantize_weight(w13, w13_scales)
    w2_ref = _dequantize_weight(w2, w2_scales)
    for expert in range(experts):
        valid = int(masked_m[expert])
        if not valid:
            continue
        gateup_ref = x_ref[expert, :valid] @ w13_ref[expert].t()
        torch.testing.assert_close(
            gateup[expert, :valid].float(),
            gateup_ref,
            rtol=2e-2,
            atol=2e-2 * gateup_ref.abs().max(),
        )
        down_ref = _dequantize(down_in, down_scales)[expert, :valid]
        reference = down_ref @ w2_ref[expert].t()
        torch.testing.assert_close(
            out[expert, :valid].float(),
            reference,
            rtol=2e-2,
            atol=2e-2 * reference.abs().max(),
        )
