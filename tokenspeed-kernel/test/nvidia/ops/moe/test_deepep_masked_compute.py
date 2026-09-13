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
    fused_swiglu_fp8_ue8m0_masked_packed,
)
from tokenspeed_kernel.thirdparty.deep_gemm.utils.layout import (  # noqa: E402
    get_mn_major_tma_aligned_packed_ue8m0_tensor,
)

_BLOCK = 128


@pytest.fixture(params=[False, True], ids=["pdl_off", "pdl_on"])
def enable_pdl(request):
    """Run the full DG1 -> activation -> DG2 chain in both launch modes."""
    if request.param and torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("PDL requires SM90+")
    previous = deep_gemm.get_pdl()
    deep_gemm.set_pdl(request.param)
    try:
        yield request.param
    finally:
        deep_gemm.set_pdl(previous)


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


def _pack_weight_scales(scales: torch.Tensor, n: int, k: int) -> torch.Tensor:
    """Expand 128x128 B scales once into DeepGEMM's packed per-N layout."""
    return deep_gemm.transform_sf_into_required_layout(
        sf=scales,
        mn=n,
        k=k,
        recipe=(1, _BLOCK, _BLOCK),
        num_groups=scales.shape[0],
        is_sfa=False,
    )


def _dequantize(q: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    return q.float() * scales.repeat_interleave(_BLOCK, dim=-1)


def _dequantize_weight(q: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    expanded = scales.repeat_interleave(_BLOCK, dim=1).repeat_interleave(_BLOCK, dim=2)
    return q.float() * expanded


def _unpack_ue8m0(packed: torch.Tensor, num_groups: int) -> torch.Tensor:
    bytes_ = torch.stack(
        [(packed >> (8 * i)) & 0xFF for i in range(4)], dim=-1
    ).flatten(-2)[..., :num_groups]
    return torch.exp2(bytes_.float() - 127.0)


def test_masked_swiglu_writes_packed_mn_major_scales_for_deep_gemm():
    """The decode kernel must need no clear or post-quant scale transform."""
    torch.manual_seed(0)
    experts, capacity, ispp = 3, 17, 640
    gateup = torch.randn(
        experts, capacity, 2 * ispp, device="cuda", dtype=torch.bfloat16
    )
    masked_m = torch.tensor([17, 5, 0], dtype=torch.int32, device="cuda")

    out, packed_scales = fused_swiglu_fp8_ue8m0_masked_packed(
        gateup, masked_m, expected_m=1
    )

    num_groups = ispp // _BLOCK
    scales = _unpack_ue8m0(packed_scales, num_groups)
    assert out.shape == (experts, capacity, ispp)
    assert packed_scales.shape == (experts, capacity, 2)
    assert packed_scales.dtype == torch.int32
    # MN-major: adjacent rows are adjacent in memory, while packed scale
    # columns stride over the aligned row extent.
    assert packed_scales.stride(1) == 1
    assert packed_scales.stride(2) % 4 == 0
    # N=640 has five groups, so bytes 1..3 in the tail word are padding.
    assert bool(((packed_scales[:2, :5, 1] >> 8) == 0).all())

    reference = (
        torch.nn.functional.silu(gateup[..., :ispp].float())
        * gateup[..., ispp:].float()
    )
    for expert in range(experts):
        valid = int(masked_m[expert])
        if not valid:
            continue
        dequantized = (
            out[expert, :valid].float().view(valid, num_groups, _BLOCK)
            * scales[expert, :valid, :, None]
        ).view(valid, ispp)
        torch.testing.assert_close(
            dequantized,
            reference[expert, :valid],
            rtol=6e-2,
            atol=6e-2 * reference[expert, :valid].abs().max(),
        )


def test_sparse_and_full_launch_mappings_match_exactly():
    torch.manual_seed(1)
    experts, capacity, ispp = 3, 33, 640
    gateup = torch.randn(
        experts, capacity, 2 * ispp, device="cuda", dtype=torch.bfloat16
    )
    masked_m = torch.tensor([33, 7, 0], dtype=torch.int32, device="cuda")

    sparse_out, sparse_scales = fused_swiglu_fp8_ue8m0_masked_packed(
        gateup, masked_m, expected_m=1
    )
    full_out, full_scales = fused_swiglu_fp8_ue8m0_masked_packed(
        gateup, masked_m, expected_m=capacity
    )

    for expert in range(experts):
        valid = int(masked_m[expert])
        assert torch.equal(sparse_out[expert, :valid], full_out[expert, :valid])
        assert torch.equal(sparse_scales[expert, :valid], full_scales[expert, :valid])


def test_low_latency_expert_compute_matches_dequantized_reference(enable_pdl):
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
        (
            recv_x,
            get_mn_major_tma_aligned_packed_ue8m0_tensor(recv_scales),
        ),
        (w13, _pack_weight_scales(w13_scales, 2 * ispp, hidden)),
        gateup,
        masked_m,
        expected_m,
        recipe=(1, 1, _BLOCK),
    )

    down_in, down_scales = fused_swiglu_fp8_ue8m0_masked_packed(
        gateup,
        masked_m,
        expected_m=expected_m,
        enable_pdl=enable_pdl,
    )

    out = torch.empty((experts, capacity, hidden), dtype=torch.bfloat16, device=device)
    deep_gemm.m_grouped_fp8_gemm_nt_masked(
        (down_in, down_scales),
        (w2, _pack_weight_scales(w2_scales, hidden, ispp)),
        out,
        masked_m,
        expected_m,
        recipe=(1, 1, _BLOCK),
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
        down_ref = (
            down_in[expert, :valid].float().view(valid, ispp // _BLOCK, _BLOCK)
            * _unpack_ue8m0(down_scales[expert, :valid], ispp // _BLOCK)[..., None]
        ).view(valid, ispp)
        reference = down_ref @ w2_ref[expert].t()
        torch.testing.assert_close(
            out[expert, :valid].float(),
            reference,
            rtol=2e-2,
            atol=2e-2 * reference.abs().max(),
        )
