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

"""The DeepEP normal-mode expert compute: permute, two contiguous grouped
GEMMs, unpermute. Runs the same tensor layouts the DeepEP FP8 apply kernel
builds, minus the all-to-all legs, against a dequantized torch reference.

Scales are rounded to UE8M0 (powers of two) because that is the only 1D1D scale
format DeepGEMM accepts on Blackwell; on Hopper it merely costs a little
precision, so one test body covers both.
"""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.ops.activation.triton import fused_swiglu_fp8_ue8m0
from tokenspeed_kernel.ops.moe.triton.deepep_permute import (
    deepep_gather,
    deepep_scatter,
)

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

deep_gemm = pytest.importorskip(
    "tokenspeed_kernel.thirdparty.deep_gemm",
    reason="DeepGEMM is an optional dependency",
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


def _quantize_blockwise(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row 1x128 block FP8 quantize with UE8M0 scales."""
    rows, cols = x.shape
    blocks = x.view(rows, cols // _BLOCK, _BLOCK).float()
    amax = blocks.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)
    scales = deep_gemm.ceil_to_ue8m0(amax / 448.0)
    quantized = (blocks / scales).to(torch.float8_e4m3fn).view(rows, cols)
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


def _dequantize_packed(q: torch.Tensor, packed: torch.Tensor) -> torch.Tensor:
    num_groups = q.shape[1] // _BLOCK
    bytes_ = torch.stack(
        [(packed >> (8 * i)) & 0xFF for i in range(4)], dim=-1
    ).flatten(-2)[:, :num_groups]
    scales = torch.exp2(bytes_.float() - 127.0)
    return q.float() * scales.repeat_interleave(_BLOCK, dim=1)


def test_fused_silu_block_quant_fills_mn_major_scales():
    """The 2-D activation between the two grouped GEMMs.

    The apply kernel hands this kernel a ``[blocks, rows]`` scale buffer viewed
    as ``[rows, blocks]``, which is both what the kernel requires (mn-major) and
    what DeepGEMM wants for the second GEMM. Getting the view backwards would
    scale the wrong rows.
    """
    from tokenspeed_kernel.thirdparty.cuda import silu_and_mul_fuse_block_quant

    torch.manual_seed(0)
    rows, ispp = 384, 256
    gateup = torch.randn(rows, 2 * ispp, device="cuda", dtype=torch.bfloat16)
    scales = torch.zeros(
        (ispp // _BLOCK, rows), dtype=torch.float32, device="cuda"
    ).permute(1, 0)
    out = torch.empty((rows, ispp), dtype=torch.float8_e4m3fn, device="cuda")
    silu_and_mul_fuse_block_quant(gateup, scales, out)

    assert scales.shape == (rows, ispp // _BLOCK)
    assert scales.stride() == (1, rows)
    reference = (
        torch.nn.functional.silu(gateup[:, :ispp].float()) * gateup[:, ispp:].float()
    )
    torch.testing.assert_close(
        _dequantize(out, scales),
        reference,
        rtol=6e-2,
        atol=6e-2 * reference.abs().max(),
    )


def test_normal_mode_expert_compute_matches_dequantized_reference(enable_pdl):
    torch.manual_seed(0)
    num_recv, hidden, ispp, top_k, num_local_experts = 96, 512, 256, 4, 4
    device = "cuda"

    recv_x, recv_scales = _quantize_blockwise(
        torch.randn(num_recv, hidden, device=device)
    )
    topk_ids = torch.full((num_recv, top_k), -1, dtype=torch.int32, device=device)
    # Give every token two local experts and leave the other slots remote, the
    # shape dispatch produces when a rank owns a slice of the expert set.
    for token in range(num_recv):
        topk_ids[token, 0] = token % num_local_experts
        topk_ids[token, 2] = (token + 1) % num_local_experts
    topk_weights = torch.rand(num_recv, top_k, device=device)
    counts = []
    for expert in range(num_local_experts):
        real = int((topk_ids == expert).sum())
        counts.append((real + _BLOCK - 1) // _BLOCK * _BLOCK)

    w13, w13_scales = _quantize_weight(
        torch.randn(num_local_experts, 2 * ispp, hidden, device=device) * 0.1
    )
    w2, w2_scales = _quantize_weight(
        torch.randn(num_local_experts, hidden, ispp, device=device) * 0.1
    )

    gemm_x, gemm_scales, m_indices, dest_index = deepep_scatter(
        recv_x,
        recv_scales,
        topk_ids,
        counts,
        expert_alignment=_BLOCK,
        pack_ue8m0_scales=True,
    )
    total_rows = gemm_x.shape[0]

    gateup = torch.empty((total_rows, 2 * ispp), dtype=torch.bfloat16, device=device)
    deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
        (gemm_x, gemm_scales),
        (w13, _pack_weight_scales(w13_scales, 2 * ispp, hidden)),
        gateup,
        m_indices,
        recipe=(1, 1, _BLOCK),
    )

    down_in, down_scales = fused_swiglu_fp8_ue8m0(gateup, enable_pdl=enable_pdl)

    expert_out = torch.empty((total_rows, hidden), dtype=torch.bfloat16, device=device)
    deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
        (down_in, down_scales),
        (w2, _pack_weight_scales(w2_scales, hidden, ispp)),
        expert_out,
        m_indices,
        recipe=(1, 1, _BLOCK),
    )

    got = deepep_gather(expert_out, topk_ids, topk_weights, dest_index)

    # Reference, stage by stage on the kernels' own operands so only the grouped
    # GEMM layout wiring is under test: a mis-tagged row or a mis-strided scale
    # block produces garbage, not quantization-level noise.
    x_ref = _dequantize(recv_x, recv_scales)
    w13_ref = _dequantize_weight(w13, w13_scales)
    w2_ref = _dequantize_weight(w2, w2_scales)
    row_to_token = {}
    for token in range(num_recv):
        for slot in range(top_k):
            if int(topk_ids[token, slot]) >= 0:
                row_to_token[int(dest_index[token, slot])] = token

    rows = sorted(row_to_token)
    tokens = [row_to_token[row] for row in rows]
    experts = m_indices[rows].long()
    gateup_ref = torch.einsum("rk,rnk->rn", x_ref[tokens], w13_ref[experts]).to(
        torch.bfloat16
    )
    torch.testing.assert_close(
        gateup[rows].float(),
        gateup_ref.float(),
        rtol=2e-2,
        atol=2e-2 * gateup_ref.float().abs().max(),
    )

    down_ref = torch.einsum(
        "rk,rnk->rn",
        _dequantize_packed(down_in, down_scales)[rows],
        w2_ref[experts],
    )
    reference = torch.zeros(num_recv, hidden, device=device, dtype=torch.float32)
    for row, token in row_to_token.items():
        slot = int((dest_index[token] == row).nonzero()[0])
        reference[token] += down_ref[rows.index(row)] * topk_weights[token, slot]

    torch.testing.assert_close(
        got.float(),
        reference,
        rtol=2e-2,
        atol=2e-2 * reference.abs().max(),
    )


def test_requantization_makes_scales_ue8m0_and_preserves_values():
    """Weight requantization must rewrite values, not just round the scale.

    Rounding the scale up alone would leave the old FP8 values scaled by up to
    2x, throwing away a bit of their mantissa. Re-quantizing keeps the
    dequantized weight close to what the checkpoint encoded.
    """
    from tokenspeed_kernel.ops.moe.deep_gemm.ue8m0 import (
        is_ue8m0,
        requantize_to_ue8m0_,
    )

    torch.manual_seed(0)
    experts, n, k = 3, 256, 512
    reference = torch.randn(experts, n, k, device="cuda") * 0.05
    tiled = reference.view(experts, n // _BLOCK, _BLOCK, k // _BLOCK, _BLOCK)
    amax = tiled.abs().amax(dim=(2, 4)).clamp(min=1e-10)
    scales = (amax / 448.0).contiguous()  # arbitrary, as FP8 checkpoints ship
    weight = (
        (tiled / scales[:, :, None, :, None])
        .clamp(-448, 448)
        .view(experts, n, k)
        .to(torch.float8_e4m3fn)
    )
    assert not is_ue8m0(scales)

    def dequantize(w, s):
        return w.float() * s.repeat_interleave(_BLOCK, 1).repeat_interleave(_BLOCK, 2)

    before = dequantize(weight, scales)
    requantize_to_ue8m0_(weight, scales, (_BLOCK, _BLOCK))
    after = dequantize(weight, scales)

    assert is_ue8m0(scales)
    # One extra FP8 rounding against a power-of-two scale; e4m3 has 3 mantissa
    # bits, so a few percent of the block amax is the expected ceiling.
    drift = (after - before).abs().max() / before.abs().max()
    assert drift < 0.10, f"requantization drifted {drift:.3f} of amax"


def test_deepep_weight_preprocessor_packs_scales_once_for_both_modes():
    """The per-forward GEMMs should receive ready-to-use B scale layouts."""
    from tokenspeed_kernel.ops.moe.deep_gemm import deepep_fp8

    torch.manual_seed(0)
    experts, hidden, ispp = 3, 512, 256
    w = torch.nn.Module()
    w.w13_weight, w.w13_weight_scale_inv = _quantize_weight(
        torch.randn(experts, 2 * ispp, hidden, device="cuda") * 0.1
    )
    w.w2_weight, w.w2_weight_scale_inv = _quantize_weight(
        torch.randn(experts, hidden, ispp, device="cuda") * 0.1
    )

    deepep_fp8.deep_gemm_deepep_fp8_moe_weights({}, w)

    assert w.w13_weight_scale_inv.dtype == torch.int32
    assert w.w2_weight_scale_inv.dtype == torch.int32
    assert w.w13_weight_scale_inv.shape[:2] == (experts, 2 * ispp)
    assert w.w2_weight_scale_inv.shape[:2] == (experts, hidden)
    assert w.w13_weight_scale_inv.stride(1) == 1
    assert w.w2_weight_scale_inv.stride(1) == 1


def test_ue8m0_quantized_activations_round_trip_through_the_grouped_gemm():
    """A UE8M0-scaled activation must survive DeepGEMM on this device.

    Non-power-of-two scales are misread by the sm100 1d1d kernel, so this pins
    the format the DeepEP normal path relies on.
    """
    from tokenspeed_kernel.ops.moe.deep_gemm.ue8m0 import (
        is_ue8m0,
        per_token_group_quant_fp8_ue8m0,
    )

    torch.manual_seed(0)
    tokens, hidden = 256, 512
    x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16)
    q, s = per_token_group_quant_fp8_ue8m0(x, _BLOCK)
    assert s.shape == (tokens, hidden // _BLOCK) and s.dtype == torch.float32
    assert is_ue8m0(s)
    dequantized = q.float() * s.repeat_interleave(_BLOCK, dim=1)
    torch.testing.assert_close(
        dequantized, x.float(), rtol=0.15, atol=0.15 * x.float().abs().max()
    )

    experts = 2
    w, w_scales = _quantize_weight(
        torch.randn(experts, hidden, hidden, device="cuda") * 0.1
    )
    m_indices = torch.repeat_interleave(
        torch.arange(experts, dtype=torch.int32, device="cuda"),
        torch.tensor([tokens // experts] * experts, dtype=torch.int32, device="cuda"),
        output_size=tokens,
    )
    out = torch.empty((tokens, hidden), dtype=torch.bfloat16, device="cuda")
    deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
        (q, deep_gemm.get_mn_major_tma_aligned_tensor(s)),
        (w, w_scales),
        out,
        m_indices,
    )
    assert torch.isfinite(out).all()
    expected = torch.einsum(
        "rk,rnk->rn", dequantized, _dequantize_weight(w, w_scales)[m_indices.long()]
    )
    torch.testing.assert_close(
        out.float(), expected, rtol=5e-2, atol=5e-2 * expected.abs().max()
    )
