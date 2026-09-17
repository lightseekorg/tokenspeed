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

"""Single-bank, sorted-scale MXFP8 SiTU expert pipeline for gfx950."""

from __future__ import annotations

import math

import torch
from tokenspeed_kernel_amd._scheduling import sched_barrier_compile_options
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.expert_mesh import sort_expert_slots
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.mxfp8_gemm import (
    _mxfp8_stage1,
    _mxfp8_stage2,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.mxfp8_quantize import quantize_mxfp8
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.n16_weights import n16_mxfp4_shape


def mxfp8_situ_prefill(
    x: torch.Tensor,
    w13: torch.Tensor,
    s13: torch.Tensor,
    w2: torch.Tensor,
    s2: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    global_experts: int,
    expert_start: int,
    situ_beta: float,
    situ_linear_beta: float,
    out: torch.Tensor | None,
) -> torch.Tensor:
    """Apply the two-stage MXFP8 expert pipeline to BF16 rows and N16 weights.

    Values stay token-major; only E8M0 scales follow sorted route order. The
    BF16 token-slot intermediate separates FP32 SiTU from its quantizer.
    Repeated expert IDs are distinct route slots. The caller must ensure output
    elements do not overlap any inputs; disjoint strided workspace views may
    share storage. Output requires nonoverlapping, dword-aligned rows with unit
    inner stride for packed BF16 atomics.
    """
    if x.ndim != 2 or x.dtype != torch.bfloat16 or not x.is_cuda:
        raise ValueError("MXFP8 prefill requires rank-2 BF16 GPU activations")
    weights = (w13, s13, w2, s2)
    e, weight_d, i = n16_mxfp4_shape(*weights)
    m, d = x.shape
    if weight_d != d or e <= 0 or expert_start < 0 or expert_start + e > global_experts:
        raise ValueError("inconsistent MXFP8 expert shapes or expert range")
    if any(t.numel() // e * t.element_size() >= 2**31 for t in weights):
        raise ValueError("MXFP8 expert cells exceed the buffer-offset range")
    if (
        topk_ids.ndim != 2
        or topk_ids.shape != topk_weights.shape
        or topk_ids.shape[0] != m
        or topk_ids.dtype not in (torch.int32, torch.int64)
        or topk_weights.dtype not in (torch.bfloat16, torch.float32)
    ):
        raise ValueError(
            "MXFP8 prefill requires matching integer IDs and BF16/FP32 route weights"
        )
    topk = topk_ids.shape[1]
    if not 0 < topk < 256 or m >= 2**24:
        raise ValueError(
            "route encoding requires fewer than 256 slots and 2**24 tokens"
        )
    if any(not math.isfinite(b) or b <= 0 for b in (situ_beta, situ_linear_beta)):
        raise ValueError("SiTU clamps must be positive finite values")
    inputs = (x, *weights, topk_ids, topk_weights)
    if any(t.device != x.device for t in inputs):
        raise ValueError("MXFP8 prefill tensors must be colocated")
    if out is None:
        out = torch.empty((m, d), dtype=x.dtype, device=x.device)
    if (
        out.shape != x.shape
        or out.dtype != x.dtype
        or out.device != x.device
        or out.stride(1) != 1
        or (m > 1 and out.stride(0) < d)
        or out.data_ptr() % 4 != 0
        or (m > 1 and out.stride(0) % 2 != 0)
    ):
        raise ValueError(
            "MXFP8 output requires nonoverlapping dword-aligned BF16 rows "
            "with unit inner stride"
        )
    if m == 0:
        return out
    compile_options = sched_barrier_compile_options()
    block_m = 32 if m <= 1024 else 128
    ids, route_weights, experts, valid = sort_expert_slots(
        topk_ids,
        topk_weights,
        out,
        global_experts=global_experts,
        local_experts=e,
        expert_start=expert_start,
        block_m=block_m,
    )
    xq, xs = quantize_mxfp8(
        x, ids, valid, tokens=m, topk=topk, slot_major=False, block_m=block_m
    )
    z = torch.empty((m * topk, i), dtype=torch.bfloat16, device=x.device)
    stage1_blocks = ids.numel() // block_m
    stage2_blocks = experts.numel()
    if block_m == 32:
        stage1_blocks = min(stage1_blocks, m * topk)
        stage2_blocks = min(stage2_blocks, m * topk)
    _mxfp8_stage1[(i // 64, stage1_blocks)](
        xq,
        xs,
        w13,
        s13,
        ids,
        experts,
        valid,
        z,
        m,
        topk,
        d,
        i,
        e,
        situ_beta,
        situ_linear_beta,
        m * d < 2**31,
        BM=block_m,
        num_warps=4,
        num_stages=1,
        enable_fp_fusion=False,
        **compile_options,
    )
    zq, zs = quantize_mxfp8(
        z, ids, valid, tokens=m, topk=topk, slot_major=True, block_m=block_m
    )
    _mxfp8_stage2[(d // 128, stage2_blocks)](
        zq,
        zs,
        w2,
        s2,
        ids,
        route_weights,
        experts,
        valid,
        out,
        m,
        topk,
        i,
        d,
        e,
        out.stride(0),
        m * topk * i < 2**31,
        (m - 1) * out.stride(0) * 2 + d * 2 < 2**31,
        BM=block_m,
        num_warps=4,
        num_stages=1,
        enable_fp_fusion=False,
        **compile_options,
    )
    return out
