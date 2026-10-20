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

"""Pointer-based FP8 experts with fused FP32 gate/up activation.

Input quantization uses the public per-token FP8 operation. FC1 retains FP32
gate/up through SiLU and multiplication, then stores BF16. Route quantization
rounds its FP32 scaled product to BF16 before saturated E4M3 conversion.
FC2 stores BF16 per route; FP32 route weights and accumulation produce BF16.
IEEE division is used; cross-vendor byte identity is not a contract.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.ops.moe._triton.common import _combine, _prepare_routed_output
from tokenspeed_kernel.ops.moe._triton.fp8_channel import _expert_tiles, _locate_tile
from tokenspeed_kernel.ops.quantization import quantize_fp8


@triton.jit
def _route_quantize_kernel(
    x_ptr, q_ptr, scale_ptr, width: tl.constexpr, BLOCK: tl.constexpr
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    values = tl.load(x_ptr + row * width + cols, cols < width, other=0).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(values), 0), 1.0e-6)
    scale = tl.maximum(tl.div_rn(amax, 448.0), 1.0 / (448.0 * 512.0))
    multiplier = tl.minimum(tl.div_rn(448.0, amax), 448.0 * 512.0)
    scaled = (values * multiplier).to(tl.bfloat16).to(tl.float32)
    codes = tl.clamp(scaled, -448.0, 448.0).to(
        tl.float8e4nv, fp_downcast_rounding="rtne"
    )
    tl.store(q_ptr + row * width + cols, codes, cols < width)
    tl.store(scale_ptr + row, scale)


def _quantize_routes(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rows, width = x.shape
    codes = torch.empty(x.shape, device=x.device, dtype=torch.float8_e4m3fn)
    scales = torch.empty(rows, device=x.device, dtype=torch.float32)
    block = triton.next_power_of_2(width)
    _route_quantize_kernel[(rows,)](
        x, codes, scales, width, block, num_warps=8 if block >= 4096 else 4
    )
    return codes, scales


@triton.jit
def _expert_matmul_kernel(
    x_ptr,
    x_scale_ptr,
    weight_ptr,
    weight_scale_ptr,
    out_ptr,
    route_ids_ptr,
    counts_ptr,
    num_tokens,
    num_programs,
    hidden: tl.constexpr,
    intermediate: tl.constexpr,
    num_experts: tl.constexpr,
    EXPERTS: tl.constexpr,
    top_k: tl.constexpr,
    FC1: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    width: tl.constexpr = intermediate if FC1 else hidden
    reduction: tl.constexpr = hidden if FC1 else intermediate
    weight_rows: tl.constexpr = 2 * intermediate if FC1 else hidden
    routes = num_tokens * top_k
    num_n_tiles: tl.constexpr = tl.cdiv(width, BLOCK_N)
    experts, counts, tiles, ends = _expert_tiles(
        counts_ptr, num_experts, EXPERTS, num_n_tiles, BLOCK_M
    )
    for tile in range(tl.program_id(0), tl.sum(tiles, 0), num_programs):
        expert, local_tile, count = _locate_tile(tile, experts, counts, tiles, ends)
        rows = local_tile // num_n_tiles * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = local_tile % num_n_tiles * BLOCK_N + tl.arange(0, BLOCK_N)
        route_ids = tl.load(
            route_ids_ptr + expert * routes + rows, rows < count, other=0
        ).to(tl.int64)
        input_rows = route_ids // top_k if FC1 else route_ids
        ks = tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        if FC1:
            up_acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        for start in range(0, reduction, BLOCK_K):
            kvals = start + ks
            x = tl.load(
                x_ptr + input_rows[:, None] * reduction + kvals[None, :],
                (rows < count)[:, None] & (kvals < reduction)[None, :],
                other=0.0,
            )
            offsets = (
                expert.to(tl.int64) * weight_rows + cols[:, None]
            ) * reduction + kvals[None, :]
            weight = tl.load(
                weight_ptr + offsets,
                (cols < width)[:, None] & (kvals < reduction)[None, :],
                other=0.0,
            )
            acc = tl.dot(x, weight.T, acc)
            if FC1:
                up = tl.load(
                    weight_ptr + offsets + intermediate * reduction,
                    (cols < width)[:, None] & (kvals < reduction)[None, :],
                    other=0.0,
                )
                up_acc = tl.dot(x, up.T, up_acc)
        input_scale = tl.load(x_scale_ptr + input_rows, rows < count, other=0)
        wscale = tl.load(
            weight_scale_ptr + expert * weight_rows + cols, cols < width, other=0
        )
        result = acc * input_scale[:, None] * wscale[None, :]
        if FC1:
            uscale = tl.load(
                weight_scale_ptr + expert * weight_rows + intermediate + cols,
                cols < width,
                other=0,
            )
            up_result = up_acc * input_scale[:, None] * uscale[None, :]
            result = result * tl.sigmoid(result) * up_result
        tl.store(
            out_ptr + route_ids[:, None] * width + cols[None, :],
            result.to(tl.bfloat16),
            (rows < count)[:, None] & (cols < width)[None, :],
        )


def _triton_fp8_channel_fused_moe_impl(
    x, w13, w13_scale, w2, w2_scale, topk_weights, topk_ids, expert_offset
):
    tokens, hidden = x.shape
    experts, twice_intermediate, _ = w13.shape
    intermediate = twice_intermediate // 2
    top_k = topk_ids.shape[1]
    if tokens == 0:
        return torch.empty_like(x)
    local_ids = (topk_ids.to(torch.int32) - expert_offset).contiguous()
    route_ids, counts, routed_output, output = _prepare_routed_output(
        x, local_ids, experts
    )
    xq, xs = quantize_fp8(
        x,
        granularity="token",
        scale_encoding="float32",
        enable_pdl=False,
        required_features=frozenset({"token_fp8_min_scaling"}),
    )
    # Nonlocal routes remain zero, including the full row read by quantization.
    activated = torch.zeros(
        (tokens * top_k, intermediate), device=x.device, dtype=torch.bfloat16
    )
    programs = torch.cuda.get_device_properties(x.device).multi_processor_count
    for first in (True, False):
        if first:
            values, scales, weight, weight_scales, destination = (
                xq,
                xs,
                w13,
                w13_scale,
                activated,
            )
        else:
            values, scales = _quantize_routes(activated)
            weight, weight_scales, destination = w2, w2_scale, routed_output
        _expert_matmul_kernel[(programs,)](
            values,
            scales,
            weight,
            weight_scales,
            destination,
            route_ids,
            counts,
            tokens,
            programs,
            hidden,
            intermediate,
            experts,
            triton.next_power_of_2(experts),
            top_k,
            first,
            16,
            64,
            64,
            num_warps=4,
            num_stages=2,
            enable_fp_fusion=False,
        )
    _combine(routed_output, topk_weights, output)
    return output
