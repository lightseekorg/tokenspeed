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

"""BF16 activation and route-rounding boundaries shared by expert GEMMs."""

from __future__ import annotations

import math

import torch
from tokenspeed_kernel._triton import tl, triton


@triton.jit
def _clamped_swiglu_bf16_kernel(
    x_ptr,
    out_ptr,
    ids_ptr,
    expert_start,
    num_local_experts,
    num_rows,
    ISPP: tl.constexpr,
    BLOCK: tl.constexpr,
    limit,
    HAS_LIMIT: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    expert = tl.load(ids_ptr + row).to(tl.int64) - expert_start
    valid = (expert >= 0) & (expert < num_local_experts) & (cols < ISPP)
    gate = tl.load(x_ptr + row * (2 * ISPP) + cols, mask=valid, other=0)
    up = tl.load(x_ptr + row * (2 * ISPP) + ISPP + cols, mask=valid, other=0)
    gate = gate.to(tl.bfloat16).to(tl.float32)
    up = up.to(tl.bfloat16).to(tl.float32)
    if HAS_LIMIT:
        gate = tl.minimum(gate, limit)
        up = tl.clamp(up, -limit, limit)
    output = gate * tl.sigmoid(gate) * up
    tl.store(out_ptr + row * ISPP + cols, output.to(tl.bfloat16), mask=cols < ISPP)


@triton.jit
def _finalize_bf16_routes_kernel(
    values_ptr,
    weights_ptr,
    ids_ptr,
    out_ptr,
    expert_start,
    num_local_experts,
    HIDDEN: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK: tl.constexpr,
    WEIGHTED_ROUTES: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for slot in tl.static_range(TOP_K):
        route = token * TOP_K + slot
        expert = tl.load(ids_ptr + route).to(tl.int64) - expert_start
        valid = (expert >= 0) & (expert < num_local_experts) & (cols < HIDDEN)
        value = tl.load(values_ptr + route * HIDDEN + cols, mask=valid, other=0)
        value = value.to(tl.bfloat16).to(tl.float32)
        if not WEIGHTED_ROUTES:
            weight = (
                tl.load(
                    weights_ptr + route,
                    mask=(expert >= 0) & (expert < num_local_experts),
                    other=0,
                )
                .to(tl.bfloat16)
                .to(tl.float32)
            )
            value = (value * weight).to(tl.bfloat16).to(tl.float32)
        acc = acc + value
    tl.store(out_ptr + token * HIDDEN + cols, acc, mask=cols < HIDDEN)


def _check_route_rows(values: torch.Tensor, ids: torch.Tensor) -> None:
    if values.ndim != 2 or ids.ndim != 2 or values.shape[0] != ids.numel():
        raise ValueError("route values must have one row per top-k slot")
    if values.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError("route values must be BF16 or FP32")
    if ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("expert IDs must be int32 or int64")
    if values.device.type != "cuda" or ids.device != values.device:
        raise ValueError("route values and IDs must share a CUDA device")
    if ids.shape[1] <= 0 or values.shape[1] <= 0:
        raise ValueError("route width and top-k must be positive")


def clamped_swiglu_bf16(
    gate_up: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    expert_start: int,
    num_local_experts: int,
    limit: float | None,
) -> torch.Tensor:
    """Apply standard SwiGLU to canonical route rows with BF16 boundaries.

    ``gate_up`` is BF16/FP32 ``[tokens * top_k, 2 * intermediate]`` in
    ``[gate; up]`` order. Operands first round to BF16; the gate is clamped
    from above and up on both sides when ``limit`` is present. SiLU and the
    product compute in FP32 and materialize BF16. Nonlocal or invalid IDs
    in ``topk_ids[tokens, top_k]`` produce zero rows without reading GEMM data.
    """
    _check_route_rows(gate_up, topk_ids)
    if gate_up.shape[1] % 2:
        raise ValueError("SwiGLU requires equally sized gate and up halves")
    if num_local_experts <= 0 or expert_start < 0:
        raise ValueError("the local expert range must be positive")
    if limit is not None and (not math.isfinite(limit) or limit <= 0):
        raise ValueError("SwiGLU clamp limit must be finite and positive")
    rows, width = gate_up.shape
    intermediate = width // 2
    out = torch.empty((rows, intermediate), dtype=torch.bfloat16, device=gate_up.device)
    if rows:
        block = 256
        _clamped_swiglu_bf16_kernel[(rows, triton.cdiv(intermediate, block))](
            gate_up.contiguous(),
            out,
            topk_ids.contiguous(),
            expert_start,
            num_local_experts,
            rows,
            ISPP=intermediate,
            BLOCK=block,
            limit=0.0 if limit is None else float(limit),
            HAS_LIMIT=limit is not None,
            num_warps=4,
        )
    return out


def finalize_bf16_routes(
    route_values: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    expert_start: int,
    num_local_experts: int,
    weighted_routes: bool,
) -> torch.Tensor:
    """Fold canonical expert outputs into a BF16 rank contribution.

    Inputs are BF16/FP32 ``route_values[tokens * top_k, hidden]`` and
    ``topk_weights/topk_ids[tokens, top_k]``. When ``weighted_routes`` is
    False, values and weights each round to BF16, their product rounds to
    BF16, then slots accumulate in increasing order in FP32. True means the
    GEMM already performed that product. Invalid and nonlocal slots contribute
    zero without reading their route outputs. Repeated expert IDs remain
    distinct slots. The caller owns communication between ranks.
    """
    _check_route_rows(route_values, topk_ids)
    if (
        topk_weights.shape != topk_ids.shape
        or topk_weights.device != route_values.device
    ):
        raise ValueError("route weights must match the expert IDs and device")
    if topk_weights.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError("route weights must be BF16 or FP32")
    if num_local_experts <= 0 or expert_start < 0:
        raise ValueError("the local expert range must be positive")
    tokens, top_k = topk_ids.shape
    hidden = route_values.shape[1]
    out = torch.empty(
        (tokens, hidden), dtype=torch.bfloat16, device=route_values.device
    )
    if tokens:
        block = 1024
        _finalize_bf16_routes_kernel[(tokens, triton.cdiv(hidden, block))](
            route_values.contiguous(),
            topk_weights.contiguous(),
            topk_ids.contiguous(),
            out,
            expert_start,
            num_local_experts,
            HIDDEN=hidden,
            TOP_K=top_k,
            BLOCK=block,
            WEIGHTED_ROUTES=weighted_routes,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return out
