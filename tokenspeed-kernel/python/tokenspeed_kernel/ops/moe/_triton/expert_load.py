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

"""Portable Triton kernel for masked EPLB expert-load accumulation."""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.ops.moe import validate_record_expert_load_inputs

_BLOCK_M = 32
_BLOCK_K = 8


@triton.jit(
    do_not_specialize=[
        "rows",
        "top_k",
        "num_physical_experts",
        "stride_ids_m",
        "stride_ids_k",
        "stride_mask",
        "stride_load",
    ],
    do_not_specialize_on_alignment=[
        "topk_ids_ptr",
        "live_rows_ptr",
        "physical_load_ptr",
        "recording_enabled_ptr",
    ],
)
def _record_expert_load_kernel(
    topk_ids_ptr,
    live_rows_ptr,
    physical_load_ptr,
    recording_enabled_ptr,
    rows,
    top_k,
    num_physical_experts,
    stride_ids_m,
    stride_ids_k,
    stride_mask,
    stride_load,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HAS_RECORDING_ENABLED: tl.constexpr,
) -> None:
    if HAS_RECORDING_ENABLED:
        if not tl.load(recording_enabled_ptr).to(tl.int1):
            return
    row_offs = tl.program_id(0).to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M).to(
        tl.int64
    )
    col_offs = tl.program_id(1).to(tl.int64) * BLOCK_K + tl.arange(0, BLOCK_K).to(
        tl.int64
    )
    row_mask = row_offs < rows
    live = tl.load(
        live_rows_ptr + row_offs * stride_mask,
        mask=row_mask,
        other=False,
    ).to(tl.int1)
    elem_mask = live[:, None] & (col_offs[None, :] < top_k)
    ids = tl.load(
        topk_ids_ptr
        + row_offs[:, None] * stride_ids_m
        + col_offs[None, :] * stride_ids_k,
        mask=elem_mask,
        other=-1,
    ).to(tl.int64)
    valid = elem_mask & (ids >= 0)
    in_bounds = ids < num_physical_experts
    tl.device_assert((~valid) | in_bounds, "physical expert id out of range")
    active = valid & in_bounds
    safe_ids = tl.where(active, ids, 0)
    tl.atomic_add(
        physical_load_ptr + safe_ids * stride_load,
        1,
        mask=active,
        sem="relaxed",
    )


def triton_record_expert_load(
    topk_ids: torch.Tensor,
    live_rows: torch.Tensor,
    physical_load: torch.Tensor,
    recording_enabled: torch.Tensor | None = None,
) -> None:
    """Count non-negative ``topk_ids`` on live rows into ``physical_load``.

    Args:
        topk_ids: Physical expert ids shaped ``[rows, top_k]`` (int32 or int64).
        live_rows: Boolean row mask covering at least ``rows`` elements.
        physical_load: 1D int32 or int64 per-physical-expert counter updated in place.
        recording_enabled: Optional 1-element boolean device toggle; None records
            every call.
    """
    rows, top_k, num_physical_experts = validate_record_expert_load_inputs(
        topk_ids, live_rows, physical_load, recording_enabled
    )
    if rows == 0 or top_k == 0:
        return

    grid = (triton.cdiv(rows, _BLOCK_M), triton.cdiv(top_k, _BLOCK_K))
    _record_expert_load_kernel[grid](
        topk_ids,
        live_rows,
        physical_load,
        recording_enabled,
        rows,
        top_k,
        num_physical_experts,
        topk_ids.stride(0),
        topk_ids.stride(1),
        live_rows.stride(0),
        physical_load.stride(0),
        BLOCK_M=_BLOCK_M,
        BLOCK_K=_BLOCK_K,
        HAS_RECORDING_ENABLED=recording_enabled is not None,
        num_warps=4,
    )
