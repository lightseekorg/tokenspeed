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


"""In-house MoE block-aligned expert sort for the gfx950 A4W4 prefill path.

Produces the block-aligned sorted routing metadata the package prefill stage
kernels consume. It launches entirely on the *caller's* CUDA stream and
performs **no** device-to-host synchronization, so the whole prefill path is
CUDA-graph capturable.

The routing buffers are sized to the worst case. Valid expert blocks and every
slot within their padded ranges are written by the scatter stage; padding token
IDs use ``(topk << 24) | M``. The stage kernels use the device-side valid count
and mask token fields ``>= M``, so launch grids remain deterministic with no
host readback.

Implemented in Triton Gluon (``@gluon.jit``) to match the AMD-kernel convention
for this package. The algorithm is a four-stage block-aligned sort: vectorized
per-chunk expert histograms, a column prefix sum, block-padded per-expert
offsets, and a vectorized scatter. Chunk prefix sums preserve source-token
locality across chunks; atomics assign ranks only among routes in the same
chunk.

Output contract::

    max_num_tokens_padded = floor((R + min(E, R) * (B - 1)) / B) * B
                            where R = M * TOPK
    max_num_m_blocks      = ceil(max_num_tokens_padded / B)
    sorted_ids            (max_num_tokens_padded,) int32
        low 24 bits = token_id, high bits = topk slot; padding = (TOPK << 24) | M
    sorted_weights        (max_num_tokens_padded,) float32; padding is unused
    sorted_expert_ids     (max_num_m_blocks,)       int32; valid blocks are written
    num_valid_ids         (2,) int32; [0] = total padded slots, [1] = M
    out                   (M, model_dim) uninitialized ``out_dtype`` buffer
                          (stage2 overwrites/zeros it; see wrapper note)
"""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._triton import gl, gluon, triton

# Warp count for the vectorized (prefix-sum) stages. Must match the hardcoded
# ``warps_per_cta`` (the ``4`` in ``gl.BlockedLayout([1], [64], [4], [0])``)
# inside stage2/stage3 -- Gluon kernels cannot read plain Python globals, so the
# layout literal and this launch constant are kept in sync by hand.
_SCAN_NUM_WARPS = 4


def _max_padded_route_capacity(
    num_routes: int,
    num_experts: int,
    block_size: int,
) -> int:
    """Bound block-padded routes by the maximum number of nonempty experts."""

    if num_routes < 0 or num_experts < 0 or block_size <= 0:
        raise ValueError(
            "route/expert counts must be nonnegative and block_size positive"
        )
    max_nonempty_experts = min(num_routes, num_experts)
    upper_bound = num_routes + max_nonempty_experts * (block_size - 1)
    # The actual padded extent is block-aligned, so the largest multiple no
    # greater than the per-expert padding bound is sufficient.
    return upper_bound // block_size * block_size


@gluon.jit
def _add(a, b):
    return a + b


@gluon.jit
def _moe_sorting_stage1_kernel(
    topk_ids_ptr,  # (numel,) int32, row-major (M, TOPK)
    tokens_cnts_ptr,  # (num_programs + 1, E) int32
    num_experts: gl.constexpr,
    numel: gl.constexpr,
    tokens_per_program: gl.constexpr,
    ROUTE_BLOCK: gl.constexpr,
    EXPERT_PAD: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    """Vectorized per-program expert histogram.

    Program ``pid`` counts the experts appearing in flat token slice
    ``[pid * tokens_per_program, (pid + 1) * tokens_per_program)`` and writes them
    to row ``pid + 1`` of ``tokens_cnts`` (row 0 is reserved as the zero base
    for the stage-2 column scan).
    """
    pid = gl.program_id(0)
    route_layout: gl.constexpr = gl.BlockedLayout([1], [64], [NUM_WARPS], [0])
    expert_layout: gl.constexpr = gl.BlockedLayout([1], [64], [NUM_WARPS], [0])
    route = gl.arange(0, ROUTE_BLOCK, layout=route_layout)
    idx = pid * tokens_per_program + route
    valid = (route < tokens_per_program) & (idx < numel)
    expert_id = gl.load(topk_ids_ptr + idx, mask=valid, other=num_experts)
    valid &= (expert_id >= 0) & (expert_id < num_experts)
    safe_expert = gl.where(valid, expert_id, num_experts)
    histogram = gl.histogram(
        safe_expert,
        EXPERT_PAD,
        mask=valid,
        layout=expert_layout,
    ).to(gl.int32)
    expert = gl.arange(0, EXPERT_PAD, layout=expert_layout)
    off_c = (pid + 1) * num_experts
    if pid == 0:
        gl.store(
            tokens_cnts_ptr + expert,
            gl.zeros([EXPERT_PAD], gl.int32, layout=expert_layout),
            mask=expert < num_experts,
        )
    gl.store(
        tokens_cnts_ptr + off_c + expert,
        histogram,
        mask=expert < num_experts,
    )


@gluon.jit
def _moe_sorting_stage2_kernel(
    tokens_cnts_ptr,  # (num_programs + 1, E) int32
    num_experts: gl.constexpr,
    num_programs: gl.constexpr,
    PROGRAM_PAD: gl.constexpr,
):
    """Column-wise inclusive prefix sum over programs (vectorized).

    Program ``pid`` owns expert column ``pid`` and turns per-program counts into
    the running start offset of each program's tokens *within* that expert:
    after this pass ``tokens_cnts[p][pid]`` == number of expert-``pid`` tokens
    contributed by programs ``0..p-1``. Row 0 stays the zero base.
    """
    pid = gl.program_id(0)
    layout: gl.constexpr = gl.BlockedLayout([1], [64], [4], [0])
    rows = gl.arange(0, PROGRAM_PAD, layout=layout)
    mask = rows < num_programs
    offs = (rows + 1) * num_experts + pid
    cnt = gl.load(tokens_cnts_ptr + offs, mask=mask, other=0)
    inclusive = gl.associative_scan(cnt, 0, _add)
    gl.store(tokens_cnts_ptr + offs, inclusive, mask=mask)


@gluon.jit
def _moe_sorting_stage3_kernel(
    num_valid_ids_ptr,  # (2,) int32
    tokens_cnts_ptr,  # (num_programs + 1, E) int32
    cumsum_ptr,  # (E + 1,) int32
    m_total,  # python int -> int32 scalar
    num_experts: gl.constexpr,
    num_programs: gl.constexpr,
    block_size: gl.constexpr,
    E_PAD: gl.constexpr,
):
    """Block-aligned per-expert slot offsets (single program, vectorized).

    ``cumsum[e + 1]`` becomes the first sorted slot owned by expert ``e + 1``
    (``cumsum[0] == 0``); each expert's token run is padded up to a multiple of
    ``block_size`` so the next expert starts on a block boundary.
    ``num_valid_ids[0]`` is the total padded extent; ``num_valid_ids[1]``
    carries ``M`` (the token count) in the second slot.
    """
    layout: gl.constexpr = gl.BlockedLayout([1], [64], [4], [0])
    e = gl.arange(0, E_PAD, layout=layout)
    mask = e < num_experts
    off_last = num_programs * num_experts
    cnt = gl.load(tokens_cnts_ptr + off_last + e, mask=mask, other=0)
    padded = ((cnt + block_size - 1) // block_size) * block_size
    inclusive = gl.associative_scan(padded, 0, _add)
    gl.store(cumsum_ptr, 0)
    gl.store(cumsum_ptr + 1 + e, inclusive, mask=mask)
    gl.store(num_valid_ids_ptr + 0, gl.sum(padded))
    gl.store(num_valid_ids_ptr + 1, m_total)


@gluon.jit
def _moe_sorting_stage4_kernel(
    topk_ids_ptr,  # (numel,) int32
    topk_weights_ptr,  # (numel,) float32
    sorted_ids_ptr,  # (max_num_tokens_padded,) int32
    sorted_weights_ptr,  # (max_num_tokens_padded,) float32
    expert_ids_ptr,  # (max_num_m_blocks,) int32
    tokens_cnts_ptr,  # (num_programs + 1, E) int32
    cumsum_ptr,  # (E + 1,) int32
    num_experts: gl.constexpr,
    num_programs: gl.constexpr,
    block_size: gl.constexpr,
    numel: gl.constexpr,
    tokens_per_program: gl.constexpr,
    TOPK: gl.constexpr,
    ROUTE_BLOCK: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    """Fill ``expert_ids`` block map and scatter routed rows + weights.

    Expert programs write their block maps and padding, while route programs
    scatter one input chunk. Stage 2 gives each chunk a disjoint sub-range of
    every expert, so atomics assign ranks only within a chunk and retain stable
    ordering between chunks.
    """
    pid = gl.program_id(0)

    if pid < num_experts:
        # Initialize only padding so these writes never overlap real routes.
        start_slot = gl.load(cumsum_ptr + pid)
        end_slot = gl.load(cumsum_ptr + pid + 1)
        padding_layout: gl.constexpr = gl.BlockedLayout([1], [64], [NUM_WARPS], [0])
        block_offsets = gl.arange(0, block_size, layout=padding_layout)
        padding_id: gl.constexpr = (TOPK << 24) | (numel // TOPK)
        for slot in range(start_slot, end_slot, block_size):
            gl.store(expert_ids_ptr + slot // block_size, pid)
        total_count = gl.load(tokens_cnts_ptr + num_programs * num_experts + pid)
        padding_start = start_slot + total_count
        padding_slots = padding_start + block_offsets
        gl.store(
            sorted_ids_ptr + padding_slots,
            padding_id,
            mask=padding_slots < end_slot,
        )

    if pid < num_programs:
        route_layout: gl.constexpr = gl.BlockedLayout([1], [64], [NUM_WARPS], [0])
        route = gl.arange(0, ROUTE_BLOCK, layout=route_layout)
        idx = pid * tokens_per_program + route
        valid = (route < tokens_per_program) & (idx < numel)
        expert_id = gl.load(topk_ids_ptr + idx, mask=valid, other=0)
        valid &= (expert_id >= 0) & (expert_id < num_experts)
        safe_expert = gl.where(valid, expert_id, 0)
        one = gl.full([ROUTE_BLOCK], 1, gl.int32, layout=route_layout)
        cursor = gl.atomic_add(
            tokens_cnts_ptr + pid * num_experts + safe_expert,
            one,
            mask=valid,
            sem="relaxed",
            scope="gpu",
        )
        rank = cursor + gl.load(cumsum_ptr + safe_expert, mask=valid, other=0)
        token_id = idx // TOPK
        topk_id = idx % TOPK
        packed = (topk_id << 24) | token_id
        gl.store(sorted_ids_ptr + rank, packed, mask=valid)
        weight = gl.load(topk_weights_ptr + idx, mask=valid, other=0.0)
        gl.store(sorted_weights_ptr + rank, weight, mask=valid)


def gluon_moe_sorting(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    num_experts: int,
    model_dim: int,
    out_dtype: torch.dtype,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Block-aligned expert sort producing sorted routing metadata.

    Runs four small Gluon kernels on the *current* CUDA stream with no
    device-to-host synchronization (see the module docstring for the output
    contract).

    Args:
        topk_ids: ``(M, TOPK)`` int32 expert assignments. ``-1`` entries are
            treated as unrouted and skipped.
        topk_weights: ``(M, TOPK)`` float32 routing weights, same layout.
        num_experts: total number of experts ``E``.
        model_dim: hidden dim of the ``out`` buffer.
        out_dtype: dtype of the ``out`` buffer (bf16 in production).
        block_size: per-expert padding granularity ``B`` (the stage BLOCK_M).

    Returns:
        ``(sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, out)``
        with shapes/semantics matching ``moe_sorting(..., accumulate=True)``.
    """
    assert topk_ids.dim() == 2, "topk_ids must be (M, TOPK)"
    assert topk_weights.shape == topk_ids.shape, "topk_weights must match topk_ids"
    device = topk_ids.device

    M, topk = topk_ids.shape
    numel = M * topk
    E = int(num_experts)
    B = int(block_size)

    # Only experts receiving at least one route contribute a padding block.
    # Bounding that count by min(E, routes) avoids scaling activation-sized
    # consumers with every expert when a small prefill touches only a few.
    max_num_tokens_padded = _max_padded_route_capacity(numel, E, B)
    max_num_m_blocks = triton.cdiv(max_num_tokens_padded, B)

    topk_ids = topk_ids.to(torch.int32).contiguous()
    topk_weights = topk_weights.to(torch.float32).contiguous()

    # Stage 4 writes every routed/padding ID and every valid expert block.
    # Padding weights are masked before stores by the consumers.
    sorted_ids = torch.empty((max_num_tokens_padded,), dtype=torch.int32, device=device)
    sorted_weights = torch.empty(
        (max_num_tokens_padded,), dtype=torch.float32, device=device
    )
    sorted_expert_ids = torch.empty(
        (max_num_m_blocks,), dtype=torch.int32, device=device
    )
    num_valid_ids = torch.empty((2,), dtype=torch.int32, device=device)
    # ``out`` does not need zeroing: stage2's reduce epilogue overwrites every
    # token row, and its small-M atomic epilogue zeroes ``out`` itself. Zeroing
    # here would redundantly clear up to tens of MB at large M.
    out = torch.empty((M, model_dim), dtype=out_dtype, device=device)

    # Keep route chunks at a practical vector width. The chunk prefix scan
    # preserves their source order even when there are fewer experts than
    # required route programs.
    max_routes_per_program = 1024
    num_programs = max(E, triton.cdiv(numel, max_routes_per_program))
    tokens_per_program = triton.cdiv(numel, num_programs)
    route_block = triton.next_power_of_2(tokens_per_program)
    expert_pad = triton.next_power_of_2(E + 1)
    program_pad = triton.next_power_of_2(num_programs)

    # Scratch: per-chunk histograms + per-expert padded prefix sums.
    tokens_cnts = torch.empty((num_programs + 1, E), dtype=torch.int32, device=device)
    cumsum = torch.empty((E + 1,), dtype=torch.int32, device=device)

    route_grid = (num_programs,)
    _moe_sorting_stage1_kernel[route_grid](
        topk_ids,
        tokens_cnts,
        E,
        numel,
        tokens_per_program,
        ROUTE_BLOCK=route_block,
        EXPERT_PAD=expert_pad,
        NUM_WARPS=_SCAN_NUM_WARPS,
        num_warps=_SCAN_NUM_WARPS,
    )
    expert_grid = (E,)
    _moe_sorting_stage2_kernel[expert_grid](
        tokens_cnts,
        E,
        num_programs,
        program_pad,
        num_warps=_SCAN_NUM_WARPS,
    )
    _moe_sorting_stage3_kernel[(1,)](
        num_valid_ids,
        tokens_cnts,
        cumsum,
        int(M),
        E,
        num_programs,
        B,
        expert_pad,
        num_warps=_SCAN_NUM_WARPS,
    )
    _moe_sorting_stage4_kernel[(max(E, num_programs),)](
        topk_ids,
        topk_weights,
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        tokens_cnts,
        cumsum,
        E,
        num_programs,
        B,
        numel,
        tokens_per_program,
        int(topk),
        ROUTE_BLOCK=route_block,
        NUM_WARPS=_SCAN_NUM_WARPS,
        num_warps=_SCAN_NUM_WARPS,
    )

    return sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, out
