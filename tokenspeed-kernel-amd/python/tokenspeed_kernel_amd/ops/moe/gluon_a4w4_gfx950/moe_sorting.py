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

The routing buffers are sized to the worst case and padding is marked with
sentinels: ``sorted_expert_ids`` padding blocks are ``-1`` and
``sorted_token_ids`` padding slots are ``(topk << 24) | M``. The stage kernels
self-skip padding on-device (expert ``-1`` early-exits; token field ``>= M`` is
masked), so the launcher sizes its grid from the deterministic worst-case shape
with no host readback.

Implemented in Triton Gluon (``@gluon.jit``) to match the AMD-kernel convention
for this package and to leave room for manual layout/scheduling optimization
later. The algorithm is a four-stage block-aligned sort: a per-program expert
histogram, a column prefix sum, block-padded per-expert offsets, and a
race-free scatter -- scalar/scan control code with no MFMA. The vectorized
prefix sums use ``gl.associative_scan``; the histogram and scatter use scalar
dynamic-index loops.

Output contract::

    max_num_tokens_padded = M * TOPK + E * B - TOPK
    max_num_m_blocks      = ceil(max_num_tokens_padded / B)
    sorted_ids            (max_num_tokens_padded,) int32
        low 24 bits = token_id, high bits = topk slot; padding = (TOPK << 24) | M
    sorted_weights        (max_num_tokens_padded,) float32; padding = 0.0
    sorted_expert_ids     (max_num_m_blocks,)       int32; padding block = -1
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


@gluon.jit
def _add(a, b):
    return a + b


@gluon.jit
def _moe_sorting_stage1_kernel(
    topk_ids_ptr,  # (numel,) int32, row-major (M, TOPK)
    tokens_cnts_ptr,  # (E + 1, E) int32, zero-initialized
    num_experts: gl.constexpr,
    numel: gl.constexpr,
    tokens_per_thread: gl.constexpr,
):
    """Per-program expert histogram.

    Program ``pid`` counts the experts appearing in flat token slice
    ``[pid * tokens_per_thread, (pid + 1) * tokens_per_thread)`` and writes them
    to row ``pid + 1`` of ``tokens_cnts`` (row 0 is reserved as the zero base
    for the stage-2 column scan). Scalar uniform loads/stores -- no tile math.
    """
    pid = gl.program_id(0)
    start_idx = pid * tokens_per_thread
    off_c = (pid + 1) * num_experts

    for i in range(tokens_per_thread):
        if start_idx + i < numel:
            expert_id = gl.load(topk_ids_ptr + start_idx + i)
            if (expert_id >= 0) and (expert_id < num_experts):
                cnt = gl.load(tokens_cnts_ptr + off_c + expert_id)
                gl.store(tokens_cnts_ptr + off_c + expert_id, cnt + 1)


@gluon.jit
def _moe_sorting_stage2_kernel(
    tokens_cnts_ptr,  # (E + 1, E) int32
    num_experts: gl.constexpr,
    E_PAD: gl.constexpr,
):
    """Column-wise inclusive prefix sum over programs (vectorized).

    Program ``pid`` owns expert column ``pid`` and turns per-program counts into
    the running start offset of each program's tokens *within* that expert:
    after this pass ``tokens_cnts[p][pid]`` == number of expert-``pid`` tokens
    contributed by programs ``0..p-1``. Row 0 stays the zero base.
    """
    pid = gl.program_id(0)
    layout: gl.constexpr = gl.BlockedLayout([1], [64], [4], [0])
    rows = gl.arange(0, E_PAD, layout=layout)  # maps to source rows 1..num_experts
    mask = rows < num_experts
    offs = (rows + 1) * num_experts + pid
    cnt = gl.load(tokens_cnts_ptr + offs, mask=mask, other=0)
    inclusive = gl.associative_scan(cnt, 0, _add)
    gl.store(tokens_cnts_ptr + offs, inclusive, mask=mask)


@gluon.jit
def _moe_sorting_stage3_kernel(
    num_valid_ids_ptr,  # (2,) int32
    tokens_cnts_ptr,  # (E + 1, E) int32
    cumsum_ptr,  # (E + 1,) int32
    m_total,  # python int -> int32 scalar
    num_experts: gl.constexpr,
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
    off_last = num_experts * num_experts  # row E == total count per expert
    cnt = gl.load(tokens_cnts_ptr + off_last + e, mask=mask, other=0)
    padded = ((cnt + block_size - 1) // block_size) * block_size
    inclusive = gl.associative_scan(padded, 0, _add)
    gl.store(cumsum_ptr + 1 + e, inclusive, mask=mask)
    gl.store(num_valid_ids_ptr + 0, gl.sum(padded))
    gl.store(num_valid_ids_ptr + 1, m_total)


@gluon.jit
def _moe_sorting_stage4_kernel(
    topk_ids_ptr,  # (numel,) int32
    topk_weights_ptr,  # (numel,) float32
    sorted_ids_ptr,  # (max_num_tokens_padded,) int32, pre-filled with sentinel
    sorted_weights_ptr,  # (max_num_tokens_padded,) float32, pre-filled with 0
    expert_ids_ptr,  # (max_num_m_blocks,) int32, pre-filled with -1
    tokens_cnts_ptr,  # (E + 1, E) int32
    cumsum_ptr,  # (E + 1,) int32
    num_experts: gl.constexpr,
    block_size: gl.constexpr,
    numel: gl.constexpr,
    tokens_per_thread: gl.constexpr,
    TOPK: gl.constexpr,
):
    """Fill ``expert_ids`` block map and scatter routed rows + weights.

    Program ``pid`` writes ``expert_ids`` for expert ``pid``'s block range and
    scatters the routed rows in its own token slice. Because each program owns a
    disjoint token slice *and* a disjoint sub-range of every expert's slots
    (via the stage-2 offsets), the scatter is race-free with no atomics.
    """
    pid = gl.program_id(0)

    # Block map: mark every block owned by expert ``pid``.
    start_slot = gl.load(cumsum_ptr + pid)
    end_slot = gl.load(cumsum_ptr + pid + 1)
    for slot in range(start_slot, end_slot, block_size):
        gl.store(expert_ids_ptr + slot // block_size, pid)

    # Scatter this program's token slice into per-expert padded slots.
    start_idx = pid * tokens_per_thread
    off_t = pid * num_experts
    for i in range(tokens_per_thread):
        idx = start_idx + i
        if idx < numel:
            expert_id = gl.load(topk_ids_ptr + idx)
            if (expert_id >= 0) and (expert_id < num_experts):
                cursor = gl.load(tokens_cnts_ptr + off_t + expert_id)
                rank = cursor + gl.load(cumsum_ptr + expert_id)
                token_id = idx // TOPK
                topk_id = idx % TOPK
                packed = (topk_id << 24) | token_id
                gl.store(sorted_ids_ptr + rank, packed)
                weight = gl.load(topk_weights_ptr + idx)
                gl.store(sorted_weights_ptr + rank, weight)
                gl.store(tokens_cnts_ptr + off_t + expert_id, cursor + 1)


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

    # Worst-case allocation: every routed slot could land in its own
    # partially-filled per-expert block.
    max_num_tokens_padded = numel + E * B - topk
    max_num_m_blocks = triton.cdiv(max_num_tokens_padded, B)

    topk_ids = topk_ids.to(torch.int32).contiguous()
    topk_weights = topk_weights.to(torch.float32).contiguous()

    # Padding sentinels let the stage kernels self-skip padding on-device
    # (token field >= M is masked; expert -1 early-exits).
    init_sorted_id = (int(topk) << 24) | int(M)
    sorted_ids = torch.full(
        (max_num_tokens_padded,), init_sorted_id, dtype=torch.int32, device=device
    )
    sorted_weights = torch.zeros(
        (max_num_tokens_padded,), dtype=torch.float32, device=device
    )
    sorted_expert_ids = torch.full(
        (max_num_m_blocks,), -1, dtype=torch.int32, device=device
    )
    num_valid_ids = torch.empty((2,), dtype=torch.int32, device=device)
    # ``out`` does not need zeroing: stage2's reduce epilogue overwrites every
    # token row, and its small-M atomic epilogue zeroes ``out`` itself. Zeroing
    # here would redundantly clear up to tens of MB at large M.
    out = torch.empty((M, model_dim), dtype=out_dtype, device=device)

    # Scratch: (E+1, E) histogram + (E+1,) padded prefix sums.
    tokens_cnts = torch.zeros((E + 1, E), dtype=torch.int32, device=device)
    cumsum = torch.zeros((E + 1,), dtype=torch.int32, device=device)
    tokens_per_thread = triton.cdiv(numel, E)
    e_pad = triton.next_power_of_2(E)

    grid = (E,)
    _moe_sorting_stage1_kernel[grid](
        topk_ids, tokens_cnts, E, numel, tokens_per_thread, num_warps=1
    )
    _moe_sorting_stage2_kernel[grid](tokens_cnts, E, e_pad, num_warps=_SCAN_NUM_WARPS)
    _moe_sorting_stage3_kernel[(1,)](
        num_valid_ids,
        tokens_cnts,
        cumsum,
        int(M),
        E,
        B,
        e_pad,
        num_warps=_SCAN_NUM_WARPS,
    )
    _moe_sorting_stage4_kernel[grid](
        topk_ids,
        topk_weights,
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        tokens_cnts,
        cumsum,
        E,
        B,
        numel,
        tokens_per_thread,
        int(topk),
        num_warps=1,
    )

    return sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, out
