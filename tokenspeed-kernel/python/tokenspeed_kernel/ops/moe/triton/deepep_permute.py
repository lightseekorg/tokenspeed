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

"""Permutation between DeepEP normal-mode receive buffers and grouped GEMMs.

DeepEP's normal (high-throughput) dispatch returns tokens ordered by source
rank, each carrying the per-token top-k slots that landed on this rank
(``recv_topk_ids``, with ``< 0`` for slots owned elsewhere). A grouped GEMM
instead wants one contiguous row block per local expert. These two kernels do
that round trip:

* :func:`deepep_scatter` expands the ``[num_recv, top_k]`` slots into the
  per-expert contiguous layout (one row per accepted slot) and records where
  every slot landed.
* :func:`deepep_gather` folds the expert outputs back onto the received tokens,
  applying the routing weights, so DeepEP's weight-less combine only has to sum
  across ranks.

The low-latency dispatch needs neither: it already hands back a padded
``[num_local_experts, capacity, hidden]`` tensor that masked grouped GEMMs
consume in place.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton

__all__ = [
    "deepep_gather",
    "deepep_scatter",
]

# Programs per grid dimension for the token loops. Both kernels stride a
# persistent grid over the received tokens rather than sizing the grid by token
# count, so the launch shape stays stable across forwards.
_MAX_TOKEN_PROGRAMS = 1024


@triton.jit
def _deepep_scatter_kernel(
    recv_x_ptr,
    recv_x_stride,
    recv_scale_ptr,
    recv_scale_stride,
    topk_ids_ptr,
    topk_ids_stride,
    cursor_ptr,
    out_x_ptr,
    out_x_stride,
    out_scale_ptr,
    out_scale_stride,
    dest_index_ptr,
    dest_index_stride,
    num_recv_tokens,
    TOP_K: tl.constexpr,
    HIDDEN: tl.constexpr,
    HIDDEN_PAD: tl.constexpr,
    NUM_SCALES: tl.constexpr,
    NUM_SCALES_PAD: tl.constexpr,
):
    program_id = tl.program_id(0)
    num_programs = tl.num_programs(0)

    off_hidden = tl.arange(0, HIDDEN_PAD)
    mask_hidden = off_hidden < HIDDEN
    off_scale = tl.arange(0, NUM_SCALES_PAD)
    mask_scale = off_scale < NUM_SCALES

    for token in range(program_id, num_recv_tokens, num_programs):
        # Read the token once, then replicate it into every expert row it was
        # routed to: dispatch deduplicates per rank, so a token that picked
        # several local experts arrives only once.
        row = tl.load(recv_x_ptr + token * recv_x_stride + off_hidden, mask=mask_hidden)
        scales = tl.load(
            recv_scale_ptr + token * recv_scale_stride + off_scale, mask=mask_scale
        )
        for slot in tl.range(0, TOP_K, 1, num_stages=4):
            expert = tl.load(topk_ids_ptr + token * topk_ids_stride + slot)
            if expert >= 0:
                # The cursor starts at the expert's block offset; the atomic
                # hands out the next free row inside that block.
                dest = tl.atomic_add(cursor_ptr + expert, 1).to(tl.int32)
                tl.store(dest_index_ptr + token * dest_index_stride + slot, dest)
                tl.store(
                    out_x_ptr + dest * out_x_stride + off_hidden,
                    row,
                    mask=mask_hidden,
                )
                tl.store(
                    out_scale_ptr + dest * out_scale_stride + off_scale,
                    scales,
                    mask=mask_scale,
                )


@triton.jit
def _deepep_gather_kernel(
    gemm_out_ptr,
    gemm_out_stride,
    topk_ids_ptr,
    topk_ids_stride,
    topk_weights_ptr,
    topk_weights_stride,
    dest_index_ptr,
    dest_index_stride,
    out_ptr,
    out_stride,
    num_recv_tokens,
    TOP_K: tl.constexpr,
    HIDDEN: tl.constexpr,
    BLOCK_HIDDEN: tl.constexpr,
):
    hidden_block = tl.program_id(0)
    program_id = tl.program_id(1)
    num_programs = tl.num_programs(1)

    off_hidden = hidden_block * BLOCK_HIDDEN + tl.arange(0, BLOCK_HIDDEN)
    mask_hidden = off_hidden < HIDDEN

    for token in range(program_id, num_recv_tokens, num_programs):
        accumulator = tl.zeros([BLOCK_HIDDEN], dtype=tl.float32)
        for slot in tl.range(0, TOP_K, 1, num_stages=4):
            expert = tl.load(topk_ids_ptr + token * topk_ids_stride + slot)
            if expert >= 0:
                source = tl.load(dest_index_ptr + token * dest_index_stride + slot)
                weight = tl.load(
                    topk_weights_ptr + token * topk_weights_stride + slot
                ).to(tl.float32)
                value = tl.load(
                    gemm_out_ptr + source * gemm_out_stride + off_hidden,
                    mask=mask_hidden,
                    other=0.0,
                )
                accumulator += value.to(tl.float32) * weight
        tl.store(
            out_ptr + token * out_stride + off_hidden,
            accumulator.to(out_ptr.dtype.element_ty),
            mask=mask_hidden,
        )


def deepep_scatter(
    recv_x: torch.Tensor,
    recv_x_scale: torch.Tensor,
    recv_topk_ids: torch.Tensor,
    num_recv_tokens_per_expert: list[int],
    expert_alignment: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Expand DeepEP normal-mode receive buffers into a grouped-GEMM layout.

    Args:
        recv_x: ``[num_recv, hidden]`` FP8 tokens returned by dispatch.
        recv_x_scale: ``[num_recv, hidden // block]`` float32 block scales that
            came alongside ``recv_x``.
        recv_topk_ids: ``[num_recv, top_k]`` local expert ids per received
            token; entries ``< 0`` belong to experts owned by another rank and
            are skipped.
        num_recv_tokens_per_expert: Per-local-expert receive counts from
            dispatch, already rounded up to ``expert_alignment``. Their sum is
            the row count of the returned buffers.
        expert_alignment: Row alignment each expert block was padded to. Must
            match the ``expert_alignment`` dispatch was called with so grouped
            GEMM tiles never straddle two experts.

    Returns:
        ``(x, x_scale, m_indices, dest_index)`` where ``x`` is
        ``[total_rows, hidden]`` FP8, ``x_scale`` is
        ``[total_rows, hidden // block]`` float32 (padding rows zeroed),
        ``m_indices`` is ``[total_rows]`` int32 mapping each row to its expert,
        and ``dest_index`` is ``[num_recv, top_k]`` int32 recording the row each
        accepted slot was written to (``-1`` for skipped slots), to be handed
        back to :func:`deepep_gather`.
    """
    num_recv, hidden = recv_x.shape
    num_scales = recv_x_scale.shape[1]
    top_k = recv_topk_ids.shape[1]
    num_local_experts = len(num_recv_tokens_per_expert)
    device = recv_x.device

    if any(count % expert_alignment for count in num_recv_tokens_per_expert):
        raise ValueError(
            f"per-expert receive counts {num_recv_tokens_per_expert} are not "
            f"aligned to {expert_alignment}; dispatch must run with "
            "expert_alignment set to the grouped-GEMM block size"
        )
    total_rows = sum(num_recv_tokens_per_expert)

    x = torch.empty((total_rows, hidden), dtype=recv_x.dtype, device=device)
    # Zeroed so the padding rows inside each expert block dequantize to zero
    # instead of feeding denormal garbage into the GEMM.
    x_scale = torch.zeros((total_rows, num_scales), dtype=torch.float32, device=device)
    dest_index = torch.full(
        (num_recv, top_k), -1, dtype=torch.int32, device=device
    )

    # Block starts double as the write cursors the scatter kernel bumps. The
    # counts are host-side already, so the prefix sum costs one small H2D copy
    # rather than a device scan plus sync.
    starts = [0] * num_local_experts
    running = 0
    for expert, count in enumerate(num_recv_tokens_per_expert):
        starts[expert] = running
        running += count
    cursor = torch.tensor(starts, dtype=torch.int32, device=device)
    m_indices = torch.repeat_interleave(
        torch.arange(num_local_experts, dtype=torch.int32, device=device),
        torch.tensor(num_recv_tokens_per_expert, dtype=torch.int32, device=device),
        output_size=total_rows,
    )

    if num_recv == 0 or total_rows == 0:
        return x, x_scale, m_indices, dest_index

    grid = (min(num_recv, _MAX_TOKEN_PROGRAMS),)
    _deepep_scatter_kernel[grid](
        recv_x,
        recv_x.stride(0),
        recv_x_scale,
        recv_x_scale.stride(0),
        recv_topk_ids,
        recv_topk_ids.stride(0),
        cursor,
        x,
        x.stride(0),
        x_scale,
        x_scale.stride(0),
        dest_index,
        dest_index.stride(0),
        num_recv,
        TOP_K=top_k,
        HIDDEN=hidden,
        HIDDEN_PAD=triton.next_power_of_2(hidden),
        NUM_SCALES=num_scales,
        NUM_SCALES_PAD=triton.next_power_of_2(num_scales),
        num_warps=4,
    )
    return x, x_scale, m_indices, dest_index


def deepep_gather(
    gemm_out: torch.Tensor,
    recv_topk_ids: torch.Tensor,
    recv_topk_weights: torch.Tensor,
    dest_index: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fold grouped expert outputs back onto DeepEP's received tokens.

    Args:
        gemm_out: ``[total_rows, hidden]`` expert outputs in the layout
            :func:`deepep_scatter` produced.
        recv_topk_ids: ``[num_recv, top_k]`` local expert ids per received
            token; entries ``< 0`` are skipped.
        recv_topk_weights: ``[num_recv, top_k]`` routing weights that came with
            the received tokens. DeepEP's normal combine sums without weights,
            so they are applied here.
        dest_index: ``[num_recv, top_k]`` int32 row map from
            :func:`deepep_scatter`.
        out: Optional ``[num_recv, hidden]`` destination.

    Returns:
        ``[num_recv, hidden]`` weighted sum over each token's local experts,
        ready for DeepEP's combine leg.
    """
    num_recv = recv_topk_ids.shape[0]
    top_k = recv_topk_ids.shape[1]
    hidden = gemm_out.shape[1]
    if out is None:
        out = torch.empty(
            (num_recv, hidden), dtype=gemm_out.dtype, device=gemm_out.device
        )
    if num_recv == 0:
        return out

    block_hidden = min(1024, triton.next_power_of_2(hidden))
    grid = (
        triton.cdiv(hidden, block_hidden),
        min(num_recv, _MAX_TOKEN_PROGRAMS),
    )
    _deepep_gather_kernel[grid](
        gemm_out,
        gemm_out.stride(0),
        recv_topk_ids,
        recv_topk_ids.stride(0),
        recv_topk_weights,
        recv_topk_weights.stride(0),
        dest_index,
        dest_index.stride(0),
        out,
        out.stride(0),
        num_recv,
        TOP_K=top_k,
        HIDDEN=hidden,
        BLOCK_HIDDEN=block_hidden,
        num_warps=4,
    )
    return out
