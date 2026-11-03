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

"""Dense-workspace DeepSeek V4 selected-attention prefill for GFX1250.

Prefill gathers SWA/global rows into a BF16 workspace first; this kernel
only runs selected attention over that dense buffer.
"""

from __future__ import annotations

import math

import torch
from tokenspeed_kernel_amd._triton import gl, gluon, tl, triton

__all__ = ["launch_gluon_dsv4_prefill_gfx1250"]

_LOG2E = 1.4426950408889634
_BLOCK_H = 16
_NUM_WARPS = 4
# Each wave owns 32 rows of a step: one K step of its PV WMMA.
_WAVE_ROWS = 32
_NUM_XCDS = 8


def _prefill_launch_metadata(grid, kernel, args):
    """Report selected-attention capacity without reading device-resident lengths."""
    q = args["q"]
    tokens, heads = q.shape[0], q.shape[1]
    width = args["SELECTED_WIDTH"]
    head_dim = args["HEAD_DIM"]
    return {
        "name": kernel.name,
        # QK^T and PV over every selected slot.
        "flops16": 4 * tokens * heads * width * head_dim,
        # Each workgroup gathers its token's selected rows (and indices) once
        # for 16 heads, and reads q / writes out once.
        "bytes": tokens * grid[1] * width * (head_dim * 2 + 4)
        + 2 * q.numel() * q.element_size(),
    }


@gluon.constexpr_function
def _warp_split_row_layout(rows, num_warps, warp_size):
    """Rows split into contiguous per-wave blocks, replicated across lanes."""
    rows_per_warp = rows // num_warps
    return gl.DistributedLinearLayout(
        reg_bases=[[1 << i] for i in range(rows_per_warp.bit_length() - 1)],
        lane_bases=[[0]] * (warp_size.bit_length() - 1),
        warp_bases=[[rows_per_warp << i] for i in range(num_warps.bit_length() - 1)],
        block_bases=[],
        shape=[rows],
    )


@gluon.jit
def _load_rows(
    indices, index_row, positions, SELECTED_WIDTH: gl.constexpr, STEP_ROWS: gl.constexpr
):
    """Load one step's raw selected-row indices."""
    # A per-element mask would split the load into one load-and-wait per row.
    if SELECTED_WIDTH % STEP_ROWS == 0:
        return gl.amd.cdna5.buffer_load(indices + index_row, positions)
    else:
        return gl.amd.cdna5.buffer_load(
            indices + index_row, positions, mask=positions < SELECTED_WIDTH, other=-1
        )


@gluon.jit
def _gather_step(kv_desc, kv_tile, rows, positions, length, num_kv_rows):
    """Gather one step's rows; unusable ones read the zero-filled OOB row."""
    # Branch-free clamp: TDM takes the rows in SGPRs, and a lane-mask select
    # per row would cost a VALU->SALU dependency each.
    past_length = ~((positions - length) >> 31)
    rows = gl.minimum((rows | past_length).to(gl.uint32), num_kv_rows.to(gl.uint32))
    gl.amd.cdna5.tdm.async_gather(kv_desc, rows.to(gl.int32), kv_tile)


@gluon.jit
def _step_valid(indices, index_row, positions, length, num_kv_rows):
    """Return whether each selected position holds a usable row."""
    in_range = positions < length
    rows = gl.amd.cdna5.buffer_load(
        indices + index_row, positions, mask=in_range, other=-1
    )
    return in_range & (rows >= 0) & (rows < num_kv_rows)


@gluon.jit
def _q_chunk(
    q_shared, start: gl.constexpr, b_layout: gl.constexpr, NUM_WARPS: gl.constexpr
):
    """One 128-dim Q^T chunk, shared by every wave."""
    q_t = (
        q_shared.slice(start, 128, dim=1)
        .permute([1, 0])
        .load(gl.SliceLayout(0, b_layout))
    )
    return q_t[None, :, :].broadcast_to(NUM_WARPS, 128, q_t.shape[1])


@gluon.jit
def _chunk_scores(
    kv_waves,
    q_shared,
    start: gl.constexpr,
    a_layout: gl.constexpr,
    b_layout: gl.constexpr,
    zeros,
):
    """S^T partial over one 128-dim chunk for each wave's rows."""
    k = kv_waves.slice(start, 128, dim=2).load(a_layout)
    return gl.amd.cdna5.wmma(
        k, _q_chunk(q_shared, start, b_layout, zeros.shape[0]), zeros
    )


@gluon.jit
def _chunk_pv(kv_waves, start: gl.constexpr, p_t, acc, alpha, a_layout: gl.constexpr):
    """O^T[start:start + 128] = alpha O^T + V^T P^T for each wave."""
    v_t = kv_waves.slice(start, 128, dim=2).permute([0, 2, 1]).load(a_layout)
    return gl.amd.cdna5.wmma(v_t, p_t, acc * alpha[:, None, :])


@gluon.jit
def _store_chunk(
    out,
    partials,
    weight,
    token_idx,
    num_heads,
    head_offset,
    start: gl.constexpr,
    sum_layout: gl.constexpr,
):
    """Merge one 128-dim chunk of the per-wave O^T partials and store it."""
    chunk = gl.sum(
        partials.slice(start, 128, dim=1).load(sum_layout) * weight[:, None, :], axis=0
    )
    out_layout: gl.constexpr = gl.SliceLayout(0, sum_layout)
    dims = gl.arange(0, 128, layout=gl.SliceLayout(1, out_layout))
    heads = head_offset + gl.arange(
        0, weight.shape[1], layout=gl.SliceLayout(0, out_layout)
    )
    gl.store(
        out
        + (token_idx.to(tl.int64) * num_heads + heads[None, :]) * 512
        + start
        + dims[:, None],
        chunk.to(out.dtype.element_ty),
        mask=(heads < num_heads)[None, :],
    )


@gluon.jit(
    launch_metadata=_prefill_launch_metadata,
    do_not_specialize=("num_heads", "num_kv_rows"),
)
def gluon_dsv4_prefill_gfx1250(
    q,
    kv,
    indices,
    lens,
    attn_sink,
    out,
    qk_scale_log2: tl.float32,
    num_heads,
    num_kv_rows,
    SELECTED_WIDTH: gl.constexpr,
    BLOCK_H: gl.constexpr,
    WAVE_ROWS: gl.constexpr,
    HEAD_DIM: gl.constexpr,
    NUM_XCDS: gl.constexpr,
):
    # One workgroup per (token, 16-head group). Each wave runs its own online
    # softmax over a private WAVE_ROWS slice of every step in the transposed
    # form S^T = K Q^T, O^T += V^T P^T, which keeps the softmax and P inside
    # the wave; the per-wave softmaxes are merged once at the end.
    NUM_WARPS: gl.constexpr = gl.num_warps()
    STEP_ROWS: gl.constexpr = WAVE_ROWS * NUM_WARPS
    wmma: gl.constexpr = gl.amd.AMDWMMALayout(
        version=3,
        transposed=True,
        warp_bases=[[1, 0, 0], [2, 0, 0]],
        reg_bases=[],
        instr_shape=[16, 16, 32],
        rank=3,
    )
    a_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=wmma, k_width=8
    )
    b_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=wmma, k_width=8
    )
    score_layout: gl.constexpr = gl.SliceLayout(2, wmma)
    stat_layout: gl.constexpr = gl.SliceLayout(1, wmma)
    row_layout: gl.constexpr = _warp_split_row_layout(STEP_ROWS, NUM_WARPS, 32)
    q_load_layout: gl.constexpr = gl.BlockedLayout(
        [1, 8], [1, 32], [NUM_WARPS, 1], [1, 0]
    )
    # A thread merges all waves' copies of 8 contiguous dims of one head.
    sum_layout: gl.constexpr = gl.BlockedLayout(
        [NUM_WARPS, 8, 1], [1, 16, 2], [1, 1, NUM_WARPS], [1, 2, 0]
    )
    kv_shared_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[512, 16]], [STEP_ROWS, HEAD_DIM], [1, 0]
    )
    # Aliasing the KV tile requires its byte padding pattern (32 B per 1 KiB).
    partial_shared_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[256, 8]], [NUM_WARPS, HEAD_DIM, BLOCK_H], [1, 2, 0]
    )
    stat_shared_layout: gl.constexpr = gl.SwizzledSharedLayout(
        vec=1, per_phase=1, max_phase=1, order=[1, 0]
    )

    # Workgroups are dispatched round-robin over the XCDs; give each XCD a
    # contiguous token range so neighbouring tokens' shared KV rows (sliding
    # window, overlapping top-k) hit in that XCD's L2.
    pid = gl.program_id(axis=0)
    num_tokens = gl.num_programs(axis=0)
    xcd = pid % NUM_XCDS
    token_idx = (
        xcd * (num_tokens // NUM_XCDS)
        + gl.minimum(xcd, num_tokens % NUM_XCDS)
        + pid // NUM_XCDS
    )
    head_offset = gl.program_id(axis=1) * BLOCK_H
    valid_len = gl.load(lens + token_idx)
    effective_len = gl.minimum(gl.maximum(valid_len, 0), SELECTED_WIDTH)
    num_steps = gl.cdiv(effective_len, STEP_ROWS)
    last_step = gl.maximum(num_steps - 1, 0)
    index_row = token_idx.to(tl.int64) * SELECTED_WIDTH

    # One KV step in LDS, wave w's rows at [w * WAVE_ROWS, (w + 1) * WAVE_ROWS).
    # A second buffer would halve the workgroups per CU.
    kv_desc = gl.amd.cdna5.tdm.make_tensor_descriptor(
        base=kv,
        shape=[num_kv_rows, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[STEP_ROWS, HEAD_DIM],
        layout=kv_shared_layout,
    )
    kv_tile = gl.allocate_shared_memory(
        gl.bfloat16, [STEP_ROWS, HEAD_DIM], layout=kv_shared_layout
    )
    kv_waves = kv_tile.reshape([NUM_WARPS, WAVE_ROWS, HEAD_DIM])
    gather_positions = gl.arange(0, STEP_ROWS, layout=row_layout)
    score_positions = (
        gl.arange(0, NUM_WARPS, layout=gl.SliceLayout(1, score_layout))[:, None]
        * WAVE_ROWS
        + gl.arange(0, WAVE_ROWS, layout=gl.SliceLayout(0, score_layout))[None, :]
    )

    # The first gather overlaps the q load. Each step's indices are loaded a
    # step ahead of their gather.
    rows = _load_rows(indices, index_row, gather_positions, SELECTED_WIDTH, STEP_ROWS)
    q_rows = head_offset + gl.arange(
        0, BLOCK_H, layout=gl.SliceLayout(1, q_load_layout)
    )
    q_cols = gl.arange(0, HEAD_DIM, layout=gl.SliceLayout(0, q_load_layout))
    q_tile = gl.load(
        q
        + (token_idx.to(tl.int64) * num_heads + q_rows[:, None]) * HEAD_DIM
        + q_cols[None, :],
        mask=(q_rows < num_heads)[:, None],
        other=0.0,
    )
    if num_steps > 0:
        _gather_step(
            kv_desc, kv_tile, rows, gather_positions, effective_len, num_kv_rows
        )
    rows = _load_rows(
        indices,
        index_row,
        gl.minimum(1, last_step) * STEP_ROWS + gather_positions,
        SELECTED_WIDTH,
        STEP_ROWS,
    )
    # Q^T stays in LDS: holding it in registers costs the second workgroup
    # per CU.
    q_shared = gl.allocate_shared_memory(
        gl.bfloat16,
        [BLOCK_H, HEAD_DIM],
        layout=gl.PaddedSharedLayout.with_identity_for(
            [[512, 16]], [BLOCK_H, HEAD_DIM], [1, 0]
        ),
        value=q_tile,
    )

    max_value = gl.full([NUM_WARPS, BLOCK_H], -float("inf"), gl.float32, stat_layout)
    denominator = gl.zeros([NUM_WARPS, BLOCK_H], gl.float32, stat_layout)
    acc_0 = gl.zeros([NUM_WARPS, 128, BLOCK_H], gl.float32, wmma)
    acc_1 = gl.zeros([NUM_WARPS, 128, BLOCK_H], gl.float32, wmma)
    acc_2 = gl.zeros([NUM_WARPS, 128, BLOCK_H], gl.float32, wmma)
    acc_3 = gl.zeros([NUM_WARPS, 128, BLOCK_H], gl.float32, wmma)
    zeros = gl.zeros([NUM_WARPS, WAVE_ROWS, BLOCK_H], gl.float32, wmma)
    for step in range(num_steps):
        score_valid = _step_valid(
            indices,
            index_row,
            step * STEP_ROWS + score_positions,
            effective_len,
            num_kv_rows,
        )
        gl.amd.cdna5.tdm.async_wait(0)
        # Four independent WMMA chains over the head dims.
        scores = (
            _chunk_scores(kv_waves, q_shared, 0, a_layout, b_layout, zeros)
            + _chunk_scores(kv_waves, q_shared, 128, a_layout, b_layout, zeros)
        ) + (
            _chunk_scores(kv_waves, q_shared, 256, a_layout, b_layout, zeros)
            + _chunk_scores(kv_waves, q_shared, 384, a_layout, b_layout, zeros)
        )
        scores = gl.where(
            score_valid[:, :, None], scores * qk_scale_log2, -float("inf")
        )
        next_max = gl.maximum(max_value, gl.max(scores, axis=1))
        safe_max = gl.where(next_max > -float("inf"), next_max, 0.0)
        alpha = gl.exp2(max_value - safe_max)
        max_value = next_max
        probabilities = gl.exp2(scores - safe_max[:, None, :])
        denominator = denominator * alpha + gl.sum(probabilities, axis=1)
        p_t = gl.convert_layout(probabilities.to(gl.bfloat16), b_layout)
        acc_0 = _chunk_pv(kv_waves, 0, p_t, acc_0, alpha, a_layout)
        acc_1 = _chunk_pv(kv_waves, 128, p_t, acc_1, alpha, a_layout)
        acc_2 = _chunk_pv(kv_waves, 256, p_t, acc_2, alpha, a_layout)
        acc_3 = _chunk_pv(kv_waves, 384, p_t, acc_3, alpha, a_layout)
        if step + 1 < num_steps:
            _gather_step(
                kv_desc,
                kv_tile,
                rows,
                (step + 1) * STEP_ROWS + gather_positions,
                effective_len,
                num_kv_rows,
            )
            rows = _load_rows(
                indices,
                index_row,
                gl.minimum(step + 2, last_step) * STEP_ROWS + gather_positions,
                SELECTED_WIDTH,
                STEP_ROWS,
            )

    # Merge the per-wave softmaxes behind a single barrier: each wave parks
    # its unnormalized O^T partial (in the dead KV tile) and its statistics
    # in LDS, and each thread merges all waves' copies of its outputs.
    partials = kv_tile.reinterpret(
        gl.float32, [NUM_WARPS, HEAD_DIM, BLOCK_H], partial_shared_layout
    )
    partials.slice(0, 128, dim=1).store(acc_0)
    partials.slice(128, 128, dim=1).store(acc_1)
    partials.slice(256, 128, dim=1).store(acc_2)
    partials.slice(384, 128, dim=1).store(acc_3)
    max_shared = gl.allocate_shared_memory(
        gl.float32, [NUM_WARPS, BLOCK_H], layout=stat_shared_layout, value=max_value
    )
    sum_shared = gl.allocate_shared_memory(
        gl.float32, [NUM_WARPS, BLOCK_H], layout=stat_shared_layout, value=denominator
    )
    wave_stat_layout: gl.constexpr = gl.SliceLayout(1, sum_layout)
    wave_max = max_shared.load(wave_stat_layout)
    wave_sum = sum_shared.load(wave_stat_layout)
    heads = head_offset + gl.arange(
        0, BLOCK_H, layout=gl.SliceLayout(0, wave_stat_layout)
    )
    # The sink joins the softmax with a unit weight and no value.
    sink = (
        gl.load(attn_sink + heads, mask=heads < num_heads, other=0.0).to(gl.float32)
        * 1.4426950408889634  # log2(e): the sink joins the base-2 softmax.
    )
    total_max = gl.maximum(gl.max(wave_max, axis=0), sink)
    # No usable row and a -inf sink leave an all-zero output.
    total_max = gl.where(total_max > -float("inf"), total_max, 0.0)
    weight = gl.exp2(wave_max - total_max[None, :])
    total = gl.sum(wave_sum * weight, axis=0) + gl.exp2(sink - total_max)
    weight = weight * gl.where(total > 0.0, 1.0 / total, 0.0)[None, :]
    _store_chunk(
        out, partials, weight, token_idx, num_heads, head_offset, 0, sum_layout
    )
    _store_chunk(
        out, partials, weight, token_idx, num_heads, head_offset, 128, sum_layout
    )
    _store_chunk(
        out, partials, weight, token_idx, num_heads, head_offset, 256, sum_layout
    )
    _store_chunk(
        out, partials, weight, token_idx, num_heads, head_offset, 384, sum_layout
    )


def _check_tensor(name: str, tensor: object) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    return tensor


def _shares_storage(tensor: torch.Tensor, other: torch.Tensor) -> bool:
    if tensor.numel() == 0 or other.numel() == 0:
        return False
    return tensor.untyped_storage().data_ptr() == other.untyped_storage().data_ptr()


def _validate_inputs(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    lens: torch.Tensor,
    attn_sink: torch.Tensor,
    out: torch.Tensor | None,
) -> None:
    if q.dtype != torch.bfloat16:
        raise TypeError(f"q must be BF16, got {q.dtype}")
    if q.dim() != 3 or q.shape[2] != 512:
        raise ValueError(
            f"q must have shape [tokens, heads, 512], got {tuple(q.shape)}"
        )
    if not q.is_cuda:
        raise ValueError("q must be on an AMD GPU")
    if not q.is_contiguous():
        raise ValueError("q must be contiguous")

    if kv.dtype != torch.bfloat16:
        raise TypeError(f"kv must be BF16, got {kv.dtype}")
    if kv.device != q.device:
        raise ValueError("kv must be on the same device as q")
    if not kv.is_contiguous() or kv.numel() % 512 != 0:
        raise ValueError("kv must be contiguous and reshapeable to [-1, 512]")

    if indices.dtype != torch.int32:
        raise TypeError(f"indices must be int32, got {indices.dtype}")
    if indices.dim() != 2 or indices.shape[0] != q.shape[0]:
        raise ValueError(
            "indices must have shape [tokens, selected_width], got "
            f"{tuple(indices.shape)}"
        )
    if indices.device != q.device or not indices.is_contiguous():
        raise ValueError("indices must be contiguous and on the same device as q")

    if lens.dtype != torch.int32:
        raise TypeError(f"lens must be int32, got {lens.dtype}")
    if lens.shape != (q.shape[0],):
        raise ValueError(f"lens must have shape [tokens], got {tuple(lens.shape)}")
    if lens.device != q.device or not lens.is_contiguous():
        raise ValueError("lens must be contiguous and on the same device as q")

    if attn_sink.dtype not in (torch.float32, torch.bfloat16):
        raise TypeError(f"attn_sink must be FP32 or BF16, got {attn_sink.dtype}")
    if attn_sink.device != q.device or not attn_sink.is_contiguous():
        raise ValueError("attn_sink must be contiguous and on the same device as q")
    if attn_sink.numel() < q.shape[1]:
        raise ValueError("attn_sink must provide at least one value per query head")

    if out is None:
        return
    if out.dtype != torch.bfloat16:
        raise TypeError(f"out must be BF16, got {out.dtype}")
    if out.shape != q.shape:
        raise ValueError(
            f"out must have exact shape {tuple(q.shape)}, got {tuple(out.shape)}"
        )
    if out.device != q.device or not out.is_contiguous():
        raise ValueError("out must be contiguous and on the same device as q")
    for name, tensor in (
        ("q", q),
        ("kv", kv),
        ("indices", indices),
        ("lens", lens),
        ("attn_sink", attn_sink),
    ):
        if _shares_storage(out, tensor):
            raise ValueError(f"out must not alias {name}")


def launch_gluon_dsv4_prefill_gfx1250(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    lens: torch.Tensor,
    attn_sink: torch.Tensor,
    softmax_scale: float,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run dense-workspace selected attention for DeepSeek V4 on GFX1250.

    Args:
        q: Contiguous BF16 queries shaped `[tokens, heads, 512]`.
        kv: Contiguous BF16 storage reshapeable to rows of 512 channels.
        indices: Contiguous int32 selected-row indices shaped
            `[tokens, selected_width]`. Negative entries are ignored.
        lens: Contiguous int32 valid selected widths shaped `[tokens]`.
        attn_sink: Contiguous FP32 or BF16 sink logits with at least one value
            per query head.
        softmax_scale: Scale applied to query-key dot products.
        out: Optional exact contiguous BF16 output shaped like `q`.

    Returns:
        The BF16 selected-attention output shaped `[tokens, heads, 512]`.
    """

    q = _check_tensor("q", q)
    kv = _check_tensor("kv", kv)
    indices = _check_tensor("indices", indices)
    lens = _check_tensor("lens", lens)
    attn_sink = _check_tensor("attn_sink", attn_sink)
    if out is not None:
        out = _check_tensor("out", out)
    _validate_inputs(q, kv, indices, lens, attn_sink, out)

    try:
        scale = float(softmax_scale)
    except (TypeError, ValueError, OverflowError) as error:
        raise TypeError("softmax_scale must be a finite real scalar") from error
    if not math.isfinite(scale):
        raise ValueError("softmax_scale must be finite")

    output = out if out is not None else torch.empty_like(q)
    if q.shape[0] == 0 or q.shape[1] == 0 or indices.shape[1] == 0:
        output.zero_()
        return output

    kv_rows = kv.reshape(-1, 512)
    grid = (q.shape[0], triton.cdiv(q.shape[1], _BLOCK_H))
    gluon_dsv4_prefill_gfx1250[grid](
        q,
        kv_rows,
        indices,
        lens,
        attn_sink.reshape(-1),
        output,
        scale * _LOG2E,
        q.shape[1],
        kv_rows.shape[0],
        SELECTED_WIDTH=indices.shape[1],
        BLOCK_H=_BLOCK_H,
        WAVE_ROWS=_WAVE_ROWS,
        HEAD_DIM=512,
        NUM_XCDS=_NUM_XCDS,
        num_warps=_NUM_WARPS,
    )
    return output
