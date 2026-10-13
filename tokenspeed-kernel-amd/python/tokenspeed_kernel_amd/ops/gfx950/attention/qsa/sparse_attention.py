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

"""QSA sparse attention over per-row selected KV-cache slots for GFX950.

A one-wave workgroup owns one query row, up to 16 query heads of one KV head's
GQA group (the MFMA M dimension) and a share of the row's selected slots. The
row's valid slots are first compacted in LDS, so invalid columns cost no
gathers whatever their order. Each 16-slot tile is gathered straight into
registers (K in the MFMA operand layout, V transposed through LDS) one tile
ahead of the compute. Small batches split the compacted slots over workgroups
and merge the partial softmax states in a companion kernel. See the "gfx950 QSA
sparse attention" section of ``ops/README.md``.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._triton import gl, gluon, triton

cdna4 = gl.amd.cdna4

__all__ = ["launch_gluon_qsa_sparse_attention_gfx950"]

_LOG2E = 1.4426950408889634
_GFX950_CU_COUNT = 256
# Selected slots per tile.
_BLOCK_N = 16
# Split each row's slots until about four one-wave workgroups run per CU.
_TARGET_WORKGROUPS = 4 * _GFX950_CU_COUNT
_MAX_SPLITS = 32


def _attention_metadata(grid, kernel, args):
    # Capacity bound without reading slots: every selected slot is valid.
    rows = args["q"].shape[0]
    heads = args["q"].shape[1]
    kv_heads = args["k_cache"].shape[1]
    width = args["WIDTH"]
    head_dim = args["HEAD_DIM"]
    value_dim = args["VALUE_DIM"]
    kv_bytes = args["k_cache"].element_size()
    splits = args["num_splits"]
    out_bytes = heads * value_dim * (2 if splits == 1 else 4 * splits)
    return {
        "name": kernel.name,
        "flops16": 2 * rows * heads * width * (head_dim + value_dim),
        "bytes": rows
        * (
            width * 4
            + kv_heads * width * (head_dim + value_dim) * kv_bytes
            + heads * head_dim * 2
            + out_bytes
        ),
    }


def _merge_metadata(grid, kernel, args):
    rows_heads = grid[0] * grid[1]
    value_dim = args["VALUE_DIM"]
    splits = args["num_splits"]
    return {
        "name": kernel.name,
        "bytes": rows_heads * (splits * (value_dim + 1) * 4 + value_dim * 2),
    }


@gluon.jit
def _load_tile(
    slots_smem,
    k_cache,
    v_cache,
    tile,
    num_valid,
    active,
    k_head,
    v_head,
    stride_k_n,
    stride_v_n,
    k_layout: gl.constexpr,
    v_load: gl.constexpr,
    s_layout: gl.constexpr,
    HEAD_DIM: gl.constexpr,
    VALUE_DIM: gl.constexpr,
    BLOCK_N: gl.constexpr,
):
    # Slot ids come from LDS: a global load here would wait for the in-order
    # gathers still in flight. K lands directly in the MFMA B-operand layout
    # (head_dim contiguous per lane); V is transposed through LDS later.
    k_n = tile * BLOCK_N + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, k_layout))
    v_n = tile * BLOCK_N + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(1, v_load))
    s_n = tile * BLOCK_N + gl.arange(0, BLOCK_N, layout=s_layout)
    # Lanes past the row's valid slots read slot 1, the finite masked-read
    # target, so the gathers need no predication.
    k_slots = gl.where((k_n < num_valid) & active, slots_smem.gather(k_n, 0), 1)
    v_slots = gl.where((v_n < num_valid) & active, slots_smem.gather(v_n, 0), 1)
    k_d = gl.arange(0, HEAD_DIM, layout=gl.SliceLayout(1, k_layout))
    v_d = gl.arange(0, VALUE_DIM, layout=gl.SliceLayout(0, v_load))
    # 64-bit addressing: per-layer caches exceed the 2 GiB reach of buffer
    # loads.
    keys = gl.load(
        k_cache + k_head + k_slots.to(gl.int64)[None, :] * stride_k_n + k_d[:, None]
    )
    values = gl.load(
        v_cache + v_head + v_slots.to(gl.int64)[:, None] * stride_v_n + v_d[None, :]
    )
    return keys, values, s_n < num_valid


@gluon.jit
def _attend_tile(
    q_op,
    k_tile,
    v_tile,
    s_valid,
    m_i,
    l_i,
    acc,
    qk_scale,
    v_smem,
    qk_mma: gl.constexpr,
    pv_mma: gl.constexpr,
    p_layout: gl.constexpr,
    v_layout: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    IS_FP8: gl.constexpr,
):
    if IS_FP8:
        k_tile = k_tile.to(q_op.dtype)
    scores = gl.zeros([BLOCK_M, BLOCK_N], gl.float32, qk_mma)
    scores = cdna4.mfma(q_op, k_tile, scores)
    scores = gl.where(s_valid[None, :], scores * qk_scale, float("-inf"))
    m_new = gl.maximum(m_i, gl.max(scores, axis=1))
    # Heads without a valid slot so far keep m = -inf; avoid -inf - -inf.
    m_safe = gl.where(m_new == float("-inf"), 0.0, m_new)
    alpha = gl.exp2(m_i - m_safe)
    p = gl.exp2(scores - m_safe[:, None])
    l_i = l_i * alpha + gl.sum(p, axis=1)
    acc = acc * gl.convert_layout(alpha, gl.SliceLayout(1, pv_mma))[:, None]
    p_op = gl.convert_layout(p.to(q_op.dtype), p_layout)
    # Transpose V through LDS: vector stores, then transposed reads.
    v_smem.store(v_tile)
    v_op = v_smem.load(v_layout)
    if IS_FP8:
        v_op = v_op.to(q_op.dtype)
    acc = cdna4.mfma(p_op, v_op, acc)
    return m_new, l_i, acc


@gluon.jit(
    launch_metadata=_attention_metadata,
    do_not_specialize=["num_splits"],
)
def gluon_qsa_sparse_attention_gfx950(
    q,
    k_cache,
    v_cache,
    selected_slots,
    output,
    partial_out,
    partial_lse,
    scale_log2,
    k_scale,
    v_scale,
    num_splits,
    stride_q_n,
    stride_q_h,
    stride_k_n,
    stride_k_h,
    stride_v_n,
    stride_v_h,
    stride_s_n,
    stride_o_n,
    stride_o_h,
    GROUP: gl.constexpr,
    NUM_KV_HEADS: gl.constexpr,
    HEAD_DIM: gl.constexpr,
    VALUE_DIM: gl.constexpr,
    WIDTH: gl.constexpr,
    WIDTH_HEAD: gl.constexpr,
    WIDTH_TAIL: gl.constexpr,
    BLOCK_N: gl.constexpr,
    SPLIT: gl.constexpr,
    num_warps: gl.constexpr,
):
    BLOCK_M: gl.constexpr = 16
    HEAD_BLOCKS: gl.constexpr = (GROUP + BLOCK_M - 1) // BLOCK_M
    gl.static_assert(num_warps == 1)
    gl.static_assert(BLOCK_N == 16)
    kv_dtype: gl.constexpr = k_cache.dtype.element_ty
    IS_FP8: gl.constexpr = kv_dtype.primitive_bitwidth == 8
    # One wave per workgroup: no softmax state or LDS buffer crosses waves,
    # so the compiler's LDS barriers cost nothing.
    qk_mma: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[1, 1]
    )
    pv_mma: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 16], transposed=True, warps_per_cta=[1, 1]
    )
    q_layout: gl.constexpr = gl.DotOperandLayout(0, qk_mma, 8)
    k_layout: gl.constexpr = gl.DotOperandLayout(1, qk_mma, 8)
    p_layout: gl.constexpr = gl.DotOperandLayout(0, pv_mma, 4)
    v_layout: gl.constexpr = gl.DotOperandLayout(1, pv_mma, 4)
    LOAD_VEC: gl.constexpr = 16 if IS_FP8 else 8
    V_THREADS: gl.constexpr = VALUE_DIM // LOAD_VEC
    v_load: gl.constexpr = gl.BlockedLayout(
        [1, LOAD_VEC], [64 // V_THREADS, V_THREADS], [1, 1], [1, 0]
    )
    head_layout: gl.constexpr = gl.BlockedLayout([4], [64], [1], [0])
    tail_layout: gl.constexpr = gl.BlockedLayout([1], [64], [1], [0])
    s_layout: gl.constexpr = gl.SliceLayout(0, qk_mma)
    slots_layout: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0])

    work = gl.program_id(0)
    split = work % num_splits
    row_head = work // num_splits
    head_block = row_head % HEAD_BLOCKS
    kv_head = row_head // HEAD_BLOCKS % NUM_KV_HEADS
    row = row_head // (HEAD_BLOCKS * NUM_KV_HEADS)
    # First query head of this workgroup's block of the GQA group.
    group_start = head_block * BLOCK_M

    # Compact the row's valid (positive) slots to the front, in LDS. The row
    # is read as a power-of-two head plus a short tail (2048 + 3 for QSA);
    # invalid entries fill the positions after the valid ones, so the two
    # scatters together write a permutation of the row.
    slot_row = selected_slots + row.to(gl.int64) * stride_s_n
    head_cols = gl.arange(0, WIDTH_HEAD, layout=head_layout)
    head = cdna4.buffer_load(slot_row, head_cols)
    head_valid = (head > 0).to(gl.int32)
    head_before = gl.cumsum(head_valid, axis=0) - head_valid
    num_valid = gl.sum(head_valid, axis=0)
    slots_smem = gl.allocate_shared_memory(
        gl.int32, [WIDTH_HEAD * 2 if WIDTH_TAIL > 0 else WIDTH_HEAD], slots_layout
    )
    if WIDTH_TAIL > 0:
        tail_cols = gl.arange(0, WIDTH_TAIL, layout=tail_layout)
        tail = cdna4.buffer_load(
            slot_row,
            WIDTH_HEAD + tail_cols,
            mask=WIDTH_HEAD + tail_cols < WIDTH,
            other=-1,
        )
        tail_valid = (tail > 0).to(gl.int32)
        tail_before = gl.cumsum(tail_valid, axis=0) - tail_valid
        num_head_valid = num_valid
        num_valid = num_valid + gl.sum(tail_valid, axis=0)
        tail_pos = gl.where(
            tail_valid > 0,
            num_head_valid + tail_before,
            num_valid + (WIDTH_HEAD - num_head_valid) + tail_cols - tail_before,
        )
        slots_smem.scatter(tail, tail_pos, 0)
    head_pos = gl.where(
        head_valid > 0, head_before, num_valid + head_cols - head_before
    )
    slots_smem.scatter(head, head_pos, 0)

    num_tiles = gl.cdiv(num_valid, BLOCK_N)
    per_split = gl.cdiv(num_tiles, num_splits)
    first = split * per_split
    last = gl.minimum(first + per_split, num_tiles)

    # A block of the GQA group's query heads, padded to the MFMA M size with
    # zeros. Q lives in LDS and is re-read per tile, leaving the arch VGPRs to
    # the gathers in flight.
    q_m = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, q_layout))
    q_d = gl.arange(0, HEAD_DIM, layout=gl.SliceLayout(0, q_layout))
    q_offsets = (
        row.to(gl.int64) * stride_q_n
        + ((kv_head * GROUP + group_start + q_m) * stride_q_h)[:, None]
        + q_d[None, :]
    )
    q_smem = gl.allocate_shared_memory(
        q.dtype.element_ty,
        [BLOCK_M, HEAD_DIM],
        gl.PaddedSharedLayout.with_identity_for(
            [[HEAD_DIM, 16]], [BLOCK_M, HEAD_DIM], [1, 0]
        ),
        gl.load(q + q_offsets, mask=(group_start + q_m < GROUP)[:, None], other=0.0),
    )
    v_smem = gl.allocate_shared_memory(
        kv_dtype,
        [BLOCK_N, VALUE_DIM],
        gl.PaddedSharedLayout.with_identity_for(
            [[VALUE_DIM, 16]], [BLOCK_N, VALUE_DIM], [1, 0]
        ),
    )

    k_head = kv_head * stride_k_h
    v_head = kv_head * stride_v_h
    m_i = gl.full([BLOCK_M], float("-inf"), gl.float32, gl.SliceLayout(1, qk_mma))
    l_i = gl.zeros([BLOCK_M], gl.float32, gl.SliceLayout(1, qk_mma))
    acc = gl.zeros([BLOCK_M, VALUE_DIM], gl.float32, pv_mma)
    qk_scale = scale_log2 * k_scale

    # Gathers run one tile ahead. Loads complete in order, so the wait for
    # tile i leaves tile i + 1 in flight.
    k_cur, v_cur, s_cur = _load_tile(
        slots_smem,
        k_cache,
        v_cache,
        first,
        num_valid,
        first < last,
        k_head,
        v_head,
        stride_k_n,
        stride_v_n,
        k_layout,
        v_load,
        s_layout,
        HEAD_DIM,
        VALUE_DIM,
        BLOCK_N,
    )
    for i in range(first, last):
        k_nxt, v_nxt, s_nxt = _load_tile(
            slots_smem,
            k_cache,
            v_cache,
            i + 1,
            num_valid,
            i + 1 < last,
            k_head,
            v_head,
            stride_k_n,
            stride_v_n,
            k_layout,
            v_load,
            s_layout,
            HEAD_DIM,
            VALUE_DIM,
            BLOCK_N,
        )
        m_i, l_i, acc = _attend_tile(
            q_smem.load(q_layout),
            k_cur,
            v_cur,
            s_cur,
            m_i,
            l_i,
            acc,
            qk_scale,
            v_smem,
            qk_mma,
            pv_mma,
            p_layout,
            v_layout,
            BLOCK_M,
            BLOCK_N,
            IS_FP8,
        )
        k_cur = k_nxt
        v_cur = v_nxt
        s_cur = s_nxt

    l_i = gl.convert_layout(l_i, gl.SliceLayout(1, pv_mma))
    m_i = gl.convert_layout(m_i, gl.SliceLayout(1, pv_mma))
    # Heads without any valid slot produce zeros.
    out = acc * (v_scale / gl.where(l_i > 0, l_i, 1.0))[:, None]
    o_m = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, pv_mma))
    o_d = gl.arange(0, VALUE_DIM, layout=gl.SliceLayout(0, pv_mma))
    o_heads = kv_head * GROUP + group_start + o_m
    o_valid = group_start + o_m < GROUP
    head_mask = o_valid[:, None]
    if SPLIT:
        part = (
            row.to(gl.int64) * (GROUP * NUM_KV_HEADS) + o_heads
        ) * num_splits + split
        gl.store(
            partial_out + part[:, None] * VALUE_DIM + o_d[None, :],
            out,
            mask=head_mask,
        )
        # Base-2 log-sum-exp; -inf marks a split without valid slots.
        lse = gl.where(l_i > 0, m_i + gl.log2(l_i), float("-inf"))
        gl.store(partial_lse + part, lse, mask=o_valid)
    else:
        o_offsets = (
            row.to(gl.int64) * stride_o_n
            + o_heads[:, None].to(gl.int64) * stride_o_h
            + o_d[None, :]
        )
        gl.store(output + o_offsets, out.to(output.dtype.element_ty), mask=head_mask)


@gluon.jit(launch_metadata=_merge_metadata, do_not_specialize=["num_splits"])
def gluon_qsa_sparse_attention_merge_gfx950(
    partial_out,
    partial_lse,
    output,
    num_splits,
    stride_o_n,
    stride_o_h,
    NUM_HEADS: gl.constexpr,
    VALUE_DIM: gl.constexpr,
    MAX_SPLITS: gl.constexpr,
    num_warps: gl.constexpr,
):
    # One program merges every split of one (row, head) in a single pass.
    layout: gl.constexpr = gl.BlockedLayout([1, 4], [1, 64], [num_warps, 1], [1, 0])
    row = gl.program_id(0)
    head = gl.program_id(1)
    splits = gl.arange(0, MAX_SPLITS, layout=gl.SliceLayout(1, layout))
    dv = gl.arange(0, VALUE_DIM, layout=gl.SliceLayout(0, layout))
    split_mask = splits < num_splits
    base = (row.to(gl.int64) * NUM_HEADS + head) * num_splits
    lse = gl.load(partial_lse + base + splits, mask=split_mask, other=float("-inf"))
    parts = gl.load(
        partial_out + (base + splits)[:, None] * VALUE_DIM + dv[None, :],
        mask=split_mask[:, None],
        other=0.0,
    )
    lse_max = gl.max(lse, axis=0)
    lse_max = gl.where(lse_max == float("-inf"), 0.0, lse_max)
    # Splits without valid slots carry lse = -inf and weigh zero.
    weights = gl.exp2(lse - lse_max)
    total = gl.sum(weights, axis=0)
    acc = gl.sum(parts * weights[:, None], axis=0)
    acc = acc / gl.where(total > 0, total, 1.0)
    gl.store(
        output + row.to(gl.int64) * stride_o_n + head * stride_o_h + dv,
        acc.to(output.dtype.element_ty),
    )


def _scalar_descale(scale: float | torch.Tensor | None, name: str) -> float:
    if scale is None:
        return 1.0
    if isinstance(scale, torch.Tensor):
        raise TypeError(f"Gluon QSA sparse attention expects a float {name}")
    return float(scale)


def _num_splits(rows: int, num_kv_heads: int, width: int) -> int:
    """Splits per (row, KV head) so small batches still fill the GPU."""
    programs = rows * num_kv_heads
    max_splits = min(_MAX_SPLITS, triton.cdiv(width, _BLOCK_N))
    return max(1, min(max_splits, triton.cdiv(_TARGET_WORKGROUPS, programs)))


def launch_gluon_qsa_sparse_attention_gfx950(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    selected_slots: torch.Tensor,
    *,
    scale: float,
    max_seqlen_q: int,
    metadata_capacity_rows: int | None,
    k_scale: float | torch.Tensor | None,
    v_scale: float | torch.Tensor | None,
) -> torch.Tensor:
    """QSA sparse attention on GFX950; see ``qsa_sparse_attention``.

    Every query row attends independently to its own selected slots, so
    ``max_seqlen_q`` does not change the computation, and the kernels keep no
    metadata workspace sized by ``metadata_capacity_rows``.
    """

    del max_seqlen_q, metadata_capacity_rows
    rows, num_heads, head_dim = q.shape
    num_kv_heads = k_cache.shape[1]
    value_dim = v_cache.shape[-1]
    width = selected_slots.shape[1]
    if num_heads % num_kv_heads:
        raise ValueError("QSA query heads must be a multiple of KV heads")
    group = num_heads // num_kv_heads
    if q.stride(-1) != 1 or k_cache.stride(-1) != 1 or v_cache.stride(-1) != 1:
        raise ValueError("QSA Q/K/V head dimensions must be contiguous")
    if selected_slots.stride(-1) != 1 or selected_slots.dtype != torch.int32:
        raise ValueError("QSA selected slots must be contiguous int32 rows")
    output = q.new_empty((rows, num_heads, value_dim))
    if rows == 0:
        return output
    # Workgroups own up to 16 query heads of one KV head's GQA group.
    head_blocks = triton.cdiv(group, 16)
    num_splits = _num_splits(rows, num_kv_heads * head_blocks, width)
    use_split = num_splits > 1
    if use_split:
        partial_out = torch.empty(
            (rows, num_heads, num_splits, value_dim),
            dtype=torch.float32,
            device=q.device,
        )
        partial_lse = torch.empty(
            (rows, num_heads, num_splits), dtype=torch.float32, device=q.device
        )
    else:
        partial_out = partial_lse = output
    # A power-of-two head and a padded tail cover the selected width.
    width_head = 1 << (width.bit_length() - 1)
    width_tail = (
        0 if width == width_head else triton.next_power_of_2(width - width_head)
    )
    gluon_qsa_sparse_attention_gfx950[
        (rows * num_kv_heads * head_blocks * num_splits,)
    ](
        q,
        k_cache,
        v_cache,
        selected_slots,
        output,
        partial_out,
        partial_lse,
        scale * _LOG2E,
        _scalar_descale(k_scale, "k_scale"),
        _scalar_descale(v_scale, "v_scale"),
        num_splits,
        q.stride(0),
        q.stride(1),
        k_cache.stride(0),
        k_cache.stride(1),
        v_cache.stride(0),
        v_cache.stride(1),
        selected_slots.stride(0),
        output.stride(0),
        output.stride(1),
        GROUP=group,
        NUM_KV_HEADS=num_kv_heads,
        HEAD_DIM=head_dim,
        VALUE_DIM=value_dim,
        WIDTH=width,
        WIDTH_HEAD=width_head,
        WIDTH_TAIL=width_tail,
        BLOCK_N=_BLOCK_N,
        SPLIT=use_split,
        num_warps=1,
    )
    if use_split:
        gluon_qsa_sparse_attention_merge_gfx950[(rows, num_heads)](
            partial_out,
            partial_lse,
            output,
            num_splits,
            output.stride(0),
            output.stride(1),
            NUM_HEADS=num_heads,
            VALUE_DIM=value_dim,
            MAX_SPLITS=_MAX_SPLITS,
            num_warps=2,
        )
    return output
