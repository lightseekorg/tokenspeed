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

"""DeepSeek V4 prefill selected-attention kernel for AMD GFX950."""

from __future__ import annotations

import math

import torch
from tokenspeed_kernel_amd._triton import gl, gluon, tl, triton
from tokenspeed_kernel_amd.ops.gfx950.attention.dsv4.sparse_prefill import (
    gluon_dsv4_sparse_prefill_gfx950,
)

__all__ = ["launch_gluon_dsv4_prefill_gfx950"]


def _use_sparse_prefill(q: torch.Tensor, indices: torch.Tensor) -> bool:
    # Compact H=64/128 helper does not skip -1 pads; width 128 uses the generic kernel.
    return q.shape[1] in (64, 128) and indices.shape[1] > 128


_LOG2E = 1.4426950408889634
_BLOCK_H = 16
_TILE_K = 32
_NUM_WARPS = 4
_NUM_XCDS = 8
# A row whose byte offset (2^21 rows x 1 KiB = 2^31) lies past every buffer
# range: buffer loads from it return zeros.
_OOB_ROW = gl.constexpr(1 << 21)


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


@gluon.jit(
    launch_metadata=_prefill_launch_metadata,
    do_not_specialize=("num_heads", "num_kv_rows"),
)
def gluon_dsv4_prefill_gfx950(
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
    TILE_K: gl.constexpr,
    HEAD_DIM: gl.constexpr,
    INDEX_GROUPS_POW2: gl.constexpr,
    NUM_XCDS: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    # One workgroup per (token, 16-head group). The 16 heads are the MFMA M
    # dimension, so each gathered KV row feeds all heads. Wave w owns head
    # channels [w * SPLIT_D, (w + 1) * SPLIT_D): it computes a partial QK^T
    # over them, the partials are summed through LDS, and wave w runs the
    # online softmax for heads [w * SPLIT_H, (w + 1) * SPLIT_H) and shares P
    # and the rescale factors back through LDS for the PV MFMA.
    SPLIT_D: gl.constexpr = HEAD_DIM // NUM_WARPS
    SPLIT_H: gl.constexpr = BLOCK_H // NUM_WARPS
    INDEX_GROUP: gl.constexpr = 64 * NUM_WARPS
    NUM_INDEX_TILES: gl.constexpr = SELECTED_WIDTH // TILE_K
    gl.static_assert(TILE_K == 32)
    gl.static_assert(SELECTED_WIDTH % TILE_K == 0)

    # QK^T is batched over the channel split (batch dim = wave).
    mfma_score: gl.constexpr = gl.amd.cdna4.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[NUM_WARPS, 1, 1],
    )
    # PV splits the output channels across waves.
    mfma_value: gl.constexpr = gl.amd.cdna4.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[1, NUM_WARPS],
    )
    q_dot_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfma_score, k_width=8
    )
    k_dot_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfma_score, k_width=8
    )
    p_dot_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfma_value, k_width=4
    )
    v_dot_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfma_value, k_width=4
    )
    # Reads every wave's partial scores for this wave's SPLIT_H heads with the
    # split axis in registers, so the sum needs no lane shuffles.
    reduce_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[NUM_WARPS, 1, TILE_K * SPLIT_H // 64],
        threads_per_warp=[1, SPLIT_H, 64 // SPLIT_H],
        warps_per_cta=[1, NUM_WARPS, 1],
        order=[2, 1, 0],
    )
    soft_layout: gl.constexpr = gl.SliceLayout(0, reduce_layout)
    # One 1 KiB KV row per async-copy instruction (64 lanes x 16 bytes).
    kv_load_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[8, 1],
        threads_per_warp=[64, 1],
        warps_per_cta=[1, NUM_WARPS],
        order=[0, 1],
    )
    kv_row_layout: gl.constexpr = gl.SliceLayout(0, kv_load_layout)
    index_load_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1],
        threads_per_warp=[64],
        warps_per_cta=[NUM_WARPS],
        order=[0],
    )
    # 512-channel rows padded by 16 elements keep both the K reads and the
    # transposed V reads free of bank conflicts.
    kv_shared_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[HEAD_DIM, 16]],
        [HEAD_DIM, TILE_K],
        [0, 1],
    )
    flat_layout: gl.constexpr = gl.SwizzledSharedLayout(
        vec=1, per_phase=1, max_phase=1, order=[0]
    )
    exchange_layout: gl.constexpr = gl.SwizzledSharedLayout(
        vec=4, per_phase=1, max_phase=8, order=[2, 1, 0]
    )
    p_shared_layout: gl.constexpr = gl.SwizzledSharedLayout(
        vec=4, per_phase=1, max_phase=8, order=[1, 0]
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
    valid_len = gl.load(lens + token_idx).to(tl.int32)
    effective_len = gl.minimum(gl.maximum(valid_len, 0), SELECTED_WIDTH)
    num_tiles = gl.cdiv(effective_len, TILE_K)

    # Stage the token's selected indices in LDS.
    index_groups = gl.allocate_shared_memory(
        gl.int32, [INDEX_GROUPS_POW2, INDEX_GROUP], layout=flat_layout
    )
    group_offsets = gl.arange(0, INDEX_GROUP, layout=index_load_layout)
    for group in gl.static_range((SELECTED_WIDTH + INDEX_GROUP - 1) // INDEX_GROUP):
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            dest=index_groups.index(group),
            ptr=indices,
            offsets=token_idx * SELECTED_WIDTH + group * INDEX_GROUP + group_offsets,
            mask=group * INDEX_GROUP + group_offsets < SELECTED_WIDTH,
        )
    gl.amd.cdna4.async_copy.commit_group()

    # Q[split, head, channel] goes straight into the A-operand registers.
    # q, out, and kv are contiguous with 512-element rows (checked on launch).
    q_split = gl.arange(
        0, NUM_WARPS, layout=gl.SliceLayout(1, gl.SliceLayout(2, q_dot_layout))
    )
    q_heads = head_offset + gl.arange(
        0, BLOCK_H, layout=gl.SliceLayout(0, gl.SliceLayout(2, q_dot_layout))
    )
    q_dims = gl.arange(
        0, SPLIT_D, layout=gl.SliceLayout(0, gl.SliceLayout(1, q_dot_layout))
    )
    q_offsets = (
        (token_idx * num_heads + q_heads[None, :, None]) * HEAD_DIM
        + q_split[:, None, None] * SPLIT_D
        + q_dims[None, None, :]
    )
    q_dot = gl.amd.cdna4.buffer_load(
        ptr=q,
        offsets=q_offsets,
        mask=(q_heads < num_heads)[None, :, None],
        other=0.0,
    )

    # Online softmax in the log2 domain. The sink seeds the running max with a
    # unit weight; denominators stay per column until after the loop.
    soft_heads = head_offset + gl.arange(
        0, BLOCK_H, layout=gl.SliceLayout(1, soft_layout)
    )
    max_value = (
        gl.load(attn_sink + soft_heads, mask=soft_heads < num_heads, other=0.0).to(
            gl.float32
        )
        * 1.4426950408889634
    )
    soft_cols = gl.arange(0, TILE_K, layout=gl.SliceLayout(0, soft_layout))
    row_sums = gl.where(soft_cols[None, :] == 0, 1.0, 0.0) + gl.zeros(
        [BLOCK_H, TILE_K], dtype=gl.float32, layout=soft_layout
    )
    accumulator = gl.zeros([BLOCK_H, HEAD_DIM], dtype=gl.float32, layout=mfma_value)

    # Sanitize the selection once: invalid, padded, and out-of-length slots
    # become _OOB_ROW, so KV copies need no mask (they load zeros) and the
    # scores test one sentinel.
    gl.amd.cdna4.async_copy.wait_group(0)
    for group in gl.static_range((SELECTED_WIDTH + INDEX_GROUP - 1) // INDEX_GROUP):
        rows = index_groups.index(group).load(index_load_layout)
        valid = (
            (group * INDEX_GROUP + group_offsets < effective_len)
            & (rows >= 0)
            & (rows < num_kv_rows)
        )
        index_groups.index(group).store(gl.where(valid, rows, _OOB_ROW))
    row_shared = index_groups.reinterpret(
        gl.int32, [INDEX_GROUPS_POW2 * INDEX_GROUP // TILE_K, TILE_K], flat_layout
    )

    kv_shared = gl.allocate_shared_memory(
        kv.dtype.element_ty, [2, HEAD_DIM, TILE_K], layout=kv_shared_layout
    )
    exchange = gl.allocate_shared_memory(
        gl.float32, [NUM_WARPS, BLOCK_H, TILE_K], layout=exchange_layout
    )
    p_shared = gl.allocate_shared_memory(
        kv.dtype.element_ty, [BLOCK_H, TILE_K], layout=p_shared_layout
    )
    alpha_shared = gl.allocate_shared_memory(gl.float32, [BLOCK_H], layout=flat_layout)
    kv_dims = gl.arange(0, HEAD_DIM, layout=gl.SliceLayout(1, kv_load_layout))

    # Double-buffered KV tiles: the next tile's copy overlaps this tile.
    rows = row_shared.index(0).load(kv_row_layout)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        dest=kv_shared.index(0),
        ptr=kv,
        offsets=rows[None, :] * HEAD_DIM + kv_dims[:, None],
    )
    gl.amd.cdna4.async_copy.commit_group()
    for tile_idx in range(num_tiles):
        rows = row_shared.index(gl.minimum(tile_idx + 1, NUM_INDEX_TILES - 1)).load(
            kv_row_layout
        )
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            dest=kv_shared.index((tile_idx + 1) % 2),
            ptr=kv,
            offsets=rows[None, :] * HEAD_DIM + kv_dims[:, None],
        )
        gl.amd.cdna4.async_copy.commit_group()
        score_rows = row_shared.index(tile_idx).load(gl.SliceLayout(0, soft_layout))

        gl.amd.cdna4.async_copy.wait_group(1)
        current_kv = kv_shared.index(tile_idx % 2)
        k_dot = current_kv.reshape([NUM_WARPS, SPLIT_D, TILE_K]).load(k_dot_layout)
        partial = gl.amd.cdna4.mfma(
            q_dot,
            k_dot,
            gl.zeros([NUM_WARPS, BLOCK_H, TILE_K], dtype=gl.float32, layout=mfma_score),
        )
        exchange.store(partial)
        scores = gl.sum(exchange.load(reduce_layout), axis=0)
        scores = gl.where(
            (score_rows != _OOB_ROW)[None, :], scores * qk_scale_log2, -float("inf")
        )
        next_max = gl.maximum(max_value, gl.max(scores, axis=1))
        safe_max = gl.where(next_max > -float("inf"), next_max, 0.0)
        alpha = gl.exp2(max_value - safe_max)
        max_value = next_max
        probabilities = gl.exp2(scores - safe_max[:, None])
        row_sums = row_sums * alpha[:, None] + probabilities
        p_shared.store(probabilities.to(kv.dtype.element_ty))
        alpha_shared.store(alpha)
        p_dot = p_shared.load(p_dot_layout)
        accumulator *= alpha_shared.load(gl.SliceLayout(1, mfma_value))[:, None]
        v_dot = current_kv.permute([1, 0]).load(v_dot_layout)
        accumulator = gl.amd.cdna4.mfma(p_dot, v_dot, accumulator)
    gl.amd.cdna4.async_copy.wait_group(0)

    denominator = gl.convert_layout(
        gl.sum(row_sums, axis=1), gl.SliceLayout(1, mfma_value)
    )
    safe_denominator = gl.where(denominator > 0.0, denominator, 1.0)
    accumulator = gl.where(
        denominator[:, None] > 0.0, accumulator / safe_denominator[:, None], 0.0
    )

    out_heads = head_offset + gl.arange(
        0, BLOCK_H, layout=gl.SliceLayout(1, mfma_value)
    )
    out_dims = gl.arange(0, HEAD_DIM, layout=gl.SliceLayout(0, mfma_value))
    out_offsets = (token_idx * num_heads + out_heads[:, None]) * HEAD_DIM + out_dims[
        None, :
    ]
    gl.amd.cdna4.buffer_store(
        stored_value=accumulator.to(out.dtype.element_ty),
        ptr=out,
        offsets=out_offsets,
        mask=(out_heads < num_heads)[:, None],
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


def launch_gluon_dsv4_prefill_gfx950(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    lens: torch.Tensor,
    attn_sink: torch.Tensor,
    softmax_scale: float,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run dense-workspace selected attention for DeepSeek V4 on GFX950.

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

    if _use_sparse_prefill(q, indices):
        return gluon_dsv4_sparse_prefill_gfx950(
            q=q,
            kv=kv,
            indices=indices,
            lens=lens,
            attn_sink=attn_sink,
            softmax_scale=scale,
            out=output,
        )

    kv_rows = kv.reshape(-1, 512)
    width = indices.shape[1]
    if width % _TILE_K != 0:
        raise ValueError(f"selected width must be a multiple of {_TILE_K}, got {width}")
    # Buffer offsets are 32-bit byte offsets, and _OOB_ROW must lie past kv.
    if kv_rows.shape[0] >= _OOB_ROW.value or q.numel() * q.element_size() >= 2**31:
        raise ValueError("q and kv must each be smaller than 2 GiB")
    grid = (q.shape[0], triton.cdiv(q.shape[1], _BLOCK_H))
    gluon_dsv4_prefill_gfx950[grid](
        q,
        kv_rows,
        indices,
        lens,
        attn_sink.reshape(-1),
        output,
        scale * _LOG2E,
        q.shape[1],
        kv_rows.shape[0],
        SELECTED_WIDTH=width,
        BLOCK_H=_BLOCK_H,
        TILE_K=_TILE_K,
        HEAD_DIM=512,
        INDEX_GROUPS_POW2=triton.next_power_of_2(triton.cdiv(width, 64 * _NUM_WARPS)),
        NUM_XCDS=_NUM_XCDS,
        NUM_WARPS=_NUM_WARPS,
        num_warps=_NUM_WARPS,
    )
    return output
