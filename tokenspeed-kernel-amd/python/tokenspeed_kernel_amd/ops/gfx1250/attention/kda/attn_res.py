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

"""Fused Kimi K3 AttnRes mixing and output RMSNorm for gfx1250."""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._triton import gl, gluon, tl, triton

cdna5 = gl.amd.cdna5

# Hidden sizes up to 8192 fit one padded row. BF16 rows are read as 32-bit
# words: each word holds an (even, odd) element pair, which converts to FP32
# with one shift or mask per element. Each lane owns 4 words (one 128-bit load)
# per 128 * NUM_WARPS words.
_BLOCK_P = gl.constexpr(4096)
_WORDS = gl.constexpr(4)


@gluon.jit
def _add_pair(a0, a1, b0, b1):
    return a0 + b0, a1 + b1


@gluon.jit
def _unpack(words):
    """Split BF16 pairs held in int32 words into (even, odd) FP32 values."""
    even = (words << 16).to(gl.float32, bitcast=True)
    odd = (words & -65536).to(gl.float32, bitcast=True)
    return even, odd


@gluon.jit
def _pack(even, odd):
    """Round (even, odd) FP32 values to BF16 pairs packed in int32 words."""
    even = even.to(gl.bfloat16).to(gl.int16, bitcast=True).to(gl.int32) & 65535
    odd = odd.to(gl.bfloat16).to(gl.int16, bitcast=True).to(gl.int32) << 16
    return even | odd


@gluon.jit
def _load_pairs(ptr, offsets, mask):
    """Load BF16 elements (2p, 2p + 1) for each pair index p as (even, odd) FP32."""
    gl.static_assert(ptr.dtype.element_ty == gl.bfloat16, "pairs are BF16")
    words = cdna5.buffer_load(
        ptr.to(gl.pointer_type(gl.int32)), offsets.to(gl.int32), mask=mask, other=0
    )
    return _unpack(words)


@gluon.jit
def _store_pairs(ptr, offsets, mask, even, odd):
    """Store (even, odd) FP32 values as BF16 pairs to a BF16 tensor."""
    cdna5.buffer_store(
        _pack(even, odd),
        ptr.to(gl.pointer_type(gl.int32)),
        offsets.to(gl.int32),
        mask=mask,
    )


@gluon.jit
def _load_snapshot(
    words,
    row: gl.constexpr,
    token_offsets,
    pair_mask,
    stride_row,
    NUM_BLOCKS: gl.constexpr,
):
    """Load one snapshot row of words; rows past NUM_BLOCKS are zero."""
    if row < NUM_BLOCKS:
        # A 64-bit scalar row base keeps the 32-bit buffer offsets small.
        return cdna5.buffer_load(
            words + row * stride_row, token_offsets, mask=pair_mask, other=0
        )
    else:
        return gl.zeros_like(token_offsets)


@gluon.jit
def _join_rows(
    words,
    token_offsets,
    pair_mask,
    stride_row,
    NUM_BLOCKS: gl.constexpr,
    ROWS: gl.constexpr,
    base: gl.constexpr,
    step: gl.constexpr,
):
    """Join rows base, base + step, ... (ROWS of them) into [P, 2, ..., 2].

    Rows pair in bit-reversed order so a row-major reshape of the trailing
    dims yields ascending rows.
    """
    if ROWS == 1:
        return _load_snapshot(
            words, base, token_offsets, pair_mask, stride_row, NUM_BLOCKS
        )
    else:
        HALF: gl.constexpr = ROWS // 2
        low = _join_rows(
            words,
            token_offsets,
            pair_mask,
            stride_row,
            NUM_BLOCKS,
            HALF,
            base,
            step * 2,
        )
        high = _join_rows(
            words,
            token_offsets,
            pair_mask,
            stride_row,
            NUM_BLOCKS,
            HALF,
            base + step,
            step * 2,
        )
        return gl.join(low, high)


def _launch_metadata(grid, kernel, args):
    """Report the modeled HBM traffic of one AttnRes launch to Proton.

    Every token reads its N candidates (plus delta, read and written back
    when present), optionally writes the new snapshot, and writes the output;
    the three weight vectors are read once.
    """
    tokens = grid[0]
    hidden = args["H"]
    row_reads = args["N"] + (1 if args["HAS_DELTA"] else 0)
    row_writes = 1 + (1 if args["HAS_DELTA"] else 0) + (1 if args["WRITE_BLOCK"] else 0)
    return {
        "name": f"{kernel.name}_n{args['N']}",
        "flops": tokens * hidden * (6 * args["N"] + 6),
        "bytes": 2 * hidden * (tokens * (row_reads + row_writes) + 3),
    }


@gluon.jit(launch_metadata=_launch_metadata)
def gluon_attn_res_fwd_gfx1250(
    layer_residual,
    delta,
    block_residual,
    res_weight,
    score_rms_weight,
    output_rms_weight,
    output,
    stride_layer_t: gl.constexpr,
    stride_delta_t: gl.constexpr,
    stride_block_t: gl.constexpr,
    stride_block_n: tl.int64,
    stride_output_t: gl.constexpr,
    H: gl.constexpr,
    N: gl.constexpr,
    BLOCK_ROWS: gl.constexpr,
    BLOCK_WRITE_IDX: gl.constexpr,
    HAS_DELTA: gl.constexpr,
    WRITE_BLOCK: gl.constexpr,
    SCORE_EPS: gl.constexpr,
    OUTPUT_EPS: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    # One workgroup per token. The N - 1 snapshots form one [BLOCK_ROWS, H/2]
    # tile of BF16 pairs whose rows all live in the same lane: every load is
    # issued up front, one reduction scores all rows, and the softmax mix is
    # lane-local. A spare tile row, when there is one, holds the current
    # residual so it is scored by the same reduction.
    token = gl.program_id(0)
    tile_layout: gl.constexpr = gl.BlockedLayout(
        [BLOCK_ROWS, _WORDS], [1, 32], [1, NUM_WARPS], [1, 0]
    )
    pair_layout: gl.constexpr = gl.SliceLayout(0, tile_layout)
    row_layout: gl.constexpr = gl.SliceLayout(1, tile_layout)
    pairs = gl.arange(0, _BLOCK_P, layout=pair_layout)
    pair_mask = pairs < H // 2
    NUM_BLOCKS: gl.constexpr = N - 1
    PREFIX_IN_TILE: gl.constexpr = NUM_BLOCKS < BLOCK_ROWS

    # Issue every load before the first store: the compiler cannot move a
    # load above a store that may alias it, and each dependent round trip
    # costs a full memory latency in this latency-bound kernel.
    if NUM_BLOCKS > 0:
        rows = gl.arange(0, BLOCK_ROWS, layout=row_layout)
        words = _join_rows(
            block_residual.to(gl.pointer_type(gl.int32)),
            (token * (stride_block_t // 2) + pairs).to(gl.int32),
            pair_mask,
            stride_block_n.to(gl.int64) // 2,
            NUM_BLOCKS,
            BLOCK_ROWS,
            0,
            1,
        )
        if BLOCK_ROWS == 1:
            words = gl.expand_dims(words, 0)
        else:
            words = gl.permute(gl.reshape(words, [_BLOCK_P, BLOCK_ROWS]), [1, 0])
        words = gl.convert_layout(words, tile_layout, assert_trivial=True)
        res_even, res_odd = _load_pairs(res_weight, pairs, pair_mask)
        score_even, score_odd = _load_pairs(score_rms_weight, pairs, pair_mask)
    prefix_words = cdna5.buffer_load(
        layer_residual.to(gl.pointer_type(gl.int32)),
        (token * (stride_layer_t // 2) + pairs).to(gl.int32),
        mask=pair_mask,
        other=0,
    )
    if HAS_DELTA:
        delta_words = cdna5.buffer_load(
            delta.to(gl.pointer_type(gl.int32)),
            (token * (stride_delta_t // 2) + pairs).to(gl.int32),
            mask=pair_mask,
            other=0,
        )
    out_even, out_odd = _load_pairs(output_rms_weight, pairs, pair_mask)

    prefix_even, prefix_odd = _unpack(prefix_words)
    if HAS_DELTA:
        delta_even, delta_odd = _unpack(delta_words)
        prefix_even = (prefix_even + delta_even).to(gl.bfloat16).to(gl.float32)
        prefix_odd = (prefix_odd + delta_odd).to(gl.bfloat16).to(gl.float32)
        _store_pairs(
            layer_residual,
            token * (stride_layer_t // 2) + pairs,
            pair_mask,
            prefix_even,
            prefix_odd,
        )
    if WRITE_BLOCK:
        _store_pairs(
            block_residual + BLOCK_WRITE_IDX * stride_block_n.to(gl.int64),
            token * (stride_block_t // 2) + pairs,
            pair_mask,
            prefix_even,
            prefix_odd,
        )

    if NUM_BLOCKS == 0:
        mixed_even = prefix_even
        mixed_odd = prefix_odd
    else:
        even, odd = _unpack(words)
        if PREFIX_IN_TILE:
            is_prefix = gl.expand_dims(rows == NUM_BLOCKS, 1)
            even = gl.where(is_prefix, gl.expand_dims(prefix_even, 0), even)
            odd = gl.where(is_prefix, gl.expand_dims(prefix_odd, 0), odd)
        scorer_even = gl.expand_dims(res_even * score_even, 0)
        scorer_odd = gl.expand_dims(res_odd * score_odd, 0)
        square_sums, dots = gl.reduce(
            (even * even + odd * odd, even * scorer_even + odd * scorer_odd),
            1,
            _add_pair,
        )
        scores = dots * gl.rsqrt(square_sums * (1.0 / H) + SCORE_EPS)
        scores = gl.where(rows < NUM_BLOCKS + PREFIX_IN_TILE, scores, -float("inf"))
        max_score = gl.max(scores, axis=0)
        if not PREFIX_IN_TILE:
            prefix_square_sum, prefix_dot = gl.reduce(
                (
                    prefix_even * prefix_even + prefix_odd * prefix_odd,
                    prefix_even * (res_even * score_even)
                    + prefix_odd * (res_odd * score_odd),
                ),
                0,
                _add_pair,
            )
            prefix_score = prefix_dot * gl.rsqrt(
                prefix_square_sum * (1.0 / H) + SCORE_EPS
            )
            max_score = gl.maximum(max_score, prefix_score)
            prefix_weight = gl.exp(prefix_score - max_score)
        weights = gl.exp(scores - max_score)
        # Normalize the few weights rather than every mixed element.
        denominator = gl.sum(weights, axis=0)
        if not PREFIX_IN_TILE:
            denominator += prefix_weight
        inverse_denominator = 1.0 / denominator
        weights = gl.expand_dims(weights * inverse_denominator, 1)
        mixed_even = gl.sum(weights * even, axis=0)
        mixed_odd = gl.sum(weights * odd, axis=0)
        if not PREFIX_IN_TILE:
            prefix_weight *= inverse_denominator
            mixed_even += prefix_weight * prefix_even
            mixed_odd += prefix_weight * prefix_odd

    # Preserve the AttnRes BF16 boundary before output RMSNorm.
    mixed_even = mixed_even.to(gl.bfloat16).to(gl.float32)
    mixed_odd = mixed_odd.to(gl.bfloat16).to(gl.float32)
    inverse_rms = gl.rsqrt(
        gl.sum(mixed_even * mixed_even + mixed_odd * mixed_odd, axis=0) * (1.0 / H)
        + OUTPUT_EPS
    )
    _store_pairs(
        output,
        token * (stride_output_t // 2) + pairs,
        pair_mask,
        mixed_even * inverse_rms * out_even,
        mixed_odd * inverse_rms * out_odd,
    )


def attn_res_rmsnorm_gfx1250(
    *,
    layer_residual: torch.Tensor,
    block_residual: torch.Tensor,
    res_weight: torch.Tensor,
    score_rms_weight: torch.Tensor,
    score_eps: float,
    output_rms_weight: torch.Tensor,
    output_eps: float,
    num_valid_blocks: int,
    delta: torch.Tensor | None = None,
    block_write_idx: int = -1,
) -> torch.Tensor:
    """Mix AttnRes candidates and apply the following RMSNorm in one launch."""
    tokens, hidden = layer_residual.shape
    if layer_residual.dtype != torch.bfloat16 or hidden not in {
        4096,
        5120,
        6144,
        7168,
        8192,
    }:
        raise ValueError(
            "gfx1250 AttnRes requires BF16 input with "
            "H in {4096, 5120, 6144, 7168, 8192}"
        )
    if not 0 <= num_valid_blocks <= 11:
        raise ValueError("gfx1250 AttnRes supports at most 11 block snapshots")
    if (
        block_residual.ndim != 3
        or block_residual.shape[0] != tokens
        or block_residual.shape[2] != hidden
    ):
        raise ValueError(
            "gfx1250 AttnRes requires token-major block storage [T, blocks, H]"
        )
    if block_residual.shape[1] < num_valid_blocks:
        raise ValueError("num_valid_blocks is outside block_residual")
    if block_write_idx >= 0 and (
        block_write_idx != num_valid_blocks
        or block_write_idx >= block_residual.shape[1]
    ):
        raise ValueError("block_write_idx must append within block_residual")
    if delta is not None and (
        delta.shape != layer_residual.shape
        or delta.dtype != layer_residual.dtype
        or delta.device != layer_residual.device
        or delta.stride(1) != 1
    ):
        raise ValueError("gfx1250 AttnRes delta must match layer_residual")
    if layer_residual.stride(1) != 1 or block_residual.stride(2) != 1:
        raise ValueError("gfx1250 AttnRes requires a contiguous hidden dimension")
    weights = (res_weight, score_rms_weight, output_rms_weight)
    if any(
        w.dtype != torch.bfloat16 or w.shape != (hidden,) or w.stride(0) != 1
        for w in weights
    ):
        raise ValueError("gfx1250 AttnRes weights must be contiguous BF16 [H]")
    delta_tensor = layer_residual if delta is None else delta
    pair_views = (layer_residual, delta_tensor, block_residual, *weights)
    if any(t.data_ptr() % 4 for t in pair_views) or any(
        stride % 2 for t in pair_views for stride in t.stride()[:-1]
    ):
        raise ValueError("gfx1250 AttnRes reads BF16 rows as 4-byte pairs")

    output = torch.empty_like(layer_residual)
    num_warps = _num_warps(tokens, num_valid_blocks)
    gluon_attn_res_fwd_gfx1250[(tokens,)](
        layer_residual,
        delta_tensor,
        block_residual,
        res_weight,
        score_rms_weight,
        output_rms_weight,
        output,
        stride_layer_t=layer_residual.stride(0),
        stride_delta_t=delta_tensor.stride(0),
        stride_block_t=block_residual.stride(0),
        stride_block_n=block_residual.stride(1),
        stride_output_t=output.stride(0),
        H=hidden,
        N=num_valid_blocks + 1,
        BLOCK_ROWS=triton.next_power_of_2(max(num_valid_blocks, 1)),
        BLOCK_WRITE_IDX=max(block_write_idx, 0),
        HAS_DELTA=delta is not None,
        WRITE_BLOCK=block_write_idx >= 0,
        SCORE_EPS=score_eps,
        OUTPUT_EPS=output_eps,
        NUM_WARPS=num_warps,
        num_warps=num_warps,
    )
    return output


def _num_warps(tokens: int, num_valid_blocks: int) -> int:
    """Pick the workgroup size; every token runs on one workgroup.

    Long prefills are bandwidth-bound: four warps (one per SIMD) give each
    lane a wide register tile, so loads stay in flight without cross-warp
    reductions. Decode grids leave most WGPs idle and each workgroup is
    VALU-bound: eight warps suit small batches and small tiles, while wider
    tiles at larger batches spread over sixteen warps.
    """
    if tokens >= 256:
        return 4
    if tokens <= 16 or num_valid_blocks <= 4:
        return 8
    return 16


__all__ = ["attn_res_rmsnorm_gfx1250"]
