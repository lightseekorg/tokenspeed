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

"""NPU DSpark sampling: resolve once, reduce INT32 keys before packing."""

from tokenspeed_kernel_npu._triton import tl, triton

_INT64_MIN = tl.constexpr(-(1 << 63))
_LOW32 = tl.constexpr(0xFFFFFFFF)


@triton.jit
def _orderable_key(values):
    bits = values.to(tl.int32, bitcast=True)
    return tl.where(bits < 0, bits ^ 0x7FFFFFFF, bits)


@triton.jit
def _score_tiles(
    base_ptr,
    base_row_stride,
    anchor_ptr,
    anchor_stride,
    embedding_ptr,
    projection_ptr,
    partials_ptr,
    rows,
    num_valid,
    vocab_start,
    vocab_size,
    RANK: tl.constexpr,
    RANK_PAD: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK_V: tl.constexpr,
    SUB_V: tl.constexpr,
    N_TILES: tl.constexpr,
):
    tile = tl.program_id(0)
    row_offsets = tl.program_id(1) * ROWS + tl.arange(0, ROWS)
    row_mask = row_offsets < rows
    previous = tl.load(
        anchor_ptr + row_offsets * anchor_stride, mask=row_mask, other=0
    ).to(tl.int32)
    # Graph padding rows carry arbitrary anchors; keep their gathers in bounds.
    previous = tl.minimum(tl.maximum(previous, 0), vocab_size - 1)

    rank_offsets = tl.arange(0, RANK_PAD)
    rank_mask = rank_offsets < RANK
    bigram = tl.load(
        embedding_ptr + previous.to(tl.int64)[:, None] * RANK + rank_offsets[None, :],
        mask=row_mask[:, None] & rank_mask[None, :],
        other=0.0,
    )
    best = tl.full((ROWS,), -2147483648, tl.int32)
    best_id = tl.full((ROWS,), 2147483647, tl.int32)
    for sub in tl.static_range(0, BLOCK_V, SUB_V):
        vocab_offsets = tile * BLOCK_V + sub + tl.arange(0, SUB_V)
        vocab_mask = vocab_offsets < num_valid
        weight = tl.load(
            projection_ptr
            + vocab_offsets.to(tl.int64)[:, None] * RANK
            + rank_offsets[None, :],
            mask=vocab_mask[:, None] & rank_mask[None, :],
            other=0.0,
        )
        bias = tl.dot(bigram, tl.trans(weight))
        base = tl.load(
            base_ptr + row_offsets[:, None] * base_row_stride + vocab_offsets[None, :],
            mask=row_mask[:, None] & vocab_mask[None, :],
            other=float("-inf"),
        )
        values = base + bias
        # Compare signed FP32 bit ordering before packing, preserving +/-zero.
        ordered = _orderable_key(values)
        ordered = tl.where(vocab_mask[None, :], ordered, -2147483648)
        max_key = tl.max(ordered, axis=1)
        token = tl.min(
            tl.where(
                vocab_mask[None, :] & (ordered == max_key[:, None]),
                vocab_start + vocab_offsets[None, :],
                2147483647,
            ),
            axis=1,
        )
        prior = best
        take = (max_key > prior) | ((max_key == prior) & (token < best_id))
        best = tl.where(take, max_key, best)
        best_id = tl.where(take, token, best_id)
    packed = (best.to(tl.int64) << 32) | (_LOW32 - best_id.to(tl.int64))
    packed = tl.where(best_id == 2147483647, _INT64_MIN, packed)
    tl.store(partials_ptr + row_offsets * N_TILES + tile, packed, mask=row_mask)


@triton.jit
def _resolve_candidates(
    C, O, ROWS, STRIDE, TP: tl.constexpr, TILES: tl.constexpr, B: tl.constexpr
):
    row = tl.program_id(0)
    i = tl.arange(0, B)
    value = tl.load(
        C + (i // TILES * ROWS + row) * TILES + i % TILES,
        i < TP * TILES,
        other=_INT64_MIN,
    )
    maximum = tl.max(value, 0)
    tl.store(O + row * STRIDE, (_LOW32 - (maximum & _LOW32)).to(tl.int32))


def resolve(candidates, output, step):
    """Write the exact packed-key winner of every row, including tie order."""
    tp, rows, tiles = candidates.shape
    if not rows:
        return
    _resolve_candidates[(rows,)](
        candidates,
        output[:, step],
        rows,
        output.stride(0),
        tp,
        tiles,
        triton.next_power_of_2(tp * tiles),
        multibuffer=False,
    )


def score_step(
    base_logits,
    step,
    anchor_ids,
    candidates,
    embedding,
    projection,
    vocab_start,
    num_valid,
    partials,
    output,
):
    """Score validated tensors using BF16 products and FP32 accumulation.

    Resolving the previous step is independent of the local vocabulary tile.
    Do it once per row, rather than repeating a TP-wide INT64 reduction in
    every matrix tile. Reduce score keys as INT32 and pack only tile winners;
    the final packed key and its tie-breaking contract remain unchanged.
    """
    rows = base_logits.shape[0]
    if not rows:
        return
    if step:
        resolve(candidates, output, step - 1)
        anchor_ids = output[:, step - 1]
    rank = embedding.shape[1]
    tiles = partials.shape[1]
    _score_tiles[(tiles, triton.cdiv(rows, 16))](
        base_logits[:, step],
        base_logits.stride(0),
        anchor_ids,
        anchor_ids.stride(0),
        embedding,
        projection,
        partials,
        rows,
        num_valid,
        vocab_start,
        embedding.shape[0],
        RANK=rank,
        RANK_PAD=max(16, triton.next_power_of_2(rank)),
        ROWS=16,
        BLOCK_V=256,
        SUB_V=256,
        N_TILES=tiles,
        num_warps=4,
        multibuffer=False,
    )
