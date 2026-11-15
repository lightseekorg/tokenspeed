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

"""One launch per round of DSpark's greedy proposal walk.

A round proposes block position k from the hoisted base logits of position
k-1 plus the Markov bias of the token proposed at k-1. Each rank packs its
vocab shard's best logit and token id into one int64 per row, so a single
all-gather of those words lets every rank, in the next round's launch, decode
the winner that the walk then conditions on. The words order by BF16 logit,
then by lower token id, as the reference's first-index max does.
"""

from __future__ import annotations

import torch

from tokenspeed.runtime.utils.triton import tl, triton

_BLOCK_V = 64
_BLOCK_K = 128


@triton.jit(do_not_specialize=["rows", "column", "vocab_start"])
def _dspark_walk_round_kernel(
    base_ptr,  # [rows, num_org] BF16: this round's hoisted base logits
    latent_table_ptr,  # [markov_vocab, RANK]: the Markov embedding
    bias_weight_ptr,  # [num_org, RANK]: this rank's rows of markov_w2
    tokens_ptr,  # [rows, tokens_stride] int32 proposal block
    packed_ptr,  # [rows] int64, zeroed: this round's shard winners
    gathered_ptr,  # [tp, rows] int64: the previous round's gathered winners
    rows,
    num_org,
    vocab_start,
    markov_vocab,
    tp,
    tokens_stride,
    column,
    RANK: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_K: tl.constexpr,
    FROM_GATHER: tl.constexpr,
    FINISH: tl.constexpr,
):
    row = tl.arange(0, ROWS)
    row_ok = row < rows
    low_bits = tl.full([], 0xFFFFFFFF, tl.int64)
    if FROM_GATHER:
        # Every program decodes the previous round's winner itself; the first
        # writes it into the block.
        best = tl.zeros([ROWS], dtype=tl.int64)
        for t in range(tp):
            word = tl.load(gathered_ptr + t * rows + row, mask=row_ok, other=0)
            best = tl.maximum(best, word)
        prev = (low_bits - (best & low_bits)).to(tl.int32)
        if tl.program_id(0) == 0:
            tl.store(tokens_ptr + row * tokens_stride + column - 1, prev, mask=row_ok)
    else:
        prev = tl.load(
            tokens_ptr + row * tokens_stride + column - 1, mask=row_ok, other=0
        )
    if not FINISH:
        # The anchor is the target's last output (garbage in warmup), so the
        # embedding row must be clamped into the table.
        prev = tl.minimum(tl.maximum(prev, 0), markov_vocab - 1).to(tl.int64)
        vocab = tl.program_id(0) * BLOCK_V + tl.arange(0, BLOCK_V)
        vocab_ok = vocab < num_org
        acc = tl.zeros([ROWS, BLOCK_V], dtype=tl.float32)
        for k0 in range(0, RANK, BLOCK_K):
            k = k0 + tl.arange(0, BLOCK_K)
            latent = tl.load(
                latent_table_ptr + prev[:, None] * RANK + k[None, :],
                mask=row_ok[:, None],
                other=0.0,
            )
            weight = tl.load(
                bias_weight_ptr + vocab[:, None].to(tl.int64) * RANK + k[None, :],
                mask=vocab_ok[:, None],
                other=0.0,
            )
            acc = tl.dot(latent, tl.trans(weight), acc)
        # Rounded as the reference rounds: the bias GEMM's output, then the sum.
        bias = acc.to(tl.bfloat16).to(tl.float32)
        valid = row_ok[:, None] & vocab_ok[None, :]
        base = tl.load(
            base_ptr + row[:, None] * num_org + vocab[None, :], mask=valid, other=0.0
        )
        logit = (base.to(tl.float32) + bias).to(tl.bfloat16)
        # BF16 bits mapped to an order-preserving unsigned key.
        bits = logit.to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF
        key = tl.where(bits >= 0x8000, 0xFFFF - bits, bits + 0x8000)
        word = (key.to(tl.int64) << 32) | (
            low_bits - (vocab_start + vocab).to(tl.int64)
        )
        word = tl.where(valid, word, 0)
        tl.atomic_max(
            packed_ptr + row, tl.max(word, axis=1), mask=row_ok, sem="relaxed"
        )


def dspark_walk_round(
    base_logits: torch.Tensor,
    latent_table: torch.Tensor,
    bias_weight: torch.Tensor,
    tokens: torch.Tensor,
    packed: torch.Tensor,
    gathered: torch.Tensor | None,
    *,
    column: int,
    vocab_start: int,
    tp: int,
) -> None:
    """Propose ``tokens[:, column]``'s shard winner into ``packed``.

    ``gathered`` holds the previous round's ``tp`` gathered words, rank-major;
    None reads the previous token from ``tokens[:, column - 1]`` instead.
    ``packed`` must be zero on entry.
    """
    rows, num_org = base_logits.shape
    rank = int(latent_table.shape[1])
    _dspark_walk_round_kernel[(triton.cdiv(num_org, _BLOCK_V),)](
        base_logits,
        latent_table,
        bias_weight,
        tokens,
        packed,
        packed if gathered is None else gathered,
        rows,
        num_org,
        vocab_start,
        int(latent_table.shape[0]),
        tp,
        tokens.stride(0),
        column,
        RANK=rank,
        ROWS=max(16, triton.next_power_of_2(rows)),
        BLOCK_V=_BLOCK_V,
        BLOCK_K=min(_BLOCK_K, rank),
        FROM_GATHER=gathered is not None,
        FINISH=False,
        num_warps=4,
    )


def dspark_walk_finish(
    tokens: torch.Tensor, gathered: torch.Tensor, *, column: int, tp: int
) -> None:
    """Write the last round's winner, decoded from ``gathered``, into
    ``tokens[:, column]``."""
    rows = int(tokens.shape[0])
    _dspark_walk_round_kernel[(1,)](
        tokens,
        tokens,
        tokens,
        tokens,
        gathered,
        gathered,
        rows,
        0,
        0,
        1,
        tp,
        tokens.stride(0),
        column + 1,
        RANK=16,
        ROWS=max(16, triton.next_power_of_2(rows)),
        BLOCK_V=_BLOCK_V,
        BLOCK_K=16,
        FROM_GATHER=True,
        FINISH=True,
        num_warps=1,
    )
